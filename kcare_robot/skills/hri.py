"""Human-robot interaction skills: ``reply``, ``ask``, ``qa``, ``announce`` and ``wait``.

All talk and listen on one side, chosen per call with ``source``:

* ``'robot'`` (default) — the robot's own speaker and microphone. Speech goes
  out through gTTS; the answer is captured with energy-based voice activity
  detection and transcribed by the ``vlms`` Whisper server. Works from every
  entry point: dashboard, CLI, HTTP, Python API.
* ``'dashboard'`` — the browser the operator is at: it voices the robot's lines
  and captures the answer with the Web Speech API. Needs a run started from the
  dashboard's Agent panel, which is the only channel back to the browser.

Listening ends ``silence_sec`` after the user stops talking, or at ``max_sec``.
Nobody speaking at all counts as no answer.

Usage:
    reply                                     # listen, then "들었어요"
    reply::need_confirm=True                  # ..., then '"<text>"라고 들었어요'
    ask::'어떤 음료 드릴까요?'                   # ask, then repeat the answer back
    ask::inputs='어떤 음료 드릴까요?', options=['물', '주스', '커피']
    ask::inputs='어디로 갈까요?', options="식탁 앞->table@kitchen, 옷방->dressroom"
    ask::inputs='Which drink?', lang='en', source='dashboard', options='water,juice'
    qa                                        # Q&A about what the head camera sees
    qa::lang='vi', input='text'               # questions typed on the dashboard
"""

import re
import unicodedata

from robot_agent.skill_configs import HRI_CONFIGS
from robot_agent.skills import log_data
from robot_agent.utils import (
    exception_handler, listen_dashboard, play_audio, record_phrase, say_to_user,
    speech_to_text,
)

SOURCES = ('robot', 'dashboard')
MAX_SEC = 8.0             # longest phrase to capture (s)
SILENCE_SEC = 1.5         # quiet after speech that ends the phrase (s)
ENERGY_THRESHOLD = 800    # robot-mic VAD level; what this robot's mic loop ran with
MATCH_THRESHOLD = 0.7     # fuzzy-match floor for `ask` options (edit similarity)
RETRIES = 2               # attempts after the first, for `ask`
# The browser voices the prompt before it opens the mic, so a dashboard listen
# needs headroom beyond the phrase itself.
DASHBOARD_SLACK_SEC = 20.0
INPUTS = ('voice', 'text')
# input='text': the answer is typed on the dashboard, which takes far longer
# than saying it — one wait covers a typed question.
TEXT_WAIT_SEC = 300.0
QA_TEXT_WAIT_SEC = 24 * 3600.0  # qa: a typed question is waited for until it comes or Cancel

_PHRASES = {
    'heard':      {'ko': '들었어요',
                   'en': 'Got it.',
                   'vi': 'Đã nghe.'},
    'heard_text': {'ko': '"{text}"라고 들었어요',
                   'en': 'I heard: {text}',
                   'vi': 'Đã nghe: {text}'},
    # Said just before the recording itself is played back (need_confirm).
    'heard_audio':{'ko': '이렇게 들었어요',
                   'en': 'This is what I heard:',
                   'vi': 'Tôi đã nghe thế này:'},
    'not_heard':  {'ko': '잘 못 들었어요',
                   'en': "Sorry, I didn't catch that.",
                   'vi': 'Xin lỗi, tôi chưa nghe rõ.'},
    'choose':     {'ko': '{options} 중에서 골라 주세요',
                   'en': 'Please choose one of: {options}',
                   'vi': 'Hãy chọn một trong: {options}'},
    'confirm':    {'ko': '{choice}, 알겠습니다',
                   'en': '{choice}, confirmed.',
                   'vi': 'Đã chọn {choice}.'},
}


def _phrase(key: str, lang: str, **kw) -> str:
    table = _PHRASES[key]
    return table.get(lang, table['en']).format(**kw)


def _truthy(v) -> bool:
    """Plan and CLI arguments arrive as strings as often as booleans."""
    if isinstance(v, str):
        return v.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
    return bool(v)


def _hri_cfg() -> dict:
    """HRI_CONFIGS (Global Configs) with defaults filled in."""
    cfg = {'source': 'robot', 'dashboard_stt': 'whisper', 'stt': '', 'stt_hint': {}}
    try:
        cfg.update(dict(HRI_CONFIGS.items()))
    except Exception:
        pass
    return cfg


def _common(kwargs: dict):
    hri = _hri_cfg()
    lang = str(kwargs.pop('lang', 'ko') or 'ko')
    source = str(kwargs.pop('source', None) or hri['source'] or 'dashboard').lower()
    if source not in SOURCES:
        raise ValueError(f'source must be one of {SOURCES}, got {source!r}')
    mode = str(kwargs.pop('input', 'voice') or 'voice').lower()
    if mode not in INPUTS:
        raise ValueError(f'input must be one of {INPUTS}, got {mode!r}')
    if mode == 'text' and source != 'dashboard':
        raise ValueError("input='text' is typed on the dashboard: use source='dashboard'")
    listen = {
        'mode': mode,
        'text_wait_sec': float(kwargs.pop('text_wait_sec', TEXT_WAIT_SEC)),
        'max_sec': float(kwargs.pop('max_sec', MAX_SEC)),
        'silence_sec': float(kwargs.pop('silence_sec', SILENCE_SEC)),
        'energy_threshold': float(kwargs.pop('energy_threshold', ENERGY_THRESHOLD)),
        # Dashboard mic: 'whisper' (browser records, the stt connection
        # transcribes) or 'browser' (the browser's recogniser).
        'capture': str(kwargs.pop('stt', None) or hri['dashboard_stt'] or 'whisper').lower(),
        'stt': str(hri.get('stt') or '') or None,
        'hint': kwargs.pop('stt_hint', None) or _hint_for(hri.get('stt_hint'), lang),
    }
    if listen['capture'] not in ('whisper', 'browser'):
        raise ValueError(f"stt must be 'whisper' or 'browser', got {listen['capture']!r}")
    return lang, source, listen


def _hint_for(hints, lang: str):
    """Whisper hint words for `lang`: stt_hint is a {lang: text} dict or one text."""
    if isinstance(hints, dict):
        return str(hints.get(lang) or '').strip() or None
    return str(hints or '').strip() or None


def _say(text: str, lang: str, source: str) -> None:
    log_data({'msg': f'robot: {text}'})
    say_to_user(text, lang=lang, source=source)


def _hear(prompt, lang: str, source: str, listen: dict):
    """Say `prompt` (if any), then capture and transcribe one phrase.

    Returns ``(transcript, audio)``: the transcript is None when nobody
    answered, and `audio` is the raw recording — only on the robot side, since
    the dashboard mic hands back text and never the sound. On the robot side
    the prompt is spoken to completion before the mic opens, or the recording
    would start with the robot's own voice; the browser does the same ordering
    itself.
    """
    audio = None
    if prompt:
        log_data({'msg': f'robot: {prompt}'})
    if source == 'dashboard':
        typed = listen.get('mode') == 'text'
        try:
            text = listen_dashboard(prompt=prompt, lang=lang, max_sec=listen['max_sec'],
                                    timeout=(listen['text_wait_sec'] if typed
                                             else listen['max_sec'] + DASHBOARD_SLACK_SEC),
                                    mode='text' if typed else 'voice',
                                    capture=listen['capture'], hint=listen['hint'],
                                    silence_sec=listen['silence_sec'], stt=listen['stt'])
        except RuntimeError as e:
            # The browser refused the mic ('not-allowed': permission denied,
            # or the page is on plain http, where the mic is never allowed).
            # Ask for a typed answer in the Q&A card instead of failing — for
            # the rest of this skill call too, since `listen` is shared.
            if typed or 'not-allowed' not in str(e):
                raise
            listen['mode'] = 'text'
            listen['text_wait_sec'] = max(listen['text_wait_sec'], listen.get('text_fallback_sec', TEXT_WAIT_SEC))
            log_data({'msg': 'the browser blocked the microphone (not-allowed) — '
                             'switching to typed input: type in the Q&A card'})
            text = listen_dashboard(prompt=prompt, lang=lang, max_sec=listen['max_sec'],
                                    timeout=listen['text_wait_sec'], mode='text')
    else:
        if prompt:
            say_to_user(prompt, lang=lang, source='robot')
        audio = record_phrase(max_sec=listen['max_sec'],
                              silence_sec=listen['silence_sec'],
                              energy_threshold=listen['energy_threshold'])
        text = (speech_to_text(audio, lang=lang, prompt=listen['hint'], stt=listen['stt'])
                if audio is not None else None)
    text = (text or '').strip() or None
    log_data({'msg': f'user: {text}' if text else 'user: (no answer)'})
    return text, audio


# ── Matching an answer against options ───────────────────────────────────────

def parse_options(options):
    """None, a list, or the string forms plans and the CLI produce
    (``"[물, 주스]"``, ``"물,주스"``) -> list of option strings, or None."""
    if options is None:
        return None
    if isinstance(options, str):
        s = options.strip()
        if s in ('', 'None', 'none', 'null'):
            return None
        items = s.strip('[]()').split(',')
    else:
        items = list(options)
    items = [str(x).strip().strip('\'"').strip() for x in items]
    items = [x for x in items if x]
    return items or None


def _words(s) -> list[str]:
    """Lower-cased words, punctuation dropped, Hangul kept as whole syllables."""
    s = unicodedata.normalize('NFC', str(s).lower())
    return re.sub(r'[\W_]+', ' ', s).split()


def _jamo(s) -> str:
    """NFD splits each Hangul syllable into its jamo, so a one-vowel slip in
    the transcript ('쥬스' for '주스') costs a letter rather than a whole
    syllable: 0.75 similarity instead of 0.5. For Latin text it strips accents."""
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFD', str(s).lower()))


def _contains(answer_words: list[str], option: str) -> bool:
    """Does the answer say this option?

    Matched per word as a PREFIX, because Korean glues particles onto the noun
    ('주스로', '물이요'). Word-anchored rather than raw substring, so 'tea' is not
    found inside 'steak' and '차' is not found inside '착해요'. A multi-word
    option is also tried with its spaces removed, for '오렌지주스로'.
    """
    ow = _words(option)
    if not ow:
        return False
    joined = ''.join(ow)
    if any(w.startswith(joined) for w in answer_words):
        return True
    n = len(ow)
    return any(all(answer_words[i + k].startswith(ow[k]) for k in range(n))
               for i in range(len(answer_words) - n + 1))


def _similarity(a: str, b: str) -> float:
    """1 - Levenshtein distance / longer length. Unlike difflib's ratio it
    charges for every extra letter, so 'steak' is only 0.6 like 'tea'."""
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


def match_option(text, options, threshold: float = MATCH_THRESHOLD):
    """The option a spoken answer picks, or None.

    Options the answer actually contains win outright — the longest one, so
    '오렌지 주스' beats '주스' when both fit. Otherwise the option closest to some
    run of the answer's words as long as the option itself ('orange juse' for
    'orange juice'), by jamo-level edit similarity, if it clears `threshold`.
    Near misses are left unmatched on purpose: `ask` then offers the list and
    asks again, which costs a few seconds, while confirming the wrong option
    sends the robot off to do the wrong thing.
    """
    words = _words(text)
    if not words:
        return None
    inside = [o for o in options if _contains(words, o)]
    if inside:
        return max(inside, key=lambda o: len(_jamo(o)))
    best, score = None, 0.0
    for o in options:
        jo, n = _jamo(o), len(_words(o))
        if not jo or not n:
            continue
        spans = {_jamo(' '.join(words[i:i + k]))
                 for k in {n, n + 1, max(1, n - 1)}
                 for i in range(max(1, len(words) - k + 1))}
        r = max(_similarity(t, jo) for t in spans)
        if r > score:
            best, score = o, r
    return best if score >= threshold else None


# ── Skills ───────────────────────────────────────────────────────────────────

@exception_handler
def reply(node, **kwargs):
    """Listen for what the user says and acknowledge it.

    Params:
        need_confirm (bool): play the recording back ('이렇게 들었어요' + the
            person's own voice) instead of a bare '들었어요', so a name the
            recogniser mangled is still checkable. With the dashboard mic there
            is no recording, so the transcript is read back instead
            ('"<text>"라고 들었어요'). Default False.
        inputs (str):  optional line to say before listening.
        source (str):  'robot' (default) or 'dashboard'.
        lang (str):    'ko' (default), 'en' or 'vi' — speech and recognition.
        max_sec, silence_sec, energy_threshold: listening limits.

    Returns ``{'isdone', 'text'}``; isdone is False when nobody spoke.
    """
    need_confirm = _truthy(kwargs.pop('need_confirm', True))
    prompt = str(kwargs.pop('inputs', '') or '').strip() or None
    lang, source, listen = _common(kwargs)

    text, audio = _hear(prompt, lang, source, listen)
    if not text:
        _say(_phrase('not_heard', lang), lang, source)
        return {'isdone': False, 'msg': 'no speech heard', 'text': ''}

    if not need_confirm:
        _say(_phrase('heard', lang), lang, source)
    else:
        # Echo the recording rather than the transcript: the person hears what
        # the microphone actually got, which is checkable even when Whisper
        # mangled a name. No recording (dashboard mic) or no speaker -> read
        # the transcript back as before.
        echoed = False
        if audio is not None:
            _say(_phrase('heard_audio', lang), lang, source)
            echoed = play_audio(audio)
        if not echoed:
            _say(_phrase('heard_text', lang, text=text), lang, source)
    return {'isdone': True, 'text': text}


@exception_handler
def ask(node, **kwargs):
    """Ask a question aloud and understand the spoken answer.

    Params:
        inputs (str):   the question.
        options (list): allowed answers. When given, the answer is matched to
            one of them and that option is confirmed aloud; an answer matching
            none gets the question asked again. An option may be
            "spoken->value" — "검은 가방->black bag" (or the older
            "검은 가방=black bag"): the robot hears, matches and confirms the
            spoken part, and `answer` is the value (for the English-only
            detector: pick::{answer}; the next step also gets the spoken part,
            as "검은 가방->black bag"). Without "->" both are the option itself. They are also Whisper's hint
            words, so it hears 신라면, not 실라면. When None, the answer is
            simply repeated back.
        read_options (bool): on a non-matching answer, read the options out
            ("… 중에서 골라 주세요") instead of repeating the question. Default False.
        retries (int):  extra attempts after the first, for no answer or no
            match. Default 2.
        match_threshold (float): fuzzy-match floor, 0..1. Default 0.7.
        source, lang, max_sec, silence_sec, energy_threshold: as for `reply`.

    Returns ``{'isdone', 'answer', 'answer_text', 'text', 'attempts'}`` —
    `answer` is the matched option's value (or the transcript when there are no
    options), `answer_text` its spoken part, `text` the
    raw transcript.
    """
    question = str(kwargs.pop('inputs', '') or '').strip()
    options = parse_options(kwargs.pop('options', None))
    values = {}                                 # spoken option → value ("검은 가방" → "black bag")
    if options:
        spoken = []
        for o in options:
            # "검은 가방->black bag" (the plan-wide said->real form); the older
            # "검은 가방=black bag" still works.
            say, _, val = o.partition('->') if '->' in o else o.partition('=')
            say, val = say.strip(), val.strip()
            if not say:                          # "=black bag": nothing to say — use the value
                say = val
            spoken.append(say)
            values[say] = val or say
        options = spoken
    retries = max(0, int(kwargs.pop('retries', RETRIES)))
    threshold = float(kwargs.pop('match_threshold', MATCH_THRESHOLD))
    read_options = _truthy(kwargs.pop('read_options', False))
    own_hint = 'stt_hint' in kwargs          # _common pops it
    lang, source, listen = _common(kwargs)
    if options and not own_hint:
        # The options are the words to expect. Without them Whisper heard
        # "실라면" for 신라면 and "자파게티" for 짜파게티 — no match, and the list
        # was read out instead of confirming.
        listen['hint'] = ', '.join(options)

    prompt, last_text = question or None, ''
    for attempt in range(1, retries + 2):
        text, _audio = _hear(prompt, lang, source, listen)   # ask confirms the option, not the audio
        if not text:
            # Nobody answered: say so, then put the question again.
            prompt = ' '.join(p for p in (_phrase('not_heard', lang), question) if p)
            continue
        last_text = text

        if options is None:
            _say(_phrase('heard_text', lang, text=text), lang, source)
            return {'isdone': True, 'answer': text, 'answer_text': text, 'text': text, 'attempts': attempt}

        choice = match_option(text, options, threshold)
        if choice is not None:
            _say(_phrase('confirm', lang, choice=choice), lang, source)
            return {'isdone': True, 'answer': values.get(choice, choice), 'answer_text': choice,
                    'text': text, 'attempts': attempt, 'options': options}
        # An answer, just not one of the options: ask again (the list only
        # with read_options).
        log_data({'msg': f'ask: "{text}" matches none of {options}'})
        prompt = (_phrase('choose', lang, options=', '.join(options)) if read_options
                  else ' '.join(p for p in (_phrase('not_heard', lang), question) if p))

    _say(_phrase('not_heard', lang), lang, source)
    msg = ('answer did not match any option' if last_text and options
           else 'no speech heard')
    return {'isdone': False, 'msg': msg, 'answer': None, 'text': last_text,
            'attempts': retries + 1, 'options': options}



# ── Announce / wait ──────────────────────────────────────────────────────────

_VI_CHARS = re.compile(r'[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]', re.IGNORECASE)


def _guess_lang(text: str) -> str:
    """ko for Hangul, vi for Vietnamese letters, else en."""
    if re.search(r'[가-힣]', text):
        return 'ko'
    if _VI_CHARS.search(text):
        return 'vi'
    return 'en'


@exception_handler
def announce(node, **kwargs):
    """Say `inputs` aloud and return once it has been said.

    Params:
        inputs (str): what to say.
        lang: 'ko' / 'en' / 'vi'; default from the text (Hangul → ko,
            Vietnamese letters → vi, else en).
        source: 'dashboard' / 'robot' — default HRI_CONFIGS['source'].
        wait (bool): True (default) — return once the line has been spoken, so
            the next step does not start under it; False — go on at once.

    Usage:
        announce::식사 준비가 다 되었어요
        announce::inputs='Lunch is ready', source='robot'
        announce::inputs='잠시만요', wait=False      # speak while the next step runs
    """
    text = str(kwargs.pop('inputs', '') or '').strip()
    if not text:
        return {'isdone': False, 'msg': 'announce: nothing to say (inputs is empty)'}
    lang = str(kwargs.pop('lang', '') or _guess_lang(text))
    source = str(kwargs.pop('source', None) or _hri_cfg()['source'] or 'dashboard').lower()
    if source not in SOURCES:
        raise ValueError(f'source must be one of {SOURCES}, got {source!r}')
    wait = _truthy(kwargs.pop('wait', True))
    log_data({'msg': f'robot: {text}'})
    say_to_user(text, lang=lang, source=source, wait=wait)
    return {'isdone': True, 'said': text, 'lang': lang, 'waited': wait}


@exception_handler
def wait(node, **kwargs):
    """Stand still for `inputs` seconds (time.sleep that Cancel / Stop ends).

    Usage:
        wait::3
        wait::inputs=1.5
    """
    import time
    from robot_agent.core.run_control import cancel_requested
    try:
        sec = float(kwargs.pop('inputs', 0) or 0)
    except (TypeError, ValueError):
        return {'isdone': False, 'msg': 'wait: inputs must be a number of seconds'}
    if sec < 0:
        return {'isdone': False, 'msg': 'wait: seconds must not be negative'}
    end = time.monotonic() + sec
    while time.monotonic() < end:
        if cancel_requested():
            return {'isdone': False, 'msg': 'cancelled', 'waited': round(sec - (end - time.monotonic()), 1)}
        time.sleep(min(0.1, max(0.0, end - time.monotonic())))
    return {'isdone': True, 'waited': sec}


# ── Visual Q&A ───────────────────────────────────────────────────────────────

@exception_handler
def qa(node, **kwargs):
    """Answer questions about what the robot sees, round after round, like
    `ask`, until the user says a stop word or the run is cancelled.

    Each round is one `ask`-style listen: the line the robot says before
    listening — the opening question, then each answer — is the prompt of the
    listen, so it is spoken to completion and the mic opens right after. No
    answer is not an end: the robot listens again (silently, so it does not
    keep talking at an empty room). Only a stop word (그만, 종료, dừng, stop …,
    on a short utterance) or Cancel / Stop on the dashboard ends it.

    The vision side (photos, model, place-restricted answers) is in skills/qa.py.

    Params:
        source, lang, max_sec, silence_sec, energy_threshold: as for `reply`.
        input: 'voice' (default) or 'text' — typed in the dashboard's Q&A card.
        Any QA_CONFIGS key (refresh_sec, views, model, url, ...) overrides the
            configured value for this call.
        loc: ENV location (key or alias) to answer for; default the nearest.
        cam: 'head' (default: a photo at each head tilt in `views`) or 'arm'
            (one photo from the arm camera, QA_CONFIGS['arm_camera']; the head
            does not move).
        Light requests ("불 꺼줘", "turn on the light") are acknowledged and
            done with turn_light (off without a place → every light).
        once: True — answer one question and end (no goodbye); silence is
            asked again up to `retries` times (default 2), as `ask` does.
            Default False: the conversation goes on until a stop word or Cancel.
        products: False turns off the product recogniser (default on): a
            question about 라면 / a product name is answered with the names it
            is sure of (configs/products), never a name the model read.

    Returns ``{'isdone', 'turns', 'ended', 'history'}`` (+ ``answer`` with
    once) — `ended` is 'stop word', 'cancelled', 'answered' or 'no answer'
    (once, nobody spoke; isdone False).
    """
    import time
    from robot_agent.core.run_control import cancel_requested
    from . import qa as vqa

    kwargs.pop('inputs', None)              # a bare "qa::" carries nothing
    loc = kwargs.pop('loc', None)
    cam = str(kwargs.pop('cam', 'head') or 'head').lower()
    once = _truthy(kwargs.pop('once', False))
    retries = max(0, int(kwargs.pop('retries', RETRIES)))
    use_products = _truthy(kwargs.pop('products', True))
    if cam not in ('head', 'arm'):
        raise ValueError(f"cam must be 'head' or 'arm', got {cam!r}")
    explicit_views = 'views' in kwargs
    cfg = vqa._config(kwargs)
    cfg['cam'] = cam
    lang, source, listen = _common(kwargs)
    # A typed question is waited for until it comes or the run is cancelled: a
    # listen that times out would leave the browser's Q&A card waiting on the
    # old request, and the line typed next would go to it and be lost.
    listen['text_fallback_sec'] = QA_TEXT_WAIT_SEC
    # Whisper hint: also the things people drop and the products the robot
    # knows (Whisper heard "손가락" for 숟가락 without them).
    extra = [w for w in vqa._drop_vocab(cfg) if re.match(r'[가-힣]', w)] if lang == 'ko' else []
    try:
        from . import _products
        extra += list((_products._catalog().get('products') or {}).keys())
    except Exception:
        pass
    if extra:
        listen['hint'] = ', '.join(x for x in [listen.get('hint') or '', ', '.join(dict.fromkeys(extra))] if x)
    if listen.get('mode') == 'text':
        listen['text_wait_sec'] = QA_TEXT_WAIT_SEC
    place, qa_cfg = vqa._place_config(node, cfg, loc)
    if qa_cfg and qa_cfg.get('views') and not explicit_views:
        cfg['views'] = list(qa_cfg['views'])
    log_data({'msg': f"qa: at {place or 'unknown place'}"
                     + (' (place layout)' if qa_cfg else '')
                     + (f" — {cfg['arm_camera']}, one photo" if cam == 'arm'
                        else f" — {cfg['camera']} at {', '.join(cfg['views'])}")})
    log_data({'msg': f"qa: VLM {vqa._vlm_label(cfg)} (url {cfg.get('url')}, num_ctx {cfg.get('num_ctx')}, "
                     f"from {cfg.get('_source', 'QA_CONFIGS')})"})

    history, shots, shot_at = [], [], 0.0
    inv, inv_at = None, None                # product inventory of the current photos
    ended = 'cancelled'
    silent = 0
    prompt = vqa._phrase('start', lang)     # said by the first listen, as `ask` does

    while not cancel_requested():
        try:
            text, _audio = _hear(prompt, lang, source, listen)
        except RuntimeError:
            if cancel_requested():          # the dashboard mic was released by Stop
                break
            raise
        prompt = None                       # nothing heard: listen again, quietly
        if not text:
            silent += 1
            if once and silent > retries:   # one-shot: give up like `ask` does
                ended = 'no answer'
                break
            if once:                        # one-shot: ask again, as `ask` does
                prompt = ' '.join(p for p in (_phrase('not_heard', lang), vqa._phrase('start', lang)) if p)
            continue
        silent = 0
        if vqa._is_stop(text):
            ended = 'stop word'
            break

        # "불 꺼줘" — say it will, then do it with turn_light (qa._light_command).
        light = vqa._light_command(text)
        if light is not None:
            action, loc = light
            ack = vqa._light_phrase('ack', action, lang)
            _say(ack, lang, source)
            try:
                ok, note = vqa._do_light(node, action, loc, lang)
            except Exception as e:                      # no SwitchBot / BLE error
                ok, note = False, f'turn_light {action}: {e}'
            log_data({'msg': f'qa: {note}'})
            answer = ack
            if not ok:
                answer = vqa._light_phrase('fail', action, lang)
                _say(answer, lang, source)
            history.append((text, answer))
            if once:
                ended = 'answered'
                break
            continue                                    # listen again (already spoken)

        # A new question: clear the previous answer's picture; the photos the
        # model is given for this answer are logged once it has answered.
        log_data({'log_image_reset': True})
        # "숟가락 어디에 떨어졌어?" — the arm camera + geometry, no head photos /
        # model (see qa._drop_answer).
        # Any "…떨어졌어?" goes here, even when the object word was not
        # understood (it asks again) — never to the left / right place answer.
        drop = vqa._is_drop_question(text)
        if not drop and (not shots or time.time() - shot_at > float(cfg['refresh_sec'])):
            _say(vqa._phrase('look', lang), lang, source)
            shots, shot_at = vqa._look_around(node, cfg), time.time()
            if cancel_requested():
                break

        facts = None
        if drop:
            t0 = time.time()
            try:
                answer = vqa._drop_answer(node, cfg, text, lang)
            except Exception as e:
                log_data({'msg': f'qa: dropped-object search failed: {e}'})
                answer = vqa._phrase('error', lang)
            log_data({'msg': f'qa: answered in {time.time() - t0:.1f}s from the arm camera (feet + object)'})
        elif not shots:
            answer = vqa._phrase('blind', lang)
        else:
            t0 = time.time()
            try:
                named = []
                if use_products and vqa._is_product_question(text):
                    if inv_at != shot_at:       # once per set of photos
                        t1 = time.time()
                        inv, inv_at = vqa._inventory(cfg, shots), shot_at
                        log_data({'msg': f'qa: products ({time.time() - t1:.1f}s): ' + (', '.join(
                            f"{p['name'] or '?'}@{p['side']} {p['sim']:.2f}" for p in inv) or 'none')})
                    from . import _products
                    named = _products.names_in(text)
                    facts = vqa._product_facts(inv)
                unknown = vqa._unknown_names(text) if facts else []
                if named:                       # "신라면 있어?" — straight from the recogniser
                    answer = vqa._product_answer(named, inv, lang)
                elif unknown:                   # "진라면 있어?" — not a product it knows: never yes / no
                    answer = vqa._unrecognised_answer(unknown, inv, lang)
                elif facts:                     # "무슨 라면 있어?" — model, with the verified names
                    answer = vqa._ask_vlm(cfg, shots, text, history, lang, facts=facts)
                    if vqa._unknown_names(answer):   # it named a product anyway: use the list
                        log_data({'msg': f'qa: model named an unverified product — {answer!r}'})
                        answer = vqa._product_list(inv, lang)
                else:
                    answer = (vqa._ask_place(cfg, shots, text, history, lang, qa_cfg) if qa_cfg
                              else vqa._ask_vlm(cfg, shots, text, history, lang))
            except Exception as e:
                log_data({'msg': f'qa: vision model failed ({vqa._vlm_label(cfg)}): {e}'})
                answer = ''
            log_data({'msg': f'qa: answered in {time.time() - t0:.1f}s by {vqa._vlm_label(cfg)} '
                             f'from {len(shots)} photo(s) taken {time.time() - shot_at:.0f}s ago '
                             f"({', '.join(v for v, _ in shots)})",
                      'log_image': vqa._mosaic(shots)})
            answer = answer or vqa._phrase('error', lang)
        if cancel_requested():
            break
        history.append((text, answer))
        if once:                            # one question: say the answer and end
            _say(answer, lang, source)
            ended = 'answered'
            break
        prompt = answer                     # spoken by the next listen, then the mic opens

    if ended not in ('cancelled', 'answered'):
        _say(vqa._phrase('bye', lang), lang, source)
    out = {'isdone': ended != 'no answer', 'turns': len(history), 'ended': ended,
           'history': [{'q': q, 'a': a} for q, a in history]}
    if once and history:
        out['answer'] = history[-1][1]
    return out
