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
    qa::lang='en', input='text'               # questions typed on the dashboard
    qa::와인잔 위치를 물어보세요                   # opens with this line instead of 무엇이 궁금하세요?

Languages: 'ko' (default) and 'en'.
"""

import base64
import json
import logging
import re
import time
import unicodedata

import cv2
import numpy as np
import requests

from robot_agent.core.run_control import cancel_requested, dialog_begin, dialog_end, end_dialog_requested
from robot_agent.skill_configs import ENV, HRI_CONFIGS, QA_CONFIGS
from robot_agent.skills import log_data as _emit
from robot_agent.utils import (
    EARCON_GAP_SEC, exception_handler, listen_dashboard, play_audio, play_earcon, record_phrase, say_to_user,
    set_speech_rate, speech_to_text,
)

from ..utils import env_key, get_closest_loc
from .head import head_state, moveh

# Under robot_agent: that logger holds the file handler (logging_config).
_logger = logging.getLogger('robot_agent.skill.hri')


def log_data(data: dict) -> None:
    """To the dashboard's Execution panel and — the text — to the log file
    (data/logs/<host>/kcare_robot.log): the panel keeps nothing, and a bad
    conversation could not be read back afterwards."""
    _emit(data)
    msg = data.get('msg') if isinstance(data, dict) else None
    if msg:
        _logger.info(msg)


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
LANGS = ('ko', 'en')
# input='text': the answer is typed on the dashboard, which takes far longer
# than saying it — one wait covers a typed question.
TEXT_WAIT_SEC = 300.0
QA_TEXT_WAIT_SEC = 24 * 3600.0  # qa: a typed question is waited for until it comes or Cancel

_PHRASES = {
    'heard':      {'ko': '들었어요',
                   'en': 'Got it.'},
    'heard_text': {'ko': '"{text}"라고 들었어요',
                   'en': 'I heard: {text}'},
    # Said just before the recording itself is played back (need_confirm).
    'heard_audio':{'ko': '이렇게 들었어요',
                   'en': 'This is what I heard:'},
    'not_heard':  {'ko': '잘 못 들었어요',
                   'en': "Sorry, I didn't catch that."},
    'choose':     {'ko': '{options} 중에서 골라 주세요',
                   'en': 'Please choose one of: {options}'},
    'confirm':    {'ko': '{choice}, 알겠습니다',
                   'en': '{choice}, confirmed.'},
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
    cfg = {'source': 'robot', 'dashboard_stt': 'whisper', 'stt': '', 'stt_hint': {}, 'earcon': True, 'earcon_volume': 0.3,
           'speech_rate': 1.3}
    try:
        cfg.update(dict(HRI_CONFIGS.items()))
    except Exception:
        pass
    return cfg


SPEECH_RATE_RANGE = (0.5, 2.0)


def _set_rate(kwargs: dict, hri: dict | None = None) -> float:
    """How fast the dashboard says this call's lines: the call's `rate=`, else HRI_CONFIGS.speech_rate,
    within SPEECH_RATE_RANGE; handed to say_to_user / listen_dashboard (robot_agent.utils.set_speech_rate)."""
    hri = hri or _hri_cfg()
    try:
        rate = float(kwargs.pop('rate', None) or hri.get('speech_rate') or 1.0)
    except (TypeError, ValueError):
        rate = 1.0
    rate = min(max(rate, SPEECH_RATE_RANGE[0]), SPEECH_RATE_RANGE[1])
    set_speech_rate(rate)
    return rate


def _common(kwargs: dict):
    hri = _hri_cfg()
    _set_rate(kwargs, hri)
    lang = str(kwargs.pop('lang', 'ko') or 'ko').lower()
    if lang not in LANGS:
        raise ValueError(f'lang must be one of {LANGS}, got {lang!r}')
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
        # "your turn" / "got it" tones around each listen (robot_agent.utils.earcon), volume 0 = off;
        # HRI_CONFIGS earcon / earcon_volume, a call's earcon=False wins
        'earcon': float(hri.get('earcon_volume') or 0.3) if _truthy(kwargs.pop('earcon', hri.get('earcon', True))) else 0.0,
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
    keep = [] if listen.get('keep_audio') else None     # qa: the recording, for replays
    listen['last_audio'] = None
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
                                    silence_sec=listen['silence_sec'], stt=listen['stt'],
                                    audio_out=keep, earcon=listen['earcon'])
            if keep:
                listen['last_audio'] = keep[0]              # (bytes, mime)
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
        if play_earcon('listen', listen['earcon']):          # "your turn" — over before the mic opens
            time.sleep(EARCON_GAP_SEC)
        audio = record_phrase(max_sec=listen['max_sec'],
                              silence_sec=listen['silence_sec'],
                              energy_threshold=listen['energy_threshold'])
        if audio is not None:
            play_earcon('heard', listen['earcon'])           # "got it"; nothing when nobody spoke
        text = (speech_to_text(audio, lang=lang, prompt=listen['hint'], stt=listen['stt'])
                if audio is not None else None)
        if keep is not None and audio is not None:
            listen['last_audio'] = (audio, 'pcm16k')       # float32 mono 16 kHz
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
        lang (str):    'ko' (default) or 'en' — speech and recognition.
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
        The options' spoken parts are read out with the question, every time
            it is asked ("어떤 가방을 가져갈까요? 베이지색 핸드백, 검은색 핸드백 중에서
            골라 주세요"): a blind user cannot answer a choice they were not told.
        read_options: no longer used (the options are always read); accepted for old plans.
        confirm (bool): True (default) — say back what was understood: the
            chosen option ("검은색핸드백, 알겠습니다") or, without options, the
            user's words ('"…"라고 들었어요'). False — say nothing, go on.
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
    kwargs.pop('read_options', None)            # the options are always read now
    confirm = _truthy(kwargs.pop('confirm', True))
    own_hint = 'stt_hint' in kwargs          # _common pops it
    lang, source, listen = _common(kwargs)
    if options and not own_hint:
        # The options are the words to expect. Without them Whisper heard
        # "실라면" for 신라면 and "자파게티" for 짜파게티 — no match, and the list
        # was read out instead of confirming.
        listen['hint'] = ', '.join(options)

    # The question with the choices: said first and with every repeat.
    asked = ' '.join(p for p in (question, options and _phrase('choose', lang, options=', '.join(options))) if p)
    prompt, last_text = asked or None, ''
    for attempt in range(1, retries + 2):
        text, _audio = _hear(prompt, lang, source, listen)   # ask confirms the option, not the audio
        if not text:
            # Nobody answered: say so, then put the question again.
            prompt = ' '.join(p for p in (_phrase('not_heard', lang), asked) if p)
            continue
        last_text = text

        if options is None:
            if confirm:
                _say(_phrase('heard_text', lang, text=text), lang, source)
            return {'isdone': True, 'answer': text, 'answer_text': text, 'text': text, 'attempts': attempt}

        choice = match_option(text, options, threshold)
        if choice is not None:
            if confirm:
                _say(_phrase('confirm', lang, choice=choice), lang, source)
            return {'isdone': True, 'answer': values.get(choice, choice), 'answer_text': choice,
                    'text': text, 'attempts': attempt, 'options': options}
        # An answer, just not one of the options: ask again, the choices with it.
        log_data({'msg': f'ask: "{text}" matches none of {options}'})
        prompt = ' '.join(p for p in (_phrase('not_heard', lang), asked) if p)

    _say(_phrase('not_heard', lang), lang, source)
    msg = ('answer did not match any option' if last_text and options
           else 'no speech heard')
    return {'isdone': False, 'msg': msg, 'answer': None, 'text': last_text,
            'attempts': retries + 1, 'options': options}



# ── Announce / wait ──────────────────────────────────────────────────────────

def _guess_lang(text: str) -> str:
    """ko for Hangul, else en."""
    return 'ko' if re.search(r'[가-힣]', text) else 'en'


@exception_handler
def announce(node, **kwargs):
    """Say `inputs` aloud and return once it has been said.

    Params:
        inputs (str): what to say.
        lang: 'ko' / 'en'; default from the text (Hangul → ko, else en).
        source: 'dashboard' / 'robot' — default HRI_CONFIGS['source'].
        wait (bool): True (default) — return once the line has been spoken, so
            the next step does not start under it; False — go on at once.
        rate (float): how fast the dashboard says it — default HRI_CONFIGS['speech_rate'].

    Usage:
        announce::식사 준비가 다 되었어요
        announce::inputs='Lunch is ready', source='robot'
        announce::inputs='잠시만요', wait=False      # speak while the next step runs
    """
    text = str(kwargs.pop('inputs', '') or '').strip()
    if not text:
        return {'isdone': False, 'msg': 'announce: nothing to say (inputs is empty)'}
    lang = str(kwargs.pop('lang', '') or _guess_lang(text)).lower()
    if lang not in LANGS:
        raise ValueError(f'lang must be one of {LANGS}, got {lang!r}')
    source = str(kwargs.pop('source', None) or _hri_cfg()['source'] or 'dashboard').lower()
    if source not in SOURCES:
        raise ValueError(f'source must be one of {SOURCES}, got {source!r}')
    wait = _truthy(kwargs.pop('wait', True))
    _set_rate(kwargs)
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
#
# `qa` answers spoken questions about what is around the robot — "what colour
# is the shirt?", "와인잔이 선반 어디에 있어?" — from photos taken by the head
# camera, using a vision-language model served by Ollama (QA_CONFIGS in
# configs/tasks.py, or the 'vlm' connection: url, model, num_ctx).
#
# Looking around: the head takes one photo at each tilt in QA_CONFIGS['views']
# (moveh up / straight / down), then goes back to where it was. Every question
# gets new photos (2026-10-13: an answer from photos taken for an earlier
# question can describe a shelf that has changed since).
#
# Where-questions at a known place: an ENV entry may carry a ``qa`` block
# describing what the robot sees there and the places an answer may name::
#
#     'dressroom': {..., 'qa': {
#         'views': ['up', 'straight', 'down'],
#         'scene': 'a dressing room: a hanging rail on the left, ...',
#         'frame': 'robot',                     # or 'person': left/right of the seated user
#         'places': {
#             'rail_left': {'desc': 'left part of the hanging rail',
#                           'ko': '행거 왼쪽', 'en': 'on the left of the rail'},
#             ...}}}
#
# At such a place the model returns JSON restricted to those place ids, and the
# spoken answer is built from the phrases ("행거 왼쪽에 있어요") instead of being
# free text. The place is the nearest ENV location with a ``qa`` block within
# `loc_radius`, or ``qa::loc='옷방'``.

_QA_PHRASES = {
    'start': {'ko': '무엇이 궁금하세요?',
              'en': 'What would you like to know?'},
    'look':  {'ko': '잠깐 둘러볼게요.',
              'en': 'Let me look around.'},
    'bye':   {'ko': '대화를 마칠게요.',
              'en': 'Ending our chat.'},
    'error': {'ko': '지금은 대답하기 어려워요.',
              'en': "Sorry, I can't answer right now."},
    'blind': {'ko': '카메라 영상을 받지 못했어요.',
              'en': "I can't get an image from my camera."},
    'no_shelf': {'ko': '선반이 잘 보이지 않아서 위치를 말씀드리기 어려워요.',
                  'en': "I can't see the shelves well enough to say where."},
    'which': {'ko': '무엇을 찾으시는지 잘 못 들었어요. 다시 말씀해 주세요.',
              'en': "I didn't catch what you are looking for. Please say it again."},
    'unseen': {'ko': '보이지 않아요.',
               'en': "I can't see it."},
    # Long silence in a conversation (idle_ask_sec / idle_end_sec).
    'idle_ask': {'ko': '더 궁금하신 게 있으세요? 대답이 없으시면 잠시 후 대화를 마칠게요.',
                 'en': "Anything else you'd like to know? If I hear nothing, I'll end our chat shortly."},
    'idle_bye': {'ko': '대답이 없으셔서 대화를 마칠게요.',
                 'en': "I didn't hear anything, so I'll end our chat."},
    'go_on':    {'ko': '네, 말씀하세요.',
                 'en': 'Sure, go ahead.'},
}

_LANG_NAME = {'ko': 'Korean', 'en': 'English'}

# Checked in every language, and only on a short utterance: "끝에 있는 컵은
# 무슨 색이야?" contains '끝' but is a question, not a goodbye.
_STOP_WORDS = ('그만', '종료', '끝', '멈춰', '됐어',
               'stop', 'bye', 'quit', 'thatsall')
_STOP_MAX_CHARS = 10

# The model answers from a few photos; everything it may do beyond that is a
# guess, which is the failure that matters most for a care robot.
_QA_PROMPT = """\
You are the eyes of a home-care robot talking with a person. You are given {n} \
photos taken just now by the robot's head camera, all facing the same \
direction at different tilts: {views}. The first shows the room at eye level \
and further away; the later ones show the table, the floor and low shelves \
closer to the robot.

Answer the person's question about the surroundings.
- Reply in {language}, in ONE short sentence of at most about 15 words. No \
lists, no markdown.
- Use only what is visible in the photos. If what they ask about is not \
visible, say you cannot see it. Never guess.
- Colours: plain everyday colour words.
- Positions: from the robot's point of view (left / middle / right, near / \
far) and what the thing is on or next to (table, chair, sofa, shelf ...).
- Shelf levels: count from the bottom; the lowest shelf is level 1.
- The photos overlap: the same shelf or object can appear in two photos. \
Count it once."""

# Runtime defaults; configs/tasks.py QA_CONFIGS overrides them, and a plan can
# override one key per call (qa::views='up').
_QA_DEFAULTS = {
    'url': 'http://localhost:11434',
    # The instruct tag. The plain `qwen3-vl:8b` tag is thinking-only: on real
    # head-cam photos it took 1-10 s and twice ran out of tokens mid-thought,
    # answering nothing; instruct answers in ~2.3 s, ~0.2 s for follow-ups on
    # the same photos (Ollama reuses the cached image prefix).
    'model': 'qwen3-vl:8b-instruct',
    'camera': 'head_rgb',
    'arm_camera': 'arm_rgb',    # qa::cam='arm' — one photo, the head does not move
    'views': ['up', 'straight', 'down'],
    # A conversation nobody talks to: after idle_ask_sec of silence the robot asks
    # whether there is anything else ('idle_ask'), and ends idle_end_sec after that
    # question if still nobody speaks. 0 = never (listen until a stop word / Cancel).
    # Not for typed input or once.
    'idle_ask_sec': 60,
    'idle_end_sec': 30,
    'history_turns': 4,
    'max_side': 1280,           # full head-cam width: small things (towels) read better
    'settle_sec': 0.4,
    'frame_timeout_sec': 2.0,
    'temperature': 0.2,
    # Generous on purpose: should `model` be switched to a thinking model, it
    # reasons before answering, and a tight cap leaves the answer empty.
    'num_predict': 1024,
    'keep_alive': '30m',
    # Context window. A few photos + the prompt are ~3.5k tokens; Ollama's own
    # default can be the model's maximum (262k for qwen3-vl), which on the A6000
    # server spread qwen3-vl:32b over 94 GB / several GPUs at 16 tok/s instead
    # of 23 GB on one GPU at 26 tok/s.
    'num_ctx': 8192,
    'timeout_sec': 60.0,
    'loc_radius': 1.0,          # m: nearest ENV location closer than this is "here"
    # Connection (type 'llm') whose url / model / num_ctx override the above.
    'connection': 'vlm',
    # Off by default (2026-10-13): each answer from new photos and the question
    # alone. memory=True sends the earlier questions / answers with it (it once
    # made "머그컵 어디?" come back with the wine glass of the question before);
    # reuse_photos=True answers from the last photos while younger than
    # refresh_sec instead of looking again.
    'memory': False,
    'reuse_photos': False,
    'refresh_sec': 20.0,
}


def _qa_phrase(key: str, lang: str) -> str:
    table = _QA_PHRASES[key]
    return table.get(lang, table['en'])


def _vlm_connection(name: str) -> tuple[dict, str]:
    """(url / model / num_ctx of the `name` connection (type 'llm', Connections
    panel), where they came from) — ({}, why) when there is none. A connection
    not connected yet (the backend restores them one by one at startup) is
    read as saved: falling back to QA_CONFIGS then switched the model."""
    try:
        from robot_agent.state import current
        dm = current().dm
    except Exception:
        return {}, 'no running agent'
    entry = dm.get_connect(name)
    if entry is not None:
        typ, conf, how = entry.type, entry.config, f"connection '{name}'"
    else:
        saved = dm.saved_connect(name) if hasattr(dm, 'saved_connect') else None
        if not saved:
            return {}, f"no connection '{name}'"
        typ, conf, how = saved.get('type'), saved.get('config') or {}, f"connection '{name}' (saved, not connected yet)"
    if typ != 'llm':
        return {}, f"connection '{name}' is {typ}, not llm"
    return {k: conf[k] for k in ('url', 'model', 'num_ctx') if conf.get(k)}, how


def _vlm_label(cfg: dict) -> str:
    """'model @ server' for the logs."""
    from urllib.parse import urlparse
    host = urlparse(str(cfg.get('url', ''))).netloc or cfg.get('url', '?')
    return f"{cfg.get('model', '?')} @ {host}"


def _qa_config(kwargs: dict) -> dict:
    """Defaults < QA_CONFIGS < the VLM connection (QA_CONFIGS['connection'],
    default 'vlm' — edited in the Connections panel) < this call's kwargs."""
    cfg = dict(_QA_DEFAULTS)
    try:
        cfg.update(dict(QA_CONFIGS.items()))
    except Exception:
        pass
    conn = str(kwargs.get('connection') or cfg.get('connection') or 'vlm')
    from_conn, how = _vlm_connection(conn)
    cfg.update(from_conn)
    cfg['_source'] = how if from_conn else f'QA_CONFIGS ({how})'
    for key in list(kwargs):
        if key in cfg:
            cfg[key] = kwargs.pop(key)
    if isinstance(cfg['views'], str):
        cfg['views'] = [v.strip() for v in cfg['views'].split(',') if v.strip()]
    return cfg


def _is_stop(text: str) -> bool:
    norm = re.sub(r'[\s\.\,\!\?\'"~…]+', '', text.lower())
    return len(norm) <= _STOP_MAX_CHARS and any(w in norm for w in _STOP_WORDS)


# ── Looking around ───────────────────────────────────────────────────────────

def _fresh_frame(cam, settle_sec: float, timeout_sec: float):
    """The first camera frame received after the head stopped, or None.

    Waits for `rev_seq` to move so the photo is not one captured mid-motion or
    before the move; returns None if the topic has stalled rather than answer
    from an old picture.
    """
    time.sleep(settle_sec)                       # let the head stop shaking
    seq0 = getattr(cam, 'rev_seq', None)
    deadline = time.monotonic() + timeout_sec
    while seq0 is not None and getattr(cam, 'rev_seq', seq0) == seq0:
        if time.monotonic() > deadline:
            return None
        time.sleep(0.03)
    data = cam.rev_data
    im = data.get('im') if isinstance(data, dict) else data
    return im if isinstance(im, np.ndarray) else None


class _Shots(list):
    """[(view, rgb)], plus `geo` {view: head-camera geometry, see _capture_geo}."""

    def __init__(self, *a):
        super().__init__(*a)
        self.geo = {}


def _look_around(node, cfg: dict) -> list:
    """[(view, rgb)] for each tilt in cfg['views']; the head returns after.
    Head photos also get the depth and camera pose (`.geo`) for _ask_shelves.

    The head is only put back when the run has not been cancelled — while
    cancelled every action agent refuses to send, and a cancel means "stop
    moving" anyway.
    """
    if cfg.get('cam') == 'arm':
        # The arm camera looks where the arm points: one photo as it is.
        cam = node.agents.get(cfg['arm_camera']) if node is not None else None
        if cam is None:
            log_data({'msg': f"qa: camera {cfg['arm_camera']!r} is not connected"})
            return _Shots()
        im = _fresh_frame(cam, 0.0, cfg['frame_timeout_sec'])
        if im is None:
            log_data({'msg': f"qa: no fresh frame from {cfg['arm_camera']}"})
            return _Shots()
        return _Shots([('arm', im)])
    cam = node.agents.get(cfg['camera']) if node is not None else None
    if cam is None:
        log_data({'msg': f"qa: camera {cfg['camera']!r} is not connected"})
        return _Shots()
    start = head_state(node=node)
    shots = _Shots()
    try:
        for view in cfg['views']:
            if cancel_requested():
                break
            ret = moveh(node=node, inputs=view)
            if isinstance(ret, dict) and not ret.get('isdone', True):
                log_data({'msg': f"qa: moveh {view} failed: {ret.get('msg', '')}"})
                continue
            im = _fresh_frame(cam, cfg['settle_sec'], cfg['frame_timeout_sec'])
            if im is None:
                log_data({'msg': f'qa: no fresh frame at head {view}'})
                continue
            shots.append((view, im))
            try:                            # while the head is still at this tilt
                geo = _capture_geo(node, im.shape[:2])
            except Exception as e:
                geo = None
                log_data({'msg': f'qa: no depth at head {view} ({e})'})
            if geo is not None:
                shots.geo[view] = geo
    finally:
        if isinstance(start, dict) and 'current_ry' in start and not cancel_requested():
            moveh(node=node, ry=start['current_ry'])
    log_data({'msg': f"qa: looked at {', '.join(v for v, _ in shots) or 'nothing'}; depth for "
                     f"{', '.join(shots.geo) or 'none'}"})
    _save_shots(shots)
    return shots


QA_SAVE_KEEP = 60     # look-arounds kept under <log dir>/qa/ (rgb, depth, camera pose)


def _save_shots(shots) -> None:
    """Keep the photos, depth and camera pose of a look-around, so a bad answer
    can be replayed offline on exactly what the robot saw."""
    try:
        from robot_agent.state import current
        root = current().log_dir / 'qa'
        d = root / time.strftime('%Y%m%d-%H%M%S')
        d.mkdir(parents=True, exist_ok=True)
        for view, im in shots:
            cv2.imwrite(str(d / f'{view}.jpg'), np.ascontiguousarray(im[..., ::-1]))
            g = shots.geo.get(view)
            if g is not None:
                np.savez_compressed(d / f'{view}_geo.npz', depth=g['depth'], K=np.asarray(g['K']),
                                    R=g['R'], t=g['t'])
        old = sorted(p for p in root.iterdir() if p.is_dir())[:-QA_SAVE_KEEP]
        for p in old:
            for f in p.iterdir():
                f.unlink()
            p.rmdir()
        log_data({'msg': f'qa: photos saved in {d}'})
    except Exception as e:
        log_data({'msg': f'qa: could not save the photos ({e})'})


QA_AUDIO_KEEP = 300   # recordings kept under <log dir>/qa_audio/


def _save_audio(listen: dict, text, lang: str) -> None:
    """Keep what the mic recorded for one listen, with what Whisper made of it
    and the hint it was given — <log dir>/qa_audio/<time>.<ext> + index.jsonl —
    so misheard Korean can be replayed with other hints offline."""
    rec = listen.get('last_audio')
    if not rec:
        return
    try:
        from robot_agent.state import current
        root = current().log_dir / 'qa_audio'
        root.mkdir(parents=True, exist_ok=True)
        data, kind = rec
        stem = time.strftime('%Y%m%d-%H%M%S') + f'-{int(time.time() * 1000) % 1000:03d}'
        if kind == 'pcm16k':                    # robot mic: float32 → 16-bit wav
            import wave
            f = root / f'{stem}.wav'
            with wave.open(str(f), 'wb') as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
                w.writeframes((np.clip(np.asarray(data), -1, 1) * 32767).astype('<i2').tobytes())
        else:
            ext = {'ogg': 'ogg', 'mp4': 'm4a', 'wav': 'wav', 'mpeg': 'mp3'}
            f = root / f"{stem}.{next((v for k, v in ext.items() if k in str(kind)), 'webm')}"
            f.write_bytes(data)
        with open(root / 'index.jsonl', 'a', encoding='utf-8') as idx:
            idx.write(json.dumps({'file': f.name, 'text': text, 'hint': listen.get('hint'),
                                  'lang': lang, 'ts': time.time()}, ensure_ascii=False) + '\n')
        for old in sorted(p for p in root.iterdir() if p.name != 'index.jsonl')[:-QA_AUDIO_KEEP]:
            old.unlink()
    except Exception as e:
        log_data({'msg': f'qa: could not save the recording ({e})'})


def _mosaic(shots: list, height: int = 240) -> np.ndarray:
    """The photos side by side, labelled — what the model saw, for the log."""
    tiles = []
    for view, im in shots:
        h, w = im.shape[:2]
        tile = cv2.resize(im, (max(1, int(w * height / h)), height), interpolation=cv2.INTER_AREA)
        tile = np.ascontiguousarray(tile)
        cv2.putText(tile, view, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        tiles.append(tile)
    return np.hstack(tiles)


# ── Asking the model ─────────────────────────────────────────────────────────

# One connection for the whole conversation: to a remote Ollama behind HTTPS a
# fresh connection costs a TLS handshake (0.3-1.3 s measured) on every answer.
_session = requests.Session()


def _post_chat(cfg: dict, body: dict):
    body['options'] = {**body.get('options', {}), 'num_ctx': int(cfg['num_ctx'])}
    r = _session.post(cfg['url'].rstrip('/') + '/api/chat', json=body,
                      timeout=float(cfg['timeout_sec']))
    r.raise_for_status()
    return r


def _b64jpeg(rgb: np.ndarray, max_side: int) -> str:
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', np.ascontiguousarray(rgb[..., ::-1]),
                           [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def _first_sentence(text: str) -> str:
    text = re.sub(r'[*_#`]+', '', text or '').strip()
    m = re.match(r'(.+?[\.\?\!。])(\s|$)', text, re.S)
    return (m.group(1) if m else text).strip()


def _view_labels(shots: list) -> str:
    """'photo 1 = head up, photo 2 = head straight' / 'photo 1 = arm camera'."""
    return ', '.join(f'photo {i + 1} = ' + ('arm camera (on the gripper)' if v == 'arm' else f'head {v}')
                     for i, (v, _) in enumerate(shots))


def _ask_vlm(cfg: dict, shots: list, question: str, history: list, lang: str,
             facts: str | None = None) -> str:
    """One short answer to `question` from the photos, or '' if the model gave none.
    `facts`: verified product names (see _product_facts), added to the prompt."""
    views = _view_labels(shots)
    system = _QA_PROMPT.format(n=len(shots), views=views,
                                   language=_LANG_NAME.get(lang, 'the language of the question'))
    if facts:
        system += '\n\n' + facts
    msgs = [{'role': 'system', 'content': system}]
    for q, a in history[-int(cfg['history_turns']):]:
        msgs += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
    msgs.append({'role': 'user', 'content': question,
                 'images': [_b64jpeg(im, int(cfg['max_side'])) for _, im in shots]})
    body = {
        'model': cfg['model'],
        'messages': msgs,
        'stream': False,
        # Honoured by hybrid / instruct models; a thinking-only tag ignores it.
        'think': False,
        'keep_alive': cfg['keep_alive'],
        'options': {'temperature': float(cfg['temperature']),
                    'num_predict': int(cfg['num_predict'])},
    }
    for attempt in (1, 2):
        r = _post_chat(cfg, body)
        answer = _first_sentence(r.json().get('message', {}).get('content', ''))
        if not _wrong_script(answer, lang):
            return answer
        # "架子上放着짜파게티、신라면和안성탕면。" (2026-10-13): the 8B model drifted
        # into Chinese with Korean product names in it.
        log_data({'msg': f'qa: answer not in {lang}, attempt {attempt} — {answer!r}'})
        body['options']['temperature'] = 0.0
    return ''


_FOREIGN_SCRIPT = re.compile(r'[\u3040-\u30ff\u4e00-\u9fff]')     # kana, Chinese characters


def _wrong_script(text: str, lang: str) -> bool:
    """A ko / en answer with Chinese characters or kana in it."""
    return lang in ('ko', 'en') and bool(_FOREIGN_SCRIPT.search(text or ''))


# ── Where-questions at a known place ─────────────────────────────────────────

_PLACE_PROMPT = """\
You are the eyes of a home-care robot talking with a person. You are given {n} \
photos taken just now by the robot's head camera, all facing the same \
direction at different tilts: {views}. The photos overlap; count a thing once.

The robot is at: {scene}.
{frame}
Places here (id: description):
{places}

Classify the person's question and reply with JSON only:
- kind "where": they ask where something is. items = each matching thing you \
can actually see, one entry per thing: what it is (a few English words) and \
the ONE place id it is in. items = [] when you cannot see it. Do not add a \
place just because it exists; only places holding a matching thing.
- kind "other": any other question. answer = one short sentence in \
{language} (at most about 15 words), from what is visible; if it is not \
visible say so. Never guess."""

# Words that make a question a where-question, in the languages qa speaks.
_WHERE = re.compile(r'어디|어느\s*쪽|어느\s*방향|위치|where')

_FRAME = {
    'robot': 'Left and right are as seen from the robot, i.e. as in the photos.',
    'person': 'Left, right and front are from the point of view of the person '
              'using the table (the seated person in the photos), not the '
              'robot: "front" is on the table right in front of them.',
}


def _qa_block(spec):
    """The `qa` block of an ENV entry, or None when it has neither places nor
    `'shelves': True`."""
    block = spec.get('qa') if isinstance(spec, dict) else None
    if not isinstance(block, dict):
        return None
    ok = (isinstance(block.get('places'), dict) and block['places']) or block.get('shelves')
    return block if ok else None


QA_SAME_SPOT = 0.1    # m: ENV entries this close to each other are one spot (sink / shelf2)


def _place_config(node, cfg: dict, loc):
    """(ENV key, its `qa` block) for where the robot is, or (key, None).

    Without `loc`: the ENV entry nearest the base within loc_radius — and of
    the entries on that same spot (QA_SAME_SPOT), the one with a `qa` block.
    sink and shelf2@kitchen share one spot and sink came first in ENV, so the
    block was hidden; but a block must not reach past the spot either: at
    shelf1 (2.55, 4) the nearest entry WITH a block, shelf2 0.55 m away, was
    picked (2026-10-13).
    """
    if loc:
        try:
            key = env_key(loc)
        except Exception as e:
            log_data({'msg': f'qa: location {loc!r} unknown ({e})'})
            key = None
        return key, _qa_block(ENV.get(key)) if key else None
    if node is None:
        return None, None
    try:
        env = dict(ENV.items())
        key = get_closest_loc(node, env, threshold=float(cfg['loc_radius']))
    except Exception as e:
        log_data({'msg': f'qa: location unknown ({e})'})
        return None, None
    if key and not _qa_block(env.get(key)):
        at = env[key]['loc']
        for k, v in env.items():
            l = v.get('loc') if isinstance(v, dict) else None
            if (_qa_block(v) and isinstance(l, dict) and 'x' in l
                    and np.hypot(l['x'] - at['x'], l['y'] - at['y']) <= QA_SAME_SPOT):
                key = k
                break
    return key, _qa_block(env.get(key)) if key else None


def _batchim(word: str) -> bool:
    """Does the Korean *word* end in a final consonant (과 vs 와)?"""
    ch = word.strip()[-1:] if word.strip() else ''
    return bool(ch) and 0xAC00 <= ord(ch) <= 0xD7A3 and (ord(ch) - 0xAC00) % 28 != 0


def _place_answer(ids: list, places: dict, lang: str) -> str:
    """Spoken answer naming `ids` with the place phrases of `lang`."""
    names = [places[i].get(lang) or places[i].get('en') or places[i].get('desc', i) for i in ids]
    if lang == 'ko':
        joined = names[0]
        for n in names[1:]:
            joined += ('과 ' if _batchim(joined) else '와 ') + n
        return f'{joined}에 있어요.'
    return f'It is {" and ".join(names)}.'


# What a where-question asks about when it names no object: the furniture.
_NOT_OBJECTS = {'shelf', 'shelve', 'cabinet', 'cupboard', 'rack', 'closet', 'furniture',
                'thing', 'object', 'item', 'it', 'something', 'anything', 'everything'}


def _no_object(asked_kind) -> bool:
    k = _norm_kind(asked_kind)
    return not k or k.split()[-1] in _NOT_OBJECTS


def _asked_items(items: list, asked_kind) -> list:
    """The items that are the kind asked about. Decided here from the kinds,
    not by the model's `asked` flag: asked "머그컵 어디?" it named the red cup a
    "tumbler" and still flagged it asked. An item matches when its kind has
    every word of the asked kind ("mug" → mug, travel mug; not tumbler; "wine
    glass" → not glass). No item matching → the flags, plus their kind-mates
    (the second of two mugs often goes unflagged)."""
    want = _norm_kind(asked_kind).split()
    if want:
        hit = [it for it in items if set(want) <= set(_norm_kind(it.get('what')).split())]
        if hit:
            return hit
    kinds = {_norm_kind(it.get('what')) for it in items if it.get('asked') is True}
    return [it for it in items if it.get('asked') is True or _norm_kind(it.get('what')) in kinds]


def _norm_kind(what) -> str:
    """'Wine glasses' / 'wine glass' → 'wine glass'."""
    w = ' '.join(str(what or '').lower().split())
    if w.endswith(('sses', 'xes', 'ches', 'shes')):
        return w[:-2]
    if w.endswith('s') and not w.endswith('ss'):
        return w[:-1]
    return w


def _ask_place(cfg: dict, shots: list, question: str, history: list, lang: str, qa_cfg: dict) -> str:
    """Answer at a place with a `qa` block: where-questions from its places."""
    places = qa_cfg['places']
    views = _view_labels(shots)
    system = _PLACE_PROMPT.format(
        n=len(shots), views=views, scene=qa_cfg.get('scene', 'an indoor place'),
        frame=_FRAME.get(qa_cfg.get('frame', 'robot'), _FRAME['robot']),
        places='\n'.join(f'- {k}: {v.get("desc", k)}' for k, v in places.items()),
        language=_LANG_NAME.get(lang, 'the language of the question'))
    item = {'type': 'object', 'required': ['what', 'place'], 'properties': {
        'what': {'type': 'string'}, 'place': {'type': 'string', 'enum': list(places)}}}
    schema = {'type': 'object', 'required': ['kind', 'items', 'answer'], 'properties': {
        'kind': {'type': 'string', 'enum': ['where', 'other']},
        'items': {'type': 'array', 'maxItems': 6, 'items': item},
        'answer': {'type': 'string'}}}
    msgs = [{'role': 'system', 'content': system}]
    for q, a in history[-int(cfg['history_turns']):]:
        msgs += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
    msgs.append({'role': 'user', 'content': question,
                 'images': [_b64jpeg(im, int(cfg['max_side'])) for _, im in shots]})
    body = {'model': cfg['model'], 'messages': msgs, 'stream': False, 'think': False,
            'format': schema, 'keep_alive': cfg['keep_alive'],
            # Short cap: the JSON is ~100 tokens, and a model that starts
            # repeating itself would otherwise run on until the JSON is cut.
            'options': {'temperature': 0.0, 'num_predict': 300}}
    r = _post_chat(cfg, body)
    try:
        out = json.loads(r.json().get('message', {}).get('content', '') or '{}')
    except ValueError:                      # cut-off JSON: answer free-form instead
        log_data({'msg': 'qa: place answer was not valid JSON, answering free-form'})
        return _ask_vlm(cfg, shots, question, history, lang)
    log_data({'msg': f'qa: {json.dumps(out, ensure_ascii=False)}'})
    # The model also files "행거에 걸린 옷은 무슨 색이야?" under where; only a
    # question that asks where gets the place answer.
    if out.get('kind') != 'where' or not _WHERE.search(question.lower()):
        return _first_sentence(out.get('answer', '')) or _ask_vlm(cfg, shots, question, history, lang)
    ids = [i for i in dict.fromkeys(it.get('place') for it in out.get('items') or []
                                    if isinstance(it, dict)) if i in places]
    return _place_answer(ids, places, lang) if ids else _qa_phrase('unseen', lang)



# ── Where on the shelves: shelf boards and dividers from depth ───────────────
#
# Place boxes drawn on a photo (above) hold for one spot only, and the 8B model
# cannot count shelf levels itself (asked for 3 boards it returned 12). So for
# a where-question the model only boxes the objects (with their own kind) and
# the shelf unit, and the head camera's depth gives the structure in metres:
#
# * every head photo also keeps its depth frame and the camera pose in the base
#   frame, fitted (Kabsch) to a few pixels sent through the `get3d` service
#   while the head is still at that tilt (12 pixels: < 3 mm off over the image);
# * inside the shelf unit's box: horizontal surfaces (boards, seen from below
#   or above) wider than `shelf_min_width` are the levels' bounds, vertical
#   surfaces facing sideways away from its sides are the dividers;
# * an object's level is the interval its lower half is in, its column the
#   number of dividers on its left — "오른쪽 선반 가운데 칸".
#
# Nothing is configured per place, so it works wherever the robot stands.
# At shelf2 (scene 1): boards at 1.61 / 1.85 / 2.12 / 2.35 m, divider found,
# camera at 1.47 m. Used for where-questions at a place without a `qa` block
# or with `'shelves': True`; nothing found → the place layout / free answer.

_SHELF_DEFAULTS = {
    'shelf_unit': 'the shelves / cabinet / rack that holds the things asked about (the whole unit, all its shelves)',
    'shelf_min_width': 0.35,    # share of the unit's width a board spans (two mug rims side by side do not)
    'shelf_min_height': 0.6,    # share of the shelves' height a divider spans (a cup's side does not)
    'shelf_merge': 0.08,        # m: surfaces closer than this are one board (its top and underside)
    'shelf_step': 8,            # px: depth grid
}

_SHELF_PROMPT = """\
You are the eyes of a home-care robot talking with a person. You are given {n} \
photos taken just now by the robot's head camera: {views}.

Classify the person's question and reply with JSON only:
- kind "where": they ask where something is. items = EVERY object you can \
see on or in {unit}, one entry per thing: what it is (its own kind in a few \
English words: mug, wine glass, tumbler, cup, bowl, plate ...), photo = the \
number of the photo (1..{n}), bbox_2d = [x1, y1, x2, y2] in that photo in \
0-1000 coordinates, and asked = true only when it is the kind of thing the \
person asks about — a tumbler is not a mug, a mug is not a cup or a glass. \
asked_kind = the kind of thing they ask about, in the same words as `what` \
(e.g. "mug"). units = for each photo showing it, photo and bbox_2d of {unit}. \
answer = "".
- kind "other": any other question. items = [], units = [], asked_kind = "", answer = one \
short sentence in {language} (at most about 15 words), from what is visible; \
if it is not visible say so. Never guess."""


def _kabsch(A, B):
    """R, t with B ≈ A @ R.T + t (least squares, no scaling)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    return R, cb - R @ ca


def _capture_geo(node, shape):
    """The head camera as it is now: {'depth' (mm, `shape`), 'K' (fx, fy, cx,
    cy), 'R', 't' (camera → base_footprint)}; raises with the reason when it
    cannot (no depth / get3d)."""
    a = node.agents if node is not None else {}
    dcam, pcam, g3 = a.get('head_depth'), a.get('head_cam_params'), a.get('get3d')
    if dcam is None or pcam is None or g3 is None:
        raise RuntimeError('needs the head_depth, head_cam_params and get3d connections')
    depth = _fresh_frame(dcam, 0.0, 1.5)
    params = pcam.rev_data.get('cam_params') if isinstance(pcam.rev_data, dict) else None
    if depth is None:
        raise RuntimeError('no fresh head_depth frame')
    if not params:
        raise RuntimeError(f'no head_cam_params ({pcam.rev_data!r})')
    if depth.shape[:2] != tuple(shape):
        depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    fx, fy, cx, cy = [float(p) for p in params]
    # 16 pixels with depth, spread over the picture
    H, W = depth.shape[:2]
    vv, uu = np.mgrid[H // 8:H:H // 4, W // 8:W:W // 4]
    pts = []
    for u, v in zip(uu.ravel(), vv.ravel()):
        win = depth[max(0, v - 20):v + 20, max(0, u - 20):u + 20]
        ys, xs = np.nonzero(win)
        if len(ys):
            pts.append((int(max(0, u - 20) + xs[len(xs) // 2]), int(max(0, v - 20) + ys[len(ys) // 2])))
    if len(pts) < 6:
        raise RuntimeError(f'depth on {len(pts)} of 16 sample pixels')
    d = [float(depth[v, u]) / 1000.0 for u, v in pts]
    ret = g3.send({'u_array': [u for u, _ in pts], 'v_array': [v for _, v in pts],
                   'd_array': d, 'base_frame': 'base_footprint'})
    if not isinstance(ret, dict) or not ret.get('isdone'):
        raise RuntimeError(f'get3d failed: {ret!r}'[:300])
    B = np.stack([np.asarray(ret['x'], float), np.asarray(ret['y'], float), np.asarray(ret['z'], float)], -1)
    C = np.array([((u - cx) * z / fx, (v - cy) * z / fy, z) for (u, v), z in zip(pts, d)])
    ok = np.isfinite(B).all(1)
    if ok.sum() < 6:
        raise RuntimeError(f'get3d gave {int(ok.sum())} usable points')
    R, t = _kabsch(C[ok], B[ok])
    err = float(np.abs(C[ok] @ R.T + t - B[ok]).max())
    if err > 0.02:                                     # not one rigid camera: do not trust it
        raise RuntimeError(f'camera pose fit off by {err:.3f} m')
    return {'depth': depth, 'K': (fx, fy, cx, cy), 'R': R, 't': t}


def _grid_xyz(geo, step: int):
    """Base-frame xyz on a pixel grid (NaN without depth), and the grid's (u, v)."""
    depth, (fx, fy, cx, cy) = geo['depth'], geo['K']
    H, W = depth.shape[:2]
    vv, uu = np.mgrid[step // 2:H:step, step // 2:W:step]
    z = depth[vv, uu].astype(float) / 1000.0
    C = np.stack([(uu - cx) * z / fx, (vv - cy) * z / fy, z], -1)
    P = C @ geo['R'].T + geo['t']
    P[z <= 0] = np.nan
    return P, uu, vv


def _clusters(vals, gap: float):
    """Indices of `vals` split where sorted neighbours are more than `gap` apart."""
    o = np.argsort(vals)
    cuts = np.where(np.diff(vals[o]) > gap)[0] + 1
    return np.split(o, cuts)


def _shelf_structure(P, uu, vv, unit_px, objects_px, cfg, edge_px=None):
    """(board heights, divider y's — left first, open_top, open_bottom) inside
    the unit's box, or None. open_*: the unit goes on past the photo's edge
    there (a level we cannot see the end of: count from the other side).
    `objects_px`: every object's box — their tops and sides are not boards /
    dividers (two mug rims made a board 12 cm above the real one). `edge_px`:
    the unit's box as the model gave it, for the cut-edge test (`unit_px` is
    grown, and touched the photo's top when the unit did not)."""
    x1, y1, x2, y2 = unit_px
    inside = (uu >= x1) & (uu <= x2) & (vv >= y1) & (vv <= y2)
    for a, b, c, d in objects_px:                      # mug rims are not boards
        inside &= ~((uu >= a - 8) & (uu <= c + 8) & (vv >= b - 8) & (vv <= d + 8))
    dx = np.full(P.shape, np.nan); dy = np.full(P.shape, np.nan)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    with np.errstate(invalid='ignore', divide='ignore'):
        n = np.cross(dx, dy)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
    ok = inside & np.isfinite(n).all(-1) & np.isfinite(P).all(-1)
    if ok.sum() < 20:
        return None
    unit_y = P[ok][:, 1]
    lo, hi = np.percentile(unit_y, [3, 97])
    width = float(cfg['shelf_min_width']) * max(hi - lo, 0.1)
    boards = []
    hz = ok & (np.abs(n[..., 2]) > 0.92)
    pts = P[hz]
    for idx in _clusters(pts[:, 2], 0.03) if len(pts) else []:
        if len(idx) >= 5 and np.ptp(pts[idx, 1]) >= width:
            boards.append(float(np.median(pts[idx, 2])))
    merged = []
    for z in sorted(boards):
        if merged and z - merged[-1] < float(cfg['shelf_merge']):
            merged[-1] = z                              # top and underside of one board
        else:
            merged.append(z)
    if len(merged) < 2:
        return None
    # dividers: surfaces facing sideways, tall, away from the unit's own sides
    tall = max(0.15, float(cfg['shelf_min_height']) * (merged[-1] - merged[0]))
    side = ok & (np.abs(n[..., 1]) > 0.9)
    pts = P[side]
    divs = []
    for idx in _clusters(pts[:, 1], 0.03) if len(pts) else []:
        y = float(np.median(pts[idx, 1]))
        if len(idx) >= 5 and np.ptp(pts[idx, 2]) >= tall and lo + 0.07 < y < hi - 0.07:
            divs.append(y)
    cols = []
    for y in sorted(divs, reverse=True):               # +y is the robot's left
        if cols and cols[-1] - y < 0.10:                # the two faces of one divider
            continue
        cols.append(y)
    # Cut by the photo: the unit's box reaches the edge and some of the unit
    # lies beyond the last board found (head tilted 10°: the top level's
    # ceiling was out of view and its middle level was called "top").
    H = float(vv.max() + vv.min())                     # image height (from the grid centres)
    ey1, ey2 = (edge_px[1], edge_px[3]) if edge_px is not None else (y1, y2)
    # Another level past the last board shows the unit's back wall there
    # (surfaces facing the robot as deep as the back wall between the boards);
    # the frame of a unit that ends there is nearer (shelf2: 1.33 vs 1.03 m).
    # The model's box alone is not enough: it reached the top for a unit that
    # did not. The robot faces the unit, so "facing" is along base x.
    facing = ok & (np.abs(n[..., 0]) > 0.85)
    fz, fx = P[facing][:, 2], P[facing][:, 0]
    inner = (fz > merged[0] + 0.05) & (fz < merged[-1] - 0.05)
    back = float(np.median(fx[inner])) if inner.sum() >= 5 else None

    def beyond(sel):
        return back is not None and int((sel & (np.abs(fx - back) < 0.06)).sum()) >= 8
    open_top = ey1 <= 0.02 * H and beyond(fz > merged[-1] + 0.04)
    open_bottom = ey2 >= 0.98 * H and beyond(fz < merged[0] - 0.04)
    return merged, cols, open_top, open_bottom


def _object_spot(P, uu, vv, box_px):
    """(z, y) of an object: its lower half's 20th-percentile height and median
    y; for a see-through glass, the board just under its box. None without depth."""
    a, b, c, d = box_px
    lower = (uu >= a) & (uu <= c) & (vv >= (b + d) / 2) & (vv <= d)
    pts = P[lower & np.isfinite(P).all(-1)]
    if len(pts) >= 3:
        return float(np.percentile(pts[:, 2], 20)), float(np.median(pts[:, 1]))
    below = (uu >= a) & (uu <= c) & (vv > d) & (vv <= d + 24)
    pts = P[below & np.isfinite(P).all(-1)]
    if len(pts) >= 3:
        return float(np.median(pts[:, 2])) + 0.03, float(np.median(pts[:, 1]))
    return None


_INVENTORY_PROMPT = """\
List EVERY object you can see in these {n} photos — every cup, glass, mug, \
bottle, box, bowl, plate, bag ... wherever it is — but not the furniture, \
shelves, doors or walls. One entry per thing: what (a few English words), \
photo (1..{n}) and bbox_2d [x1, y1, x2, y2] in that photo, 0-1000. JSON only."""


def _inventory_boxes(cfg: dict, use: list) -> dict:
    """{photo number: [0-1000 boxes]} of every object in the photos: asked
    about one thing, the model lists only that one, and the others' rims and
    sides then look like boards and dividers."""
    box = {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 4, 'maxItems': 4}
    schema = {'type': 'object', 'required': ['items'], 'properties': {'items': {
        'type': 'array', 'maxItems': 40, 'items': {'type': 'object', 'required': ['what', 'photo', 'bbox_2d'],
                                                   'properties': {'what': {'type': 'string'},
                                                                  'photo': {'type': 'integer'}, 'bbox_2d': box}}}}}
    body = {'model': cfg['model'], 'stream': False, 'think': False, 'format': schema,
            'keep_alive': cfg['keep_alive'], 'options': {'temperature': 0.0, 'num_predict': 1500},
            'messages': [{'role': 'user', 'content': _INVENTORY_PROMPT.format(n=len(use)),
                          'images': [_b64jpeg(im, int(cfg['max_side'])) for _, im in use]}]}
    try:
        items = json.loads(_post_chat(cfg, body).json().get('message', {}).get('content', '') or '{}').get('items')
    except ValueError:
        return {}
    out = {}
    for it in items or []:
        if isinstance(it, dict) and isinstance(it.get('bbox_2d'), list) and len(it['bbox_2d']) == 4:
            out.setdefault(int(it.get('photo') or 1), []).append(it['bbox_2d'])
    return out


def _shelf_words(col: int, ncols: int, level: int, nlevels: int, lang: str, count: str = '') -> str:
    """"오른쪽 선반 가운데 칸" — `col` from the left, `level` from the bottom
    (1-based). count 'bottom' / 'top': the other end is out of view, so the
    level is counted from this end ("아래에서 두 번째 칸"); 'none': both are."""
    if count == 'top':
        level, nlevels = nlevels - level + 1, nlevels
    if lang == 'ko':
        c = {1: [''], 2: ['왼쪽', '오른쪽'], 3: ['왼쪽', '가운데', '오른쪽']}.get(ncols)
        c = c[col - 1] if c else f'왼쪽에서 {col}번째'
        nth = {1: '첫', 2: '두', 3: '세', 4: '네', 5: '다섯'}.get(level, str(level))
        if count in ('bottom', 'top'):
            lv = f"{'아래' if count == 'bottom' else '위'}에서 {nth} 번째"
        elif count == 'none':
            lv = ''
        else:
            lv = {1: [''], 2: ['아래', '위'], 3: ['아래', '가운데', '위']}.get(nlevels)
            lv = lv[level - 1] if lv else f'아래에서 {nth} 번째'
        return ' '.join(x for x in (c, '선반', f'{lv} 칸' if lv else '') if x)
    c = {1: [''], 2: ['left', 'right'], 3: ['left', 'middle', 'right']}.get(ncols)
    c = c[col - 1] if c else f'column {col} from the left'
    if count in ('bottom', 'top'):
        lv = f"shelf {level} from the {count}"
        return f'{lv}, on the {c}' if c else lv
    lv = '' if count == 'none' else {1: [''], 2: ['bottom', 'top'], 3: ['bottom', 'middle', 'top']}.get(nlevels)
    if isinstance(lv, list):
        lv = lv[level - 1]
    elif lv is None:
        lv = f'shelf {level} from the bottom'
    if lv and c:
        return f'the {lv} shelf on the {c}' if nlevels <= 3 else f'{lv}, on the {c}'
    return f'the {lv} shelf' if lv else (f'the {c} side' if c else 'the shelf')


def _shelf_answer(spots: list, lang: str) -> str:
    if lang == 'ko':
        out = spots[0]
        for s in spots[1:]:
            out += ('과 ' if _batchim(out) else '와 ') + s
        return f'{out}에 있어요.'
    return 'It is on ' + ' and on '.join(spots) + '.'


def _ask_shelves(cfg: dict, shots, question: str, history: list, lang: str, qa_cfg) -> str | None:
    """Where-question answered from the shelf structure (see above); None when
    no shelf unit / structure was found, so the caller answers another way."""
    scfg = {**_SHELF_DEFAULTS, **{k: v for k, v in (qa_cfg or {}).items() if k in _SHELF_DEFAULTS},
            **{k: cfg[k] for k in _SHELF_DEFAULTS if k in cfg}}
    geo = getattr(shots, 'geo', {}) or {}
    use = [(v, im) for v, im in shots if v in geo]
    if not use:
        return None
    n = len(use)
    unit = str(scfg['shelf_unit'])
    box = {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 4, 'maxItems': 4}
    schema = {'type': 'object', 'required': ['kind', 'asked_kind', 'items', 'units', 'answer'], 'properties': {
        'kind': {'type': 'string', 'enum': ['where', 'other']},
        'asked_kind': {'type': 'string'},
        'items': {'type': 'array', 'maxItems': 16, 'items': {
            'type': 'object', 'required': ['what', 'photo', 'bbox_2d', 'asked'], 'properties': {
                'what': {'type': 'string'}, 'photo': {'type': 'integer'}, 'bbox_2d': box,
                'asked': {'type': 'boolean'}}}},
        'units': {'type': 'array', 'maxItems': 6, 'items': {
            'type': 'object', 'required': ['photo', 'bbox_2d'], 'properties': {
                'photo': {'type': 'integer'}, 'bbox_2d': box}}},
        'answer': {'type': 'string'}}}
    system = _SHELF_PROMPT.format(n=n, views=_view_labels(use), unit=unit,
                                  language=_LANG_NAME.get(lang, 'the language of the question'))
    # History only with memory=True (qa passes [] otherwise): after "와인잔
    # 어디?" the model answered "머그컵 어디?" with the wine glasses again.
    msgs = [{'role': 'system', 'content': system}]
    for q, a in history[-int(cfg['history_turns']):]:
        msgs += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
    msgs.append({'role': 'user', 'content': question,
                 'images': [_b64jpeg(im, int(cfg['max_side'])) for _, im in use]})
    body = {'model': cfg['model'], 'messages': msgs, 'stream': False, 'think': False,
            'format': schema, 'keep_alive': cfg['keep_alive'],
            'options': {'temperature': 0.0, 'num_predict': 1000}}
    try:
        out = json.loads(_post_chat(cfg, body).json().get('message', {}).get('content', '') or '{}')
    except ValueError:
        return None
    if out.get('kind') != 'where':
        return None

    def px(b, im):                                     # 0-1000 → pixels of that photo
        h, w = im.shape[:2]
        return [b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000]

    items = [it for it in out.get('items') or [] if isinstance(it, dict)
             and isinstance(it.get('bbox_2d'), list) and len(it['bbox_2d']) == 4]
    # "선반 어디에 있어?" — Whisper lost the object word: the model asked for
    # "shelf", nothing matched, and every object was read out. Say so instead.
    if _no_object(out.get('asked_kind')):
        log_data({'msg': f"qa: no object in the question ({question!r}, asked_kind "
                         f"{out.get('asked_kind')!r}) — asking again"})
        return _qa_phrase('which', lang)
    asked = _asked_items(items, out.get('asked_kind'))
    spots, notes, found_any = [], [], False
    step = int(scfg['shelf_step'])
    inv = getattr(shots, 'objects', None)               # every object: once per set of photos
    if inv is None:
        inv = _inventory_boxes(cfg, use)
        try:
            shots.objects = inv
        except AttributeError:
            pass
        notes.append(f'inventory: {sum(len(v) for v in inv.values())} objects')
    for i, (view, im) in enumerate(use, 1):
        units = [u['bbox_2d'] for u in out.get('units') or []
                 if isinstance(u, dict) and u.get('photo') == i and isinstance(u.get('bbox_2d'), list)
                 and len(u['bbox_2d']) == 4]
        mine = [it for it in asked if int(it.get('photo') or 1) == i]
        if not units or not mine:
            continue
        P, uu, vv = _grid_xyz(geo[view], step)
        objects_px = [px(it['bbox_2d'], im) for it in items if int(it.get('photo') or 1) == i] \
            + [px(b, im) for b in inv.get(i, [])]
        for ub in units:
            # The unit's own top / bottom boards lie on its box's edges, and the
            # model's box moves by 1-2 %: 738 instead of 750 lost the bottom
            # board. 5 % more each side (a board just outside merges with it).
            mx, my = 0.05 * (ub[2] - ub[0]), 0.05 * (ub[3] - ub[1])
            grown = [max(0, ub[0] - mx), max(0, ub[1] - my), min(1000, ub[2] + mx), min(1000, ub[3] + my)]
            st = _shelf_structure(P, uu, vv, px(grown, im), objects_px, scfg, edge_px=px(ub, im))
            if st is None:
                notes.append(f'photo {i}: no shelf boards in {ub}')
                continue
            found_any = True
            boards, divs, open_top, open_bottom = st
            # levels: between boards, plus one past a board at a cut edge
            lv0 = 0 if open_bottom else 1
            nlev = len(boards) - 1 + int(open_top) + int(open_bottom)
            count = ('none' if open_top and open_bottom else 'bottom' if open_top
                     else 'top' if open_bottom else '')
            notes.append(f"photo {i}: boards {[round(z, 2) for z in boards]} m, dividers "
                         f"{[round(y, 2) for y in divs]}" + (' (cut at the top)' if open_top else '')
                         + (' (cut at the bottom)' if open_bottom else ''))
            ux1, uy1, ux2, uy2 = ub
            for it in mine:
                b = it['bbox_2d']
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if not (ux1 <= cx <= ux2 and uy1 <= cy <= uy2):
                    continue
                spot = _object_spot(P, uu, vv, px(b, im))
                if spot is None:
                    notes.append(f"{it.get('what')}: no depth")
                    continue
                z, y = spot
                below = sum(1 for bz in boards if bz <= z + 0.02)     # boards under it
                if not (1 - int(open_bottom)) <= below <= len(boards) - 1 + int(open_top):
                    notes.append(f"{it.get('what')}: z {z:.2f} m outside the shelves")
                    continue
                level = below + 1 - lv0                               # 1 = lowest level
                col = 1 + sum(1 for dy in divs if dy > y)
                words = _shelf_words(col, len(divs) + 1, level, nlev, lang, count)
                notes.append(f"{it.get('what')}: z {z:.2f} y {y:.2f} → {words}")
                if words not in spots:
                    spots.append(words)
    log_data({'msg': 'qa: shelves — ' + '; '.join(notes) + ' | ' + json.dumps(
        {'asked_kind': out.get('asked_kind'),
         'items': [{k: it.get(k) for k in ('what', 'asked', 'photo', 'bbox_2d')} for it in items],
         'units': out.get('units')}, ensure_ascii=False)})
    if not found_any:
        return None
    if spots:
        return _shelf_answer(spots, lang)
    if asked:                                          # seen, but not placed on a shelf
        return _qa_phrase('no_shelf', lang)
    return _qa_phrase('unseen', lang)

# ── Products (신라면, 짜파게티 …): names from the product recogniser ───────────
#
# The VLMs cannot read pack labels reliably: on the arm-camera shelf photo they
# answered with colours only, or — given example names in the prompt — repeated
# an example that was not on the shelf (진라면). So product names come from
# skills/_products.py (detector + SigLIP match against reference photos), which
# names a pack only when it is sure; the robot serves blind users and must not
# guess a product.

# Words that make a question about products even without a product name.
_PRODUCT_WORDS = re.compile(r'라면|컵라면|봉지면|ramen|ramyun|ramyeon|noodle', re.IGNORECASE)

_SIDE = {'ko': {'left': '왼쪽', 'middle': '가운데', 'right': '오른쪽'},
         'en': {'left': 'on the left', 'middle': 'in the middle', 'right': 'on the right'}}


def _is_product_question(text: str) -> bool:
    from . import _products
    return bool(_products.names_in(text) or _PRODUCT_WORDS.search(text))


def _inventory(cfg: dict, shots: list) -> list:
    """Packs in the photos: [{name|None, side, view, ...}] — see _products.inventory.
    Head photos overlap, so a product seen in several is listed once (its
    first view); unnamed packs are counted from the photo with the most."""
    from . import _products
    from ._recognition_helpers import _object_det_select_configs, _predict_detect, _vs_client
    det, _sel = _object_det_select_configs()
    client = _vs_client()
    detect = lambda rgb, prompt, th: _predict_detect(rgb, prompt, {**det, **th})  # noqa: E731
    named, unnamed = {}, []
    for view, im in shots:
        packs = _products.inventory(client, im, detect)
        for p in packs:
            if p['name'] and p['name'] not in named:
                named[p['name']] = {**p, 'view': view}
        anon = [{**p, 'view': view} for p in packs if not p['name']]
        if len(anon) > len(unnamed):
            unnamed = anon
    return list(named.values()) + unnamed


def _join(words: list, lang: str) -> str:
    if lang == 'ko':
        out = words[0]
        for w in words[1:]:
            out += ('과 ' if _batchim(out) else '와 ') + w
        return out
    sep = {'en': ' and '}.get(lang, ', ')
    return ', '.join(words[:-1]) + sep + words[-1] if len(words) > 1 else words[0]


def _product_answer(names: list, inv: list, lang: str) -> str:
    """Spoken answer for a question naming catalogue products: where each one
    is, or that it is not seen — straight from the recogniser, no model."""
    side = _SIDE.get(lang, _SIDE['en'])
    parts = []
    for n in names:
        sides = sorted({p['side'] for p in inv if p['name'] == n}, key=['left', 'middle', 'right'].index)
        topic = n + ('은' if _batchim(n) else '는')
        if sides:
            where = _join([side[x] for x in sides], lang)
            parts.append({'ko': f'{topic} {where}에 있어요.'}.get(lang, f'{n} is {where}.'))
        else:
            parts.append({'ko': f'{topic} 보이지 않아요.'}.get(lang, f"I can't see {n}."))
    unknown = sum(1 for p in inv if not p['name'])
    if unknown and any('보이지' in x or "can't" in x for x in parts):
        parts.append({'ko': f'이름을 알 수 없는 라면이 {unknown}개 있어요.'}.get(
            lang, f"There {'is' if unknown == 1 else 'are'} {unknown} pack(s) I can't name."))
    return ' '.join(parts)


def _product_facts(inv: list) -> str:
    """The recogniser's findings as prompt text for a general product question."""
    named = [f"{p['name']} ({p['side']})" for p in inv if p['name']]
    unknown = sum(1 for p in inv if not p['name'])
    return ('Product packs, identified by the robot\'s product recogniser (trust these over the photos): '
            + (', '.join(named) if named else 'none identified')
            + (f'; plus {unknown} pack(s) it could not identify' if unknown else '') + '.\n'
            + '- Name products ONLY with these names, exactly as written. Never name any other product, '
              'even if you think you can read one.\n'
            + '- A pack that could not be identified: say there is one you cannot name.\n'
            + '- left / middle / right are from the camera\'s point of view.')


# "무슨 라면 있어?" / "라면 있어?" / "what noodles are there?" — answered with the
# recogniser's list (_product_list); other product questions go to the model.
_LIST_QUESTION = re.compile(r'무슨|어떤|뭐|무엇|있어|있나|있니|있습니까|있어요'
                            r'|\b(what|which|any)\b|\b(is|are) there\b', re.IGNORECASE)


# Words ending in 라면 / 면 that name no product ("라면 몇 개", "컵라면").
_GENERIC_NOODLE = {'라면', '컵라면', '봉지라면', '봉지면', '면', '이면', '라면은', '라면이', '라면을'}
_NOODLE_NAME = re.compile(r'([가-힣A-Za-z]{1,8}(?:라면|탕면|면))')


def _unknown_names(text: str) -> list:
    """Noodle names in `text` that are not in the product catalogue (진라면 …)."""
    from . import _products
    known = set()
    for key, entry in (_products._catalog().get('products') or {}).items():
        known |= {_products._norm(n) for n in [key, *entry.get('aliases', [])]}
    out = []
    for m in _NOODLE_NAME.findall(text):
        n = _products._norm(m)
        # "무슨라면" / "어떤라면" written together are the question, not a name.
        if n in _GENERIC_NOODLE or n in known or re.match(r'(무슨|어떤|어느|몇|뭔|무엇|이런|그런|저런)', n):
            continue
        out.append(m)
    return out


def _product_list(inv: list, lang: str) -> str:
    """Every pack the recogniser sees, by side: the safe answer when the
    question or the model names a product it does not know."""
    side = _SIDE.get(lang, _SIDE['en'])
    groups = []
    for sd in ('left', 'middle', 'right'):
        names = [p['name'] for p in inv if p['name'] and p['side'] == sd]
        if names:
            groups.append((side[sd], _join(names, lang)))
    unknown = sum(1 for p in inv if not p['name'])
    if lang == 'ko':
        txt = ', '.join(f'{s}에 {n}' for s, n in groups)
        txt = (txt + '이 있어요.' if txt and _batchim(txt) else txt + '가 있어요.') if txt else '알아볼 수 있는 라면이 없어요.'
        return txt + (f' 이름을 알 수 없는 라면이 {unknown}개 더 있어요.' if unknown else '')
    txt = '; '.join(f'{n} {s}' for s, n in groups) or "I can't identify any pack"
    return txt + (f"; and {unknown} pack(s) I can't name." if unknown else '.')


def _unrecognised_answer(names: list, inv: list, lang: str) -> str:
    """A question about a product the robot cannot recognise (not in the
    catalogue): say so — never yes or no — then what it does see."""
    n = names[0]
    head = {'ko': f"{n}{'은' if _batchim(n) else '는'} 확인할 수 없어요."}.get(lang, f"I can't recognise {n}.")
    return head + ' ' + _product_list(inv, lang)


# ── "어디에 떨어졌어?" — a dropped object, as a clock direction from the person ──
#
# The VLMs could not do it from the photos (8 / 8 wrong on the head camera, 3–11
# o'clock on the arm camera for a spoon at 2). So it is geometry: the detector
# on the ARM camera finds the person's feet ('foot' / 'slipper' — it labels the
# two feet either way), their legs and the object; depth gives each a point on
# the floor (base frame), and:
#   two feet → centre between them; 12 o'clock is perpendicular to the line
#              between them, on the side the feet stick out from the legs;
#   one foot → centre on that foot; 12 o'clock is the way it points (leg → foot).
# No feet, or no leg to tell front from back → say the direction is unknown:
# the user is blind, an answer relative to the ROBOT is no use to them.

_DROP = re.compile(r'떨어|떨궜|떨어뜨|\bdrop|\bfall|\bfell\b', re.IGNORECASE)

# Spoken word → detector prompt. QA_CONFIGS['drop_objects'] adds / overrides.
_DROP_OBJECTS = {
    '숟가락': 'spoon', '수저': 'spoon', '스푼': 'spoon', '포크': 'fork', '젓가락': 'chopsticks',
    '컵': 'cup', '휴대폰': 'phone', '핸드폰': 'phone', '리모컨': 'remote control', '안경': 'glasses',
    '열쇠': 'key', '펜': 'pen', '약': 'pill bottle', '지갑': 'wallet',
    'spoon': 'spoon', 'fork': 'fork', 'chopsticks': 'chopsticks', 'cup': 'cup', 'phone': 'phone',
    'remote': 'remote control', 'glasses': 'glasses', 'key': 'key', 'pen': 'pen',
}


def _is_drop_question(text: str) -> bool:
    return bool(_DROP.search(text or ''))


def _drop_vocab(cfg: dict) -> dict:
    return {**_DROP_OBJECTS, **(cfg.get('drop_objects') or {})}


def _drop_object(cfg: dict, text: str):
    """(word, detector prompt) for the object asked about, or None. A word
    Whisper misheard is matched by Hangul-letter similarity — "손가락" (finger)
    for 숟가락 came back on a real voice and fell through to the left / right
    place answer."""
    vocab = _drop_vocab(cfg)
    t = (text or '').lower()
    hits = [(w, p) for w, p in vocab.items() if w.lower() in t]
    if hits:
        return max(hits, key=lambda h: len(h[0]))                      # "숟가락" over "약"
    words = re.findall(r'[가-힣]{2,}|[a-z]{3,}', t)
    best, score = None, 0.0
    for tok in words:
        for w, p in vocab.items():
            if len(w) < 2:
                continue
            for cand in (tok, tok[:len(w)]):                           # "숟가락이" → "숟가락"
                s = _similarity(_jamo(cand), _jamo(w.lower()))
                if s > score:
                    best, score = (w, p), s
    return best if score >= 0.72 else None


def _box_inside(a, b, frac=0.6) -> bool:
    """xyxy box a lies mostly inside b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy > frac * max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def _shin_xy(node, cam, box):
    """Where the leg above a foot stands on the floor plan: base-frame (x, y)
    median of the depth points just above the foot box, 12–60 cm above the
    floor and within 25 cm of the foot. None when there is no leg there (an
    empty slipper) or no depth. The detector's own 'leg' boxes came and went
    from frame to frame (0, 0, 2, 1 in four runs), flipping front and back."""
    import numpy as np
    from .pointcloud import get3d_arm
    x0, y0, x1, y1 = [int(v) for v in box]
    h = max(8, y1 - y0)
    depth = np.asarray(cam.depth)
    pts = []
    for v in range(max(0, y0 - int(1.5 * h)), max(1, y0), 6):
        for u in range(max(0, x0), min(depth.shape[1], x1), 6):
            d = float(depth[v, u])
            if d > 0:
                pts.append([u, v, d])
    if len(pts) < 5:
        return None
    xyz = np.asarray(get3d_arm(node=node, points=pts)['pose'], dtype=float)
    return xyz


def _clock_from_feet(feet: list, obj_xy):
    """(hour, distance_m, how) or (None, None, why). `feet`: [(foot xy on the
    floor, shin xy or None)]; only feet with a leg above them count."""
    import math
    worn = [(f, s) for f, s in feet if s is not None]
    if not worn:
        return None, None, 'no foot with a leg above it'
    if len(worn) >= 2:
        # the pair closest to a normal stance (feet 8–50 cm apart)
        pairs = [(a, b) for i, a in enumerate(worn) for b in worn[i + 1:]
                 if 0.08 <= math.dist(a[0], b[0]) <= 0.5]
        if not pairs:
            worn = [max(worn, key=lambda w: math.dist(w[0], w[1]))]
        else:
            (a, la), (b, lb) = min(pairs, key=lambda p: abs(math.dist(p[0][0], p[1][0]) - 0.25))
            c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            n = (-(b[1] - a[1]), b[0] - a[0])               # ⟂ to the line between the feet
            fwd = (c[0] - (la[0] + lb[0]) / 2, c[1] - (la[1] + lb[1]) / 2)   # legs → feet: toes' side
            if n[0] * fwd[0] + n[1] * fwd[1] < 0:
                n = (-n[0], -n[1])
            how = 'two feet'
    if len(worn) == 1:
        (c, leg), = worn
        n = (c[0] - leg[0], c[1] - leg[1])                   # one foot: the way it points
        how = 'one foot'
    if math.hypot(*n) < 1e-3:
        return None, None, 'feet direction unclear'
    v = (obj_xy[0] - c[0], obj_xy[1] - c[1])
    left = math.degrees(math.atan2(n[0] * v[1] - n[1] * v[0], n[0] * v[0] + n[1] * v[1]))
    hour = round(((-left) % 360) / 30) % 12 or 12          # clockwise from the front
    return hour, math.hypot(*v), how


def _drop_measure(node, cfg: dict, prompt: str):
    """One frame: (hour | None, distance, how, n_feet, object found)."""
    import numpy as np
    from .recognition import _detect_objects
    from ._recognition_helpers import fetch_camera_data
    camera = cfg.get('drop_camera', 'arm')
    # A person has two feet and one thing was dropped: with 'all', extra
    # "feet" (3 in one frame, 12:56) moved the answer from 2 to 4 o'clock.
    res = _detect_objects(node, ['foot', 'slipper', prompt], camera=camera, num_trials=2,
                          max_instances={'foot': 2, 'slipper': 2, prompt: 1})
    ins = res.get('ins', {})

    def pts(name):
        e = ins.get(name)
        return [(i['pose_3d'][:2], list(i['box'])) for i in (e.get('instances', [e]) if e else [])]
    feet = []
    for f in sorted(pts('foot') + pts('slipper'), key=lambda f: -(f[1][2] - f[1][0]) * (f[1][3] - f[1][1])):
        if not any(_box_inside(f[1], g[1], 0.3) or _box_inside(g[1], f[1], 0.3) for g in feet):
            feet.append(f)                                    # one foot found as foot AND slipper: once
    feet = feet[:2]                                           # the two largest: foot + slipper can give 4
    objs = pts(prompt)
    if not objs:
        return None, None, 'object not seen', len(feet), False
    cam = fetch_camera_data(node, camera)
    with_shin = []
    for xy, box in feet:
        shin = None
        xyz = _shin_xy(node, cam, box)
        if xyz is not None and len(xyz):
            near = xyz[(xyz[:, 2] > 0.12) & (xyz[:, 2] < 0.6)
                       & (np.hypot(xyz[:, 0] - xy[0], xyz[:, 1] - xy[1]) < 0.25)]
            if len(near) >= 5:
                shin = (float(np.median(near[:, 0])), float(np.median(near[:, 1])))
        with_shin.append((xy, shin))
    hour, dist, how = _clock_from_feet(with_shin, objs[0][0])
    return hour, dist, how, len(feet), True


def _drop_answer(node, cfg: dict, text: str, lang: str) -> str:
    """Spoken answer to "<object> 어디에 떨어졌어?" from the arm camera."""
    obj = _drop_object(cfg, text)
    if obj is None:
        return {'ko': '무엇이 떨어졌는지 다시 말씀해 주세요.'}.get(
            lang, 'What did you drop? Please say it again.')
    word, prompt = obj
    topic = word + ('은' if _batchim(word) else '는')
    # Two frames must agree (±1 hour): a wrong direction is worse than none.
    runs = [_drop_measure(node, cfg, prompt) for _ in range(2)]
    if not any(r[4] for r in runs):
        return {'ko': f'{topic} 보이지 않아요.'}.get(lang, f"I can't see the {word}.")
    hours = [r[0] for r in runs if r[0]]
    agree = len(hours) == 2 and min(abs(hours[0] - hours[1]), 12 - abs(hours[0] - hours[1])) <= 1
    if len(hours) == 2 and not agree:
        runs.append(_drop_measure(node, cfg, prompt))          # a third frame decides
        h3 = runs[-1][0]
        for h in hours:
            if h3 and min(abs(h - h3), 12 - abs(h - h3)) <= 1:
                hours, agree = [h, h3], True
                break
    log_data({'msg': f'qa: dropped {word} ({prompt}): ' + ' | '.join(
        f"{r[3]} feet → {f'{r[0]}h {r[1]:.2f}m ({r[2]})' if r[0] else r[2]}" for r in runs)})
    if not agree:
        return {'ko': f'{topic} 보이지만, 방향을 정확히 알 수 없어요.'}.get(
            lang, f"I can see the {word}, but I can't tell the direction for sure.")
    hour = hours[0]
    dist = next(r[1] for r in runs if r[0] == hour)
    cm = max(10, int(round(dist * 10)) * 10)
    return {'ko': f'{topic} {hour}시 방향, 약 {cm}센티미터 거리에 있어요.'}.get(
        lang, f"The {word} is at your {hour} o'clock, about {cm} cm away.")


# ── "불 꺼줘" / "세탁실 불 켜져 있어?" — lights, with turn_light / light_state ───
#
#   (H) 나 외출할거니까 불켜진거 있으면 꺼놔
#   (R) 네. 세탁실 불 끄겠습니다.  → turn_light::inputs='off', loc='laundry room'
#
#   (H) 세탁실 불 켜져 있어?
#   (R) 세탁실 불은 켜져 있어요. 꺼 드릴까요?      → light_state::loc='laundry room'
#   (H) 응  /  꺼줘
#   (R) 네. 세탁실 불 끄겠습니다.                 → turn_light::inputs='off', loc='laundry room'
#
# The robot says the acknowledgement, then acts, and says so if it failed.
# Which light: a place named in the request (a switchbot connection's loc, id
# or alias — 세탁실 …); else the light just asked about; else the default
# light (the switchbot connection marked default: kcare_bucheon's laundry
# room) — never every light at once; the others are reached by name only. A state it could not hear now is told with
# its age ("10분 전에는 …"), never as the present state.

_LIGHT_NOUN = re.compile(r'불|전등|조명|형광등|스탠드|\blights?\b|\blamps?\b', re.IGNORECASE)
_LIGHT_OFF = re.compile(r'꺼|끄|끌|끕|\boff\b', re.IGNORECASE)
_LIGHT_ON = re.compile(r'켜|켤|켭|키|\bon\b', re.IGNORECASE)
# "켜져 있어?" asks; "켜진 거 있으면 꺼놔" / "켜 줄래?" / "turn it on" asks for an action.
_LIGHT_ASK = re.compile(r'켜져|꺼져|켜졌|꺼졌|켜\s?있|꺼\s?있|켜진|꺼진|상태'
                        r'|\bon or off\b|\boff or on\b|\bstatus\b|^\s*(is|are|was|were)\b', re.IGNORECASE)
_LIGHT_DO = re.compile(r'줘|주세|줄래|줄\s?수|놔|놓|봐|\bturn\b|\bswitch\b|\bput\b', re.IGNORECASE)
_YES = re.compile(r'^\s*(네|넵|예|응|그래|좋아|부탁|yes\b|yeah\b|yep\b|sure\b|ok\b|okay\b|please\b)'
                  r'|^\s*어\s*[.!,]?\s*$', re.IGNORECASE)
_NO = re.compile(r'아니|아뇨|괜찮|그대로|그냥\s?둬|놔둬|\bno\b|\bnope\b|\bdon\'?t\b|\bleave\b', re.IGNORECASE)
# A new question rather than an answer to "꺼 드릴까요?".
_QUESTION_WORD = re.compile(r'어디|뭐|무엇|무슨|몇|어떤|어때|누구|언제'
                            r'|\b(where|what|which|how|who|when)\b|\b(is|are) there\b', re.IGNORECASE)
# A short "no more" to "더 궁금하신 게 있으세요?" (_NO, or 없어 / 없습니다 alone).
_NONE_LEFT = re.compile(r'^\s*(없어|없어요|없습니다|없음|없다)\s*[.!]?\s*$')
OFFER_RETRIES = 2     # unclear answers to "꺼 드릴까요?" asked again before leaving the light


def _offer_reply(text: str):
    """An answer to "꺼 드릴까요?": 'yes' | 'no' | 'on' | 'off', or None when it
    is not clearly one of them — mixed signals ("네 아니요", Whisper repeating
    its hint) included. Never guessed: None is asked again."""
    t = text or ''
    yes, no = bool(_YES.search(t) or re.search(r'해\s?(줘|주세요|줄래)', t)), bool(_NO.search(t))
    off = bool(re.search(r'꺼|끄|끌|\boff\b', t, re.IGNORECASE))
    on = bool(re.search(r'켜|켤|켭|\bon\b', t, re.IGNORECASE))
    if (no and (yes or on or off)) or (on and off):
        return None
    return 'off' if off else 'on' if on else 'yes' if yes else 'no' if no else None


def _light_hint(lang: str):
    """Whisper hint words for light questions / commands (ko only)."""
    name = _light_name(_default_light() or '', lang) if lang == 'ko' else ''
    if lang != 'ko' or not name or not re.search(r'[가-힣]', name):
        return None
    return f'{name} 불 켜져 있어?, 불 꺼 줘, 불 켜 줘'


def _offer_hint(action: str, lang: str):
    """Whisper hint while waiting for the answer to "꺼 드릴까요?"."""
    if lang != 'ko':
        return None
    return '네, 꺼 주세요. 아니요, 그대로 두세요.' if action == 'off' else '네, 켜 주세요. 아니요, 그대로 두세요.'


def _light_device(text: str):
    """(loc, the name as said) of the light a text names, else (None, None)."""
    try:
        from .switchbot import _devices, _norm as sb_norm
        nt = sb_norm(text or '')
        for d in _devices():
            for n in (*(d.get('aliases') or []), d['loc'], d['id']):
                if sb_norm(n) and sb_norm(n) in nt:
                    return d['loc'], str(n)
    except Exception:
        pass
    return None, None


def _default_light():
    """loc of the switchbot connection marked default (else the first), or None."""
    try:
        from .switchbot import _devices
        devices = _devices()
    except Exception:
        return None
    d = next((d for d in devices if d['default']), devices[0] if devices else None)
    return d['loc'] if d else None


def _light_name(loc: str, lang: str) -> str:
    """How to say a light's place: ko → its first Korean alias, else its loc."""
    if lang == 'ko':
        try:
            from .switchbot import _devices
            for d in _devices():
                if d['loc'] == loc:
                    ko = next((a for a in d.get('aliases') or [] if re.search(r'[가-힣]', a)), None)
                    return ko or loc
        except Exception:
            pass
    return loc


def _light_command(text: str, follow_up: bool = False):
    """(action 'on' | 'off', loc or None) for a light request, else None.
    "불 켜진 거 있으면 꺼" holds both 켜 and 꺼: the verb that comes last wins.
    `follow_up`: right after a light was talked about, "꺼줘" needs no 불."""
    t = text or ''
    if not _LIGHT_NOUN.search(t) and not follow_up:
        return None
    if _light_question(t):
        return None
    offs = [m.start() for m in _LIGHT_OFF.finditer(t)]
    ons = [m.start() for m in _LIGHT_ON.finditer(t)]
    if follow_up and not _LIGHT_NOUN.search(t):     # no 불: only the plain verbs
        ons = [m.start() for m in re.finditer(r'켜|켤|켭|\bon\b', t, re.IGNORECASE)]
    if not offs and not ons:
        return None
    action = 'off' if (max(offs) if offs else -1) > (max(ons) if ons else -1) else 'on'
    return action, _light_device(t)[0]


def _light_question(text: str) -> bool:
    """"세탁실 불 켜져 있어?" / "are the lights on?" — the state, not an action."""
    t = text or ''
    return bool(_LIGHT_NOUN.search(t) and _LIGHT_ASK.search(t) and not _LIGHT_DO.search(t))


def _light_phrase(key: str, action: str, lang: str, name: str = '') -> str:
    on = action == 'on'
    where = f'{name} ' if name else ''
    return {
        'ack':  {'ko': f'네. {where}불 켜겠습니다.' if on else f'네. {where}불 끄겠습니다.',
                 'en': f"OK, I'll turn the {where}lights on." if on else f"OK, I'll turn the {where}lights off."},
        'done': {'ko': '불을 켰어요.' if on else '불을 껐어요.',
                 'en': 'The lights are on.' if on else 'The lights are off.'},
        'fail': {'ko': '죄송해요, 불을 켜지 못했어요.' if on else '죄송해요, 불을 끄지 못했어요.',
                 'en': "Sorry, I couldn't turn the lights on." if on else "Sorry, I couldn't turn the lights off."},
        'offer': {'ko': '켜 드릴까요?' if on else '꺼 드릴까요?',
                  'en': 'Shall I turn it on?' if on else 'Shall I turn it off?'},
        'leave': {'ko': '알겠어요. 그대로 둘게요.', 'en': "OK, I'll leave it."},
        'again': {'ko': f'잘 못 들었어요. {where}불을 {"켤" if on else "끌"}까요? 네 또는 아니요로 말씀해 주세요.',
                  'en': f"Sorry, I didn't catch that. Shall I turn the {where}light {action}? Please say yes or no."},
        'gave_up': {'ko': '잘 못 들어서 그대로 둘게요.', 'en': "I couldn't hear you, so I'll leave it."},
        'check': {'ko': '확인해 볼게요.', 'en': 'Let me check.'},
        'unknown': {'ko': f'{where}불 상태를 확인하지 못했어요.',
                    'en': f"Sorry, I couldn't check the {where}lights."},
    }[key].get(lang) or ''


def _do_light(node, action: str, loc, lang: str):
    """Run turn_light; (ok, note for the log)."""
    from .switchbot import turn_light
    target = loc or _default_light()
    r = turn_light(node=node, inputs=action, **({'loc': target} if target else {}))
    per = r.get('results') or {}
    note = ', '.join(f"{k}: {v.get('result')}" for k, v in per.items()) or str(r.get('msg', ''))
    return bool(r.get('isdone')), f'turn_light {action} {target or "default"} → {note}'


def _ago(sec: float, lang: str) -> str:
    sec = max(0, int(sec))
    if lang == 'ko':
        return ('방금' if sec < 60 else f'{sec // 60}분 전에' if sec < 3600
                else f'{sec // 3600}시간 전에' if sec < 86400 else f'{sec // 86400}일 전에')
    return ('a moment ago' if sec < 60 else f'{sec // 60} min ago' if sec < 3600
            else f'{sec // 3600} h ago' if sec < 86400 else f'{sec // 86400} days ago')


def _light_state_answer(node, loc, said, lang: str):
    """(answer, offer) for "불 켜져 있어?" from light_state: `loc` None → every
    light. `offer` = (action, loc) the answer proposes ("꺼 드릴까요?"), or None.
    A state not heard now is told as of when it was ("10분 전에는 켜져 있었어요")."""
    from .switchbot import light_state
    r = light_state(node=node, loc=loc or _default_light() or 'all')
    states = r.get('states') or {}
    log_data({'msg': 'qa: light_state ' + ', '.join(
        f"{k}: {'on' if st.get('is_on') else 'off' if 'is_on' in st else '?'}"
        + (f" (stale {st.get('age_sec')} s, {st.get('source')})" if st.get('stale') else '')
        for k, st in states.items()) or f"qa: light_state → {r.get('msg', 'nothing')}"})
    if not states:
        return _light_phrase('unknown', 'on', lang, said or ''), None
    parts, known = [], {}
    for name, st in states.items():
        spoken = said if (said and len(states) == 1) else _light_name(name, lang)
        if 'is_on' not in st:
            parts.append(_light_phrase('unknown', 'on', lang, spoken).rstrip('.'))
            continue
        on = bool(st['is_on'])
        known[name] = on
        if lang == 'ko':
            state = '켜져' if on else '꺼져'
            ago = _ago(st.get('age_sec', 0), 'ko')
            parts.append(f"{ago}{'은' if ago == '방금' else '는'} {spoken} 불이 {state} 있었어요"
                         if st.get('stale') else f'{spoken} 불은 {state} 있어요')
        else:
            state = 'on' if on else 'off'
            parts.append(f"the {spoken} light was {state} {_ago(st.get('age_sec', 0), 'en')}"
                         if st.get('stale') else f'the {spoken} light is {state}')
    if lang == 'ko':
        answer = ', '.join(p[:-2] + '고' if p.endswith('어요') and i < len(parts) - 1 else p
                           for i, p in enumerate(parts)) + '.'
        if any(st.get('stale') for st in states.values()):
            answer = '지금은 신호가 안 잡혀서 확인하지 못했어요. ' + answer
    else:
        answer = '; '.join(parts) + '.'
        answer = answer[0].upper() + answer[1:]
        if any(st.get('stale') for st in states.values()):
            answer = "I can't hear it right now. " + answer
    offer = None
    if len(known) == 1 and len(states) == 1:           # one light: offer the other state
        name, on = next(iter(known.items()))
        offer = ('off' if on else 'on', name)
    elif any(known.values()):                          # several: offer to turn the lit ones off
        lit = [n for n, on in known.items() if on]
        offer = ('off', lit[0] if len(lit) == 1 else 'all')
    if offer:
        answer += ' ' + _light_phrase('offer', offer[0], lang)
    return answer, offer


# ── The qa skill ─────────────────────────────────────────────────────────────

@exception_handler
def qa(node, **kwargs):
    """Answer questions about what the robot sees, round after round, like
    `ask`, until the user says a stop word or the run is cancelled.

    Each round is one `ask`-style listen: the line the robot says before
    listening — the opening question, then each answer — is the prompt of the
    listen, so it is spoken to completion and the mic opens right after. No
    answer is not an end: the robot listens again (silently, so it does not
    keep talking at an empty room). A stop word (그만, 종료, stop …, on a
    short utterance) or Cancel / Stop on the dashboard ends it, and so does a
    long silence: after `idle_ask_sec` (60 s) nobody spoke, the robot asks "더
    궁금하신 게 있으세요? …"; "아니요" / "없어요" ends it, "네" carries on, and
    `idle_end_sec` (30 s) more of silence ends it ("대답이 없으셔서 …").
    The dashboard's "End Q&A" button (POST /agent/qa/end) ends it like a stop
    word: an answer already worked out is still said, then "대화를 마칠게요.";
    success, so the plan goes on. Not during the emergency exchange.

    Params:
        inputs (str): the opening question, said instead of the default
            ('무엇이 궁금하세요?'); also what `once` asks again after silence.
        source, lang, max_sec, silence_sec, energy_threshold: as for `reply`.
        input: 'voice' (default) or 'text' — typed in the dashboard's Q&A card.
        Any QA_CONFIGS key (views, model, url, ...) overrides the
            configured value for this call.
        memory: True — the model also gets the earlier questions and answers
            of this conversation ("그건 무슨 색이야?"). Default False: each
            question on its own.
        reuse_photos: True — answer from the last photos while they are
            younger than `refresh_sec` (20 s) instead of looking again.
            Default False: new photos for every question.
        loc: ENV location (key or alias) to answer for; default the nearest.
        cam: 'head' (default: a photo at each head tilt in `views`) or 'arm'
            (one photo from the arm camera, QA_CONFIGS['arm_camera']; the head
            does not move).
        Light requests ("불 꺼줘", "turn on the light") are acknowledged and
            done with turn_light (no place named → the default light).
            "세탁실 불 켜져 있어?" is answered from light_state with an offer
            ("꺼 드릴까요?"); "응" or "꺼줘" in the next turn does it.
        once: True — answer one question and end (no goodbye); silence is
            asked again up to `retries` times (default 2), as `ask` does.
            Default False: the conversation goes on until a stop word or Cancel.
        products: False turns off the product recogniser (default on): a
            question about 라면 / a product name is answered with the names it
            is sure of (configs/products), never a name the model read.
        context: what an earlier step hands over. The context of check_fall /
            check_restroom (``qa::context={answer}, cam=None``): nothing
            detected → a short notice; detected → the notice and the
            "괜찮으신가요?" exchange (emergency.dialog: confirm, or call 119 —
            simulated). Then qa ends and stops the plan (``stop``); with a
            camera a "fine" carries on into the usual conversation instead.
        cam=None: no camera at all — only with such a context.

    Returns ``{'isdone', 'turns', 'ended', 'history'}`` (+ ``answer`` with
    once) — `ended` is 'stop word', 'cancelled', 'answered', 'idle' (long
    silence), 'nothing more' (no to "더 궁금하신 게 있으세요?"), 'operator'
    ("End Q&A" on the dashboard) or 'no answer'
    (once, nobody spoke; isdone False). With an emergency context ended may
    also be 'fine' / 'emergency', with ``outcome``, ``called_119`` and the
    exchange in ``emergency_check``.
    """

    opening = str(kwargs.pop('inputs', '') or '').strip()   # a bare "qa::" carries nothing
    loc = kwargs.pop('loc', None)
    context = kwargs.pop('context', None)
    emergency = isinstance(context, dict) and bool(context.get('emergency'))
    cam_arg = kwargs.pop('cam', 'head')
    cam = None if cam_arg is None or str(cam_arg).strip().lower() in ('', 'none', 'null') \
        else str(cam_arg).strip().lower()
    once = _truthy(kwargs.pop('once', False))
    retries = max(0, int(kwargs.pop('retries', RETRIES)))
    use_products = _truthy(kwargs.pop('products', True))
    if cam not in ('head', 'arm', None):
        raise ValueError(f"cam must be 'head', 'arm' or None, got {cam!r}")
    if cam is None and not emergency:
        raise ValueError('qa: cam=None (no camera) needs the context of check_fall / check_restroom '
                         '(qa::context={answer}, cam=None)')
    if emergency:                           # the detector's mic / language, unless this call names one
        for k in ('source', 'lang'):
            if k not in kwargs and context.get(k):
                kwargs[k] = context[k]
    explicit_views = 'views' in kwargs
    cfg = _qa_config(kwargs)
    cfg['cam'] = cam
    lang, source, listen = _common(kwargs)
    if emergency:
        from . import emergency as _em      # emergency imports this module
        over = {k: kwargs.pop(k) for k in list(kwargs) if k in _em._DEFAULTS}
        check = _em.dialog(node, context, lang, source, listen, **over)
        log_data({'msg': f"qa: {check['kind']} → {check['outcome']}"
                         + (' (119 called, simulated)' if check['called_119'] else '')})
        if cam is None or check['outcome'] != 'fine':
            # Said what had to be said: the plan ends here (stop), whatever the outcome.
            ended = {'fine': 'fine', 'nothing': 'nothing detected',
                     'cancelled': 'cancelled'}.get(check['outcome'], 'emergency')
            return {'isdone': check['outcome'] != 'cancelled', 'ended': ended, 'outcome': check['outcome'],
                    'called_119': check['called_119'], 'answer': check['answer'], 'emergency_check': check,
                    'turns': len(check['turns']), 'history': [],
                    'stop': check['outcome'] != 'cancelled', 'stop_reason': f"{check['kind']}: {ended}"}
        opening = None                      # fine, and a camera: the usual conversation, silently listening
    # A typed question is waited for until it comes or the run is cancelled: a
    # listen that times out would leave the browser's Q&A card waiting on the
    # old request, and the line typed next would go to it and be lost.
    listen['text_fallback_sec'] = QA_TEXT_WAIT_SEC
    # Whisper hint: also the things people drop and the products the robot
    # knows (Whisper heard "손가락" for 숟가락 without them).
    extra = [w for w in _drop_vocab(cfg) if re.match(r'[가-힣]', w)] if lang == 'ko' else []
    try:
        from . import _products
        extra += list((_products._catalog().get('products') or {}).keys())
    except Exception:
        pass
    lights = _light_hint(lang)              # "시대에 불" for 세탁실 불 without it
    if lights:
        extra.append(lights)
    if extra:
        listen['hint'] = ', '.join(x for x in [listen.get('hint') or '', ', '.join(dict.fromkeys(extra))] if x)
    base_hint = listen.get('hint')
    listen['keep_audio'] = True             # each recording → <log dir>/qa_audio/ (_save_audio)
    if listen.get('mode') == 'text':
        listen['text_wait_sec'] = QA_TEXT_WAIT_SEC
    place, qa_cfg = _place_config(node, cfg, loc)
    if qa_cfg and qa_cfg.get('views') and not explicit_views:
        cfg['views'] = list(qa_cfg['views'])
    log_data({'msg': f"qa: at {place or 'unknown place'}"
                     + (' (place layout)' if qa_cfg else '')
                     + (f" — {cfg['arm_camera']}, one photo" if cam == 'arm'
                        else f" — {cfg['camera']} at {', '.join(cfg['views'])}")})
    log_data({'msg': f"qa: VLM {_vlm_label(cfg)} (url {cfg.get('url')}, num_ctx {cfg.get('num_ctx')}, "
                     f"from {cfg.get('_source', 'QA_CONFIGS')}); memory {_truthy(cfg['memory'])}, "
                     f"reuse_photos {_truthy(cfg['reuse_photos'])}"})

    history, shots, shot_at = [], [], 0.0
    inv, inv_at = None, None                # product inventory of the current photos
    light_ctx = None                        # the light just asked about, for "응" / "꺼줘"
    ended = 'cancelled'
    silent = 0
    idle_ask = 0.0 if once else float(cfg.get('idle_ask_sec') or 0)
    idle_end = float(cfg.get('idle_end_sec') or 0)
    quiet_since = time.monotonic()          # the robot's last line / the user's last words
    asked_idle = False                      # "더 궁금하신 게 있으세요?" said, no answer yet
    if opening is None:                     # after the emergency exchange: it has spoken already
        opening, prompt = _qa_phrase('start', lang), None
    else:
        opening = opening or _qa_phrase('start', lang)
        prompt = opening                    # said by the first listen, as `ask` does

    pending_answer = None
    dialog_begin()                          # "End Q&A" may end it from here on
    _emit({'qa_state': {'active': True}})
    try:
        while not cancel_requested():
            if end_dialog_requested():          # "End Q&A" on the dashboard: as a stop word
                if pending_answer:              # the answer already worked out is still said
                    _say(pending_answer, lang, source)
                ended = 'operator'
                break
            # After "꺼 드릴까요?" Whisper is told to expect 네 / 아니요 (_offer_hint).
            pending = (light_ctx or {}).get('offer')
            listen['hint'] = (_offer_hint(pending[0], lang) if pending else None) or base_hint
            had_prompt = bool(prompt)
            try:
                text, _audio = _hear(prompt, lang, source, listen)
            except RuntimeError:
                if cancel_requested():          # the dashboard mic was released by Stop
                    break
                raise
            _save_audio(listen, text, lang)
            pending_answer = None
            if end_dialog_requested():          # pressed while listening: what was heard is dropped
                ended = 'operator'
                break
            prompt = None                       # nothing heard: listen again, quietly
            if text or had_prompt:              # the silence is counted from here
                quiet_since = time.monotonic()
            if not text:
                silent += 1
                if once and silent > retries:   # one-shot: give up like `ask` does
                    ended = 'no answer'
                    break
                if once:                        # one-shot: ask again, as `ask` does
                    prompt = ' '.join(p for p in (_phrase('not_heard', lang), opening) if p)
                elif idle_ask > 0 and listen.get('mode') != 'text':   # typed input waits as long as it takes
                    quiet = time.monotonic() - quiet_since
                    if asked_idle and quiet >= idle_end:
                        ended = 'idle'
                        break
                    if not asked_idle and quiet >= idle_ask:
                        log_data({'msg': f'qa: nobody spoke for {quiet:.0f} s — asking whether to go on'})
                        prompt, asked_idle = _qa_phrase('idle_ask', lang), True
                continue
            silent = 0
            if _is_stop(text):
                ended = 'stop word'
                break
            if asked_idle:                      # the answer to "더 궁금하신 게 있으세요?"
                asked_idle = False
                short = len(re.sub(r'\W', '', text)) <= 10 and not _QUESTION_WORD.search(text)
                if short and (_NO.search(text) or _NONE_LEFT.match(text)):
                    ended = 'nothing more'
                    break
                if short and _YES.search(text):
                    prompt = _qa_phrase('go_on', lang)
                    continue
                # anything else: a question, answered as usual

            # Lights. "세탁실 불 켜져 있어?" → light_state, answered with an offer
            # ("꺼 드릴까요?"); "불 꺼줘" — or right after, "응" / "꺼줘" — → say it
            # will, then turn_light (_light_command).
            last_light, light_ctx = light_ctx, None
            if _light_question(text):
                loc, said = _light_device(text)
                _say(_light_phrase('check', 'on', lang), lang, source)
                try:
                    answer, offer = _light_state_answer(node, loc, said, lang)
                except Exception as e:                      # no SwitchBot / BLE error
                    log_data({'msg': f'qa: light_state: {e}'})
                    answer, offer = _light_phrase('unknown', 'on', lang, said or ''), None
                history.append((text, answer))
                if once:
                    _say(answer, lang, source)
                    ended = 'answered'
                    break
                if offer:
                    name = said or (_light_name(offer[1], lang) if offer[1] != 'all' else '')
                    light_ctx = {'loc': loc, 'said': said, 'offer': offer, 'name': name, 'tries': 0}
                prompt = pending_answer = answer                 # spoken by the next listen, then the mic opens
                continue
            light = None
            offer = (last_light or {}).get('offer')
            if offer and not _LIGHT_NOUN.search(text):
                # The answer to "꺼 드릴까요?": yes / no / 꺼줘 do it; anything unclear
                # ("택시, 고죠") is asked again, never guessed and never sent to the
                # camera; a new question ("와인잔 어디 있어?") is answered as usual.
                reply = _offer_reply(text)
                if reply == 'no':
                    answer = _light_phrase('leave', 'on', lang)
                    history.append((text, answer))
                    prompt = pending_answer = answer
                    continue
                if reply == 'yes':
                    light = offer
                elif reply in ('on', 'off'):
                    light = (reply, offer[1] if reply == offer[0] else last_light['loc'])
                elif not _QUESTION_WORD.search(text):
                    tries = last_light['tries'] + 1
                    if tries > OFFER_RETRIES:
                        answer = _light_phrase('gave_up', 'on', lang)
                    else:
                        answer = _light_phrase('again', offer[0], lang, last_light['name'])
                        light_ctx = {**last_light, 'tries': tries}
                    log_data({'msg': f'qa: unclear answer to the light offer ({tries}/{OFFER_RETRIES})'})
                    history.append((text, answer))
                    prompt = pending_answer = answer
                    continue
            if light is None:
                light = _light_command(text, follow_up=last_light is not None and not offer)
            if light is not None:
                action, loc = light
                said = _light_device(text)[1] if loc else None
                if loc is None and last_light:              # "꺼줘" about the light just asked about
                    loc = offer[1] if offer and offer[0] == action else last_light['loc']
                    said = last_light['said'] if loc == last_light['loc'] else None
                loc = loc or _default_light()
                name = said or (_light_name(loc, lang) if loc and loc != 'all' else '')
                ack = _light_phrase('ack', action, lang, name)
                _say(ack, lang, source)
                try:
                    ok, note = _do_light(node, action, loc, lang)
                except Exception as e:                      # no SwitchBot / BLE error
                    ok, note = False, f'turn_light {action}: {e}'
                log_data({'msg': f'qa: {note}'})
                answer = ack
                if not ok:
                    answer = _light_phrase('fail', action, lang)
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
            # model (see _drop_answer).
            # Any "…떨어졌어?" goes here, even when the object word was not
            # understood (it asks again) — never to the left / right place answer.
            drop = _is_drop_question(text)
            fresh = (shots and _truthy(cfg['reuse_photos'])
                     and time.time() - shot_at <= float(cfg['refresh_sec']))
            if not drop and not fresh:          # new photos unless reuse_photos says otherwise
                _say(_qa_phrase('look', lang), lang, source)
                shots, shot_at = _look_around(node, cfg), time.time()
                if cancel_requested():
                    break

            facts = None
            recall = history if _truthy(cfg['memory']) else []     # earlier turns only with memory=True
            if drop:
                t0 = time.time()
                try:
                    answer = _drop_answer(node, cfg, text, lang)
                except Exception as e:
                    log_data({'msg': f'qa: dropped-object search failed: {e}'})
                    answer = _qa_phrase('error', lang)
                log_data({'msg': f'qa: answered in {time.time() - t0:.1f}s from the arm camera (feet + object)'})
            elif not shots:
                answer = _qa_phrase('blind', lang)
            else:
                t0 = time.time()
                try:
                    named = []
                    if use_products and _is_product_question(text):
                        if inv_at != shot_at:       # once per set of photos
                            t1 = time.time()
                            inv, inv_at = _inventory(cfg, shots), shot_at
                            log_data({'msg': f'qa: products ({time.time() - t1:.1f}s): ' + (', '.join(
                                f"{p['name'] or '?'}@{p['side']} {p['sim']:.2f}" for p in inv) or 'none')})
                        from . import _products
                        named = _products.names_in(text)
                        facts = _product_facts(inv)
                    unknown = _unknown_names(text) if facts else []
                    if named:                       # "신라면 있어?" — straight from the recogniser
                        answer = _product_answer(named, inv, lang)
                    elif unknown:                   # "진라면 있어?" — not a product it knows: never yes / no
                        answer = _unrecognised_answer(unknown, inv, lang)
                    elif facts and _LIST_QUESTION.search(text):
                        # "무슨 라면 있어?" — the recogniser's list, no model: the model
                        # answered it in Chinese and dropped the unnamed pack.
                        answer = _product_list(inv, lang)
                        log_data({'msg': 'qa: answer by product list'})
                    elif facts:                     # other product questions — model, with the verified names
                        answer = _ask_vlm(cfg, shots, text, recall, lang, facts=facts)
                        if _unknown_names(answer):   # it named a product anyway: use the list
                            log_data({'msg': f'qa: model named an unverified product — {answer!r}'})
                            answer = _product_list(inv, lang)
                    else:
                        answer, route = None, ''
                        where = bool(_WHERE.search(text.lower()))
                        if where and getattr(shots, 'geo', None) and (not qa_cfg or qa_cfg.get('shelves')):
                            answer = _ask_shelves(cfg, shots, text, recall, lang, qa_cfg)
                            route = 'shelves' if answer is not None else 'shelves found nothing → '
                        elif where:
                            route = ('no depth → ' if not getattr(shots, 'geo', None)
                                     else 'place block without shelves → ')
                        if answer is None:
                            use_place = bool(qa_cfg and qa_cfg.get('places'))
                            answer = (_ask_place(cfg, shots, text, recall, lang, qa_cfg) if use_place
                                      else _ask_vlm(cfg, shots, text, recall, lang))
                            route += 'place layout' if use_place else 'free answer'
                        log_data({'msg': f'qa: answer by {route}'})
                except Exception as e:
                    log_data({'msg': f'qa: vision model failed ({_vlm_label(cfg)}): {e}'})
                    answer = ''
                log_data({'msg': f'qa: answered in {time.time() - t0:.1f}s by {_vlm_label(cfg)} '
                                 f'from {len(shots)} photo(s) taken {time.time() - shot_at:.0f}s ago '
                                 f"({', '.join(v for v, _ in shots)})",
                          'log_image': _mosaic(shots)})
                answer = answer or _qa_phrase('error', lang)
            if cancel_requested():
                break
            history.append((text, answer))
            if once:                            # one question: say the answer and end
                _say(answer, lang, source)
                ended = 'answered'
                break
            prompt = pending_answer = answer                     # spoken by the next listen, then the mic opens

        if ended not in ('cancelled', 'answered'):
            _say(_qa_phrase('idle_bye' if ended == 'idle' else 'bye', lang), lang, source)
    finally:
        dialog_end()
        _emit({'qa_state': {'active': False}})
    out = {'isdone': ended != 'no answer', 'turns': len(history), 'ended': ended,
           'history': [{'q': q, 'a': a} for q, a in history]}
    if once and history:
        out['answer'] = history[-1][1]
    return out
