"""Human-robot interaction skills: ``reply`` and ``ask``.

Both talk and listen on one side, chosen per call with ``source``:

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
    ask::inputs='Which drink?', lang='en', source='dashboard', options='water,juice'
"""

import re
import unicodedata

from robot_agent.skills import log_data
from robot_agent.utils import (
    exception_handler, listen_dashboard, record_phrase, say_to_user,
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

_PHRASES = {
    'heard':      {'ko': '들었어요',
                   'en': 'Got it.',
                   'vi': 'Đã nghe.'},
    'heard_text': {'ko': '"{text}"라고 들었어요',
                   'en': 'I heard: {text}',
                   'vi': 'Đã nghe: {text}'},
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


def _common(kwargs: dict):
    lang = str(kwargs.pop('lang', 'ko') or 'ko')
    source = str(kwargs.pop('source', 'robot') or 'robot').lower()
    if source not in SOURCES:
        raise ValueError(f'source must be one of {SOURCES}, got {source!r}')
    listen = {
        'max_sec': float(kwargs.pop('max_sec', MAX_SEC)),
        'silence_sec': float(kwargs.pop('silence_sec', SILENCE_SEC)),
        'energy_threshold': float(kwargs.pop('energy_threshold', ENERGY_THRESHOLD)),
    }
    return lang, source, listen


def _say(text: str, lang: str, source: str) -> None:
    log_data({'msg': f'robot: {text}'})
    say_to_user(text, lang=lang, source=source)


def _hear(prompt, lang: str, source: str, listen: dict):
    """Say `prompt` (if any), then capture and transcribe one phrase.

    Returns the transcript, or None when nobody answered. On the robot side the
    prompt is spoken to completion before the mic opens, or the recording would
    start with the robot's own voice; the browser does the same ordering itself.
    """
    if prompt:
        log_data({'msg': f'robot: {prompt}'})
    if source == 'dashboard':
        text = listen_dashboard(prompt=prompt, lang=lang, max_sec=listen['max_sec'],
                                timeout=listen['max_sec'] + DASHBOARD_SLACK_SEC)
    else:
        if prompt:
            say_to_user(prompt, lang=lang, source='robot')
        audio = record_phrase(max_sec=listen['max_sec'],
                              silence_sec=listen['silence_sec'],
                              energy_threshold=listen['energy_threshold'])
        text = speech_to_text(audio, lang=lang) if audio is not None else None
    text = (text or '').strip() or None
    log_data({'msg': f'user: {text}' if text else 'user: (no answer)'})
    return text


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
        need_confirm (bool): repeat the transcript back ('"<text>"라고 들었어요')
            instead of a bare '들었어요'. Default False.
        inputs (str):  optional line to say before listening.
        source (str):  'robot' (default) or 'dashboard'.
        lang (str):    'ko' (default), 'en' or 'vi' — speech and recognition.
        max_sec, silence_sec, energy_threshold: listening limits.

    Returns ``{'isdone', 'text'}``; isdone is False when nobody spoke.
    """
    need_confirm = _truthy(kwargs.pop('need_confirm', False))
    prompt = str(kwargs.pop('inputs', '') or '').strip() or None
    lang, source, listen = _common(kwargs)

    text = _hear(prompt, lang, source, listen)
    if not text:
        _say(_phrase('not_heard', lang), lang, source)
        return {'isdone': False, 'msg': 'no speech heard', 'text': ''}

    _say(_phrase('heard_text', lang, text=text) if need_confirm else _phrase('heard', lang),
         lang, source)
    return {'isdone': True, 'text': text}


@exception_handler
def ask(node, **kwargs):
    """Ask a question aloud and understand the spoken answer.

    Params:
        inputs (str):   the question.
        options (list): allowed answers. When given, the answer is matched to
            one of them and that option is confirmed aloud; an answer matching
            none gets the list read out and the question asked again. When
            None, the answer is simply repeated back.
        retries (int):  extra attempts after the first, for no answer or no
            match. Default 2.
        match_threshold (float): fuzzy-match floor, 0..1. Default 0.7.
        source, lang, max_sec, silence_sec, energy_threshold: as for `reply`.

    Returns ``{'isdone', 'answer', 'text', 'attempts'}`` — `answer` is the
    matched option (or the transcript when there are no options), `text` the
    raw transcript.
    """
    question = str(kwargs.pop('inputs', '') or '').strip()
    options = parse_options(kwargs.pop('options', None))
    retries = max(0, int(kwargs.pop('retries', RETRIES)))
    threshold = float(kwargs.pop('match_threshold', MATCH_THRESHOLD))
    lang, source, listen = _common(kwargs)

    prompt, last_text = question or None, ''
    for attempt in range(1, retries + 2):
        text = _hear(prompt, lang, source, listen)
        if not text:
            # Nobody answered: say so, then put the question again.
            prompt = ' '.join(p for p in (_phrase('not_heard', lang), question) if p)
            continue
        last_text = text

        if options is None:
            _say(_phrase('heard_text', lang, text=text), lang, source)
            return {'isdone': True, 'answer': text, 'text': text, 'attempts': attempt}

        choice = match_option(text, options, threshold)
        if choice is not None:
            _say(_phrase('confirm', lang, choice=choice), lang, source)
            return {'isdone': True, 'answer': choice, 'text': text,
                    'attempts': attempt, 'options': options}
        # An answer, just not one of the options: offer the list this time.
        prompt = _phrase('choose', lang, options=', '.join(options))

    _say(_phrase('not_heard', lang), lang, source)
    msg = ('answer did not match any option' if last_text and options
           else 'no speech heard')
    return {'isdone': False, 'msg': msg, 'answer': None, 'text': last_text,
            'attempts': retries + 1, 'options': options}
