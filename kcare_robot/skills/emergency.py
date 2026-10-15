"""Emergency checks (SIMULATED — no real detector, nothing is really dialled). Public API:

  - `check_fall`     : simulated fall detector. inputs True / False.
  - `check_restroom` : simulated "in the restroom too long" detector. inputs True / False.
  - `call_119`       : the simulated 119 call: siren on the robot speaker + spoken notice + log.
  - `dialog`         : the "are you all right?" exchange — run by `qa::context={answer}` (hri.qa).

A detector skill only hands over what it found; qa decides what to say and ends the plan:

    move::table@living room
    check_fall::True                    # (head looks down) answer = the context, says nothing
    qa::context={answer}, cam=None      # "낙상이 감지되었습니다." + 괜찮으신가요? … 119 when needed — plan ends

  - check_fall::True / False → ``answer`` = {emergency, detected, question, lang, source}; never stops a plan;
  - qa with that context: nothing detected → a short notice ("낙상이 감지되지 않았습니다."); detected → the
    notice ("낙상이 감지되었습니다.") and the exchange below. Either way qa returns ``stop``: the plan
    (direct mode or plan skill) ends there as a success, steps after it are skipped.

`dialog` asks the context's question, then listens in windows of `reprompt_sec` (20 s; 5 s in demo mode)
until `timeout_sec` (60 s; 15 s in demo) is used up, re-asking with the seconds left:

  - a clear "I'm fine" (괜찮아 / 문제없어 / 별일 없어 …)          → confirm, no call;
  - asks for help (도와줘 / 아파 / 119 / 넘어졌 …), or denies
    being fine (안 괜찮아), or says both                            → call 119 at once;
  - unclear words                                                 → ask again (the clock keeps running);
  - nothing / Whisper's silence phrases (감사합니다 …)              → ask again;
  - time used up without a clear "fine"                           → call 119.

Only a clear "fine" stops the call: a false alarm costs less than a missed fall. The time left is counted
in listen windows, not wall clock, so what the robot says ("10초 뒤") matches what happens; the robot's own
speech comes on top. The mic / speaker are HRI_CONFIGS.source's (EMERGENCY_CONFIGS.source overrides): on kcare-3 the dashboard,
since the robot has no speaker or mic of its own wired (Jetson APE + HDMI only). Cancel / Stop ends the
exchange without calling.

Config: Global Config `EMERGENCY_CONFIGS` (kcare `configs/tasks.py`), each key overridable per call.
"""
import re
import time
import unicodedata

from robot_agent.core.run_control import cancel_requested
from robot_agent.skill_configs import _ConfigProxy
from robot_agent.utils import exception_handler, play_alarm, say_to_user

from .hri import _common, _hear, _hri_cfg, _say, _set_rate, _truthy, log_data

EMERGENCY_CONFIGS = _ConfigProxy('EMERGENCY_CONFIGS', {})
_DEFAULTS = {
    'timeout_sec': 60, 'reprompt_sec': 20,             # real use
    'demo': True, 'demo_timeout_sec': 15, 'demo_reprompt_sec': 5,
    # None = HRI_CONFIGS.source (the dashboard on kcare-3: its sound devices are the Jetson's APE, wired to no
    # speaker, and HDMI — 'robot' played every line into nothing, 2026-10-15).
    'source': None, 'energy_threshold': None,          # None = hri's default
    'detect_sec': 2,                                   # how long the simulated detector "looks"
    'simulate': True, 'alarm_volume': 0.6, 'alarm_cycles': 3,
}

_PHRASES = {
    'fall_check':     {'ko': '낙상 여부를 확인하고 있습니다.',
                       'en': 'Checking for a fall.'},
    'restroom_check': {'ko': '화장실 상황을 확인하고 있습니다.',
                       'en': 'Checking the restroom.'},
    'fall_found':     {'ko': '낙상이 감지되었습니다.',
                       'en': 'A fall has been detected.'},
    'fall_none':      {'ko': '낙상이 감지되지 않았습니다.',
                       'en': 'No fall has been detected.'},
    'restroom_found': {'ko': '화장실에 오래 계신 것이 감지되었습니다.',
                       'en': 'You have been in the restroom for a long time.'},
    'restroom_none':  {'ko': '화장실에서 이상이 감지되지 않았습니다.',
                       'en': 'Nothing unusual in the restroom.'},
    'question':       {'ko': '괜찮으신가요?',
                       'en': 'Are you all right?'},
    'how':            {'ko': "괜찮으시면 '괜찮아'라고 말씀해 주세요.",
                       'en': "If you are fine, please say 'I am OK'."},
    'deadline':       {'ko': '응답 없으시면 {t} 뒤 일일구에 구조요청 하겠습니다.',
                       'en': 'If you do not answer, I will call 119 in {t}.'},
    'again':          {'ko': '괜찮으신가요? {t} 뒤 일일구에 구조요청 하겠습니다.',
                       'en': 'Are you all right? I will call 119 in {t}.'},
    'unclear':        {'ko': "잘 못 들었어요. 괜찮으시면 '괜찮아'라고 말씀해 주세요. {t} 뒤 일일구에 구조요청 하겠습니다.",
                       'en': "Sorry, I didn't catch that. If you are fine, please say 'I am OK'. I will call 119 in {t}."},
    'ok':             {'ko': '괜찮으시다니 다행이에요. 도움이 필요하시면 언제든 불러 주세요.',
                       'en': "I'm glad you're all right. Call me any time you need help."},
    'help':           {'ko': '알겠습니다. 지금 일일구에 구조요청 하겠습니다.',
                       'en': 'All right. I am calling 119 now.'},
    'timeout':        {'ko': '응답이 없으셔서 일일구에 구조요청 하겠습니다.',
                       'en': 'There is no answer, so I am calling 119.'},
    'called':         {'ko': '일일구에 구조요청을 했습니다. 곧 도움이 올 거예요. 제가 옆에 있을게요.',
                       'en': 'I have called 119. Help is on the way. I will stay with you.'},
}
# Whisper's prompt while listening: the answers to expect.
# Korean says 119 as 일일구: the spoken lines write it so (TTS read "119" as 백십구), and both forms are help.
_HINT = {'ko': '괜찮아요. 도와주세요. 아파요. 일일구 불러 주세요.',
         'en': "I'm OK. Help me. It hurts. Call 119."}

# ── Reading the answer ───────────────────────────────────────────────────────
# Negated forms first: they contain the plain words ("안 괜찮아" holds 괜찮, "안 아파" holds 아파).
_NOT_FINE = re.compile(r'안\s*괜찮|괜찮지\s*않|안\s*좋|not\s+(ok|okay|fine|good)')
_NO_HELP = re.compile(r'(도움|119|일일구|구급차|병원)\S*\s*(필요\s*없|부르지\s*마|안\s*불러)|'
                      r'안\s*아파|아프지\s*않|안\s*다쳤|다치지\s*않|no\s+help|don\S*\s+call')
_FINE = ('괜찮', '문제없', '문제 없', '별일 없', '별일없', '아무 일 없', '아무일 없', '아무일없',
         "i'm ok", 'i am ok', 'im ok', 'okay', "i'm fine", 'i am fine', '__fine__')
_HELP = ('도와', '도움', '살려', '아파', '아프', '다쳤', '다쳐', '넘어졌', '넘어져', '못 일어나', '못일어나',
         '일어날 수 없', '일어날수없', '119', '일일구', '구급차', '병원', '쓰러', 'help', 'hurt', 'ambulance', 'pain')
# What Whisper writes for silence or noise (subtitle phrases it was trained on).
_SILENCE = ('시청해 주셔서 감사합니다', '시청해주셔서 감사합니다', '감사합니다', '구독과 좋아요', '구독',
            '좋아요', '알림 설정', '다음 영상에서 만나요', '영상 끝까지', '자막', 'mbc 뉴스',   # a 1 s tone → "다음 영상에서 만나요."
            'thank you for watching', 'thanks for watching', 'see you in the next video', 'thank you')


def _norm(text) -> str:
    return ' '.join(unicodedata.normalize('NFC', str(text or '')).lower().split())


def reply_kind(text, hint: str = '') -> str:
    """'fine' | 'help' | 'unclear' | 'none' for one transcript (see the module doc for the rules)."""
    t = _norm(text)
    for p in _SILENCE:
        t = t.replace(p, ' ')
    if not re.search(r'\w', t):
        return 'none'
    sentences = [_norm(s) for s in re.split(r'[.?!]', hint or '') if _norm(s)]
    if sum(s in t for s in sentences) >= 2:             # the hint read back to us, not an answer
        return 'none'
    if _NOT_FINE.search(t):
        return 'help'
    t = _NO_HELP.sub(' __fine__ ', t)
    fine = any(w in t for w in _FINE)
    help_ = any(w in t for w in _HELP)
    if help_:                                           # "괜찮은데 좀 아파" → help
        return 'help'
    return 'fine' if fine else 'unclear'


def _secs(sec: float, lang: str) -> str:
    sec = int(round(sec))
    m, s = divmod(sec, 60)
    if lang == 'ko':
        return ' '.join(p for p in (f'{m}분' if m else '', f'{s}초' if s else '') if p) or '0초'
    return ' '.join(p for p in (f"{m} minute{'s' if m > 1 else ''}" if m else '',
                                f"{s} second{'s' if s > 1 else ''}" if s else '') if p) or '0 seconds'


def _phrase(key, lang, **kw) -> str:
    table = _PHRASES[key]
    return table.get(lang, table['en']).format(**kw)


def _cfg(kwargs) -> dict:
    cfg = dict(_DEFAULTS)
    try:
        cfg.update(dict(EMERGENCY_CONFIGS.items()))
    except Exception:
        pass
    for k in list(_DEFAULTS):
        if k in kwargs and k not in ('source', 'energy_threshold'):   # those two go to hri._common
            cfg[k] = kwargs.pop(k)
    return cfg


def _source(kwargs, cfg) -> str:
    """The call's source, else EMERGENCY_CONFIGS.source, else HRI_CONFIGS.source."""
    return str(kwargs.pop('source', None) or cfg.get('source') or _hri_cfg().get('source') or 'dashboard').lower()


def _where():
    try:
        from robot_agent.state import current
        return getattr(current().world, 'arrived', None)
    except Exception:
        return None


# ── Detectors (simulated) ────────────────────────────────────────────────────

def _detect(node, kind: str, kwargs: dict) -> dict:
    detected = _truthy(kwargs.pop('inputs', True))
    lang = str(kwargs.pop('lang', 'ko') or 'ko').lower()
    cfg = _cfg(kwargs)
    source = _source(kwargs, cfg)
    _set_rate(kwargs)
    flag = 'fall' if kind == 'check_fall' else 'restroom_too_long'
    key = 'fall' if kind == 'check_fall' else 'restroom'
    # "Checking…", spoken to the end (the dashboard reports it), then the detector "looks" for detect_sec and
    # says nothing more: what it found is for the next step (qa) to tell.
    say_to_user(_phrase(f'{key}_check', lang), lang=lang, source=source, wait=True)
    log_data({'msg': f'robot: {_phrase(f"{key}_check", lang)}'})
    if kind == 'check_fall' and node is not None:          # look at the floor, where a person would be
        try:
            from .head import moveh
            ret = moveh(node=node, inputs='down')
            if not ret.get('isdone'):
                log_data({'msg': f'check_fall: head down failed ({ret.get("msg")})'})
        except Exception as e:
            log_data({'msg': f'check_fall: head down failed ({e})'})
    time.sleep(max(0.0, float(cfg['detect_sec'])))
    log_data({'msg': f'{kind}: {"detected" if detected else "nothing detected"} (simulated) — context for the next step'})
    context = {'emergency': kind, 'detected': detected, 'question': _phrase('question', lang),
               'lang': lang, 'source': source}
    return {'isdone': True, flag: detected, 'answer': context}


@exception_handler
def check_fall(node=None, **kwargs):
    """SIMULATED fall detector. inputs True (default) / False. Says "낙상 여부를 확인하고 있습니다.", turns the
    head down, "looks" for detect_sec (2 s), says nothing more and never stops a plan: it hands ``answer`` = the
    context {emergency, detected, question, lang, source} to the next step (``qa::context={answer}, cam=None``),
    which tells what was found, asks and ends the plan. Returns {isdone, fall, answer}."""
    return _detect(node, 'check_fall', kwargs)


@exception_handler
def check_restroom(node=None, **kwargs):
    """SIMULATED "in the restroom too long" detector, as check_fall ("화장실 상황을 확인하고 있습니다."; the head does
    not move).
    Returns {isdone, restroom_too_long, answer}."""
    return _detect(node, 'check_restroom', kwargs)


# ── The exchange (run by qa) ─────────────────────────────────────────────────

def dialog(node, context: dict, lang: str, source: str, listen: dict, **over) -> dict:
    """Say what the detector found, then (if it found something) ask the context's question and decide
    (module doc). `listen` is the caller's hri listen settings (copied: the window, Whisper hint and mic level
    are set here). `over`: EMERGENCY_CONFIGS keys.
    Returns {kind, outcome: 'nothing' | 'fine' | 'help' | 'unclear' | 'no_answer' | 'cancelled', called_119,
    emergency, answer, turns: [{robot, user, kind}], call?, elapsed_sec, demo}."""
    cfg = _cfg(over)
    kind = context.get('emergency', 'emergency')
    key = 'fall' if kind == 'check_fall' else 'restroom'
    if not _truthy(context.get('detected', True)):
        _say(_phrase(f'{key}_none', lang), lang, source)
        log_data({'msg': f'{kind}: nothing detected — said so'})
        return {'kind': kind, 'outcome': 'nothing', 'called_119': False, 'emergency': False, 'answer': None,
                'turns': [], 'elapsed_sec': 0.0, 'demo': _truthy(cfg['demo'])}
    _say(_phrase(f'{key}_found', lang), lang, source)
    demo = _truthy(cfg['demo'])
    timeout = float(cfg['demo_timeout_sec' if demo else 'timeout_sec'])
    reprompt = float(cfg['demo_reprompt_sec' if demo else 'reprompt_sec'])
    windows = max(1, int(round(timeout / reprompt)))
    listen = dict(listen, max_sec=reprompt, hint=_HINT.get(lang, _HINT['en']), keep_audio=False)
    if cfg.get('energy_threshold') is not None:
        listen['energy_threshold'] = float(cfg['energy_threshold'])
    hint = listen['hint']

    log_data({'msg': f'{kind}: asking — {timeout:.0f} s, again every {reprompt:.0f} s'
                     f'{" (demo)" if demo else ""}, {source} mic'})
    t0, turns, kinds = time.time(), [], []
    prompt = ' '.join([context.get('question') or _phrase('question', lang), _phrase('how', lang),
                       _phrase('deadline', lang, t=_secs(timeout, lang))])
    outcome = None
    for k in range(windows):
        if cancel_requested():
            break
        t_listen = time.time()
        try:
            text, _audio = _hear(prompt, lang, source, listen)
            rk = reply_kind(text, hint)
        except RuntimeError as e:
            if cancel_requested():                      # the dashboard mic was released by Stop
                break
            # Whisper down / mic error (2026-10-15: the speaches container had not come back after a reboot).
            # Not a reason to stop an emergency: count it as not understood, keep the clock going.
            log_data({'msg': f'{kind}: could not hear ({e}) — counted as unclear'})
            text, rk = None, 'unclear'
            time.sleep(max(0.0, reprompt - (time.time() - t_listen)))   # the window still takes its time
        turns.append({'robot': prompt, 'user': text, 'kind': rk})
        kinds.append(rk)
        log_data({'msg': f'{kind}: heard {text!r} → {rk}'})
        if rk in ('fine', 'help'):
            outcome = rk
            break
        left = timeout - (k + 1) * reprompt
        if left <= 0:
            break
        prompt = _phrase('unclear' if rk == 'unclear' else 'again', lang, t=_secs(left, lang))

    result = {'kind': kind, 'turns': turns, 'demo': demo,
              'answer': next((t['user'] for t in reversed(turns) if t['kind'] != 'none'), None)}
    if cancel_requested():
        result.update(outcome='cancelled', called_119=False, emergency=None)
    elif outcome == 'fine':
        _say(_phrase('ok', lang), lang, source)
        result.update(outcome='fine', called_119=False, emergency=False)
    else:
        if outcome is None:
            outcome = 'unclear' if 'unclear' in kinds else 'no_answer'
        _say(_phrase('help' if outcome == 'help' else 'timeout', lang), lang, source)
        reason = {'help': f'asked for help ({result["answer"]!r})', 'unclear': 'no clear answer',
                  'no_answer': 'no answer'}[outcome]
        call = call_119(node=node, inputs=f'{kind}: {reason}', lang=lang, source=source,
                        alarm_volume=cfg['alarm_volume'], alarm_cycles=cfg['alarm_cycles'],
                        simulate=cfg['simulate'])
        result.update(outcome=outcome, called_119=bool(call.get('isdone')), emergency=True, call=call)
    result['elapsed_sec'] = round(time.time() - t0, 1)
    log_data({'msg': f"{kind}: {result['outcome']}" + (' → 119 called (simulated)' if result['called_119'] else '')})
    return result


@exception_handler
def call_119(node=None, **kwargs):
    """SIMULATED 119 call: the siren (dashboard: the browser plays it; robot: the robot speaker) — the user,
    and anyone else in the house, hears it —
    then "119에 구조요청을 했습니다 …", and an EMERGENCY line in the log. inputs = the reason.
    EMERGENCY_CONFIGS.simulate False is refused: there is no real call yet.
    Returns {isdone, simulated, number, reason, location, time, alarm}."""
    cfg = _cfg(kwargs)
    reason = str(kwargs.pop('inputs', None) or 'requested')
    lang = str(kwargs.pop('lang', 'ko') or 'ko').lower()
    source = _source(kwargs, cfg)
    _set_rate(kwargs)
    if not _truthy(cfg['simulate']):
        raise NotImplementedError('call_119: only the simulation exists (EMERGENCY_CONFIGS.simulate must be true)')
    when = time.strftime('%Y-%m-%d %H:%M:%S')
    where = _where()
    log_data({'msg': f'EMERGENCY (simulated): 119 called — {reason}; robot at {where or "?"}; {when}'})
    if source == 'dashboard':       # played by the browser, in order with the spoken lines (page.tsx playAlarm)
        log_data({'alarm': {'volume': float(cfg['alarm_volume']), 'cycles': int(cfg['alarm_cycles'])}})
        alarm = 'dashboard'
    else:
        alarm = play_alarm(float(cfg['alarm_volume']), int(cfg['alarm_cycles']))
        if not alarm:
            log_data({'msg': 'call_119: the siren could not play on the robot speaker'})
    _say(_phrase('called', lang), lang, source)
    return {'isdone': True, 'simulated': True, 'number': '119', 'reason': reason, 'location': where,
            'time': when, 'alarm': alarm}
