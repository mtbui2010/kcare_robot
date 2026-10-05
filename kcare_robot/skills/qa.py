"""Visual Q&A about the robot's surroundings: ``qa``.

Answers spoken questions about what is around the robot — "what colour is the
shirt and where is it?", "which shelf level is the handbag on?" — from photos
taken by the head camera, using a local vision-language model served by Ollama
(QA_CONFIGS in configs/tasks.py: url, model, ...).

One call runs a whole conversation: listen -> look around if needed -> answer
-> listen ..., until the person says a stop word, stays silent `idle_turns`
times in a row, or the run is cancelled. The dashboard's Q&A button starts the
run and, pressed again, sends POST /agent/cancel.

Looking around: the head takes one photo at each tilt in QA_CONFIGS['views']
(moveh up / straight / down), then goes back to where it was. The photos are
reused for follow-up questions for `refresh_sec`, so the head does not nod
before every single answer.

Where-questions at a known place: an ENV entry may carry a ``qa`` block
describing what the robot sees there and the places an answer may name::

    'dressroom': {..., 'qa': {
        'views': ['up', 'straight', 'down'],
        'scene': 'a dressing room: a hanging rail on the left, corner shelves on the right',
        'frame': 'robot',                     # or 'person': left/right of the seated user
        'places': {
            'rail_left': {'desc': 'left part of the hanging rail',
                          'ko': '행거 왼쪽', 'en': 'on the left of the rail', 'vi': 'ở bên trái giá treo'},
            ...}}}

At such a place the model returns JSON restricted to those place ids, and the
spoken answer is built from the phrases ("행거 왼쪽에 있어요") instead of being
free text. The place is the ENV location nearest the base (`loc_radius`), or
``qa::loc='옷방'``. Other questions (colours, "is there ...") stay free-form.

Usage:
    qa                                  # mic and language from hri defaults
    qa::source='dashboard', lang='vi'
    qa::source='robot', refresh_sec=60
    qa::loc='dressroom'                 # answer with that place's layout
    qa::source='dashboard', input='text'   # questions typed on the dashboard
"""

import base64
import json
import re
import time

import cv2
import numpy as np
import requests

from robot_agent.core.run_control import cancel_requested
from robot_agent.skill_configs import ENV, QA_CONFIGS
from robot_agent.skills import log_data
from robot_agent.utils import exception_handler

from ..utils import env_key, get_closest_loc
from .head import head_state, moveh
from .hri import _common, _hear, _say


_PHRASES = {
    'start': {'ko': '무엇이 궁금하세요?',
              'en': 'What would you like to know?',
              'vi': 'Bạn muốn hỏi gì?'},
    'look':  {'ko': '잠깐 둘러볼게요.',
              'en': 'Let me look around.',
              'vi': 'Để tôi nhìn xung quanh.'},
    'bye':   {'ko': '대화를 마칠게요.',
              'en': 'Ending our chat.',
              'vi': 'Kết thúc hỏi đáp.'},
    'error': {'ko': '지금은 대답하기 어려워요.',
              'en': "Sorry, I can't answer right now.",
              'vi': 'Xin lỗi, bây giờ tôi chưa trả lời được.'},
    'blind': {'ko': '카메라 영상을 받지 못했어요.',
              'en': "I can't get an image from my camera.",
              'vi': 'Tôi không nhận được hình ảnh từ camera.'},
    'unseen': {'ko': '보이지 않아요.',
               'en': "I can't see it.",
               'vi': 'Tôi không thấy nó.'},
}

_LANG_NAME = {'ko': 'Korean', 'en': 'English', 'vi': 'Vietnamese'}

# Checked in every language, and only on a short utterance: "끝에 있는 컵은
# 무슨 색이야?" contains '끝' but is a question, not a goodbye.
_STOP_WORDS = ('그만', '종료', '끝', '멈춰', '됐어',
               'stop', 'bye', 'quit', 'thatsall',
               'dừng', 'kếtthúc', 'thôi', 'tạmbiệt')
_STOP_MAX_CHARS = 10

# The model answers from a few photos; everything it may do beyond that is a
# guess, which is the failure that matters most for a care robot.
_SYSTEM_PROMPT = """\
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
# override one key per call (qa::refresh_sec=60).
_DEFAULTS = {
    'url': 'http://localhost:11434',
    # The instruct tag. The plain `qwen3-vl:8b` tag is thinking-only: on real
    # head-cam photos it took 1-10 s and twice ran out of tokens mid-thought,
    # answering nothing; instruct answers in ~2.3 s, ~0.2 s for follow-ups on
    # the same photos (Ollama reuses the cached image prefix).
    'model': 'qwen3-vl:8b-instruct',
    'camera': 'head_rgb',
    'views': ['up', 'straight', 'down'],
    'refresh_sec': 20.0,
    'idle_turns': 6,
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
}


def _phrase(key: str, lang: str) -> str:
    table = _PHRASES[key]
    return table.get(lang, table['en'])


def _vlm_connection(name: str) -> dict:
    """url / model / num_ctx of the `name` connection (type 'llm', Connections
    panel), or {} when there is none or no running agent."""
    try:
        from robot_agent.state import current
        entry = current().dm.get_connect(name)
    except Exception:
        return {}
    if entry is None or entry.type != 'llm':
        return {}
    return {k: entry.config[k] for k in ('url', 'model', 'num_ctx') if entry.config.get(k)}


def _vlm_label(cfg: dict) -> str:
    """'model @ server' for the logs."""
    from urllib.parse import urlparse
    host = urlparse(str(cfg.get('url', ''))).netloc or cfg.get('url', '?')
    return f"{cfg.get('model', '?')} @ {host}"


def _config(kwargs: dict) -> dict:
    """Defaults < QA_CONFIGS < the VLM connection (QA_CONFIGS['connection'],
    default 'vlm' — edited in the Connections panel) < this call's kwargs."""
    cfg = dict(_DEFAULTS)
    try:
        cfg.update(dict(QA_CONFIGS.items()))
    except Exception:
        pass
    conn = str(kwargs.get('connection') or cfg.get('connection') or 'vlm')
    from_conn = _vlm_connection(conn)
    cfg.update(from_conn)
    cfg['_source'] = f"connection '{conn}'" if from_conn else 'QA_CONFIGS'
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


def _look_around(node, cfg: dict) -> list:
    """[(view, rgb)] for each tilt in cfg['views']; the head returns after.

    The head is only put back when the run has not been cancelled — while
    cancelled every action agent refuses to send, and a cancel means "stop
    moving" anyway.
    """
    cam = node.agents.get(cfg['camera']) if node is not None else None
    if cam is None:
        log_data({'msg': f"qa: camera {cfg['camera']!r} is not connected"})
        return []
    start = head_state(node=node)
    shots = []
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
    finally:
        if isinstance(start, dict) and 'current_ry' in start and not cancel_requested():
            moveh(node=node, ry=start['current_ry'])
    return shots


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


def _ask_vlm(cfg: dict, shots: list, question: str, history: list, lang: str) -> str:
    """One short answer to `question` from the photos, or '' if the model gave none."""
    views = ', '.join(f'photo {i + 1} = head {v}' for i, (v, _) in enumerate(shots))
    system = _SYSTEM_PROMPT.format(n=len(shots), views=views,
                                   language=_LANG_NAME.get(lang, 'the language of the question'))
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
    r = _post_chat(cfg, body)
    return _first_sentence(r.json().get('message', {}).get('content', ''))


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
_WHERE = re.compile(r'어디|어느\s*쪽|어느\s*방향|위치|ở\s*đâu|chỗ\s*nào|where')

_FRAME = {
    'robot': 'Left and right are as seen from the robot, i.e. as in the photos.',
    'person': 'Left, right and front are from the point of view of the person '
              'using the table (the seated person in the photos), not the '
              'robot: "front" is on the table right in front of them.',
}


def _place_config(node, cfg: dict, loc):
    """(ENV key, its `qa` block) for where the robot is, or (key, None)."""
    try:
        key = env_key(loc) if loc else (
            get_closest_loc(node, ENV, threshold=float(cfg['loc_radius'])) if node is not None else None)
    except Exception as e:
        log_data({'msg': f'qa: location unknown ({e})'})
        key = None
    spec = ENV.get(key) if key else None
    block = spec.get('qa') if isinstance(spec, dict) else None
    ok = isinstance(block, dict) and isinstance(block.get('places'), dict) and block['places']
    return key, (block if ok else None)


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
    if lang == 'vi':
        return f'Nó {" và ".join(names)}.'
    return f'It is {" and ".join(names)}.'


def _ask_place(cfg: dict, shots: list, question: str, history: list, lang: str, qa_cfg: dict) -> str:
    """Answer at a place with a `qa` block: where-questions from its places."""
    places = qa_cfg['places']
    views = ', '.join(f'photo {i + 1} = head {v}' for i, (v, _) in enumerate(shots))
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
    return _place_answer(ids, places, lang) if ids else _phrase('unseen', lang)


# ── Skill ────────────────────────────────────────────────────────────────────

@exception_handler
def qa(node, **kwargs):
    """Hold a spoken Q&A about the surroundings until told to stop.

    Params:
        source, lang, max_sec, silence_sec, energy_threshold: as for `reply`.
        input: 'voice' (default) or 'text' — typed on the dashboard; the
            conversation then ends after one `text_wait_sec` (300 s) without
            a question instead of `idle_turns` short listens.
        Any QA_CONFIGS key (refresh_sec, views, idle_turns, model, url, ...)
            overrides the configured value for this call.
        loc: ENV location (key or alias) to answer for; default the nearest.

    Returns ``{'isdone', 'turns', 'ended', 'history'}`` — `ended` is 'stop
    word', 'idle' or 'cancelled'.
    """
    kwargs.pop('inputs', None)              # a bare "qa::" carries nothing
    loc = kwargs.pop('loc', None)
    explicit_views = 'views' in kwargs
    cfg = _config(kwargs)
    lang, source, listen = _common(kwargs)
    place, qa_cfg = _place_config(node, cfg, loc)
    if qa_cfg and qa_cfg.get('views') and not explicit_views:
        cfg['views'] = list(qa_cfg['views'])
    log_data({'msg': f"qa: at {place or 'unknown place'}"
                     + (' (place layout)' if qa_cfg else '')})
    log_data({'msg': f"qa: VLM {_vlm_label(cfg)} (url {cfg.get('url')}, num_ctx {cfg.get('num_ctx')}, "
                     f"from {cfg.get('_source', 'QA_CONFIGS')})"})

    history, shots, shot_at = [], [], 0.0
    quiet, ended = 0, 'cancelled'
    _say(_phrase('start', lang), lang, source)

    while not cancel_requested():
        try:
            text, _audio = _hear(None, lang, source, listen)
        except RuntimeError:
            if cancel_requested():          # the dashboard mic was released by Stop
                break
            raise
        if not text:
            quiet += 1
            if quiet >= (1 if listen.get('mode') == 'text' else int(cfg['idle_turns'])):
                ended = 'idle'
                break
            continue
        quiet = 0
        if _is_stop(text):
            ended = 'stop word'
            break

        if not shots or time.time() - shot_at > float(cfg['refresh_sec']):
            _say(_phrase('look', lang), lang, source)
            shots, shot_at = _look_around(node, cfg), time.time()
            if cancel_requested():
                break
            if shots:
                log_data({'log_image': _mosaic(shots)})

        if not shots:
            answer = _phrase('blind', lang)
        else:
            t0 = time.time()
            try:
                answer = (_ask_place(cfg, shots, text, history, lang, qa_cfg) if qa_cfg
                          else _ask_vlm(cfg, shots, text, history, lang))
            except Exception as e:
                log_data({'msg': f'qa: vision model failed ({_vlm_label(cfg)}): {e}'})
                answer = ''
            log_data({'msg': f'qa: answered in {time.time() - t0:.1f}s by {_vlm_label(cfg)}'})
            answer = answer or _phrase('error', lang)
        if cancel_requested():
            break
        _say(answer, lang, source)
        history.append((text, answer))

    if ended != 'cancelled':
        _say(_phrase('bye', lang), lang, source)
    return {'isdone': True, 'turns': len(history), 'ended': ended,
            'history': [{'q': q, 'a': a} for q, a in history]}
