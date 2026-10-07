"""Vision side of the ``qa`` skill (the skill itself is in hri.py, next to
``ask``): looking around, asking the vision model, place-restricted answers.

Visual Q&A about the robot's surroundings: ``qa``.

Answers spoken questions about what is around the robot — "what colour is the
shirt and where is it?", "which shelf level is the handbag on?" — from photos
taken by the head camera, using a local vision-language model served by Ollama
(QA_CONFIGS in configs/tasks.py: url, model, ...).

One call runs a whole conversation: listen -> look around if needed -> answer
-> listen ..., until the person says a stop word or the run is cancelled
(silence just means listening again; `idle_turns` is no longer used). The dashboard's Q&A button starts the
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

from ..utils import env_key, get_closest_loc
from .head import head_state, moveh
from .lift import lift


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
    'arm_camera': 'arm_rgb',    # qa::cam='arm' — one photo, the head does not move
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
    if cfg.get('cam') == 'arm':
        # The arm camera looks where the arm points: one photo as it is.
        cam = node.agents.get(cfg['arm_camera']) if node is not None else None
        if cam is None:
            log_data({'msg': f"qa: camera {cfg['arm_camera']!r} is not connected"})
            return []
        im = _fresh_frame(cam, 0.0, cfg['frame_timeout_sec'])
        if im is None:
            log_data({'msg': f"qa: no fresh frame from {cfg['arm_camera']}"})
            return []
        return [('arm', im)]
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


def _view_labels(shots: list) -> str:
    """'photo 1 = head up, photo 2 = head straight' / 'photo 1 = arm camera'."""
    return ', '.join(f'photo {i + 1} = ' + ('arm camera (on the gripper)' if v == 'arm' else f'head {v}')
                     for i, (v, _) in enumerate(shots))


def _ask_vlm(cfg: dict, shots: list, question: str, history: list, lang: str,
             facts: str | None = None) -> str:
    """One short answer to `question` from the photos, or '' if the model gave none.
    `facts`: verified product names (see _product_facts), added to the prompt."""
    views = _view_labels(shots)
    system = _SYSTEM_PROMPT.format(n=len(shots), views=views,
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
    return _place_answer(ids, places, lang) if ids else _phrase('unseen', lang)


# ── Products (신라면, 짜파게티 …): names from the product recogniser ───────────
#
# The VLMs cannot read pack labels reliably: on the arm-camera shelf photo they
# answered with colours only, or — given example names in the prompt — repeated
# an example that was not on the shelf (진라면). So product names come from
# skills/_products.py (detector + SigLIP match against reference photos), which
# names a pack only when it is sure; the robot serves blind users and must not
# guess a product.

# Words that make a question about products even without a product name.
_PRODUCT_WORDS = re.compile(r'라면|컵라면|봉지면|ramen|ramyun|ramyeon|noodle|\bmì\b|\bmỳ\b', re.IGNORECASE)

_SIDE = {'ko': {'left': '왼쪽', 'middle': '가운데', 'right': '오른쪽'},
         'en': {'left': 'on the left', 'middle': 'in the middle', 'right': 'on the right'},
         'vi': {'left': 'bên trái', 'middle': 'ở giữa', 'right': 'bên phải'}}


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
    sep = {'en': ' and ', 'vi': ' và '}.get(lang, ', ')
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
            parts.append({'ko': f'{topic} {where}에 있어요.', 'vi': f'{n} {where}.'}.get(lang, f'{n} is {where}.'))
        else:
            parts.append({'ko': f'{topic} 보이지 않아요.', 'vi': f'Tôi không thấy {n}.'}.get(lang, f"I can't see {n}."))
    unknown = sum(1 for p in inv if not p['name'])
    if unknown and any('보이지' in x or 'không thấy' in x or "can't" in x for x in parts):
        parts.append({'ko': f'이름을 알 수 없는 라면이 {unknown}개 있어요.',
                      'vi': f'Có {unknown} gói mì tôi không biết tên.'}.get(
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
    if lang == 'vi':
        txt = '; '.join(f'{s}: {n}' for s, n in groups) or 'Tôi không nhận ra gói mì nào'
        return txt + (f'; và {unknown} gói tôi không biết tên.' if unknown else '.')
    txt = '; '.join(f'{n} {s}' for s, n in groups) or "I can't identify any pack"
    return txt + (f"; and {unknown} pack(s) I can't name." if unknown else '.')


def _unrecognised_answer(names: list, inv: list, lang: str) -> str:
    """A question about a product the robot cannot recognise (not in the
    catalogue): say so — never yes or no — then what it does see."""
    n = names[0]
    head = {'ko': f"{n}{'은' if _batchim(n) else '는'} 확인할 수 없어요.",
            'vi': f'Tôi không nhận biết được {n}.'}.get(lang, f"I can't recognise {n}.")
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

_DROP = re.compile(r'떨어|떨궜|떨어뜨|rơi|đánh rơi|\bdrop|\bfall|\bfell\b', re.IGNORECASE)

# Spoken word → detector prompt. QA_CONFIGS['drop_objects'] adds / overrides.
_DROP_OBJECTS = {
    '숟가락': 'spoon', '수저': 'spoon', '스푼': 'spoon', '포크': 'fork', '젓가락': 'chopsticks',
    '컵': 'cup', '휴대폰': 'phone', '핸드폰': 'phone', '리모컨': 'remote control', '안경': 'glasses',
    '열쇠': 'key', '펜': 'pen', '약': 'pill bottle', '지갑': 'wallet',
    'spoon': 'spoon', 'fork': 'fork', 'chopsticks': 'chopsticks', 'cup': 'cup', 'phone': 'phone',
    'remote': 'remote control', 'glasses': 'glasses', 'key': 'key', 'pen': 'pen',
    'thìa': 'spoon', 'muỗng': 'spoon', 'nĩa': 'fork', 'dĩa': 'fork', 'đũa': 'chopsticks',
    'cốc': 'cup', 'điện thoại': 'phone', 'kính': 'glasses', 'chìa khóa': 'key', 'bút': 'pen',
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
    from .hri import _jamo, _similarity
    words = re.findall(r'[가-힣]{2,}|[a-zà-ỹ]{3,}', t)
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
    res = _detect_objects(node, ['foot', 'slipper', prompt], camera=camera, num_trials=2, max_instances='all')
    ins = res.get('ins', {})

    def pts(name):
        e = ins.get(name)
        return [(i['pose_3d'][:2], list(i['box'])) for i in (e.get('instances', [e]) if e else [])]
    feet = []
    for f in sorted(pts('foot') + pts('slipper'), key=lambda f: -(f[1][2] - f[1][0]) * (f[1][3] - f[1][1])):
        if not any(_box_inside(f[1], g[1], 0.3) or _box_inside(g[1], f[1], 0.3) for g in feet):
            feet.append(f)                                    # one foot found as foot AND slipper: once
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
        return {'ko': '무엇이 떨어졌는지 다시 말씀해 주세요.', 'vi': 'Bạn làm rơi gì vậy? Hãy nói lại.'}.get(
            lang, 'What did you drop? Please say it again.')
    word, prompt = obj
    topic = word + ('은' if _batchim(word) else '는')
    # Two frames must agree (±1 hour): a wrong direction is worse than none.
    runs = [_drop_measure(node, cfg, prompt) for _ in range(2)]
    if not any(r[4] for r in runs):
        return {'ko': f'{topic} 보이지 않아요.', 'vi': f'Tôi không thấy {word}.'}.get(lang, f"I can't see the {word}.")
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
        return {'ko': f'{topic} 보이지만, 방향을 정확히 알 수 없어요.',
                'vi': f'Tôi thấy {word} nhưng không xác định được hướng.'}.get(
            lang, f"I can see the {word}, but I can't tell the direction for sure.")
    hour = hours[0]
    dist = next(r[1] for r in runs if r[0] == hour)
    cm = max(10, int(round(dist * 10)) * 10)
    return {'ko': f'{topic} {hour}시 방향, 약 {cm}센티미터 거리에 있어요.',
            'vi': f'{word} ở hướng {hour} giờ của bạn, cách khoảng {cm} cm.'}.get(
        lang, f"The {word} is at your {hour} o'clock, about {cm} cm away.")


# ── "불 꺼줘" — light commands, done with turn_light ───────────────────────────
#
#   (H) 나 외출할거니까 불켜진거 있으면 꺼놔
#   (R) 네. 불 끄겠습니다.        → turn_light::inputs='off', loc='all'
#
# The robot says the acknowledgement, then acts, and says so if it failed.
# Which light: a place named in the request (a switchbot connection's loc, id
# or alias — 세탁실 …); else off → every light, on → the default light.

_LIGHT_NOUN = re.compile(r'불|전등|조명|형광등|스탠드|\blights?\b|\blamps?\b|đèn', re.IGNORECASE)
_LIGHT_OFF = re.compile(r'꺼|끄|끌|끕|\boff\b|tắt', re.IGNORECASE)
_LIGHT_ON = re.compile(r'켜|켤|켭|키|\bon\b|bật|mở đèn', re.IGNORECASE)


def _light_command(text: str):
    """(action 'on' | 'off', loc or None) for a light request, else None.
    "불 켜진 거 있으면 꺼" holds both 켜 and 꺼: the verb that comes last wins."""
    t = text or ''
    if not _LIGHT_NOUN.search(t):
        return None
    offs = [m.start() for m in _LIGHT_OFF.finditer(t)]
    ons = [m.start() for m in _LIGHT_ON.finditer(t)]
    if not offs and not ons:
        return None
    action = 'off' if (max(offs) if offs else -1) > (max(ons) if ons else -1) else 'on'
    loc = None
    try:
        from .switchbot import _devices, _norm as sb_norm
        nt = sb_norm(t)
        for d in _devices():
            names = [d['loc'], d['id'], *(d.get('aliases') or [])]
            if any(sb_norm(n) and sb_norm(n) in nt for n in names):
                loc = d['loc']
                break
    except Exception:
        pass
    return action, loc


def _light_phrase(key: str, action: str, lang: str) -> str:
    on = action == 'on'
    return {
        'ack':  {'ko': '네. 불 켜겠습니다.' if on else '네. 불 끄겠습니다.',
                 'en': "OK, I'll turn the lights on." if on else "OK, I'll turn the lights off.",
                 'vi': 'Vâng, tôi sẽ bật đèn.' if on else 'Vâng, tôi sẽ tắt đèn.'},
        'done': {'ko': '불을 켰어요.' if on else '불을 껐어요.',
                 'en': 'The lights are on.' if on else 'The lights are off.',
                 'vi': 'Đã bật đèn.' if on else 'Đã tắt đèn.'},
        'fail': {'ko': '죄송해요, 불을 켜지 못했어요.' if on else '죄송해요, 불을 끄지 못했어요.',
                 'en': "Sorry, I couldn't turn the lights on." if on else "Sorry, I couldn't turn the lights off.",
                 'vi': 'Xin lỗi, tôi không bật được đèn.' if on else 'Xin lỗi, tôi không tắt được đèn.'},
    }[key].get(lang) or ''


def _do_light(node, action: str, loc, lang: str):
    """Run turn_light; (ok, note for the log)."""
    from .switchbot import turn_light
    target = loc or ('all' if action == 'off' else None)
    r = turn_light(node=node, inputs=action, **({'loc': target} if target else {}))
    per = r.get('results') or {}
    note = ', '.join(f"{k}: {v.get('result')}" for k, v in per.items()) or str(r.get('msg', ''))
    return bool(r.get('isdone')), f'turn_light {action} {target or "default"} → {note}'
