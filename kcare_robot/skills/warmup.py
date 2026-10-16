"""warmup_servers — load the models on the AI servers before a scenario, so the first answer is not slow.

Whisper (speaches) unloads its model after 5 min unused and Ollama after its keep_alive; visionserve loads each
model on its first request. On 2026-10-15 Whisper could not load again at all (the GPU was full) and `ask` failed.
Run it from the dashboard (DevicePanel: 🔥 Warm up) or a plan; it moves nothing and says nothing.

Every server of every AI connection (robot_agent/connect/pool.py: vlms / visions / voices, and the older
llm / visionserve / stt) is warmed at the same time. A cloud API (OpenAI / Gemini / Claude) has nothing to load:
its key and model are checked instead. Per kind:

- error (red): every server of the kind failed;
- ok (green): the main (first) server of every connection of the kind works;
- warn (amber): a main server failed but another server of the kind works — calls will fall back.

The result is also the pools' state, so the Connections panel shows the same colours.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from robot_agent.skill_configs import FIND_CONFIGS
from robot_agent.utils import exception_handler

from . import hri

KIND_NAME = {'vlms': 'VLMs', 'visions': 'Visions', 'voices': 'Voices'}
LEVEL_RANK = {'ok': 0, 'warn': 1, 'error': 2}


def _vision_models():
    """The visionserve models kcare's skills call: find / find_grasp's detector, the background check's, the
    product embedder."""
    models = []
    for key in ('detect_object', 'detect_grasp'):
        m = (FIND_CONFIGS.get(key) or {}).get('model')
        if m:
            models.append(m)
    bg = FIND_CONFIGS.get('detect_background') or {}
    if bg.get('use') and bg.get('avoid_model'):
        models.append(bg['avoid_model'])
    models.append('siglip-image')
    return list(dict.fromkeys(models))


def _pools(dm):
    """[(kind, connection id, pool)], plus qa's VLM from QA_CONFIGS when no connection serves it."""
    from robot_agent.connect.pool import make_pool
    pools = dm.pools() if hasattr(dm, 'pools') else []
    cfg = hri._qa_config({})
    if cfg.get('_pool') is None and not any(k == 'vlms' and cid == cfg.get('connection', 'vlm')
                                            for k, cid, _ in pools):
        try:
            pools.append(('vlms', 'QA_CONFIGS',
                          make_pool('QA_CONFIGS', 'vlms', [{'provider': 'ollama', 'url': cfg['url'].rstrip('/'),
                                                            'model': cfg['model'],
                                                            'keep_alive': cfg.get('keep_alive', '30m')}])))
        except Exception:
            pass
    return pools


@exception_handler
def warmup_servers(node=None, **kwargs):
    """Warm every server of every AI connection at once. Returns {isdone, level, kinds: {kind: {level,
    servers: [...]}}, msg}; isdone False only when every server of some kind failed."""
    from robot_agent.skills import log_data
    from robot_agent.state import current
    dm = current().dm
    pools = _pools(dm)
    vmodels = _vision_models()
    jobs = [(kind, cid, pool, i) for kind, cid, pool in pools for i in pool.enabled()]
    if not jobs:
        msg = 'no server to warm up (no vlms, visions or voices connection)'
        log_data({'msg': msg})
        return {'isdone': False, 'level': 'error', 'kinds': {}, 'msg': msg}

    def one(job):
        kind, cid, pool, i = job
        t0 = time.time()
        try:
            pool.warm(i, vmodels if kind == 'visions' else ())
            return job, True, time.time() - t0, ''
        except Exception as e:
            return job, False, time.time() - t0, pool.state[i].error or str(e)[:120]

    with ThreadPoolExecutor(max_workers=min(16, len(jobs))) as ex:
        results = list(ex.map(one, jobs))

    kinds = {}
    for (kind, cid, pool, i), ok, sec, err in results:
        k = kinds.setdefault(kind, {'servers': [], 'primary_failed': False, 'any_ok': False})
        primary = i == pool.primary()
        k['servers'].append({'connection': cid, 'idx': i, 'label': pool.label(i), 'primary': primary,
                             'ok': ok, 'sec': round(sec, 1), 'error': err})
        k['any_ok'] |= ok
        k['primary_failed'] |= primary and not ok
        log_data({'msg': f"warmup: {cid} {pool.label(i)} " + (f'ready ({sec:.1f} s)' if ok else f'FAILED — {err}')})
    for k in kinds.values():
        k['level'] = 'error' if not k.pop('any_ok') else ('warn' if k.pop('primary_failed') else 'ok')
        k.pop('primary_failed', None)
    level = max((k['level'] for k in kinds.values()), key=LEVEL_RANK.get)

    parts = []
    for kind in ('voices', 'vlms', 'visions'):
        if kind not in kinds:
            continue
        k = kinds[kind]
        bad = [s for s in k['servers'] if not s['ok']]
        txt = f"{KIND_NAME[kind]} {'✓' if k['level'] == 'ok' else ('fallback' if k['level'] == 'warn' else 'FAILED')}"
        txt += f" ({len(k['servers']) - len(bad)}/{len(k['servers'])})"
        if bad:
            txt += ': ' + '; '.join(f"{s['connection']} {s['label']}: {s['error']}" for s in bad)
        parts.append(txt)
    msg = ' · '.join(parts)
    log_data({'msg': f'warmup: {level} — {msg}'})
    return {'isdone': level != 'error', 'level': level, 'kinds': kinds, 'msg': msg}
