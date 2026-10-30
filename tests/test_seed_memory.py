"""planner/seed_memory.jsonl: the curated seed pool SAGE retrieves for KAAIR before it has a
live memory (manifest `planner.seed_memory`). Every episode uses KAAIR place names and passes
embodiplan's action model under the manifest's capabilities. Skipped without embodiplan."""

import json
from pathlib import Path

import pytest

yaml = pytest.importorskip('yaml')

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'kcare_robot' / 'configs'
MANIFEST = CONFIGS / 'manifest.yaml'
SEED = ROOT / 'kcare_robot' / 'planner' / 'seed_memory.jsonl'
LOC_ACTIONS = {'MoveTo', 'Place', 'PutIn', 'Open', 'Close'}


@pytest.fixture(scope='module')
def manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def episodes():
    return [json.loads(l) for l in SEED.read_text(encoding='utf-8').splitlines() if l.strip()]


def _steps(rec):
    return json.loads(rec['plan_text'])['steps']


def test_manifest_points_at_the_seed_file(manifest):
    rel = manifest['planner']['seed_memory']
    assert (MANIFEST.parent / rel).resolve() == SEED.resolve()


def test_seed_is_small_and_well_formed(episodes):
    assert 5 <= len(episodes) <= 40
    tasks = [e['task'] for e in episodes]
    assert len(set(tasks)) == len(tasks)
    for e in episodes:
        assert e['task'].strip() and e['reasoning'].strip() and e.get('verified') is True
        assert all(s.get('action') and s.get('object') for s in _steps(e))


def test_places_are_kaair_env_keys(episodes):
    keys = set()
    for f in (CONFIGS / 'locations').glob('*/skill_configs_override.json'):
        keys |= set(json.loads(f.read_text(encoding='utf-8')).get('ENV', {}))
    used = {s['object'] for e in episodes for s in _steps(e) if s['action'] in LOC_ACTIONS}
    assert used and used <= keys, sorted(used - keys)
    assert any('drawer@' in u for u in used)


def test_every_plan_passes_the_action_model(episodes, manifest):
    mt = pytest.importorskip('embodiplan.memory_tools')
    for e in episodes:
        ok, why = mt.verify_episode(e, manifest['capabilities'], require_verified=True)
        assert ok, f"{e['task']}: {why}"
    rep = mt.clean(episodes, capabilities=manifest['capabilities'], require_verified=True)
    assert rep.dropped == []                     # no duplicates either
