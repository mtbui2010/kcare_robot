"""configs/manifest.yaml: structure, and agreement with grace_namemap /
skills_config / the core's defaults. Pure python (pyyaml); the contract loader
from robot_agent is exercised only when its checkout is around."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

yaml = pytest.importorskip('yaml')

from kcare_robot.configs import grace_namemap as nm
from kcare_robot.configs.skills_config import SKILL_CONFIGS

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'kcare_robot' / 'configs' / 'manifest.yaml'

# robot_agent.contract.FailureCode values (the shared contract)
FAILURE_CODES = {'absent', 'perception_unavailable', 'unreachable', 'blocked', 'grasp_failed', 'dropped',
                 'hw_fault', 'estop', 'timeout', 'cancelled', 'software', 'unknown'}
VERIFY_KINDS = {'symbolic', 'sensor', 'vlm'}
SKILL_KEYS = {'name', 'action', 'params', 'pre', 'hold', 'effects', 'verify', 'interruptible', 'errors', 'version'}
PREDICATES = {'arrived', 'found', 'holding', 'hand_empty', 'opened'}
_PRED = re.compile(r'^(not )?([a-z_]+)\((\?[a-z_]+|\*)?\)$')


@pytest.fixture(scope='module')
def manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def skills(manifest):
    return {s['name']: s for s in manifest['skills']}


def test_top_level(manifest):
    assert manifest['robot'] == 'kaair'
    assert isinstance(manifest['version'], str)
    assert manifest['language'] == 'ko'
    caps = manifest['capabilities']
    assert caps['base'] is True and caps['arms'] == 1 and caps['drawer'] is True
    geo = manifest['geometry']
    # the values robot_agent's assist_loop uses today (ROBOT_AGENT_ARRIVE_TOL / _REACH_LATERAL / _REACH_FAR)
    assert (geo['arrive_tol_m'], geo['reach_m'], geo['reach_far_m']) == (0.3, 0.78, 1.0)


def test_skill_entries(skills, manifest):
    assert len(skills) == len(manifest['skills']), 'duplicate skill names'
    actions = [s['action'] for s in skills.values()]
    assert len(actions) == len(set(actions)), 'one skill per action (skill_for(action))'
    for name, s in skills.items():
        assert SKILL_KEYS <= set(s), f'{name}: missing {SKILL_KEYS - set(s)}'
        assert name in SKILL_CONFIGS, f'{name} is not a registered kcare skill'
        assert isinstance(s['version'], str)
        assert isinstance(s['interruptible'], bool)
        assert s['errors'] and set(s['errors']) <= FAILURE_CODES, f'{name}: {set(s["errors"]) - FAILURE_CODES}'
        for v in s['verify']:
            assert v['kind'] in VERIFY_KINDS, f'{name}: verify kind {v["kind"]}'
            if v['kind'] == 'sensor':
                assert v.get('skill') in skills, f'{name}: verify skill {v.get("skill")} not in the manifest'
        p = s['params']
        assert p['type'] == 'object' and isinstance(p['properties'], dict)
        assert set(p['required']) <= set(p['properties'])
        for pred in s['pre'] + s['hold'] + s['effects']:
            m = _PRED.match(pred)
            assert m, f'{name}: bad predicate {pred!r}'
            assert m.group(2) in PREDICATES, f'{name}: unknown predicate {pred!r}'
            var = m.group(3)
            if var and var != '*':
                assert var[1:] in p['properties'], f'{name}: {pred!r} uses a variable that is not a param'


def test_covers_the_planner(skills):
    by_action = {s['action']: n for n, s in skills.items()}
    for s in skills.values():
        for a in s.get('alias_actions', []):
            by_action[a] = s['name']
    # every GRACE / SAGE action the namemap maps must have its skill here
    for action, skill in nm.SKILL_MAP.items():
        assert by_action.get(action) == skill, f'{action} -> {skill} missing from the manifest'
    # skills the core calls by name
    for name in ('grasp_succeed', 'mobile_pose', 'ask', 'llm'):
        assert name in skills


def test_not_interruptible_while_holding_or_pulling(skills):
    for name in ('pick', 'place', 'open_drawer', 'close_drawer'):
        assert skills[name]['interruptible'] is False, name
    assert {'kind': 'sensor', 'skill': 'grasp_succeed'} in skills['pick']['verify']


def _world(**kw):
    w = dict(found=None, found_pose=None, holding=None, holding_since=None, holding_pose=None, opened=set())
    w.update(kw)
    return SimpleNamespace(**w)


def _holds(world, pred, arg):
    """Evaluate one manifest predicate on the namemap's WorldState fields.
    None = not tracked by apply_skill_effect (left to the sensor reconcile)."""
    neg, name, var = _PRED.match(pred).groups()
    val = arg if var and var != '*' else None
    if name == 'arrived':
        return None
    if name == 'hand_empty':
        r = world.holding is None
    elif name == 'found':
        r = world.found == val if val else world.found is not None
    elif name == 'holding':
        r = world.holding == val if val else world.holding is not None
    elif name == 'opened':
        r = val in world.opened
    return (not r) if neg else r


@pytest.mark.parametrize('skill,before', [
    ('find', {}),
    ('pick', {'found': 'cup'}),
    ('place', {'holding': 'cup'}),
    ('open_drawer', {}),
    ('close_drawer', {'opened': {'cup'}}),
])
def test_effects_match_apply_skill_effect(skills, monkeypatch, skill, before):
    monkeypatch.setattr(nm, '_robot_pose_xyz', lambda node: None)
    monkeypatch.setattr(nm, '_container_key', lambda params, fallback: fallback)
    arg = 'cup'
    w = _world(**before)
    nm.apply_skill_effect(w, skill, {'inputs': arg}, {'isdone': True})
    for eff in skills[skill]['effects']:
        got = _holds(w, eff, arg)
        assert got in (True, None), f'{skill}: effect {eff!r} does not hold after apply_skill_effect'


def test_skills_json_registers_them(skills):
    entries = json.loads((ROOT / 'kcare_robot' / 'configs' / 'common' / 'skills.json').read_text(encoding='utf-8'))
    registered = {e['name'] for e in entries}
    assert set(skills) <= registered


# robot_agent's contract loader, when the contract checkout is available. Run in
# a subprocess with a bare `robot_agent` package (no robot_agent/__init__.py, which
# pulls in FastAPI), so it never shadows anything in this process.
def _contract_root() -> Path:
    """KCARE_CONTRACT_ROOT, else the robot_agent checkout on sys.path, else ../robot_agent."""
    if os.environ.get('KCARE_CONTRACT_ROOT'):
        return Path(os.environ['KCARE_CONTRACT_ROOT'])
    for p in sys.path:
        if p and (Path(p) / 'robot_agent' / 'contract').is_dir():
            return Path(p)
    return ROOT.parent / 'robot_agent'


CONTRACT_ROOT = _contract_root()
_LOAD = r'''
import sys, types
pkg = types.ModuleType('robot_agent'); pkg.__path__ = [sys.argv[1] + '/robot_agent']
sys.modules['robot_agent'] = pkg
try:
    from robot_agent.contract import Manifest
except ImportError as e:
    print('SKIP', e); sys.exit(0)
m = Manifest.from_yaml(sys.argv[2])
assert m.robot == 'kaair' and m.language == 'ko', (m.robot, m.language)
pick = m.skill_for('Pick')
assert pick is not None and pick.name == 'pick' and pick.interruptible is False
assert m.skill_for('MoveTo').name == 'move'
print('OK', len(m.skills))
'''


@pytest.mark.skipif(not (CONTRACT_ROOT / 'robot_agent' / 'contract').is_dir(),
                    reason='robot_agent contract checkout not found')
def test_loads_with_robot_agent_contract():
    r = subprocess.run([sys.executable, '-c', _LOAD, str(CONTRACT_ROOT), str(MANIFEST)],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stdout + r.stderr
    if r.stdout.startswith('SKIP'):
        pytest.skip('robot_agent.contract has no Manifest yet: ' + r.stdout.strip())
    assert r.stdout.startswith('OK'), r.stdout
