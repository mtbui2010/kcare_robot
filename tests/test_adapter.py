"""configs/adapter.py: the KaairAdapter robot_agent's core loads (load_adapter), what it tells the
core, and apply_skill_effect on the shared action model. Needs the robot_agent checkout on the
path (it is the contract); skipped without it."""

from pathlib import Path
from types import SimpleNamespace

import pytest

contract = pytest.importorskip('robot_agent.contract')

from kcare_robot.configs import grace_namemap as nm

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def adapter():
    return contract.load_adapter('kcare_robot')


@pytest.fixture(scope='module')
def robot(adapter):
    from robot_agent.core.planning.robot import RobotView
    return RobotView(adapter)


def test_load_adapter_finds_this_module(adapter):
    from kcare_robot.configs.adapter import KaairAdapter
    assert isinstance(adapter, KaairAdapter)
    assert isinstance(adapter, contract.RobotAdapter)
    assert adapter.manifest.robot == 'kaair' and adapter.manifest.language == 'ko'
    assert adapter.namemap is nm


def test_bind_matches_the_namemap(adapter):
    w = SimpleNamespace(found='cup', arrived=None, holding=None)
    assert adapter.bind({'action': 'MoveTo', 'object': 'Kitchen'}, w) == ('move', {'inputs': 'kitchen'})
    assert adapter.bind({'action': 'Pick', 'object': ''}, w) == ('pick', {'inputs': 'cup', 'type': 'fine_move'})
    assert adapter.bind({'action': 'PutIn', 'object': 'Drawer'}, w)[0] == 'place'
    assert adapter.bind({'action': 'Open', 'object': 'Drawer'}, w)[0] == 'open_drawer'
    assert adapter.bind({'action': 'Wait', 'object': ''}, w) is None
    with pytest.raises(contract.Unbindable):
        adapter.bind({'action': 'TurnOn', 'object': 'Light'}, w)


def test_what_the_core_reads_from_kaair(robot):
    # the skills the core used to call by name, now found through the manifest
    assert robot.holding_sensor() == 'grasp_succeed'
    assert robot.pose_sensor() == 'mobile_pose'
    assert robot.ask_spec().name == 'ask'
    place = {v.skill: v.expected for v in robot.sensors('Place')}
    assert place == {'grasp_succeed': False}          # the release passes on an empty gripper
    # geometry the core used to hard-code
    g = robot.geometry
    assert (g.arrive_tol_m, g.reach_m, g.reach_far_m) == (0.3, 0.78, 1.0)
    # body: one hand, Find only looks (a found drawer still needs the MoveTo)
    caps = robot.model_capabilities()
    assert caps.hands == 1 and caps.find_reaches is False
    assert robot.plan_actions == {'MoveTo', 'Find', 'Pick', 'Place', 'PutIn', 'Open', 'Close',
                                  'Sit', 'LieOn', 'Serve', 'Wait'}
    # a KAAIR ENV marks no drawer: its key names it
    assert robot.openable('drawer@kitchen', {}) and not robot.openable('shelf@kitchen', {})
    assert robot.openable('chest@bedroom', {'openable': True})
    assert {'check_fall', 'call_119', 'ask'} <= robot.quiet_skills


def test_korean_phrases_live_here(robot):
    from robot_agent.core.planning.announcer import Announcer
    a = Announcer(lang='ko', speak_backend=False, i18n=robot.i18n, quiet=robot.quiet_skills)
    assert a.announce('task_start') == '작업을 시작합니다'
    assert a.announce('step_start', action='MoveTo', object='식탁') == '식탁 이동을 진행합니다'
    assert a.announce('step_start', action='Find', object='컵') == '컵 찾기를 진행합니다'
    assert a.announce('step_start', action='check_fall', object='') == ''
    # the task-understanding description the core used to hard-code
    assert robot.robot_description().startswith('The robot is a mobile manipulator. It can:\n')
    assert robot.text('verify_failed.Pick') == 'pick reported success but the gripper is empty'


def test_manifest_matches_the_skills_strict():
    from robot_agent.contract.check_manifest import assert_manifest_matches
    rep = assert_manifest_matches(ROOT / 'kcare_robot', strict=True)
    assert not rep.errors


# ── apply_skill_effect on the action model, kcare extras kept ──
def _world(**kw):
    w = dict(arrived=None, found=None, found_pose=None, holding=None, holding_since=None, holding_pose=None,
             opened=set(), on=set())
    w.update(kw)
    return SimpleNamespace(**w)


@pytest.fixture(autouse=True)
def _no_robot(monkeypatch):
    monkeypatch.setattr(nm, '_robot_pose_xyz', lambda node: [1.0, 2.0, 90.0])
    monkeypatch.setattr(nm, '_container_key', lambda params, fallback: {'서랍장': 'drawer@bedroom'}.get(
        (params or {}).get('inputs'), fallback))


def test_effects_and_extras():
    w = _world(arrived='table@kitchen')
    nm.apply_skill_effect(w, 'find', {'inputs': 'water bottle'},
                          {'isdone': True, 'ins': {'water bottle': {'pose_3d': [0.5, 0, 0.8], 'grasppose': [1]}}})
    assert w.found == 'water bottle' and w.found_pose['pose_3d'] == [0.5, 0, 0.8]
    assert w.found_pose['robot_pose'] == [1.0, 2.0, 90.0]
    # the grasp takes what find found (not the first word of the arg)
    nm.apply_skill_effect(w, 'pick', {'inputs': 'water bottle'}, {'isdone': True})
    assert (w.holding, w.found, w.found_pose) == ('water bottle', None, None)
    assert w.holding_since and w.holding_pose['grasppose'] == [1]
    assert w.arrived == 'table@kitchen'                    # arrived stays with the sensor reconcile
    nm.apply_skill_effect(w, 'place', {'inputs': 'shelf'}, {'isdone': True})
    assert (w.holding, w.holding_since, w.holding_pose) == (None, None, None)
    nm.apply_skill_effect(w, 'open_drawer', {'inputs': '서랍장'}, {'isdone': True})
    assert w.opened == {'drawer@bedroom'}
    nm.apply_skill_effect(w, 'close_drawer', {'inputs': '서랍장'}, {'isdone': True})
    assert w.opened == set()
    nm.apply_skill_effect(w, 'move', {'inputs': 'kitchen'}, {'isdone': True})
    assert w.arrived == 'table@kitchen'


def test_failed_or_unknown_skill_changes_nothing():
    w = _world(found='cup')
    nm.apply_skill_effect(w, 'pick', {'inputs': 'cup'}, {'isdone': False})
    nm.apply_skill_effect(w, 'reply', {'inputs': 'hi'}, {'isdone': True})
    assert (w.found, w.holding) == ('cup', None)


@pytest.mark.parametrize('in_hand, after', [(False, None), (True, 'cup'), (RuntimeError('no wrist cam'), 'cup')])
def test_reconcile_reads_the_gripper(adapter, monkeypatch, in_hand, after):
    """A believed-held object the gripper check does not see is dropped from the belief; a check that
    holds it, or fails, leaves the belief."""
    import robot_agent.state as state_mod
    from robot_agent.core.planning import loop as LP
    from robot_agent.core.planning.base import WorldState

    def execute(skill, params, node=None, log_fn=None):
        assert skill == 'grasp_succeed'
        if isinstance(in_hand, Exception):
            raise in_hand
        return {'isdone': in_hand}
    st = SimpleNamespace(sr=SimpleNamespace(execute=execute), dm=SimpleNamespace(_ros_node=object()))
    monkeypatch.setattr(state_mod, 'current', lambda: st)
    monkeypatch.setattr(LP, 'reconcile_world', lambda node, world, threshold=0.8: world)   # no pose here
    monkeypatch.setattr(nm, 'reconcile_world', None, raising=False)
    w = WorldState(holding='cup', holding_since=1.0)
    adapter.reconcile(w)
    assert w.holding == after
