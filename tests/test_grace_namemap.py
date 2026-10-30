"""grace_namemap imports without a runtime (only re / time / typing at module
level) and its pure helpers behave as the closed loop expects."""

from types import SimpleNamespace

import pytest

from kcare_robot.configs import grace_namemap as nm


@pytest.fixture(autouse=True)
def _no_robot(monkeypatch):
    # pick records the base pose through the mobile_pose skill (ROS); keep it out
    monkeypatch.setattr(nm, '_robot_pose_xyz', lambda node: None)


def _world(**kw):
    w = dict(found=None, found_pose=None, holding=None, holding_since=None, holding_pose=None, opened=set())
    w.update(kw)
    return SimpleNamespace(**w)


def test_public_names():
    for name in ('to_loc', 'to_obj', 'SKILL_MAP', 'SUPPORTED_ACTIONS', 'NOOP_ACTIONS', 'VLM_ACTIONS',
                 'build_params', 'observe', 'vlm_hook', 'apply_skill_effect'):
        assert hasattr(nm, name), name


def test_name_resolvers():
    assert nm.to_loc('LivingRoom') == 'living room'
    assert nm.to_loc('BigCloset') == 'big_closet'
    assert nm.to_obj('WaterBottle') == 'water bottle'
    assert nm.to_obj('RedApple') == 'red apple'
    assert nm.to_loc('') == '' and nm.to_obj('') == ''


def test_build_params():
    w = _world(found='cup')
    assert nm.build_params('MoveTo', 'Kitchen', w) == {'inputs': 'kitchen'}
    assert nm.build_params('Pick', 'Mug', w) == {'inputs': 'cup', 'type': 'fine_move'}
    assert nm.build_params('Place', 'SofaTable', w) == {'inputs': 'sopha table'}
    assert nm.build_params('Wait', '', w) == {}


def test_skill_effects():
    w = _world()
    nm.apply_skill_effect(w, 'find', {'inputs': 'cup'}, {'isdone': True})
    assert w.found == 'cup'
    nm.apply_skill_effect(w, 'pick', {'inputs': 'cup'}, {'isdone': True})
    assert (w.holding, w.found) == ('cup', None)
    nm.apply_skill_effect(w, 'place', {'inputs': 'table'}, {'isdone': False})
    assert w.holding == 'cup', 'a failed skill must not change the belief'
    nm.apply_skill_effect(w, 'place', {'inputs': 'table'}, {'isdone': True})
    assert w.holding is None


def test_observe_without_runtime():
    # no booted AgentState here -> the safe default, never an exception
    assert nm.observe() == ('', [])
