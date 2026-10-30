"""KaairAdapter.health() (frame / pose age from the ROS agents' receive times, against the manifest's
limits) and vlm_ask() without a runtime; the manifest keeps the vlm verify specs off by default."""

import time
from types import SimpleNamespace

import pytest

contract = pytest.importorskip('robot_agent.contract')


@pytest.fixture(scope='module')
def adapter():
    return contract.load_adapter('kcare_robot')


def topic(age, period=1 / 30, data=True):
    return SimpleNamespace(rev_time=time.time() - age, rev_period=period, rev_data={'im': 0} if data else None)


def test_health_fresh(adapter):
    node = SimpleNamespace(agents={'head_rgb': topic(0.05), 'head_depth': topic(0.05), 'mobile_pose': topic(0.1)})
    h = adapter.health(node)
    assert h['ok'] is True and h['frame_age_s'] < 1 and h['pose_age_s'] < 1 and h['depth'] is True
    assert 25 < h['fps'] < 35


def test_health_stale_frame_and_pose(adapter):
    node = SimpleNamespace(agents={'head_rgb': topic(10.0), 'mobile_pose': topic(10.0)})
    h = adapter.health(node)
    assert h['ok'] is False and 'frame_age_s' in h['error'] and 'pose_age_s' in h['error'] and h['depth'] is None


def test_health_unknown_without_receive_times(adapter):
    assert adapter.health(SimpleNamespace(agents={'head_rgb': SimpleNamespace()}))['ok'] is None
    assert adapter.health(SimpleNamespace(agents={}))['ok'] is None


def test_manifest_limits_and_vlm_off(adapter):
    g = adapter.manifest.geometry
    assert g['frame_max_age_s'] == 2.0 and g['pose_max_age_s'] == 2.0
    assert adapter.manifest.capabilities['vlm_verify'] is False
    kinds = {s.action: [v.kind for v in s.verify] for s in adapter.manifest.skills}
    assert 'vlm' in kinds['Place'] and 'vlm' in kinds['Open']


def test_vlm_ask_without_a_runtime_raises(adapter):
    with pytest.raises(Exception):
        adapter.vlm_ask('Is the drawer open?', step={'action': 'Open'}, node=None)


def test_llm_budget_phrase(adapter):
    assert adapter.i18n.phrases['llm_budget']['ko']
