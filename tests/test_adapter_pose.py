"""KaairAdapter.pose() for the core's SAGE-v2 reconcile: x / y from the mobile_pose agent's last message, ts its
receive time; {} without a runtime; no container_states() (KAAIR has no drawer state sensor)."""

import time
from types import SimpleNamespace

import pytest

contract = pytest.importorskip('robot_agent.contract')


@pytest.fixture(scope='module')
def adapter():
    return contract.load_adapter('kcare_robot')


def test_pose_from_a_ros_pose(adapter):
    t = time.time() - 0.2
    pose = SimpleNamespace(position=SimpleNamespace(x=1.5, y=-2.0, z=0.0))
    p = adapter.pose(SimpleNamespace(agents={'mobile_pose': SimpleNamespace(rev_time=t, rev_data={'pose': pose})}))
    assert p == {'ts': t, 'x': 1.5, 'y': -2.0}


def test_pose_from_a_list_and_without_data(adapter):
    p = adapter.pose(SimpleNamespace(agents={'mobile_pose': SimpleNamespace(rev_time=None, rev_data={'pose': [1, 2, 90]})}))
    assert p == {'x': 1.0, 'y': 2.0, 'yaw': 90.0}
    assert adapter.pose(SimpleNamespace(agents={'mobile_pose': SimpleNamespace(rev_time=5.0, rev_data={'im': None})})) == {'ts': 5.0}
    assert adapter.pose(SimpleNamespace(agents={})) == {}


def test_no_container_states(adapter):
    assert not callable(getattr(adapter, 'container_states', None))
