"""KAAIR's RobotAdapter: everything robot_agent's planning core needs to know about this robot.

``robot_agent.contract.load_adapter('kcare_robot')`` finds this module (``make_adapter``). It
wraps the existing ``grace_namemap`` (name maps, ``build_params``, ``observe``, ``vlm_hook``,
``apply_skill_effect``) and ``manifest.yaml`` (skills, verify sensors, capabilities, geometry,
language), and adds what the core used to hard-code for KAAIR:

* ``i18n``: the Korean narration (the announcer's ``ko`` phrasebook and verbs), the
  task-understanding robot description and the post-check failure messages;
* ``is_openable``: a KAAIR ENV marks no drawer, its key names it (``drawer@kitchen``);
* ``quiet_skills``: skills that speak to the user themselves, so their steps are not narrated;
* ``health()``: head-camera frame age / fps / depth and base-pose age from the ROS agents
  (``rev_time`` / ``rev_period``), against the manifest's ``frame_max_age_s`` / ``pose_max_age_s``;
* ``vlm_ask()``: a yes / no question on the head camera frame for the manifest's vlm verify specs
  (only asked when they are turned on: ``capabilities.vlm_verify`` / ``ROBOT_AGENT_VLM_VERIFY``).

Only robot_agent's public contract is used (``robot_agent.contract``).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from robot_agent.contract import Manifest, NamemapAdapter

from . import grace_namemap

MANIFEST = Path(__file__).with_name("manifest.yaml")

# narration milestones: {kind: {lang: template}}; {object} {verb} {reason} {eul} (을/를)
_KO_PHRASES = {
    'task_start':   '작업을 시작합니다',
    'step_start':   '{object} {verb}{eul} 진행합니다',
    'step_success': '{object} {verb}{eul} 완료했습니다',
    'step_fail':    '{object} {verb}에 실패했습니다. {reason}',
    'replan':       '계획을 다시 세웁니다',
    'done_success': '작업을 완료했습니다',
    'done_fail':    '작업에 실패했습니다',
    'plan_ready':   '계획을 생성했습니다',
    'unsupported':  '지원하지 않는 동작입니다',
    'llm_down':     '지금 언어 모델에 연결할 수 없어서 작업을 중단합니다',
    'generic':      '진행 중입니다',
}

# per-action verbs, GRACE actions and the skill names of the direct path (``find::apple``)
_KO_VERBS = {
    'MoveTo': '이동', 'GoTo': '이동', 'Navigate': '이동',
    'Find': '찾기', 'Search': '찾기', 'Detect': '찾기',
    'Pick': '집기', 'PickUp': '집기', 'Grasp': '집기',
    'Place': '놓기', 'PlaceAt': '놓기', 'PutDown': '놓기', 'PutIn': '놓기',
    'Open': '열기', 'Close': '닫기',
    'navigate': '이동', 'move': '이동', 'moveto': '이동',
    'find': '찾기', 'detect': '찾기', 'detect_object': '찾기',
    'pick': '집기', 'grasp': '집기',
    'place': '놓기', 'put': '놓기',
    'open': '열기', 'close': '닫기',
}

PHRASES: dict = {k: {'ko': v} for k, v in _KO_PHRASES.items()}
PHRASES.update({f'verb.{a}': {'ko': v} for a, v in _KO_VERBS.items()})
PHRASES.update({
    # how the task-understanding prompt (English) describes this robot
    'robot_description': {'en': 'The robot is a mobile manipulator. It can:\n'
                                'move to a place, find an object, pick it up, place it on a surface, '
                                'open / close a drawer, put an object in an open drawer.'},
    # a skill reported success but its sensor verify says otherwise (English: replan reason + log)
    'verify_failed.Pick': {'en': 'pick reported success but the gripper is empty'},
    'verify_failed.Place': {'en': 'place reported success but the object is still in the gripper'},
    'verify_failed.PutIn': {'en': 'place reported success but the object is still in the gripper'},
    # the run used its LLM-call budget (robot_agent budget.budget_end)
    'llm_budget': {'ko': '정해진 시도 횟수 안에 작업을 해결하지 못해서 작업을 중단합니다'},
})

# ROS agents the health report reads (TopicAgent rev_time / rev_period)
HEALTH_AGENTS = {'frame': 'head_rgb', 'depth': 'head_depth', 'pose': 'mobile_pose'}

# skills that speak for themselves (hri reply / ask, the emergency skills) or have nothing to tell
QUIET_SKILLS = {'reply', 'ask', 'qa', 'announce', 'wait', 'check_fall', 'check_restroom', 'call_119',
                'warmup_servers'}


class KaairAdapter(NamemapAdapter):
    """grace_namemap + manifest.yaml, with KAAIR's phrases and ENV conventions."""

    quiet_skills = QUIET_SKILLS

    def __init__(self, manifest: Manifest | None = None):
        super().__init__(grace_namemap, manifest if manifest is not None else Manifest.from_yaml(MANIFEST))
        phrases = dict(PHRASES)
        phrases.update(self.i18n.phrases)          # grace_namemap.PHRASES, if it ever has any, wins
        self.i18n.phrases = phrases

    def reconcile(self, world, node=None):
        """The belief brought in line with KAAIR at a run's start / on resume: where the base is (the
        namemap / the core's pose + ENV match, as before), then the gripper: a believed-held object the
        wrist depth check (``grasp_succeed``) does not see is dropped from the belief. A check that
        cannot run (no runtime, no camera) leaves the belief as it was."""
        super().reconcile(world, node)
        if getattr(world, 'holding', None) and _gripper_holds(node) is False:
            world.holding = None
            for k in ('holding_since', 'holding_pose'):
                if hasattr(world, k):
                    setattr(world, k, None)
        return world

    def health(self, node=None) -> dict:
        """Head-camera frame age / fps / depth and base-pose age (seconds), from when the ROS agents
        last received a message; ``ok`` False when one is past the manifest's limits, None when
        nothing could be read (no runtime, an agent that keeps no receive time). Never raises."""
        try:
            node = node if node is not None else _node()
            agents = dict(getattr(node, 'agents', None) or {})
            now = time.time()

            def age(name):
                t = getattr(agents.get(name), 'rev_time', None)
                return round(now - t, 3) if isinstance(t, (int, float)) else None
            period = getattr(agents.get(HEALTH_AGENTS['frame']), 'rev_period', None)
            dep = agents.get(HEALTH_AGENTS['depth'])
            h = {'frame_age_s': age(HEALTH_AGENTS['frame']),
                 'fps': round(1.0 / period, 1) if isinstance(period, (int, float)) and period > 0 else None,
                 'depth': (getattr(dep, 'rev_data', None) is not None) if dep is not None else None,
                 'pose_age_s': age(HEALTH_AGENTS['pose'])}
            lim = self.geometry.extra
            bad = [f'{k} {h[k]:.1f} s > {lim[m]} s' for k, m in (('frame_age_s', 'frame_max_age_s'),
                                                                  ('pose_age_s', 'pose_max_age_s'))
                   if h[k] is not None and lim.get(m) is not None and h[k] > float(lim[m])]
            known = any(v is not None for v in h.values())
            h['ok'] = (not bad) if known else None
            if bad:
                h['error'] = '; '.join(bad)
            return h
        except Exception as e:
            return {'ok': None, 'error': f'health not readable: {e}'}

    def vlm_ask(self, prompt: str, step: dict | None = None, node=None):
        """``prompt`` (a yes / no question) on the current head-camera frame, through the VLM
        connection the qa skill uses: 'yes' / 'no' / 'unsure'. Raises when there is no runtime,
        camera or VLM (the core then looks with find instead)."""
        from kcare_robot.skills import hri as qa
        from kcare_robot.skills._recognition_helpers import fetch_camera_data
        node = node if node is not None else _node()
        if node is None:
            raise RuntimeError('no ROS node')
        rgb = fetch_camera_data(node, 'head').rgb
        cfg = qa._qa_config({})
        body = {'model': cfg['model'], 'stream': False, 'think': False, 'keep_alive': cfg['keep_alive'],
                'format': {'type': 'object', 'properties': {'answer': {'type': 'string',
                                                                       'enum': ['yes', 'no', 'unsure']}},
                           'required': ['answer']},
                'options': {'temperature': 0.0, 'num_predict': 16},
                'messages': [{'role': 'user', 'images': [qa._b64jpeg(rgb, 640)],
                              'content': f'{prompt} Answer yes, no, or unsure if you cannot tell.'}]}
        return json.loads(qa._post_chat(cfg, body).json()['message']['content']).get('answer', 'unsure')

    @staticmethod
    def is_openable(key: str, entry: dict | None = None) -> bool:
        """An ENV place opens when it says so, or when its key names a drawer."""
        return bool((entry or {}).get('openable')) or 'drawer' in str(key).lower()


# the skill that tells whether something is in the gripper (the sensor that verifies a Pick in manifest.yaml)
HOLDING_SENSOR = 'grasp_succeed'


def _gripper_holds(node=None):
    """True / False from the gripper check through the running agent's skill registry; None when it
    cannot be read (no runtime, no ROS node, the check failed)."""
    try:
        from robot_agent.state import current
        state = current()
        if node is None:
            node = getattr(state.dm, '_ros_node', None)
        sr = getattr(state, 'sr', None)
        if sr is None or node is None:
            return None
        return bool((sr.execute(HOLDING_SENSOR, {}, node) or {}).get('isdone'))
    except Exception:
        return None


def _node():
    """The running agent's ROS node, or None outside a runtime."""
    try:
        from robot_agent.state import current
        return getattr(current().dm, '_ros_node', None)
    except Exception:
        return None


def make_adapter() -> KaairAdapter:
    return KaairAdapter()
