"""Helpers for pick.py. Pure refactor of inline logic into named steps; flow
matches the original module 1:1."""

import numpy as np

from robot_agent.skill_configs import ARM_CONFIGS
from robot_agent.utils import run_parallel_check

from kcare_robot.skills.arm import movet, movel, movelf, movej
from kcare_robot.skills.lift import lift, dlift
from kcare_robot.skills.grip import grip
from kcare_robot.skills.mobile import forward
from kcare_robot.skills.recognition import find_arm, grasp_succeed


# Drawer pull/push distance and drawer-handle search window.
DPULL = 0.25
DEFAULT_HANDLE = 'brown handle'
DRAWER_HEIGHT = 0.75
DRAWER_RANGE = [0.2, 0.3]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def is_inside_workspace(dx, dy, dz):
    """Check (dx, dy, dz) against ARM_CONFIGS tool-frame workspace bounds."""
    r = ARM_CONFIGS['tool_range']
    return (r['x'][0] <= dx <= r['x'][1] and
            r['y'][0] <= dy <= r['y'][1] and
            r['z'][0] <= dz <= r['z'][1])


def fix_angle(angle):
    """Wrap an angle to (-90, 90] so the wrist takes the shortest rotation."""
    if abs(angle) <= 90:
        return angle
    if angle > 90:
        return angle - 180
    return angle + 180


# ---------------------------------------------------------------------------
# Pick name parsing
# ---------------------------------------------------------------------------

def split_loc(loc_name):
    """Split `'caption@loc'` into `(caption, loc_or_None)`."""
    splits = loc_name.split('@')
    if len(splits) >= 2:
        return splits[0], '@'.join(splits[1:])
    return loc_name, None


# ---------------------------------------------------------------------------
# Drawer handle helpers (shared by open/close)
# ---------------------------------------------------------------------------

def compute_drawer_mforward(node, handle_name, robot_mode):
    """Locate the handle, then return how far the base should drive forward to
    centre it within `DRAWER_RANGE`. Returns `(err_or_None, mforward)`."""
    ret = find_arm(node=node, inputs=handle_name, estimate_grasp=False, detector='groundingdino')
    if not ret['isdone']:
        return ret, 0
    x0 = ret['ins'][handle_name]['loc_3d'][0]
    xmin = DRAWER_RANGE[0] if robot_mode == 'right' else -DRAWER_RANGE[1]
    xmax = DRAWER_RANGE[1] if robot_mode == 'right' else -DRAWER_RANGE[0]
    mforward = np.clip(x0, xmin, xmax) - x0
    return None, (0 if abs(mforward) < 50 else mforward)


def drive_forward_if_needed(node, mforward):
    """Drive forward by `mforward` mm if it exceeds the 50 mm dead zone."""
    if abs(mforward) > 50:
        return forward(node=node, inputs=mforward, wait=True)
    return {'isdone': True}


def retract_from_drawer(node, lift_height, mforward, robot_mode=None):
    """Lift back to `lift_height`, fold the arm, and undo the forward drive."""
    funcs = [
        lambda: lift(node=node, inputs=lift_height, wait=True),
        (lambda: movej(node=node, inputs='fold', mode=robot_mode))
        if robot_mode is not None else
        (lambda: movej(node=node, inputs='fold')),
        lambda: forward(node=node, inputs=-mforward, wait=True) if abs(mforward) > 50 else {'isdone': True},
    ]
    return run_parallel_check(funcs=funcs)


# ---------------------------------------------------------------------------
# Pick step helpers
# ---------------------------------------------------------------------------

def pickup_motion_direct(node, dz):
    """Direct grasp: dive `dz`, close gripper, retract `dz`, lift 50 mm."""
    ret = movet(node=node, dz=dz, speed=0.5)
    if not ret['isdone']:
        return ret
    ret = grip(node=node, inputs='close', wait=True)
    if not ret['isdone']:
        return ret
    ret = movet(node=node, dz=-dz)
    if not ret['isdone']:
        return ret
    return movel(node=node, dz=50, wait=True)


def post_pick_retract(node, robot_mode, lift_height, mforward):
    """Post-pick retract: keep gripper closed while lifting + going to 'give',
    then fold and drive the base back."""
    ret = run_parallel_check(funcs=[
        lambda: grip(node=node, inputs='close', wait=False),
        lambda: lift(node=node, inputs=lift_height, wait=True),
        lambda: movej(node=node, inputs='give', mode=robot_mode, wait=False),
    ])
    if not ret['isdone']:
        return ret
    return run_parallel_check(funcs=[
        lambda: grip(node=node, inputs='close', wait=False),
        lambda: movej(node=node, inputs='fold', mode=robot_mode, wait=True),
        lambda: forward(node=node, inputs=-mforward, wait=True),
    ])


def post_drawer_open_prep(node, robot_mode):
    """After opening a drawer: ease the lift down + fold, then assume the
    'approach_lying' pose so the gripper is angled into the drawer."""
    ret = run_parallel_check(funcs=[
        lambda: dlift(node=node, inputs=0.1, wait=True),
        lambda: movej(node=node, inputs='fold', mode=robot_mode, wait=True),
    ])
    if not ret['isdone']:
        return ret
    return movej(node=node, inputs='approach_lying', mode=robot_mode, wait=True)


# ---------------------------------------------------------------------------
# fine_move sub-step: compute grasp from detected pose
# ---------------------------------------------------------------------------

def grasp_pose_from_ins(ins, dpull):
    """Extract (dx, dy, dz, angle, width, dpull) from a find_arm `ins` entry.
    `dz` is forced positive; `angle` is wrist-wrapped; `width` is scaled ×10
    to match the gripper command range. Returns `None` if outside workspace."""
    dx, dy, dz, angle, width = ins['grasppose']
    # width *= 10
    angle = fix_angle(angle)
    dz = abs(dz)
    if not is_inside_workspace(dx, dy, dz):
        print(f'Out of move tool range : {(dx, dy, dz)}')
        return None
    effective_dpull = min(dz, 0.3) if dpull is None else dpull
    return dx, dy, dz, angle, width, effective_dpull


def execute_fine_grasp(node, dx, dy, dz, angle, width, dpull):
    """Open gripper to `width+200`, move to (dx, dy), rotate to `angle`,
    descend `dz`, close, then pull back `dpull` while keeping the gripper
    closed. Returns the final ret (may be a failure or success dict)."""
    ret = grip(node=node, inputs=width + 0.02, wait=False)
    if not ret['isdone']:
        return ret
    ret = movet(node=node, dx=dx, dy=dy, wait=True)
    if not ret['isdone']:
        return ret
    ret = movej(node=node, dr6=angle, wait=True)
    if not ret['isdone']:
        return ret
    ret = movet(node=node, dz=dz, speed=0.5, wait=True)
    if not ret['isdone']:
        return ret
    ret = grip(node=node, inputs='close', wait=True)
    if not ret['isdone']:
        return ret
    return run_parallel_check(funcs=[
        lambda: grip(node=node, inputs='close', wait=False),
        lambda: movet(node=node, dz=-dpull, wait=True),
    ])
