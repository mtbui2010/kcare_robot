"""Helpers for approach.py. Pure refactor: behavior is 1:1 with the original
inline code; functions here exist only to break up the orchestration."""

import time
import numpy as np

from robot_agent.skill_configs import ARM_CONFIGS, LIFT_CONFIGS, ENV, CALIB_PARAMS, MOBILE_CONFIGS
from robot_agent.utils import get_env_specs, run_parallel_check

from kcare_robot.skills.arm import movel, movej, get_wrist_angle, arm_pose
from kcare_robot.skills.lift import lift, dlift, lift_state
from kcare_robot.skills.grip import grip
from kcare_robot.skills.recognition import find, find_arm
from kcare_robot.skills.mobile import move, forward


# ---------------------------------------------------------------------------
# Small numeric / config helpers
# ---------------------------------------------------------------------------

def lift_dz(node, z=None, dz=None):
    """Move the lift to an absolute `z` (or by delta `dz`), spilling overflow
    into the arm via `movel(dz=...)` when `z` falls outside the lift range."""
    assert z is not None or dz is not None
    lm_state = lift_state(node=node)['current_position']
    z = lm_state + dz if z is None else z
    if z <= LIFT_CONFIGS['range'][0]:
        ret = lift(node=node, inputs='lowest')
        if not ret['isdone']:
            return ret
        ret = movel(node=node, dz=z - LIFT_CONFIGS['range'][0])
        if not ret['isdone']:
            return ret
    elif z >= LIFT_CONFIGS['range'][1]:
        ret = lift(node=node, inputs='highest')
        if not ret['isdone']:
            return ret
        ret = movel(node=node, dz=z - LIFT_CONFIGS['range'][1])
        if not ret['isdone']:
            return ret
    else:
        ret = lift(node=node, inputs=z)
        if not ret['isdone']:
            return ret
    return {'isdone': True}


def is_inside_workspace(x, y, z):
    """Check `(y, z)` against ARM_CONFIGS workspace bounds (x is unconstrained)."""
    return (ARM_CONFIGS['range']['y'][0] <= y <= ARM_CONFIGS['range']['y'][1] and
            ARM_CONFIGS['range']['z'][0] <= z <= ARM_CONFIGS['range']['z'][1])


def get_dmove_approach(d_approach, wrist_angle, robot_mode, moveback=True):
    """Decompose an approach distance into (dx, dy, dz) along the wrist axis,
    using `robot_mode` to choose horizontal direction."""
    sint, cost = np.sin(wrist_angle * np.pi / 180), np.cos(wrist_angle * np.pi / 180)
    dxr, dyr = (-1, 0) if robot_mode == 'front' else (0, 1) if robot_mode == 'left' else (0, -1)
    dz, dxy = d_approach * sint, d_approach * cost
    dx, dy = dxr * dxy, dyr * dxy
    return (dx, dy, dz) if moveback else (-dx, dy, dz)


def get_placepose(placepose, target_height, robot_mode):
    """Resolve a placepose dict + target height into a [x, y, z, wrist_angle] list."""
    assert placepose is not None or target_height is not None, \
        f'placepose: {placepose}, target height: {target_height}'
    # x, y, z, dx, dy = ARM_CONFIGS['base_x'], 0.7, target_height, 0, 0
    x, y, z, dx, dy = -MOBILE_CONFIGS['dshift'], 0.7, target_height, 0, 0
    wrist_angle = 15 if z > 0.45 else 30 if z > 0.2 else 45
    if placepose is not None:
        x = placepose.get('x', x) + placepose.get('dx', dx)
        y = abs(placepose.get('y', y)) + placepose.get('dy', dy)
        z = placepose.get('z', z)
        wrist_angle = placepose.get('wrist_angle', wrist_angle)
    y = y if robot_mode == 'right' else -y
    return [x, y, z, wrist_angle]


# ---------------------------------------------------------------------------
# Input resolution: `inputs` → pose_3d
# ---------------------------------------------------------------------------

def resolve_pose_3d(node, inputs, not_move, kwargs):
    """Resolve `inputs` (either an explicit pose, an ENV location name, or an
    object spec like 'cup@table') into a `pose_3d`.

    Returns `(err, pose_3d, env, obj_name, ins_obj)`. If `err` is not None it is
    a result dict the caller should `return` immediately. `ins_obj` is the
    per-object info dict (empty unless inputs is an object spec).
    """
    if not isinstance(inputs, str):
        return None, inputs, {}, None, {}

    loc_name = inputs
    env = get_env_specs(loc_name, ENV)

    if len(env) > 0:
        # Known location: optionally drive there, then build a placepose.
        if not not_move:
            ret = move(node=node, inputs=loc_name)
            if not ret['isdone']:
                return ret, None, env, None, {}
        target_height = env.get('height', 0.75)
        robot_mode = env['default_mode']
        pose_3d = get_placepose(env.get('placepose', None), target_height, robot_mode)
        if pose_3d is None:
            raise Exception(f'No grasppose in {loc_name}')
        return None, pose_3d, env, None, {}

    # Object spec: retry find() up to 2 times to populate ins.
    obj_name = loc_name.split('@')[0]
    # NOTE: matches original logic — caller already popped 'ins' from kwargs, so
    # this initial get() yields {} and the loop runs at least once.
    ins_obj = kwargs.get('ins', {}).get(obj_name, {})
    find_trial = 0
    while len(ins_obj) == 0 and find_trial < 2:
        print(f'{obj_name} not found. Finding {obj_name}')
        kwargs.update(find(**kwargs, node=node, inputs=loc_name))
        ins_obj = kwargs.pop('ins', {}).get(obj_name, {})
        find_trial += 1
    if len(ins_obj) == 0:
        raise Exception(f'Find and refind 2 times failed. Terminated ...')
    return None, ins_obj['pose_3d'], {}, obj_name, ins_obj


# ---------------------------------------------------------------------------
# Pre-approach: forward to clip x into workspace + fold arm
# ---------------------------------------------------------------------------

def compute_forward_distance(x0, not_move):
    """Clip `x0` into ARM workspace and return `(x_used, mforward)`. `mforward`
    is zero when the residual is < 50 mm (treated as not worth driving)."""
    if not_move:
        x = x0
    else:
        x = np.clip(x0, a_min=ARM_CONFIGS['range']['x'][0], a_max=ARM_CONFIGS['range']['x'][1])
    mforward = x0 - x
    mforward = mforward if abs(mforward) > 0.05 else 0
    return x, mforward


def move_forward_and_fold(node, mforward, robot_mode):
    """Drive the base forward (if needed) and fold the arm — in parallel."""
    return run_parallel_check(funcs=[
        lambda: forward(node=node, inputs=mforward, wait=True) if abs(mforward) > 0.05 else {'isdone': True},
        lambda: movej(node=node, inputs='fold', mode=robot_mode),
    ])


# ---------------------------------------------------------------------------
# init_pose_fixed=True branch (find-and-grasp from a stowed pose)
# ---------------------------------------------------------------------------

def init_pose_lift_and_unfold(node, robot_mode, z):
    """Lift to `z + 150` and unfold the arm in parallel. Returns
    `(err_or_none, lift_to_limit)` where `lift_to_limit` is how much the lift
    fell short of the requested target (used downstream to size dz)."""
    lift_to = z + 0.1
    ret = run_parallel_check(funcs=[
        lambda: lift(node=node, inputs=lift_to),
        lambda: movej(node=node, inputs='unfold', mode=robot_mode),
    ])
    if not ret['isdone']:
        return ret, None
    time.sleep(0.5)
    current_lift = lift_state(node=node)['current_position']
    lift_to_limit = abs(current_lift - lift_to)
    return None, lift_to_limit


def init_pose_detect_posture(node, obj_name, lift_to_limit):
    """Detect whether the target object is lying down and which side carries
    its mass. Skipped (returns defaults) when the lift couldn't reach target.

    Returns `(err_or_none, islying, left_mass_percent, wrist_angle)`.
    """
    islying, left_mass_percent = False, 50
    if lift_to_limit < 50:
        ret = {'isdone': False}
        num_trial = 0
        while not ret['isdone'] and num_trial < 2:
            ret = find_arm(node=node, inputs=obj_name, detector='groundingdino', estimate_grasp=False)
            num_trial += 1
        if not ret['isdone']:
            return ret, None, None, None
        left_mass_percent = ret['ins'][obj_name]['mass_percents'][0]
        islying = ret['ins'][obj_name]['islying']
    wrist_angle = ARM_CONFIGS['wrist_angle']['lying'] if islying else ARM_CONFIGS['wrist_angle']['standing']
    return None, islying, left_mass_percent, wrist_angle


def init_pose_handle_lying(node, robot_mode, y, agents):
    """Lying-object path: fold → 'approach_lying' pose → side-step to align y."""
    ret = movej(node=node, inputs='fold', mode=robot_mode, wait=True)
    if not ret['isdone']:
        return ret
    ret = movej(node=node, inputs='approach_lying', mode=robot_mode, wait=True)
    if not ret['isdone']:
        return ret
    current_pose = arm_pose(node=node)['pose']
    dy = y - (0.1 if robot_mode == 'right' else -0.1) - current_pose[1]
    return movel(node=node, dy=np.clip(dy, -0.1, 0.1), wait=True, acc=0.5, speed=0.5)


def init_pose_handle_standing(node, robot_mode, wrist_angle, lift_to_limit,
                              left_mass_percent, islying):
    """Standing-object path: rotate wrist, lower + step in + turn base, then
    bias left/right based on detected mass distribution."""
    # ret = movelf(rx=-(90 + wrist_angle), node=node, wait=True)
    ret = movel(ry=-90 + wrist_angle, node=node, wait=True)
    if not ret['isdone']:
        return ret

    dx = -0.03 if robot_mode == 'front' else 0
    dz = -0.15 + lift_to_limit
    turn_angle = 35 if robot_mode == 'right' else -35
    ret = run_parallel_check(funcs=[
        lambda: lift_dz(node=node, dz=dz),
        lambda: movel(node=node, wait=True, dx=dx),
        lambda: movej(node=node, dr0=turn_angle, wait=True) if not islying else {'isdone': True},
    ])
    if not ret['isdone']:
        return ret

    # NOTE: the final `if not ret['isdone']` in the original is intentionally
    # outside the elif (see approach.py history); preserved verbatim.
    if left_mass_percent < 46:
        ret = movel(node=node, dx=0.2, drz=30, wait=True)
        if not ret['isdone']:
            return ret
    elif left_mass_percent > 54:
        ret = run_parallel_check(funcs=[
            lambda: forward(node=node, inputs=-0.3),
            lambda: movel(node=node, dx=0.2, drz=-30, wait=True),
        ])
    if not ret['isdone']:
        return ret
    return {'isdone': True}


# ---------------------------------------------------------------------------
# Normal approach branch
# ---------------------------------------------------------------------------

def resolve_wrist_angle_early(wrist_angle, pose_3d):
    """First resolution pass (runs before the init_pose branch): if the caller
    did not supply a wrist angle, take it from `pose_3d[3]` or the configured
    default."""
    if wrist_angle is None:
        wrist_angle = ARM_CONFIGS['approach_wrist_angle'] if len(pose_3d) == 3 else pose_3d[3]
    return wrist_angle


def resolve_wrist_angle_for_motion(node, action_type, wrist_angle, ins_obj, z):
    """Second resolution pass (normal branch only): fill from `ins_obj.islying`
    if still None, override with live wrist for place actions, then clamp by z
    (very low → lying angle, very high → standing angle)."""
    if wrist_angle is None:
        wrist_angle = ARM_CONFIGS['wrist_angle']['lying'] if ins_obj.get('islying', False) \
            else ARM_CONFIGS['wrist_angle']['standing']
        if action_type == 'place':
            wrist_angle = get_wrist_angle(node=node)
    wrist_angle = ARM_CONFIGS['wrist_angle']['lying'] if z < 0 \
        else ARM_CONFIGS['wrist_angle']['standing'] if z > 0.6 else wrist_angle
    return wrist_angle


def compute_shift_values(node, wrist_angle, robot_mode, dlift_up):
    """Build the dict of (dx, dy, dz) shifts that map the target pose to an
    arm-frame command: approach offset, base+lift compensation, calibration
    correction, and any pre-place lift-up the caller requested."""
    dx, dy, dz = get_dmove_approach(
        d_approach=ARM_CONFIGS['approach_d'],
        wrist_angle=wrist_angle,
        robot_mode=robot_mode,
    )
    return {
        'approach_close': [dx, dy, dz],
        'calib_offset': ARM_CONFIGS['calib_offset'],
        'dlift_up': dlift_up,
    }


def compute_target_and_lift(x, y, z, shift_values, lm_state):
    """Sum all shifts onto (x, y, z), then split the z-component between the
    lift (snapped to 50 mm grid + clamped to lift range) and the arm.

    Returns `(dmove, target_xyz, lift_to, dlift_to)`. `dmove` is the raw shift
    sum (pre-dlift split); `target_xyz` has the lift portion removed.
    """
    dx, dy, dz = np.sum(np.array(list(shift_values.values())), axis=0).tolist()
    target_x, target_y, target_z = x + dx, y + dy, z + dz
    lift_to = z + 0.15
    return (dx, dy, dz), (target_x, target_y, target_z), lift_to


def execute_approach_motion(node, robot_mode, lift_to, wrist_angle, target_xyz):
    """Run the standard approach sequence: lift → give → rotate wrist →
    swing arm sideways → set z → set (x, y). Returns the first failing ret,
    or `{'isdone': True}` on success."""
    target_x, target_y, target_z = target_xyz

    ret = lift(node=node, inputs=lift_to)
    if not ret['isdone']:
        return ret
    ret = movej(node=node, inputs='give', mode=robot_mode)
    if not ret['isdone']:
        return ret
    ret = movel(ry=-90 + wrist_angle, node=node)
    if not ret['isdone']:
        return ret
    ret = movel(node=node, dy=-0.15 if robot_mode == 'left' else 0.15)
    if not ret['isdone']:
        return ret
    ret = movel(node=node, z=target_z)
    if not ret['isdone']:
        return ret
    ret = movel(node=node, y=target_y, x=target_x)
    if not ret['isdone']:
        return ret
    return {'isdone': True}


# ---------------------------------------------------------------------------
# Place / retract
# ---------------------------------------------------------------------------

def release_object(node, dlift_up):
    """Place-without-wipe: ease down, open gripper, settle, open again."""
    dlift_up -= 0.05
    ret = movel(node=node, dz=-dlift_up, wait=False, speed=0.5)
    if not ret['isdone']:
        return ret
    ret = grip(node=node, inputs='open', wait=False)
    if not ret['isdone']:
        return ret
    ret = movel(node=node, dz=-0.02, wait=True, speed=0.2)
    if not ret['isdone']:
        return ret
    return grip(node=node, inputs='open', wait=True)


def execute_wipe(node, dlift_up):
    """Wipe pattern: descend a bit then sweep `dx` while creeping `dy`."""
    dlift_up -= 0.03
    ret = movel(node=node, dz=-dlift_up, wait=False, speed=0.2)
    if not ret['isdone']:
        return ret
    dx, dy = 0.3, 0.05
    movel(node=node, dy=-1.5 * dy)
    movel(node=node, dx=dx // 2)
    for _ in range(4):
        movel(node=node, dx=-dx)
        movel(node=node, dx=dx, dy=dy)
    movel(node=node, dx=-0.05)
    return {'isdone': True}


def retract_after_place(node, robot_mode, mforward, lift_height):
    """Post-place retract: lift up + raise arm, go to 'give', fold, then drive
    base back. `lift` only runs when there was forward movement to undo."""
    # ret = run_parallel_check(funcs=[
    #     lambda: dlift(node=node, inputs=0.1, wait=True),
    #     lambda: movel(node=node, dz=0.05, wait=True),
    # ])
    ret = movel(node=node, dz=0.1, wait=True)
    if not ret['isdone']:
        return ret
    ret = movej(node=node, inputs='give', mode=robot_mode, wait=False)
    if not ret['isdone']:
        return ret
    ret = movej(node=node, inputs='fold', mode=robot_mode, wait=True)
    if not ret['isdone']:
        return ret
    ret = run_parallel_check(funcs=[
        lambda: lift(node=node, inputs=lift_height) if abs(mforward) > 0 else {'isdone': True},
        lambda: forward(node=node, inputs=-mforward, wait=True),
    ])
    if not ret['isdone']:
        return ret
    return {'isdone': True}
