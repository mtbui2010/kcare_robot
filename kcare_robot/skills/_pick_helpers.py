"""Helpers for pick.py. Pure refactor of inline logic into named steps; flow
matches the original module 1:1."""

import numpy as np

from robot_agent.skill_configs import ARM_CONFIGS
from robot_agent.utils import run_parallel_check
from robot_agent.skills import log_data

from kcare_robot.skills.arm import movet, movel, movelf, movej, arm_pose
from kcare_robot.skills.lift import lift, dlift
from kcare_robot.skills.grip import grip
from kcare_robot.skills.mobile import forward
from kcare_robot.skills.recognition import find_grasp , grasp_succeed, detect


# Drawer pull/push distance and drawer-handle search window.
DPULL = 0.3
GRASP_OPEN_MARGIN = 0.03     # the gripper opens to the grasp width + this [m] before closing on it
DEFAULT_HANDLE = 'brown handle'
DRAWER_HEIGHT = 0.75
DRAWER_RANGE = [0.1, 0.4]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def is_inside_workspace(dx, dy, dz):
    """Check (dx, dy, dz) against ARM_CONFIGS tool-frame workspace bounds."""
    r = ARM_CONFIGS['tool_range']
    return (r['x'][0] <= dx <= r['x'][1] and
            r['y'][0] <= dy <= r['y'][1] and
            r['z'][0] <= dz <= r['z'][1])

def is_vertical_gripper(node):
    gripper_angle = arm_pose(node=node)['pose'][-2]
    return abs(gripper_angle)<20


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

# open_drawer never drives more than this to bring the handle into its x band (m)
DRAWER_MAX_FORWARD = 0.3


def drawer_x_band():
    """Where along the heading (base-frame x, m) the handle must be for the arm to pull it: ARM_CONFIGS
    `drawer_x`, else pick's measured `approach_x.lying` ([0.15, 0.7]) — the same on both sides. DRAWER_RANGE
    ([0.1, 0.4], mirrored to [-0.4, -0.1] on the left) came from carerobotapp, whose handle point was in the
    wrist CAMERA frame (mm, [150, 200]); here pose_3d is base frame, and on kcare_bucheon the handle at
    x 0.57 m was refused as 0.67 m off (2026-10-26)."""
    band = ARM_CONFIGS.get('drawer_x') or (ARM_CONFIGS.get('approach_x') or {}).get('lying') or [0.15, 0.7]
    return float(band[0]), float(band[1])


def locate_drawer_handle(node, handle_name, robot_mode, handle_height=None, **select):
    """Locate the handle with the wrist camera and work out how far the base should drive forward (+) /
    backward (-) so the handle's x (base frame, along the heading) lands in `drawer_x_band()` — not too
    close, not too far. Returns `(err_or_None, mforward, xyz)`: `xyz` is the chosen
    handle's base-frame point AFTER that drive, for the grasp look to take the same handle (`near_xyz`).
    `handle_height` (ENV, m): of several handles take the one whose height (base-frame z) is nearest it — a
    cabinet's drawers share the handle name, so this is what tells `drawer` (0.415) from a drawer above it;
    a ⚠ when the nearest is > 8 cm off (set handle_height to the z the log shows). Without it: `select`
    (weights, target_distance …)."""
    if handle_height is not None:
        select = dict(select, near_xyz=[None, None, float(handle_height)], near_xyz_warn=0.08,
                      near_xyz_what='ENV handle_height')
    ret = detect(node=node, inputs=handle_name, camera='arm', **select)
    if not ret['isdone']:
        return ret, 0, None
    xyz = [float(v) for v in ret['ins'][handle_name]['pose_3d'][:3]]
    x0 = xyz[0]
    xmin, xmax = drawer_x_band()
    # forward (+) moves the handle back along x: drive by how far it is past the band (pick's base_shift sign;
    # the old commented-out line had it reversed — clip - x0 drove away from the band)
    mforward = float(x0 - np.clip(x0, xmin, xmax))
    mforward = 0.0 if abs(mforward) < 0.1 else mforward
    if abs(mforward) > DRAWER_MAX_FORWARD:
        return ({'isdone': False, 'msg': f'drawer: the {handle_name} is at x {x0:.2f} m, {abs(mforward):.2f} m outside '
                 f'[{xmin}, {xmax}] — more than {DRAWER_MAX_FORWARD} m to drive; check the ENV loc'}, 0, xyz)
    log_data({'msg': f'drawer: {handle_name} at [{", ".join(f"{v:.2f}" for v in xyz)}] (base), keep x in '
                     f'[{xmin}, {xmax}] → base {"stays" if not mforward else f"{mforward:+.2f} m"}'})
    xyz[0] -= mforward          # the base moves, the handle does not
    return None, mforward, xyz


def compute_drawer_mforward(node, handle_name, robot_mode, handle_height=None):
    """`locate_drawer_handle` without the handle point: `(err_or_None, mforward)`."""
    err, mforward, _ = locate_drawer_handle(node, handle_name, robot_mode, handle_height=handle_height)
    return err, mforward


def drive_forward_if_needed(node, mforward):
    """Drive forward by `mforward` mm if it exceeds the 50 mm dead zone."""
    if abs(mforward) > 0.05:
        return forward(node=node, inputs=mforward, wait=True)
    return {'isdone': True}


def retract_from_drawer(node, lift_height, mforward,  robot_mode=None):
    """Lift back to `lift_height`, fold the arm, and undo the forward drive."""
    funcs = [
        # lambda: lift(node=node, inputs=lift_height, wait=True),
        (lambda: movej(node=node, inputs='fold', mode=robot_mode))
        if robot_mode is not None else
        (lambda: movej(node=node, inputs='fold')),
        # lambda: forward(node=node, inputs=-mforward, wait=True) if abs(mforward) > 0.5 else {'isdone': True},
    ]
    ret =  run_parallel_check(funcs=funcs)
    assert ret['isdone'], f'{ret}'

    # back by what open_drawer drove (was "> 0.5": a 0.1-0.4 m drive was never undone — harmless while
    # mforward was always 0)
    ret = forward(node=node, inputs=-mforward, wait=True) if abs(mforward) > 0.05 else {'isdone': True}
    assert ret['isdone'], f'{ret}'

    return lift(node=node, inputs=lift_height, wait=True)


# ---------------------------------------------------------------------------
# Pick step helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# fine_move sub-step: compute grasp from detected pose
# ---------------------------------------------------------------------------

def grasp_pose_from_ins(ins, dpull):
    """Extract (dx, dy, dz, angle, width, dpull) from a find_grasp `ins` entry.
    `dz` is forced positive; `angle` is wrist-wrapped; `width` in metres
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


def execute_fine_grasp(node, dx, dy, dz, angle, width, dpull, pull_speed=1.0):
    """Open gripper to `width + GRASP_OPEN_MARGIN`, move to (dx, dy), rotate to `angle`,
    descend `dz`, close, then pull back `dpull` while keeping the gripper
    closed. Returns the final ret (may be a failure or success dict).

    `pull_speed` controls the dz=-dpull retraction speed (default 1.0).
    """

    ret = run_parallel_check(funcs=[
        lambda: grip(node=node, inputs=width + GRASP_OPEN_MARGIN, wait=False),
        lambda: (movet(node=node, dx=dx, dy=dy, wait=True), movej(node=node, dr6=angle, wait=True))[-1]
    ])
    assert ret['isdone'], f'{ret}'
    
    # ret = movej(node=node, dr6=angle, wait=True)
    # assert ret['isdone'], f'{ret}'

    ret = movet(node=node, dz=dz, acc=0.25, wait=True)
    assert ret['isdone'], f'{ret}'

    ret = grip(node=node, inputs='close', wait=True)
    if not ret['isdone']:
        return ret
        
    ret = run_parallel_check(funcs=[
        lambda: grip(node=node, inputs='close', wait=False),
        lambda: movet(node=node, dz=-dpull, acc=0.25, wait=True),
    ])
    assert ret['isdone'], f'{ret}'

    return {'isdone': True}
