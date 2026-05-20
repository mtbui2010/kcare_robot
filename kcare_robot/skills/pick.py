"""Pick / drawer / stacking skills. Public API:

  - `fine_move`     : detect grasp pose with the wrist camera then execute it.
  - `open_drawer`   : approach a drawer, grab its handle, pull open.
  - `close_drawer`  : push a known drawer shut.
  - `pick_no_sound` : full pick flow (no TTS).
  - `pick_card`     : domain-specific card pick.
  - `stack`         : domain-specific stacking demo.
  - `pick`          : `pick_no_sound` wrapped with picking/picked announcements.

Heavy lifting is split into helpers in `_pick_helpers.py`. This file is the
orchestration only — flow matches the original module 1:1.
"""

import time

from robot_agent.skill_configs import ENV, ARM_CONFIGS, NO_ACTION
from robot_agent.utils import (
    run_parallel_check,
    get_env_specs,
    get_lift_height,
    announce_picked,
    announce_picking,
)

from kcare_robot.skills.approach import approach_close, __TURN_ANGLE, placeat, placep  # noqa: F401
from kcare_robot.skills.arm import movet, movej, arm_exception_handler, movel, movelf
from kcare_robot.skills.lift import lift, dlift
from kcare_robot.skills.grip import grip
from kcare_robot.skills.head import get_robot_mode
from kcare_robot.skills.mobile import move, forward
from kcare_robot.skills.recognition import find_arm, grasp_succeed

from kcare_robot.skills import _pick_helpers as _h
# Re-export for parity with the original module's public surface.
from kcare_robot.skills._pick_helpers import is_inside_workspace, fix_angle


# Module-level constants (kept name-mangled to match original imports).
__DPULL = _h.DPULL
__DEFAULT_HANDLE = _h.DEFAULT_HANDLE
__DRAWER_HEIGHT = _h.DRAWER_HEIGHT
__DRAWER_RANGE = _h.DRAWER_RANGE


@arm_exception_handler
def fine_move(**kwargs):
    """Grasp using the wrist-camera grasppose. Up to 2 trials: open gripper,
    detect, pre-open to detected width, align, descend, close, pull back. Each
    trial re-detects so a mis-grasp can self-correct."""
    node = kwargs.pop('node', None)
    obj_name = kwargs.pop('inputs')
    dpull = kwargs.pop('dpull', None)

    for _ in range(2):
        ret = grip(node=node, inputs='open', wait=False)
        if not ret['isdone']:
            return ret

        ret = find_arm(node=node, inputs=obj_name, **kwargs)
        if not ret['isdone']:
            continue

        grasp = _h.grasp_pose_from_ins(ret['ins'][obj_name], dpull)
        if grasp is None:
            continue
        dx, dy, dz, angle, width, eff_dpull = grasp

        ret = _h.execute_fine_grasp(node, dx, dy, dz, angle, width, eff_dpull)
        if not ret['isdone']:
            return ret

        if grasp_succeed(node=node)['isdone']:
            return {'isdone': True}

    return {'isdone': False, 'dd': [dx, dy, dz]}


@arm_exception_handler
def open_drawer(**kwargs):
    """Open the drawer at `inputs` (or the configured default location).

    Steps: move to the location, lift + go-to 'give' in parallel, dive the arm
    down with the wrist rotated, locate the handle, drive forward to centre it,
    grasp+pull, release, back out, retract.
    """
    node = kwargs.pop('node', None)
    inp = kwargs.pop('inputs', None)
    handle_name = __DEFAULT_HANDLE
    height = __DRAWER_HEIGHT
    lift_height = node.agents['lm_state'].get()['current_position']

    if inp is not None:
        env = get_env_specs(inp, ENV)
        handle_name = env.get('handle', None)
        if handle_name is None:
            print(f'No handle at {inp}')
            return {'isdone': False}
        ret = move(node=node, inputs=inp, wait=True)
        if not ret['isdone']:
            return ret
        height = env.get('height', __DRAWER_HEIGHT)

    robot_mode = get_robot_mode(node=node)

    ret = run_parallel_check(funcs=[
        lambda: lift(node=node, inputs=height, wait=True),
        lambda: movej(node=node, inputs='give', wait=True),
    ])
    if not ret['isdone']:
        return ret

    ret = movel(node=node, dz=-0.2, rx=-90, wait=True)
    if not ret['isdone']:
        return ret

    err, mforward = _h.compute_drawer_mforward(node, handle_name, robot_mode)
    if err is not None:
        return err
    ret = _h.drive_forward_if_needed(node, mforward)
    if not ret['isdone']:
        return ret

    ret = movel(node=node, dz=-0.1, wait=True)
    if not ret['isdone']:
        return ret

    ret = fine_move(node=node, inputs=handle_name, keep_orientation=True, dpull=__DPULL, **kwargs)
    if not ret['isdone']:
        return ret

    ret = grip(node=node, inputs='open', wait=True)
    if not ret['isdone']:
        return ret

    ret = movet(node=node, dz=-0.1)
    if not ret['isdone']:
        return ret

    ret = movel(node=node, dx=0.1)
    if not ret['isdone']:
        return ret

    ret = movel(node=node, dz=0.2)
    if not ret['isdone']:
        return ret

    # NOTE: matches original — `movej('fold')` here has no `mode` argument,
    # unlike close_drawer below.
    return _h.retract_from_drawer(node, lift_height, mforward, robot_mode=None)


@arm_exception_handler
def close_drawer(**kwargs):
    """Close the drawer at `inputs`. Mirrors `open_drawer` but pushes instead
    of pulling."""
    node = kwargs.pop('node', None)
    inp = kwargs.pop('inputs', None)
    handle_name = __DEFAULT_HANDLE
    target_height = __DRAWER_HEIGHT
    lift_height = node.agents['lm_state'].get()['current_position']
    env = {}

    if inp is not None:
        env = get_env_specs(inp, ENV)
        handle_name = env.get('handle', None)
        if handle_name is None:
            print(f'No handle at {inp}')
            return {'isdone': False}
        ret = move(node=node, inputs=inp, wait=True)
        if not ret['isdone']:
            return ret
        target_height = env.get('height', __DRAWER_HEIGHT)

    robot_mode = get_robot_mode(node=node, env=env)

    ret = movej(node=node, inputs='give', mode=robot_mode, wait=True)
    if not ret['isdone']:
        return ret

    ret = movel(node=node, dy=0.3 if robot_mode == 'left' else -0.3, wait=True)
    if not ret['isdone']:
        return ret

    ret = run_parallel_check(funcs=[
        lambda: lift(node=node, inputs=target_height, wait=True),
        lambda: movelf(node=node, dz=-0.25, rx=-90, wait=True),
    ])
    if not ret['isdone']:
        return ret

    err, mforward = _h.compute_drawer_mforward(node, handle_name, robot_mode)
    if err is not None:
        return err
    ret = _h.drive_forward_if_needed(node, mforward)
    if not ret['isdone']:
        return ret

    ret = movelf(node=node, dz=-0.1, wait=True)
    if not ret['isdone']:
        return ret

    ret = find_arm(node=node, inputs=handle_name, keep_orientation=True, **kwargs)
    if not ret['isdone']:
        return ret

    x, y, z = ret['ins'][handle_name]['grasppose'][:3]

    dpush = kwargs.pop('dpush', __DPULL - 30)
    ret = movet(node=node, dx=x, dy=y)
    if not ret['isdone']:
        return ret

    ret = movet(node=node, dz=z + dpush, wait=True)
    if not ret['isdone']:
        return ret

    ret = movet(node=node, dz=-0.1, wait=False)
    if not ret['isdone']:
        return ret

    ret = movel(node=node, dz=0.1)
    if not ret['isdone']:
        return ret

    return _h.retract_from_drawer(node, lift_height, mforward, robot_mode=robot_mode)


@arm_exception_handler
def pick_no_sound(**kwargs):
    """Full pick flow (no TTS). Steps:

      1. Parse `caption@loc`; if `loc` is set, open that drawer first.
      2. Open gripper.
      3. If drawer was opened, prep arm for the 'approach_lying' pose;
         otherwise run `approach_close` with `init_pose_fixed`.
      4. Execute the grasp (`direct` straight dive, or `fine_move` re-detect).
      5. Retract: lift to carry height, go to 'give', fold, drive base back.
      6. If drawer was opened, place-back, close drawer, then re-pick.
      7. Return success based on `grasp_succeed`.
    """
    kwargs['action_type'] = 'pick'
    node = kwargs.pop('node', None)
    move_type = kwargs.pop('type', 'fine_move')  # 'direct' or 'fine_move'
    loc_name = kwargs.pop('inputs', None)
    caption, loc = _h.split_loc(loc_name)

    env = get_env_specs(loc, ENV)
    robot_mode = get_robot_mode(node=node, env=env)
    lift_height = get_lift_height(env, robot_mode)

    # Step 1: open drawer if a location is supplied.
    drawer_opened = False
    if loc is not None:
        ret = open_drawer(node=node, inputs=loc, **kwargs)
        kwargs['inputs'] = caption if ret['isdone'] else f'{caption}@{loc}'
        drawer_opened = ret['isdone']
    else:
        kwargs['inputs'] = caption

    # Step 2: open gripper.
    ret = grip(node=node, inputs='open', wait=False)
    if not ret['isdone']:
        return ret

    # Step 3: get arm into pre-grasp pose.
    if drawer_opened:
        ret = _h.post_drawer_open_prep(node, robot_mode)
        if not ret['isdone']:
            return ret
    else:
        init_pose_fixed = kwargs.pop('init_pose_fixed', True)
        kwargs.update(approach_close(**kwargs, node=node, init_pose_fixed=init_pose_fixed))
        if not kwargs['isdone']:
            movej(node=node, mode=robot_mode, inputs='fold')
            return kwargs

    # Step 4: execute the grasp.
    dz = ARM_CONFIGS['approach_d']
    if move_type == 'direct':
        ret = _h.pickup_motion_direct(node, dz)
        if not ret['isdone']:
            return ret
    elif move_type == 'fine_move':
        time.sleep(0.5)
        obj_name = loc_name.split('@')[0]
        ret = fine_move(node=node, inputs=obj_name, mode=robot_mode,
                        wrist_angle=kwargs['wrist_angle'])
        if not ret['isdone']:
            return ret
    else:
        raise Exception(f'move_type [{move_type}] not implemented ...')

    # Step 5: retract.
    mforward = kwargs.pop('mforward', 0)
    ret = _h.post_pick_retract(node, robot_mode, lift_height, mforward)
    if not ret['isdone']:
        return ret

    # Step 6: drawer-pick recovery — place inside, close drawer, re-pick.
    if drawer_opened:
        ret = placep(node=node, inputs=loc)
        if not ret['isdone']:
            return ret
        if loc is not None:
            ret = close_drawer(node=node, inputs=loc)
            if not ret['isdone']:
                return ret
        ret = pick(node=node, inputs=caption)
        if not ret['isdone']:
            return ret

    # Step 7: report success based on actual grasp state.
    kwargs['isdone'] = grasp_succeed(node=node)['isdone']
    kwargs.pop('wrist_angle', None)
    return kwargs


@arm_exception_handler
def pick_card(**kwargs):
    """Domain-specific: approach a card box's blue surface, then fine-grasp
    the white handle. Used by the robot-world card demo."""
    node = kwargs.pop('node')
    loc = kwargs.pop('inputs', None)
    loc = 'box@source' if loc is None else loc

    kwargs['inputs'] = f'blue surface@{loc}'
    kwargs['wrist_angle'] = 0
    ret = approach_close(node=node, **kwargs)
    if not ret['isdone']:
        return ret

    ret = movel(node=node, dz=-50, wait=True)
    if not ret['isdone']:
        return ret

    kwargs['inputs'] = 'white handle'
    kwargs['deeper'] = True
    ret = fine_move(node=node, **kwargs)
    if not ret['isdone']:
        return ret

    ret = movel(node=node, dz=0.1, wait=True)
    if not ret['isdone']:
        return ret

    return movej(node=node, inputs='fold', wait=True)


@arm_exception_handler
def stack(**kwargs):
    """Domain-specific stacking demo with hardcoded waypoints."""
    node = kwargs.pop('node')
    _inp = kwargs.get('inputs')  # currently unused; matches original signature

    ret = lift(node=node, inputs=0.3)
    if not ret['isdone']:
        return ret
    ret = movej(node=node, inputs='give', wait=True)
    if not ret['isdone']:
        return ret

    dx, dy, dz = 0.086, 0.1, 0.223
    ret = movet(node=node, dx=dx, dy=dy, wait=True)
    if not ret['isdone']:
        return ret
    ret = movet(node=node, dz=dz, wait=True)
    if not ret['isdone']:
        return ret
    ret = grip(node=node, inputs='close', wait=True)
    if not ret['isdone']:
        return ret
    ret = movej(node=node, inputs='give', wait=True)
    if not ret['isdone']:
        return ret

    dx, dy, dz = -75, 85, 317
    ret = movet(node=node, dx=dx, dy=dy - 150, dz=dz, wait=True)
    if not ret['isdone']:
        return ret
    ret = movel(node=node, dz=-20, wait=True)
    if not ret['isdone']:
        return ret
    ret = grip(node=node, inputs='open', wait=True)
    if not ret['isdone']:
        return ret
    ret = movej(node=node, inputs='give', wait=True)
    if not ret['isdone']:
        return ret

    dx, dy, dz = -75, 85, 357
    ret = movet(node=node, dx=dx, dy=dy, wait=True)
    if not ret['isdone']:
        return ret
    ret = movet(node=node, dz=dz, wait=True)
    if not ret['isdone']:
        return ret
    ret = grip(node=node, inputs='close', wait=True)
    if not ret['isdone']:
        return ret
    ret = movej(node=node, inputs='give', wait=True)
    if not ret['isdone']:
        return ret

    return movej(node=node, inputs='fold', wait=True)


@arm_exception_handler
def pick(**kwargs):
    """`pick_no_sound` wrapped with picking/picked voice announcements."""
    loc_name = kwargs.get('inputs', None)
    caption, _loc = _h.split_loc(loc_name)

    ret = run_parallel_check(funcs=[
        lambda: announce_picking(caption),
        lambda: {'isdone': True} if NO_ACTION else pick_no_sound(**kwargs),
    ])
    ret = ret['rets'][-1]
    if ret['isdone']:
        announce_picked()
    return ret
