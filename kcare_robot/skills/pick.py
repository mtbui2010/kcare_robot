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
import math

import numpy as np

from robot_agent.skill_configs import ENV, ARM_CONFIGS, NO_ACTION
from robot_agent.utils import (
    run_parallel_check,
    get_env_specs,
    get_lift_height,
    announce_picked,
    announce_picking,
)

from kcare_robot.skills.approach import approach_close, __TURN_ANGLE, placeat, placep  # noqa: F401
from kcare_robot.skills.arm import movet, movej, arm_exception_handler, movel, movelf, arm_joints
from kcare_robot.skills.lift import lift, dlift, lift_state
from kcare_robot.skills.grip import grip
from kcare_robot.skills.head import get_robot_mode
from kcare_robot.skills.mobile import move, forward
from kcare_robot.skills.recognition import find_grasp, grasp_succeed, find

from kcare_robot.skills import _pick_helpers as _h
# Re-export for parity with the original module's public surface.
from kcare_robot.skills._pick_helpers import is_inside_workspace, fix_angle
from robot_agent.skills import log_data


# Module-level constants (kept name-mangled to match original imports).
__DPULL = _h.DPULL
__DEFAULT_HANDLE = _h.DEFAULT_HANDLE
__DRAWER_HEIGHT = _h.DRAWER_HEIGHT
__DRAWER_RANGE = _h.DRAWER_RANGE


@arm_exception_handler
def fine_move(**kwargs):
    """Grasp using the wrist-camera grasppose.

    Flow:
      1. Coarse detect (estimate_grasp=False) to read pose_3d[0]; pre-move the
         wrist `dx` so the second detect sees the object closer to the centre
         of the wrist FOV (mirrors carerobotapp behaviour). Skipped silently
         if the coarse step fails.
      2. Up to `num_trials` attempts: open gripper, re-detect with grasp pose,
         align, descend, close, pull back. Each trial re-detects so a mis-grasp
         can self-correct.

    Kwargs:
        num_trials  (int):  number of grasp attempts. Default 2.
        pull_speed  (float): retraction speed (m/s-ish; passed to movet).
                              Default 1.0.
        dpull       (float | None): explicit pull distance (m). When None,
                                     uses min(dz, 0.3).
        Remaining kwargs forwarded to find_grasp.
    """
    node       = kwargs.pop('node', None)
    obj_name   = kwargs.pop('inputs')
    dpull      = kwargs.pop('dpull', None)
    num_trials = int(kwargs.pop('num_trials', 2))
    pull_speed = float(kwargs.pop('pull_speed', 1.0))

    # Step 1: coarse detect + dx nudge. Failure here is non-fatal — we just
    # fall through to the per-trial loop without nudging.
    # try:
    #     coarse = find(node=node, inputs=obj_name, camera='arm', target_distance=0.3)
    #     if coarse.get('isdone'):
    #         pose = coarse['ins'][obj_name].get('pose_3d') \
    #             or coarse['ins'][obj_name].get('loc_3d')
    #         if pose is not None:
    #             # pose components are in mm (from Ixy2xyz / camera frame);
    #             # movet expects metres → divide by 1000. ±0.2 m matches the
    #             # ±200 mm carerobotapp clamp.
    #             dx_pre = float(np.clip(pose[0], -0.2, 0.2))
    #             if abs(dx_pre) > 1e-4:
    #                 ret = movet(node=node, dx=dx_pre, wait=True)
    #             if _h.is_vertical_gripper(node=node):
    #                 dz_pre = float(np.clip(0.3 - pose[-1], -0.2, 0.))
    #                 if abs(dz_pre) > 1e-4:
    #                     dlift(node=node, inputs=dz_pre, wait=True)

    # except Exception:
    #     pass

    dx = dy = dz = None
    for _ in range(num_trials):
        ret = grip(node=node, inputs='open', wait=False)
        if not ret['isdone']:
            return ret

        ret = find_grasp(node=node, inputs=obj_name, **kwargs)
        if not ret['isdone']:
            continue
        ins = ret['ins'][obj_name]

        grasp = _h.grasp_pose_from_ins(ins, dpull)
        if grasp is None:
            continue
        dx, dy, dz, angle, width, eff_dpull = grasp

        ret = _h.execute_fine_grasp(node, dx, dy, dz, angle, width, eff_dpull,
                                    pull_speed=pull_speed)
        if not ret['isdone']:
            return ret

        if grasp_succeed(node=node)['isdone']:
            # Report the grasp actually executed so the world state can record
            # how/where the held object was picked (consumed by the world's
            # `apply_skill_effect` → `holding_pose`).
            return {'isdone': True, 'grasppose': [dx, dy, dz, angle, width]}

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
    lift_height = lift_state(node=node)['current_position']

    stay_here = kwargs.pop('stay_here', False)

    if inp is not None:
        env = get_env_specs(inp, ENV)
        handle_name = env.get('handle', None)
        if handle_name is None:
            print(f'No handle at {inp}')
            return {'isdone': False}
        if not stay_here:
            ret = move(node=node, inputs=inp, wait=True)
            if not ret['isdone']:
                return ret
        lift_height = env.get('handle_height', lift_height)
        height = env.get('handle_height', __DRAWER_HEIGHT)

    robot_mode = get_robot_mode(node=node)

    ret = run_parallel_check(funcs=[
        lambda: lift(node=node, inputs=height, wait=True),
        lambda: movej(node=node, inputs='approach_drawer', wait=True),
    ])
    if not ret['isdone']:
        return ret
    


    # ret = movel(node=node, ry=-90, wait=True)
    # if not ret['isdone']:
    #     return ret
    
    # ret = movet(node=node, dz=-0.1)
    # assert ret['isdone'], f'{ret}'
    
    # ret = movel(node=node, z=height-abs(ARM_CONFIGS['wrist_cam_offset'][1]) + 0.05)
    # assert ret['isdone'], f'{ret}'
    

    err, mforward = _h.compute_drawer_mforward(node, handle_name, robot_mode)
    if err is not None:
        return err
    ret = _h.drive_forward_if_needed(node, mforward)
    if not ret['isdone']:
        return ret

    # ret = movel(node=node, dz=-0.1, wait=True)
    # if not ret['isdone']:
    #     return ret

    # Mirror carerobotapp: pull slowly and bias the grasp depth shallow so the
    # handle isn't crushed. `deep_ratio=0.4` favours obj_median over bound.
    ret = fine_move(node=node, inputs=handle_name, keep_orientation=True,
                    dpull=__DPULL, deep_ratio=0.4, pull_speed=0.5, target_distance=0.4, **kwargs)
    if not ret['isdone']:
        return ret

    # Capture state right after the pull so callers (e.g. drawer-aware
    # planners) can later restore the arm/base/lift configuration.
    
    lift_after_open    = lift_state(node=node).get('current_position')
    forward_after_open = mforward

    ret = grip(node=node, inputs='open', wait=True)
    if not ret['isdone']:
        return ret

    ret = movet(node=node, dz=-0.05)
    if not ret['isdone']:
        return ret
    
    try:
        # pose_after_open = float(node.agents['robot_pose'].get()['pose'][3] * 180 / math.pi)
        pose_after_open = arm_joints(node=node)['joints']
    except Exception:
        pose_after_open = None


    # ret = movel(node=node, dz=0.2)
    # if not ret['isdone']:
    #     return ret
    ret = dlift(node=node, inputs=0.2)
    assert ret['isdone'], f'{ret}'

    # NOTE: matches original — `movej('fold')` here has no `mode` argument,
    # unlike close_drawer below.
    ret = _h.retract_from_drawer(node, lift_height, mforward, robot_mode=None)
    if not ret.get('isdone'):
        return ret

    kwargs.update({
        'isdone':             True,
        'object_from_drawer': True,
        'pose_after_open':    pose_after_open,
        'lift_after_open':    lift_after_open,
        'forward_after_open': forward_after_open,
    })
    return kwargs


@arm_exception_handler
def close_drawer(**kwargs):
    """Close the drawer at `inputs`. Mirrors `open_drawer` but pushes instead
    of pulling."""
    node = kwargs.pop('node', None)
    inp = kwargs.pop('inputs', None)
    handle_name = __DEFAULT_HANDLE
    target_height = __DRAWER_HEIGHT
    lift_height = lift_state(node=node)['current_position']
    env = {}
    stay_here = kwargs.pop('stay_here', True)

    if inp is not None:
        env = get_env_specs(inp, ENV)
        handle_name = env.get('handle', None)
        if handle_name is None:
            print(f'No handle at {inp}')
            return {'isdone': False}
        if not stay_here:
            ret = move(node=node, inputs=inp, wait=True)
            if not ret['isdone']:
                return ret
        lift_height = env.get('height', lift_height)
        target_height = env.get('handle_height', __DRAWER_HEIGHT)
    pose_after_open = kwargs.pop('pose_after_open', None)
    lift_after_open = kwargs.pop('lift_after_open', None)
    forward_after_open = kwargs.pop('forward_after_open', None)
    
    robot_mode = get_robot_mode(node=node, env=env)

    # ret = movej(node=node, inputs='give', mode=robot_mode, wait=True)
    # if not ret['isdone']:
    #     return ret
    
    # ret = movel(node=node, ry = -90)
    # assert ret['isdone'], f'{ret}'

    # ret = movet(node=node, dz=-0.1)
    # assert ret['isdone'], f'{ret}'


    ret = lift(node=node, inputs=target_height if lift_after_open is None else lift_after_open, wait=True) 
    assert ret['isdone'], f'{ret}'
    

    
    if forward_after_open is None:
        ret = movej(node=node, inputs='approach_drawer', wait=True)
        assert ret['isdone'], f'{ret}'

        err, mforward = _h.compute_drawer_mforward(node, handle_name, robot_mode)
        if err is not None:
            return err
        ret = _h.drive_forward_if_needed(node, mforward)
        if not ret['isdone']:
            return ret
    else:
        ret = movej(node=node, inputs=pose_after_open)
        assert ret['isdone'], f'{ret}'

        ret = forward(node=node, inputs=-forward_after_open)
        assert ret['isdone'], f'{ret}'

    # brand 1: after open drawer

    if pose_after_open is None:

        ret = find(node=node, inputs=handle_name, camera='arm', **kwargs)
        if not ret['isdone']:
            return ret

        x, y, z = ret['ins'][handle_name]['pose_3d'][:3]
        ret = movet(node=node, dx=x, dy=y)
        if not ret['isdone']:
            return ret

        dpush = z + kwargs.pop('dpush', __DPULL - 0.03)

        # ret = movet(node=node, dz=z + dpush, wait=True)
        # if not ret['isdone']:
        #     return ret
    else:
        dpush = __DPULL+0.03
        # ret = movet(node=node, dz=__DPULL+0.03)
        # assert ret['isdone'], f'{ret}'
        

    ret = movet(node=node, dz=dpush)
    assert ret['isdone'], f'{ret}'

    ret = movet(node=node, dz=-0.2, wait=False)
    if not ret['isdone']:
        return ret

    # ret = movel(node=node, dz=0.1)
    # if not ret['isdone']:
    #     return ret

    ret = _h.retract_from_drawer(node, lift_height, mforward if forward_after_open is None else forward_after_open,  robot_mode=robot_mode)
    assert ret['isdone'], f'{ret}'

    kwargs['isdone']=True
    return kwargs


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

    # `caption|N` lets the caller override the per-grasp trial count, e.g.
    # `pick::apple|3`. Matches carerobotapp's syntax.
    splits = caption.split('|')
    if len(splits) >= 2:
        caption = splits[0]
        try:
            num_trials = int(splits[-1])
        except ValueError:
            num_trials = 2
    else:
        num_trials = kwargs.pop('num_trials', 2)

    env = get_env_specs(loc, ENV)
    robot_mode = get_robot_mode(node=node, env=env)
    lift_height = get_lift_height(env, robot_mode)

    # # Step 1: open drawer if a location is supplied.
    # drawer_opened = False
    # if loc is not None:
    #     ret = open_drawer(node=node, inputs=loc, **kwargs)
    #     kwargs['inputs'] = caption if ret['isdone'] else f'{caption}@{loc}'
    #     drawer_opened = ret['isdone']
    # else:
    #     kwargs['inputs'] = caption

    # Step 2: open gripper.
    ret = grip(node=node, inputs='open', wait=False)
    if not ret['isdone']:
        return ret

    # # Step 3: get arm into pre-grasp pose.
    # if drawer_opened:
    #     ret = _h.post_drawer_open_prep(node, robot_mode)
    #     if not ret['isdone']:
    #         return ret
    # else:
    init_pose_fixed = kwargs.pop('init_pose_fixed', True)
    # kwargs.update(approach_close(**kwargs, node=node, init_pose_fixed=init_pose_fixed))
    kwargs.update(approach_close(**kwargs, node=node, inputs=f'{caption}{"" if loc is None else f"@{loc}"}', init_pose_fixed=init_pose_fixed))
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
        ret = fine_move(node=node, inputs=caption, mode=robot_mode,
                        wrist_angle=kwargs.get('wrist_angle'),
                        num_trials=num_trials)
        if not ret['isdone']:
            return ret
    else:
        raise Exception(f'move_type [{move_type}] not implemented ...')

    # Step 5: retract.
    mforward = kwargs.pop('mforward', 0)
    ret = _h.post_pick_retract(node, robot_mode, lift_height, mforward)
    if not ret['isdone']:
        return ret

    # # Step 6: drawer-pick recovery — place inside, close drawer, re-pick.
    # if drawer_opened:
    #     ret = placep(node=node, inputs=loc)
    #     if not ret['isdone']:
    #         return ret
    #     if loc is not None:
    #         ret = close_drawer(node=node, inputs=loc)
    #         if not ret['isdone']:
    #             return ret
    #     ret = pick(node=node, inputs=caption)
    #     if not ret['isdone']:
    #         return ret

    # Step 7: report success based on actual grasp state.
    kwargs['isdone'] = grasp_succeed(node=node)['isdone']
    for k in ['wrist_angle', 'inputs']:
        kwargs.pop(k, None)
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
def approach_pick(node, **kwargs) -> dict:
    inp = kwargs.pop('inputs')
    obj_name = inp.split('@')[0]

    ret = find(node=node, inputs=inp, **kwargs)
    assert ret['isdone'], f'{ret}'

    kwargs.update(ret['ins'][obj_name])
    ret = run_parallel_check(funcs=[
        lambda: forward(node=node, inputs=kwargs['mforward'], wait=True),
        # lambda: movej(node=node, dr0=kwargs['base_rotate'], wait=True),
        lambda: lift(node=node, inputs=kwargs['lift_to'], wait=True)
    ])
    assert ret['isdone'], f'{ret}'

    if kwargs['islying']:
        ret = movej(node=node, inputs='approach_lying', wait=True)
        assert ret['isdone'], f'{ret}'

    ret = movel(node=node, **kwargs['approach_pose'])
    assert ret['isdone'], f'{ret}'

    return kwargs

@arm_exception_handler
def pick(node, **kwargs):
    """`pick_no_sound` wrapped with picking/picked voice announcements."""
    loc_name = kwargs.pop('inputs', None)
    splits = loc_name.split('|')
    loc_name, num_trials = splits if len(splits)==2 else (splits[0], 1)
    num_trials = int(num_trials)



    caption, _loc = _h.split_loc(loc_name)
    robot_mode = get_robot_mode(node=node)

    announce_picking(caption)
    if not NO_ACTION:

        if kwargs.pop("object_from_drawer", False):
            ret = movej(node=node, inputs="approach_lying")
            assert ret['isdone'], f'{ret}'
            
            ret = movet(node=node, dx=-0.3, dy=0.15) if robot_mode=='left' else movet(node=node, dx=0.3, dy=-0.15)
            assert ret['isdone'], f'{ret}'
            
            kwargs['islying'] = True
        else:
            kwargs.update(approach_pick(node=node, inputs=loc_name, **kwargs))
            assert  kwargs['isdone'], f'{kwargs}'

        kwargs['inputs'] = caption
        kwargs.update(fine_move(node=node, num_trials=num_trials, **kwargs))
        assert  kwargs['isdone'], f'{kwargs}'

        ret = movel(node=node, dz=0.15)
        assert  ret['isdone'], f'{ret}'

        if kwargs['islying']:
            ret = movej(node=node, inputs='approach_lying')
            assert  ret['isdone'], f'{ret}'

        ret = movej(node=node, inputs='fold')
        assert  ret['isdone'], f'{ret}'

        ret = forward(node=node, inputs=-kwargs.get('mforward', 0 )) 
        assert  ret['isdone'], f'{ret}'
        
        kwargs.pop('inputs', None)
        return kwargs

    return {'isdone': True}


@arm_exception_handler
def pushing(**kwargs):
    """Push (rather than pick) a target along the detected grasp axis.

    Flow:
      1. find_grasp to get a grasppose for `inputs` (object name).
      2. Close the gripper (no grasp; we use the closed fingers as a pusher).
      3. Translate to (dx, dy), rotate wrist to `angle`, descend `dz`.
      4. Open the gripper, retract `-dz`.

    Units: grasppose dx/dy/dz come from kcare_robot's recognition in metres
    (after the per-skill /1000 conversion in `estimate_grasp`), so they're
    passed straight to `movet`. `angle` is in degrees, width in mm.

    Not registered in skills_config — call as a Python function.
    """
    node = kwargs.pop('node', None)
    obj_name = kwargs.get('inputs')

    ret = find_grasp(node=node, **kwargs)
    if not ret['isdone']:
        return ret
    ins = ret['ins'][obj_name]

    dx, dy, dz, angle, width = ins['grasppose']
    angle = fix_angle(angle)
    dz = abs(dz)

    ret = grip(node=node, inputs='close', wait=False)
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

    ret = grip(node=node, inputs='open', wait=True)
    if not ret['isdone']:
        return ret

    return movet(node=node, dz=-dz)
