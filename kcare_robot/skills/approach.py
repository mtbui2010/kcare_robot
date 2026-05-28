"""Approach + place skills. Public API:

  - `approach_close`  : move arm to the approach pose of a target.
  - `approach_closef` : `approach_close` with `init_pose_fixed=True`.
  - `approach`        : `approach_close` followed by a short forward push.
  - `placeat_no_sound`: approach + release/wipe + retract (no TTS).
  - `placeat`         : `placeat_no_sound` with placing/placed announcements.
  - `placep`          : `placeat` without driving the base.
  - `__TURN_ANGLE`    : module-level constant re-exported for callers.

Heavy lifting is split into single-purpose helpers in `_approach_helpers.py`.
This file is the orchestration only — the flow matches the original module 1:1.
"""

from robot_agent.skill_configs import ARM_CONFIGS, ENV, NO_ACTION
from robot_agent.utils import (
    get_env_specs,
    get_lift_height,
    run_parallel_check,
    announce_placing,
    announce_placed,
)

from kcare_robot.skills.arm import movel, movej, movet, arm_exception_handler
from kcare_robot.skills.lift import lift, dlift, lift_state  # noqa: F401  (kept for module surface)
from kcare_robot.skills.grip import grip  # noqa: F401
from kcare_robot.skills.mobile import forward  # noqa: F401
from kcare_robot.skills.head import get_robot_mode

from kcare_robot.skills import _approach_helpers as _h
# Re-export helpers that callers (or this module's own docstrings) reference.
from kcare_robot.skills._approach_helpers import (
    lift_dz,
    is_inside_workspace,
    get_dmove_approach,
    get_placepose,
)

__TURN_ANGLE = 0


def _run_init_pose_fixed(node, agents, robot_mode, x, y, z, obj_name, kwargs):
    """Drive the init_pose_fixed branch end-to-end and write results back into
    `kwargs`. Returns either an error dict or the mutated `kwargs`."""
    err, lift_to_limit = _h.init_pose_lift_and_unfold(node, robot_mode, z)
    if err is not None:
        return err

    err, islying, left_mass_percent, wrist_angle = _h.init_pose_detect_posture(
        node, obj_name, lift_to_limit, **kwargs
    )
    if err is not None:
        return err

    if islying:
        ret = _h.init_pose_handle_lying(node, robot_mode, y, agents)
        if not ret['isdone']:
            return ret
    else:
        ret = _h.init_pose_handle_standing(
            node, robot_mode, x, wrist_angle, lift_to_limit, left_mass_percent, islying,
        )
        if not ret['isdone']:
            return ret

    kwargs['wrist_angle'] = wrist_angle
    kwargs['isdone'] = True
    kwargs['islying']  =islying
    return kwargs


@arm_exception_handler
def approach_close(**kwargs):
    """Move the robot arm to the approach pose of `inputs`.

    `inputs` may be:
      - an explicit pose `[x, y, z]` or `[x, y, z, wrist_angle]`,
      - an ENV location name (drives there + uses configured placepose),
      - an object spec (`'obj@loc'` / `'obj'`) — calls find() to locate it.

    Returns the mutated `kwargs` with `isdone=True`, `wrist_angle`, `mforward`,
    and (on normal branch) `shift_values` + `approach_data`. On failure returns
    the failing sub-step's result dict.
    """
    node = kwargs.pop('node', None)
    action_type = kwargs.pop('action_type', 'pick')
    agents = node.agents
    # Pop 'ins' from kwargs (matches original ordering — resolve_pose_3d reads
    # via kwargs.get(), which then yields {} for the object-finding branch).
    kwargs.pop('ins', {})
    # `stay_here` is the carerobotapp name; treat it as a synonym for our
    # existing `stay_here` kwarg. Either suppresses the drive-to-location step.
    stay_here = kwargs.pop('stay_here', False)
    inputs = kwargs.pop('inputs', None)

    env = get_env_specs(inputs, ENV)
    robot_mode = get_robot_mode(node=node, env=env)

    # Step 1a: init_pose_fixed if object_from_drawer
    if kwargs.pop("object_from_drawer", False):
        return _h.run_init_object_from_drawer(node=node, robot_mode=robot_mode, **kwargs)
    

    # Step 1b: resolve target pose.
    err, pose_3d, env, obj_name, ins_obj, islying = _h.resolve_pose_3d(node, inputs, env, stay_here, kwargs)
    if err is not None:
        return err
    x0, y, z = pose_3d[:3]
    kwargs['islying'] =  islying

    # Step 2: early wrist angle (init_pose branch may overwrite).
    wrist_angle = _h.resolve_wrist_angle_early(kwargs.pop('wrist_angle', None), pose_3d)

    if not is_inside_workspace(x0, y, z):
        raise Exception(f'Out of workspace: x,y,z = {(x0, y, z)}')

    # Step 3: pre-approach — clip x and drive base forward + fold arm.
    x, mforward = _h.compute_forward_distance(x0, stay_here)
    kwargs['mforward'] = mforward
    ret = _h.move_forward_and_fold(node, mforward, robot_mode)
    if not ret['isdone']:
        return ret

    # Step 4a: init_pose_fixed branch returns directly.
    if kwargs.pop('init_pose_fixed', False):
        return _run_init_pose_fixed(node, agents, robot_mode, x, y, z, obj_name, kwargs)

    # Step 4b: normal branch.
    print(f'Approaching to : {x, y, z}')

    wrist_angle = _h.resolve_wrist_angle_for_motion(node, action_type, wrist_angle, ins_obj, z)
    kwargs['wrist_angle'] = wrist_angle

    shift_values = _h.compute_shift_values(
        node, wrist_angle, robot_mode,
        dlift_up=kwargs.pop('dlift_up', [0, 0, 0]),
        islying=kwargs.get('islying', False)
    )
    lm_state = lift_state(node=node)['current_position']
    dmove, target_xyz, lift_to = _h.compute_target_and_lift(
        x, y, z, shift_values, lm_state
    )

    ret = _h.execute_approach_motion(node, robot_mode, lift_to, wrist_angle, target_xyz, kwargs.get('islying', False))
    if not ret['isdone']:
        return ret

    kwargs['isdone'] = True
    return kwargs


@arm_exception_handler
def approach_closef(**kwargs):
    """`approach_close` with `init_pose_fixed=True` (find-and-grasp from stowed)."""
    kwargs['init_pose_fixed'] = True
    return approach_close(**kwargs)


@arm_exception_handler
def approach(**kwargs):
    """`approach_close` followed by a short forward push (`approach_d`)."""
    node = kwargs.pop('node', None)
    kwargs.update(approach_close(**kwargs, node=node))
    if not kwargs['isdone']:
        return kwargs

    ret = movet(node=node, dz=ARM_CONFIGS['approach_d'], speed=0.5)
    if not ret['isdone']:
        return ret

    kwargs['isdone'] = True
    return kwargs


@arm_exception_handler
def placeat_no_sound(**kwargs):
    """Approach the target, release (or wipe), then retract — without TTS."""
    node = kwargs.pop('node')
    to_wipe = kwargs.pop('to_wipe', False)

    inp = kwargs.get('inputs')
    env = get_env_specs(inp, ENV)
    robot_mode = get_robot_mode(node=node, env=env)
    lift_height = get_lift_height(env, robot_mode)

    # Reserve a vertical buffer above the placement so the gripper has room to
    # open without dragging on the object below it.
    dlift_up = max(ARM_CONFIGS['place_liftup'], kwargs.pop('holding_obj_height', 0) / 2)
    kwargs['dlift_up'] = [0, 0, dlift_up]
    kwargs.update(approach(**kwargs, node=node))
    if not kwargs['isdone']:
        return kwargs

    if not to_wipe:
        ret = _h.release_object(node, dlift_up)
    else:
        ret = _h.execute_wipe(node, dlift_up)
    if not ret['isdone']:
        return ret

    return _h.retract_after_place(
        node, robot_mode,
        mforward=kwargs.pop('mforward', 0),
        lift_height=lift_height,
    )


@arm_exception_handler
def placeat(**kwargs):
    """`placeat_no_sound` wrapped with placing/placed voice announcements."""
    to_wipe = kwargs.get('to_wipe', False)
    inp = kwargs.get('inputs')

    announce_placing(inp, to_wipe=to_wipe)
    if not NO_ACTION:
        ret = placeat_no_sound(**kwargs)
        announce_placed(inp, to_wipe=to_wipe)
        return ret
    # ret = run_parallel_check(funcs=[
    #     lambda: announce_placing(inp, to_wipe=to_wipe),
    #     lambda: {'isdone': True} if NO_ACTION else placeat_no_sound(**kwargs),
    # ])
    # ret = ret['rets'][-1]
    # if ret['isdone']:
    #     announce_placed(inp, to_wipe=to_wipe)
    return {'isdone': True}


@arm_exception_handler
def placep(**kwargs):
    """`placeat` without driving the mobile base."""
    kwargs['stay_here'] = True
    return placeat(**kwargs)
