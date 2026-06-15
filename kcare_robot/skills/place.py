"""Place / wipe / card skills. Public API:

  - `place`        : pick (if needed) then `placeat` at the reverse location.
  - `collect_card` : pick_card then deposit at destination box.
  - `return_card`  : ensure holding then deposit back at source box.
  - `wipe`         : `place` with `to_wipe=True` and wrist tilted for wiping.

Heavy lifting is split into helpers in `_place_helpers.py`. This file is the
orchestration only — flow matches the original module 1:1.
"""

import time

from robot_agent.skill_configs import ENV, ME
from robot_agent.utils import get_closest_loc, run_parallel_check, get_env_specs
from kcare_robot.skills.head import get_robot_mode

from kcare_robot.skills.calibrattion import Head2BaseCalibration
from kcare_robot.skills.approach import approach, __TURN_ANGLE, placeat  # noqa: F401
from kcare_robot.skills.arm import movet, movel, movej, arm_exception_handler
from kcare_robot.skills.grip import grip
from kcare_robot.skills.mobile import move
from kcare_robot.skills.pick import pick, pick_card
from kcare_robot.skills.recognition import grasp_succeed, find_place
from kcare_robot.skills.mobile import forward
from kcare_robot.skills.lift import lift


from kcare_robot.skills import _place_helpers as _h
from kcare_robot.skills._place_helpers import loc2text  # noqa: F401
from robot_agent.skills import log_data


# Calibration must exist at import time — preserved from the original module.
calib = Head2BaseCalibration()


def approach_place(node, **kwargs) -> dict:
    inp = kwargs.pop('inputs')
    obj_name = inp.split('@')[0]

    kwargs.update(find_place(node=node, inputs=inp, **kwargs))
    assert kwargs['isdone'], f'{kwargs}'

    ret = run_parallel_check(funcs=[
        lambda: forward(node=node, inputs=kwargs['mforward'], wait=True),
        lambda: movej(node=node, dr0=kwargs['base_rotate'], wait=True),
        lambda: lift(node=node, inputs=kwargs['lift_to'], wait=True)
    ])
    assert ret['isdone'], f'{ret}'

    if kwargs['islying']:
        ret = movej(node=node, inputs='approach_lying', wait=True)
        assert ret['isdone'], f'{ret}'

    ret = movel(node=node, **kwargs['approach_pose'])
    assert ret['isdone'], f'{ret}'


    kwargs['isdone'] = True
    return kwargs

@arm_exception_handler
def place(node, **kwargs):
    """Place the held object at `inputs` (format: `'[target>>]rev_loc'`).

    Behavior:
      - `inputs=None` → place at the closest known location with `force=True`.
      - If nothing is currently grasped: pick `target_loc` first (if given),
        otherwise raise unless `force=True` was set.
      - `rev_loc=='me'` is rewritten to the `ME` constant.
      - `to_wipe` is auto-set when `'spill'` appears in `rev_loc`.
    """
    for k in ['object_from_drawer', 'pose_after_open', 'lift_after_open', 'forward_after_open', 'init_pose_fixed']:
        kwargs.pop(k, None)
    inputs = kwargs.pop('inputs', None)

    splits = inputs.split('>>')
    target, destination = splits  if len(splits)==2 else (None, splits[-1])
    if target is not None:
        kwargs.update(pick(node=node, inputs=target,  **kwargs))
        assert kwargs['isdone'], f'{kwargs}'


    env = get_env_specs(destination, ENV)
    if len(env)>0:
        move(node=node, inputs=destination, **kwargs)
        assert kwargs['isdone'], f'{kwargs}'
    
    kwargs.update(approach_place(node=node, inputs=destination, **kwargs))
    assert kwargs['isdone'], f'{kwargs}'

    if kwargs['islying']:
        ret = movet(node=node, dz=kwargs['dapproach']+kwargs['dz_up']/2, wait=True)
    else:
        ret = movet(node=node, dz=kwargs['dapproach'], wait=True)
        assert ret['isdone'], f'{ret}'

    ret = run_parallel_check(funcs=[
        lambda: movet(node=node, dz=kwargs['dz_up']/2) if kwargs['islying'] else movel(node=node, dz=-kwargs['dz_up']),
        lambda: grip(node=node, inputs='open', wait=True)
    ])
    assert ret['isdone'], f'{ret}'

    ret = movel(node=node, dz=0.15)
    assert  ret['isdone'], f'{ret}'

    if kwargs['islying']:
        ret = movej(node=node, inputs='approach_lying')
        assert  ret['isdone'], f'{ret}'

    ret = movej(node=node, inputs='fold')
    assert  ret['isdone'], f'{ret}'

    ret = forward(node=node, inputs=-kwargs.get('mforward', 0 )) 
    assert  ret['isdone'], f'{ret}'


    return kwargs
    


@arm_exception_handler
def collect_card(**kwargs):
    """Pick a card (if not already held) then deposit at the destination box.
    Includes a 180° wrist twist + counter-twist before retract to register the
    card in the slot."""
    node = kwargs.pop('node')
    inp = kwargs.pop('inputs', None)
    inp = 'box@destination' if inp is None else inp
    target_loc, rev_loc = _h.parse_card_loc(inp)

    if not grasp_succeed(node=node)['isdone']:
        ret = pick_card(node=node, inputs=target_loc, dpull=50)
        if not ret['isdone']:
            return ret

    ret = approach(node=node, inputs=rev_loc, **kwargs)
    if not ret['isdone']:
        return ret

    ret = movet(node=node, drz=180, wait=True)
    if not ret['isdone']:
        return ret

    time.sleep(1)

    ret = movet(node=node, drz=-180, wait=True)
    if not ret['isdone']:
        return ret

    robot_mode = get_robot_mode(node=node)
    turn_angle = _h.turn_signed(kwargs.pop('turn_angle', __TURN_ANGLE), robot_mode)

    ret = _h.turn_base(node, turn_angle)
    if not ret['isdone']:
        return ret

    ret = movej(node=node, inputs='fold', dr0=0 if turn_angle is None else turn_angle, wait=True)
    if not ret['isdone']:
        return ret

    if turn_angle is not None:
        ret = movej(node=node, dr0=-turn_angle, wait=True)
        if not ret['isdone']:
            return ret
    return {'isdone': True}


@arm_exception_handler
def return_card(**kwargs):
    """Return the held card to the source box. If nothing is held, collects
    one first via `collect_card`."""
    node = kwargs.pop('node')
    inp = kwargs.pop('inputs', None)
    inp = 'box@source' if inp is None else inp

    if not grasp_succeed(node=node)['isdone']:
        ret = collect_card(node=node)
        if not ret['isdone']:
            return ret

    ret = approach(node=node, inputs=inp, **kwargs)
    if not ret['isdone']:
        return ret

    ret = movel(node=node, dz=-40, wait=False)
    if not ret['isdone']:
        return ret

    ret = grip(node=node, inputs='open', wait=True)
    if not ret['isdone']:
        return ret

    ret = movel(node=node, dz=-30, speed=2., wait=False)
    if not ret['isdone']:
        return ret

    ret = movet(node=node, dz=-150)
    if not ret['isdone']:
        return ret

    return movej(node=node, inputs='fold', wait=True)


@arm_exception_handler
def wipe(**kwargs):
    """`place` with wiping enabled. Defaults `inputs='towel>>spill'`."""
    inp = kwargs.pop('inputs', 'towel>>spill')
    splits = inp.split('>>')
    target_loc, rev_loc = splits if len(splits) == 2 else ('towel', splits[-1])

    kwargs['inputs'] = f'{target_loc}>>{rev_loc}'
    kwargs['to_wipe'] = True
    kwargs['wrist_angle'] = 30
    return place(**kwargs)



