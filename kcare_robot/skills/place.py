"""Place / wipe skills. Public API:

  - `place`        : pick (if needed), then approach the destination and release.
  - `wipe`         : `place` with `to_wipe=True` and wrist tilted for wiping.

Heavy lifting is split into helpers in `_place_helpers.py`. This file is the
orchestration only — flow matches the original module 1:1.
"""

import math
import time

from robot_agent.skill_configs import ENV, ME
from robot_agent.utils import run_parallel_check
from kcare_robot.utils import get_closest_loc, get_env_specs
from kcare_robot.skills.head import get_robot_mode

from kcare_robot.skills.calibrattion import Head2BaseCalibration
from kcare_robot.skills.arm import movet, movel, movej, arm_exception_handler
from kcare_robot.skills.grip import grip
from kcare_robot.skills.mobile import move
from kcare_robot.skills.pick import pick
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
    # obj_name = inp.split('@')[0]

    env = get_env_specs(inp, ENV)
    if len(env)==0:
        splits = inp.split('@')
        dest_loc = None if len(splits)==1 else '@'.join(splits[1:])
        env = get_env_specs(dest_loc, ENV)
    else: 
        dest_loc = inp
    
    if dest_loc is not None:
        ret = move(node=node, inputs=dest_loc, **kwargs)
        assert ret['isdone'], f'{ret}'

    kwargs.update(find_place(node=node, inputs=inp, **kwargs))
    assert kwargs['isdone'], f'{kwargs}'

    ret = run_parallel_check(funcs=[
        lambda: forward(node=node, inputs=kwargs['mforward'], wait=True),
        # lambda: movej(node=node, dr0=kwargs['base_rotate'], wait=True),
        lambda: lift(node=node, inputs=kwargs['lift_to'], wait=True)
    ])
    assert ret['isdone'], f'{ret}'

    ret = movej(node=node, inputs='give')
    assert ret['isdone'], f'{ret}'

    ret = movej(node=node, inputs='approach_lying' if kwargs['islying'] else 'approach_standing', wait=True)
    assert ret['isdone'], f'{ret}'

    ret = movel(node=node, **kwargs['approach_pose'])
    assert ret['isdone'], f'{ret}'


    kwargs['isdone'] = True
    return kwargs

def _held_name():
    """What the robot believes it holds (Robot State `holding`, set by pick), without the '@place' part."""
    try:
        from robot_agent.state import current
        h = getattr(current().world, 'holding', None)
        return str(h).split('@')[0].strip() or None if h else None
    except Exception:
        return None


def _verify_placed(node, obj_name=None, **_):
    """After the release and the lift: the gripper ROI must be empty (the same presence check pick uses)."""
    if grasp_succeed(node=node)['isdone']:
        return {'isdone': False, 'msg': f'place: {obj_name or "the object"} seems to be still in the gripper'}
    return {'isdone': True, 'verified': 'gripper empty after the release'}


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
    lower_before_open = kwargs.pop('lower_before_open', None)

    for k in ['object_from_drawer', 'pose_after_open', 'lift_after_open', 'forward_after_open', 'init_pose_fixed']:
        kwargs.pop(k, None)
    inputs = kwargs.pop('inputs', None)
    held = _held_name()

    splits = inputs.split('>>')
    target, destination = splits  if len(splits)==2 else (None, splits[-1])
    if target is not None or not grasp_succeed(node=node):
        kwargs.update(pick(node=node, inputs=target,  **kwargs))
        assert kwargs['isdone'], f'{kwargs}'
        held = (target or '').split('@')[0].strip() or _held_name() or held
    
    kwargs.update(approach_place(node=node, inputs=destination, **kwargs))
    assert kwargs['isdone'], f'{kwargs}'

    if kwargs['islying']:
        ret = movet(node=node, dz=kwargs['dapproach']+kwargs['dz_up']/2, wait=True)
    else:
        ret = movet(node=node, dz=kwargs['dapproach'], wait=True)
        assert ret['isdone'], f'{ret}'

    if lower_before_open is None:
        lower_before_open = kwargs['islying']

    if lower_before_open:
        ret = run_parallel_check(funcs=[
            lambda: (time.sleep(0. if kwargs['islying'] else 0.15), movet(node=node, dz=kwargs['dz_up']/2) if kwargs['islying'] else movel(node=node, dz=-kwargs['dz_up'],acc= 0.3 if kwargs['islying'] else 0.2))[-1],
            lambda: (time.sleep(0.3 if kwargs['islying'] else 0.), grip(node=node, inputs='open', wait=True))
        ])
        assert ret['isdone'], f'{ret}'
    else:
        grip(node=node, inputs="open")
        assert ret['isdone'], f'{ret}'
    # ret = grip(node=node, inputs='open')
    # assert ret['isdone'], f'{ret}'

    # ret = movet(node=node, dz=kwargs['dz_up']/2) if kwargs['islying'] else movel(node=node, dz=-kwargs['dz_up'])
    # assert ret['isdone'], f'{ret}'


    ret = movel(node=node, dz=0.15 if kwargs['islying'] else 0.25)
    assert  ret['isdone'], f'{ret}'

    # look back before leaving: the wrist camera is above the spot it just let go of
    check = _verify_placed(node, held, lifted=0.15 if kwargs['islying'] else 0.25)

    if kwargs['islying']:
        ret = movej(node=node, inputs='approach_lying')
        assert  ret['isdone'], f'{ret}'

    ret = run_parallel_check(funcs={
        lambda: (movej(node=node, inputs='give'), movej(node=node, inputs='fold', wait=True))[-1],
        lambda: forward(node=node, inputs=-kwargs.get('mforward', 0 )) 
    })
    assert  ret['isdone'], f'{ret}'

    if not check['isdone']:                 # retreated safely, but the object is not where it was put
        return check
    kwargs['place_check'] = check.get('verified')

    # ret = movej(node=node, inputs='give')
    # assert ret['isdone'], f'{ret}'

    # ret = movej(node=node, inputs='fold')
    # assert  ret['isdone'], f'{ret}'

    # ret = forward(node=node, inputs=-kwargs.get('mforward', 0 )) 
    # assert  ret['isdone'], f'{ret}'


    return kwargs
    


@arm_exception_handler
def wipe(node, **kwargs):
    for k in ['object_from_drawer', 'pose_after_open', 'lift_after_open', 'forward_after_open', 'init_pose_fixed']:
        kwargs.pop(k, None)
    """`place` with wiping enabled. Defaults `inputs='towel>>spill'`."""
    inp = kwargs.pop('inputs', 'towel>>spill')
    splits = inp.split('>>')
    target, destination = splits if len(splits) == 2 else ('towel', splits[-1])
    robot_mode = get_robot_mode(node=node)
    

    if not grasp_succeed(node=node)['isdone']:
        kwargs.update(pick(node=node, inputs=target,  **kwargs))
        assert kwargs['isdone'], f'{kwargs}'
    #
    kwargs['islying'] = False
    kwargs.update(approach_place(node=node, inputs=destination, **kwargs))
    assert kwargs['isdone'], f'{kwargs}'


    ret = movet(node=node, dz=kwargs['dapproach'], wait=True)
    assert ret['isdone'], f'{ret}'

    movel(node=node, dz=-kwargs['dz_up']+0.005)

    #wipe here
    mul = 1 if robot_mode=='right' else -1
    dx, dy = 0.15, mul*0.05
    
    movel(node=node, dy=dy, dx=-dx)
    for i in range(3):
        movel(node=node, dx=2*dx, dy=-dy, acc=0.25)
        movel(node=node, dx=-2*dx, acc=0.25)
    #

    ret = movel(node=node, dz=0.15)
    assert  ret['isdone'], f'{ret}'

    if kwargs['islying']:
        ret = movej(node=node, inputs='approach_lying')
        assert  ret['isdone'], f'{ret}'

    ret = movej(node=node, inputs='give')
    assert ret['isdone'], f'{ret}'
    
    ret = movej(node=node, inputs='fold')
    assert  ret['isdone'], f'{ret}'

    ret = forward(node=node, inputs=-kwargs.get('mforward', 0 )) 
    assert  ret['isdone'], f'{ret}'


    return kwargs



