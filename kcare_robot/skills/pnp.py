"""Pick-and-place at ONE location as a single motion: pick an object and put it on / into another object there.

    pnp::apple>>bowl@table@kitchen          # pick 'apple', put it on / into 'bowl', both at table@kitchen
    pnp::inputs='사과->apple>>그릇->bowl@table@kitchen'

Built from the pick / place steps (find, approach, grasp, find_place, release, verify), with the in-between
finishing steps of the two skills dropped, so the arm goes straight from the grasp to the place:

  pick alone ends with:   approach_lying (lying) -> give -> fold -> base back by -mforward
  place alone starts with: move -> head find -> base forward / lift -> give -> approach pose

pnp instead
  1. moves to the location once and looks once, both cameras at the same time: the wrist (arm at give) finds the
     object to pick, the head finds the destination object (head only if the wrist misses the object);
  2. picks: base forward + give + lift together, approach pose, grasp (`_grasp`: fine_move's steps with the
     wrist turn merged into the centring move);
  3. goes straight to the place pose: lifts the object clear (not needed after a top-down grasp: the pull-back
     already lifted it), base / lift / place-approach joint pose together, approach pose (no give / fold);
  4. places as `place` does (descend, release, lift, gripper-empty check);
  5. folds and drives back once, at the very end.
Step timings come back as 'timings' [s] (and in the log).

Timings in the house sim (2026-10-14, banana -> red plate at the counter, robot already there): 48.1 s with
pick's fine_move and sequential finds; see the 'timings' of a run for the current split.
"""
import threading
import time

from robot_agent.skill_configs import ENV, NO_ACTION
from robot_agent.utils import run_parallel_check
from robot_agent.skills import log_data, spoken
from kcare_robot.utils import get_env_specs, announce_picking

from kcare_robot.skills.arm import movej, movel, movet, arm_exception_handler
from kcare_robot.skills.grip import grip
from kcare_robot.skills.lift import lift
from kcare_robot.skills.mobile import move, forward
from kcare_robot.skills.recognition import find, find_place, find_grasp, grasp_succeed
from kcare_robot.skills.place import _verify_placed
from kcare_robot.skills import _pick_helpers as _h

CLEAR_DZ = 0.08          # after a side (standing) grasp: lift the object this much before swinging to the place pose [m]


def _parse(inputs):
    """'obj>>dest@loc' (or 'obj@loc>>dest') -> (obj, dest, loc)."""
    if not inputs or '>>' not in inputs:
        raise ValueError(f"pnp: inputs must be 'object>>destination_object@location', got {inputs!r}")
    src, dst = (s.strip() for s in inputs.split('>>', 1))
    obj, loc_a = _h.split_loc(src)
    dest, loc_b = _h.split_loc(dst)
    loc = loc_b or loc_a
    if loc_a and loc_b and get_env_specs(loc_a, ENV) != get_env_specs(loc_b, ENV):
        raise ValueError(f'pnp: both objects must be at one location ({loc_a!r} vs {loc_b!r}); use pick + place')
    return obj.strip(), dest.strip(), loc


def _find_both(node, obj, dest, loc):
    """(object entry, destination pose): the wrist (arm at give) looks for the object while the head looks for
    the destination — two devices, one at a time each, so no camera's background thread can move the head while
    the other detection looks (find's default 'arm, head' leaves its head thread running after it returns)."""
    q = lambda n: f'{n}@{loc}' if loc else n
    out = {}

    def head():
        out['dest'] = find(node=node, inputs=q(dest), cameras='head', simple_return=True)
    th = threading.Thread(target=head, daemon=True)
    th.start()
    ret = find(node=node, inputs=q(obj), cameras='arm')
    th.join()
    if not ret.get('isdone') or obj not in ret.get('ins', {}):
        ret = find(node=node, inputs=q(obj), cameras='head')          # the wrist missed it: the head's estimate
    assert ret.get('isdone') and obj in ret.get('ins', {}), f'pnp: {obj} not found {ret}'
    d = out.get('dest') or {}
    assert d.get('isdone') and dest in d.get('ins', {}), f'pnp: {dest} not found {d}'
    return dict(ret['ins'][obj]), d['ins'][dest]['pose']


def _grasp(node, obj, num_trials=2):
    """fine_move's grasp (wrist re-detect, centre, turn the wrist, descend, close, check, pull back, check).
    Merging the centring move and the wrist turn into one movet(dx, dy, rz) saved ~1.6 s but missed the banana next
    to the pot in the sim (the turning straight-line path) and sometimes failed IK: kept as two moves.
    Returns {'isdone', 'pulled'} (pull-back distance)."""
    for _ in range(num_trials):
        ret = grip(node=node, inputs='open', wait=False)
        assert ret['isdone'], f'{ret}'
        ret = find_grasp(node=node, inputs=obj)
        if not ret['isdone']:
            continue
        g = _h.grasp_pose_from_ins(ret['ins'][obj], None)
        if g is None:
            continue
        dx, dy, dz, angle, width, dpull = g
        grip(node=node, inputs=width + 0.02, wait=False)
        ret = movet(node=node, dx=dx, dy=dy, wait=True)
        assert ret['isdone'], f'{ret}'
        ret = movej(node=node, dr6=angle, wait=True)
        assert ret['isdone'], f'{ret}'
        ret = movet(node=node, dz=dz, acc=0.25, wait=True)
        assert ret['isdone'], f'{ret}'
        ret = grip(node=node, inputs='close', wait=True)      # wait: right after a wait=False close the check passed empty grasps
        assert ret['isdone'], f'{ret}'
        held = grasp_succeed(node=node)['isdone']
        if not held:
            grip(node=node, inputs='open', wait=False)
        ret = movet(node=node, dz=-dpull, acc=0.25, wait=True)
        assert ret['isdone'], f'{ret}'
        if held and grasp_succeed(node=node)['isdone']:
            return {'isdone': True, 'pulled': dpull}
    return {'isdone': False, 'msg': f'pnp: could not grasp {obj} in {num_trials} tries'}


@arm_exception_handler
def pnp(node, **kwargs):
    """Pick `object` and put it on / into `destination_object` at the same location, in one motion.

    inputs: 'object>>destination_object@location'. Options: num_trials (grasp tries, 2), placeside (passed to
    find_place: place beside instead of on the destination object), lower_before_open (as place)."""
    inputs = kwargs.pop('inputs', None)
    said = spoken('inputs', inputs)
    num_trials = int(kwargs.pop('num_trials', 2))
    lower_before_open = kwargs.pop('lower_before_open', None)
    placeside = kwargs.pop('placeside', None)
    for k in ['object_from_drawer', 'pose_after_open', 'lift_after_open', 'forward_after_open', 'init_pose_fixed']:
        kwargs.pop(k, None)
    obj, dest, loc = _parse(inputs)
    T, t0 = {}, [time.time()]

    def lap(name):                        # step timings, returned as 'timings' [s] and logged
        t = time.time(); T[name] = round(t - t0[0], 2); t0[0] = t
        log_data({'msg': f'pnp {name}: {T[name]:.1f} s'})

    announce_picking((said or obj).split('>>')[0].split('@')[0])
    if NO_ACTION:
        return {'isdone': True}

    # 1. one trip, one look (wrist: the object, head: the destination, at the same time)
    if loc and get_env_specs(loc, ENV):
        ret = move(node=node, inputs=loc, wait=True)
        assert ret['isdone'], f'{ret}'
    lap('move')
    pk, dest_pose = _find_both(node, obj, dest, loc)
    pl = find_place(node=node, inputs=f'{dest}@{loc}' if loc else dest, islying=pk['islying'], dest_pose=dest_pose,
                    **({'placeside': placeside} if placeside is not None else {}))
    assert pl['isdone'], f'{pl}'
    lap('find')
    log_data({'msg': f"pnp: pick {obj} (lying {pk['islying']}, forward {pk['mforward']:.2f}) -> "
                     f"{dest} (forward {pl['mforward']:.2f}, lift {pl['lift_to']:.2f})"})

    # 2. pick
    ret = run_parallel_check(funcs=[
        lambda: forward(node=node, inputs=pk['mforward'], wait=True),
        lambda: movej(node=node, inputs='give', wait=True),
        lambda: lift(node=node, inputs=pk['lift_to'], wait=True),
    ])
    assert ret['isdone'], f'{ret}'
    if pk['islying']:
        ret = movej(node=node, inputs='approach_lying')
        assert ret['isdone'], f'{ret}'
    ret = movel(node=node, **pk['approach_pose'])
    assert ret['isdone'], f'{ret}'
    lap('pick_approach')
    ret = _grasp(node, obj, num_trials)
    assert ret['isdone'], f'{ret}'
    lap('grasp')

    # 3. straight over to the place pose: no approach_lying / give / fold / drive back in between
    if not pk['islying']:                 # top-down: the pull-back was straight up already
        ret = movel(node=node, dz=CLEAR_DZ)
        assert ret['isdone'], f'{ret}'
    ret = run_parallel_check(funcs=[
        lambda: forward(node=node, inputs=pl['mforward'] - pk['mforward'], wait=True),
        lambda: lift(node=node, inputs=pl['lift_to'], wait=True),
        lambda: movej(node=node, inputs='approach_lying' if pl['islying'] else 'approach_standing', wait=True),
    ])
    assert ret['isdone'], f'{ret}'
    ret = movel(node=node, **pl['approach_pose'])
    assert ret['isdone'], f'{ret}'
    lap('place_approach')

    # 4. place (as place)
    if lower_before_open is None:
        lower_before_open = pl['islying']
    ret = movet(node=node, dz=pl['dapproach'] + (pl['dz_up'] / 2 if pl['islying'] and lower_before_open else 0),
                wait=True)
    assert ret['isdone'], f'{ret}'
    if lower_before_open:
        ret = run_parallel_check(funcs=[
            lambda: movet(node=node, dz=pl['dz_up'] / 2) if pl['islying'] else
            (time.sleep(0.15), movel(node=node, dz=-pl['dz_up'], acc=0.2))[-1],
            lambda: (time.sleep(0.3 if pl['islying'] else 0.), grip(node=node, inputs='open', wait=True))[-1],
        ])
    else:
        ret = grip(node=node, inputs='open', wait=True)
    assert ret['isdone'], f'{ret}'
    up = 0.15 if pl['islying'] else 0.25
    ret = movel(node=node, dz=up)
    assert ret['isdone'], f'{ret}'
    lap('place')

    # 5. the only finishing steps: fold + drive back. The gripper-empty check runs at give, away from the spot:
    # right above it the wrist ROI still saw the object just put down / the destination (false "still held"),
    # while an object really still in the gripper moves with it and is seen at give as well.
    if pl['islying']:
        ret = movej(node=node, inputs='approach_lying')
        assert ret['isdone'], f'{ret}'
    check = {}

    def arm_home():
        r = movej(node=node, inputs='give')
        if not r['isdone']:
            return r
        check.update(_verify_placed(node, obj))
        return movej(node=node, inputs='fold', wait=True)
    ret = run_parallel_check(funcs=[arm_home, lambda: forward(node=node, inputs=-pl['mforward'])])
    assert ret['isdone'], f'{ret}'
    lap('fold_back')
    if not check['isdone']:
        return dict(check, timings=T)
    return {'isdone': True, 'picked': obj, 'placed_on': dest, 'location': loc, 'islying': pk['islying'],
            'place_check': check.get('verified'), 'timings': T}
