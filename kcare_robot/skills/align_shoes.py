"""align_shoes: push a pair of shoes side by side, pick both at once, carry them to a place and put them down.

    align_shoes::black shoes>>entrance        carry the pair to entrance and put it down there
    align_shoes::black shoes                  no destination: lifted and put down again where it is, no move

1. look: wrist camera straight down (HOME_JOINTS) with the lift at its lowest — both shoes are in view there and the
   depth is sharper than higher up. Per shoe, from the 3D points of its box: centre, long axis and the OPENING: a dip
   in the height along the shoe, with the collar rim and the heel back higher on both sides (the toe end only slopes
   down). The toe is the other end.
2. push the shoe farther from the destination toward the other one, `min_pushes` (2) to `max_pushes` (2) pushes in all:
   a point push off its centre first if the headings differ by more than `align_deg`, then a straight push to `gap`.
   Two pushes even when the first one looks done (2026-10-30: after one push the pair often read as touching, and
   the pinch then failed); a push of a pair still apart goes `last_push_extra` / `first_push_extra` past the spot,
   one of a pair already within `near_gap` only `near_press` (1 cm) past it.
3. pinch: one finger in each opening next to the touching side walls, close (the two walls between the fingers), lift
   it a little and check with grasp_succeed (wrist depth near the fingers); a miss: open, look again, one more try.
4. back to HOME_JOINTS (the jaw there lies across the robot, so the shoes come round pointing
   straight ahead), drive to the destination if there is one, and put it down straight below (movet along the tool).
"""
import functools
import math
import time

import numpy as np

from robot_agent.skills import log_data
from robot_agent.skill_configs import ENV
from robot_agent.utils import exception_handler, run_parallel_check
from kcare_robot.skills.arm import movej, movel, movet, arm_pose
from kcare_robot.skills.grip import grip
from kcare_robot.skills.lift import lift
from kcare_robot.skills.mobile import forward, move, mobile_pose
from kcare_robot.skills.pointcloud import get3d_arm
from kcare_robot.skills.recognition import detect, grasp_succeed
from kcare_robot.utils import get_env_specs

# wrist camera straight down, tool at (0.61, 0, 0.42) with the lift lowest: 5 cm back from [0, 0, 0, 20, 0, 20, -180]
# so the gripper in the camera's view hides less floor near the robot. (10 cm back, [0, -19, 0, 0.6, 0, 19.7, -180],
# the elbow was almost straight and straight-line moves from it failed to plan, 2026-10-21)
HOME_JOINTS = [0, -8.3, 0, 10.32, 0, 18.61, -180]

SHOES = dict(
    lift='lowest',          # look and work here (2026-10-21: at 0.8 the depth was blurred, at 0.57 the push start had no IK)
    max_pushes=2, gap=0.015, gap_ok=0.025,       # pushed until ~1.5 cm apart; up to 2.5 cm is pinched as it is
    min_pushes=2,           # 2026-10-30: one push and the pinch often failed — the pair is always pushed twice
    # one turn push corrects up to ~26 deg (max_turn_push / turn_offset): a pair up to max_turn_deg apart gets one,
    # then is pinched still up to pinch_deg apart (the fingers' spread between the openings is what decides the pinch)
    # 2026-10-21: the straight push turned the pushed shoe back ~7 deg against the slanted one (25 -> 32 deg)
    max_push=0.30, max_turn_deg=90, align_deg=5.0, pinch_deg=35.0,
    last_push_extra=0.05,   # the last push goes 5 cm farther (it stopped 1.4 cm short: the walls must touch for the pinch)
    first_push_extra=0.05,  # so does the first (2026-10-21: the turn push turned the shoe only 4-10 deg)
    # a pair already within near_gap gets no extra: pushed to its spot + near_press (2026-10-30: +5 cm there pushed
    # the other shoe askew)
    near_gap=0.03, near_press=0.01,
    push_z=0.025, turn_offset=0.08, max_turn_push=0.04,
    # the push starts where every finger tip is >= finger_clear from every shoe point (2026-10-21, real robot: 1.0-2.7 cm
    # came down on the shoe, 5.6-6.7 cm was a little far)
    finger_clear=0.045,     # 3.5 cm: the closed fingers still came down almost on the shoe (turn push)
    min_z=0.02, body_frac=0.4,  # points below 40 % of the shoe's height are the outline's blurred depth (doubled the width)
    min_points=30, opening_dip=0.025,            # the opening is >= 2.5 cm below the rim / heel on both sides
    finger_in=0.01, finger_margin=0.012, max_open=0.1,
    pinch_above=0.005, min_tip_z=0.03,  # finger tips 5 mm above the opening's seen depth (1 cm below: too deep; 2 cm above: missed twice)
    pair_x=[0.64, 0.74], max_drive=0.3,   # pair centre outside this x band (or heels hidden): drive the base forward /
                            # back to its middle, look again (2026-10-21: done at x 0.64 and 0.70; 0.80 out of reach)
    view_min_x=0.48,        # the gripper, in the wrist camera's view, hides the floor nearer than ~0.46 m (tool x 0.61):
                            # a shoe reaching it has its heel, and the opening, out of view (2026-10-21: measured
                            # 0.51 with the tool at x 0.66, 0.425 at x 0.56)
    # lifted 12 cm, then grasp_succeed (2026-10-27; was the finger spread after closing twice + 0.8 s, min_hold_w 1 cm):
    # checked at the floor, the floor and the shoes are within its margin (6 cm past the tips) even after a miss
    # its ROI without the bottom 10 rows (FIND_CONFIGS grasp_check_roi [415, 365, 476, 390]): the CLOSED empty
    # fingers fill them, 27-31 % "near" on the 2026-10-21 photos vs its 30 % threshold; above them 0 %
    grasp_roi=[415, 365, 476, 380],
    lift_dz=0.12, place_above=0.005)   # put down 5 mm above where it was picked


def _jaw_axis(node):
    """Gripper jaw (tool y) direction, base xy, unit. arm_pose (roll, pitch, yaw) [deg], R = Rz Ry Rx."""
    a, b, c = (math.radians(v) for v in arm_pose(node=node)['pose'][3:6])
    y = np.array([math.cos(c) * math.sin(b) * math.sin(a) - math.sin(c) * math.cos(a),
                  math.sin(c) * math.sin(b) * math.sin(a) + math.cos(c) * math.cos(a)])
    return y / (np.linalg.norm(y) + 1e-9)


def _turn_angle(node, want):
    """Angle [deg] about the base z that lays the (symmetric) jaw along `want`."""
    j = _jaw_axis(node)
    return (math.degrees(math.atan2(j[0] * want[1] - j[1] * want[0], float(np.dot(j, want)))) + 90) % 180 - 90


def _ik(node, xyz, drz=0.0, timeout=3.0, collide=True):
    """MoveIt /compute_ik for the tool at (x, y, z) [base_footprint], pointing as now turned `drz` deg about the base
    z, from the current joints -> joint1..7 [deg], None when unreachable, or False when /compute_ik is not there."""
    from moveit_msgs.srv import GetPositionIK
    from robot_agent.utils import deg2quaternion
    from kcare_robot.skills.arm import _qmul
    if not hasattr(node, '_align_shoes_ik'):
        node._align_shoes_ik = node.create_client(GetPositionIK, '/compute_ik')
    cli = node._align_shoes_ik
    if not cli.wait_for_service(timeout_sec=timeout):
        return False
    o = node.agents['mobile_base_tool_pose'].get()['pose'].orientation
    req = GetPositionIK.Request()
    r = req.ik_request
    r.group_name, r.ik_link_name, r.avoid_collisions = 'arm', 'tool_tcp_link', bool(collide)
    r.pose_stamped.header.frame_id = 'base_footprint'
    p = r.pose_stamped.pose
    p.position.x, p.position.y, p.position.z = (float(v) for v in xyz)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = _qmul(deg2quaternion(0, 0, drz), (o.x, o.y, o.z, o.w))
    r.timeout.nanosec = 200_000_000
    fut = cli.call_async(req); t0 = time.time()
    while not fut.done() and time.time() - t0 < timeout:
        time.sleep(0.01)
    if not fut.done() or fut.result().error_code.val != 1:          # 1 = SUCCESS
        return None
    js = dict(zip(fut.result().solution.joint_state.name, fut.result().solution.joint_state.position))
    return [round(math.degrees(js[f'joint{i}']), 2) for i in range(1, 8)]


def _reachable(node, pts, drz=0.0, collide=True):
    """Index of the first (x, y, z) the tool cannot reach (see _ik), or None when all are reachable. (2026-10-21: a
    movel with no IK solution left the arm hanging halfway through a push.)"""
    for i, xyz in enumerate(pts):
        j = _ik(node, xyz, drz=drz, collide=collide)
        if j is False:
            log_data({'msg': '[align_shoes] /compute_ik not available: reach not checked'})
            return None
        if j is None:
            return i
    return None


def _go_above(node, xyz, drz=0.0):
    """Joint move (PTP) to the tool at `xyz`, turned `drz` deg: from HOME_JOINTS (elbow 10 deg) straight-line moves
    there failed to plan ("Planning 실패", 2026-10-21), joint moves to the same pose do not need a straight path."""
    j = _ik(node, xyz, drz=drz)
    if j is False:                                      # no /compute_ik: the straight move, as before
        ret = movel(node=node, x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])); assert ret['isdone'], f'{ret}'
        return _turn_to_angle(node, drz)
    assert j, f'align_shoes: {np.round(xyz, 3).tolist()} turned {drz:+.0f} deg is out of reach'
    ret = movej(node=node, inputs=j, speed=0.5, acc=0.3, wait=True); assert ret['isdone'], f'above {np.round(xyz, 3).tolist()}: {ret}'
    return drz


def _together(*funcs):
    """run_parallel_check, but an exception in one of the threads fails it too (run_parallel loses it: a failed
    _go_above went on with the push down from wherever the arm was) -> the results, in order."""
    def safe(f):
        try:
            return f()
        except Exception as e:
            return {'isdone': False, 'msg': f'{type(e).__name__}: {e}'}
    ret = run_parallel_check(funcs=[functools.partial(safe, f) for f in funcs])
    assert ret['isdone'], f"align_shoes: {[r for r in ret['rets'] if isinstance(r, dict) and not r.get('isdone', True)]}"
    return ret['rets']


def _turn_to_angle(node, ang):
    if abs(ang) > 0.5:
        ret = movel(node=node, drz=ang); assert ret['isdone'], f'{ret}'
    return ang


def _shoe(P, cfg):
    """One shoe's 3D points (base frame) -> dict(c, axis, body, opening, open_pts, open_z, toe) or None."""
    P = P[P[:, 2] > cfg['min_z']]
    if len(P) < cfg['min_points']:
        return None
    top = float(np.percentile(P[:, 2], 90))
    body = P[P[:, 2] > cfg['body_frac'] * top]
    c = body[:, :2].mean(0)
    axis = np.linalg.svd(body[:, :2] - c)[2][0]
    u, v = (P[:, :2] - c) @ axis, (P[:, :2] - c) @ np.array([-axis[1], axis[0]])
    # height along the centre line, 1 cm bins; the opening = the deepest bin with higher bins on BOTH sides
    bins = np.arange(np.percentile(u, 2), np.percentile(u, 98), 0.01)
    line = [(np.abs(u - b) < 0.005) & (np.abs(v) < 0.02) for b in bins]
    prof = np.array([np.median(P[m, 2]) if m.sum() > 2 else np.nan for m in line])
    s = dict(c=c, axis=axis, body=body, top=top, opening=None, open_pts=None, open_z=None, toe=None)
    best, bi = cfg['opening_dip'], None
    for i in range(1, len(bins) - 1):
        if np.isnan(prof[i]) or np.all(np.isnan(prof[:i])) or np.all(np.isnan(prof[i + 1:])):
            continue
        dip = min(np.nanmax(prof[:i]), np.nanmax(prof[i + 1:])) - prof[i]
        if dip >= best:
            best, bi = dip, i
    if bi is None:
        return s
    ui, oz = bins[bi], float(prof[bi])
    O = P[(np.abs(u - ui) < 0.025) & (np.abs(v) < 0.03) & (P[:, 2] < oz + 0.01)]
    if len(O) < 5:
        return s
    s.update(opening=O[:, :2].mean(0), open_pts=O, open_z=oz, toe=-axis if ui > 0 else axis)
    return s


def _split(P, seeds, iters=12, half_len=0.14):
    """Two shoes from one cloud (touching: their boxes overlap, or one box holds both), seeded with (centre, axis)
    each: every point goes to the nearer shoe 'line', refit, repeat."""
    lines = [(np.asarray(c, float), np.asarray(a, float)) for c, a in seeds]
    for _ in range(iters):
        cost = [np.abs((P[:, :2] - c) @ np.array([-a[1], a[0]])) + np.maximum(0, np.abs((P[:, :2] - c) @ a) - half_len)
                for c, a in lines]
        lab = np.argmin(np.stack(cost, 1), 1)
        if min((lab == 0).sum(), (lab == 1).sum()) < 5:
            break
        lines = [(P[lab == k, :2].mean(0), np.linalg.svd(P[lab == k, :2] - P[lab == k, :2].mean(0))[2][0]) for k in (0, 1)]
    return [P[lab == 0], P[lab == 1]]


def _look(node, name, cfg):
    """Detect the shoes on the wrist camera (at HOME_JOINTS) -> two shoe dicts (see _shoe)."""
    for _ in range(3):                                  # the detector now and then returns nothing on a good view
        ret = detect(node=node, inputs=name, camera='arm', max_instances=3)
        if ret['isdone']:
            break
        time.sleep(1.0)
    assert ret['isdone'], f'detect {name}: {ret}'
    ins = ret['ins'][name]
    boxes = [s['box'] for s in ins.get('instances', [ins])]
    inside = lambda A, B: A[0] - 4 <= B[0] and A[1] - 4 <= B[1] and A[2] + 4 >= B[2] and A[3] + 4 >= B[3]
    boxes = [B for B in boxes if not any(B is not A and inside(B, A) for A in boxes)][:2]   # drop a box around the pair
    boxes = [[int(v) for v in B] for B in boxes]
    # one 4-px grid over all boxes; each point knows which boxes hold it
    X0, Y0, X1, Y1 = min(B[0] for B in boxes), min(B[1] for B in boxes), max(B[2] for B in boxes), max(B[3] for B in boxes)
    pix = np.array([[u, v] for v in range(Y0, Y1 + 1, 4) for u in range(X0, X1 + 1, 4)
                    if any(B[0] <= u <= B[2] and B[1] <= v <= B[3] for B in boxes)])
    P = np.asarray(get3d_arm(node=node, points=pix.tolist())['pose'])
    ok = ~np.isnan(P).any(axis=-1) & (np.nan_to_num(P[:, 2]) > cfg['min_z'])
    P, pix = P[ok], pix[ok]
    inb = np.stack([(pix[:, 0] >= B[0]) & (pix[:, 0] <= B[2]) & (pix[:, 1] >= B[1]) & (pix[:, 1] <= B[3]) for B in boxes], 1)
    log_data({'msg': f'[align_shoes] {name} boxes {boxes}, points {inb.sum(0).tolist()}'})
    if len(boxes) == 1 and len(P) > 20:                 # one box for both: seed the split across the cloud
        c = P[:, :2].mean(0); a, w = np.linalg.svd(P[:, :2] - c)[2]
        clouds = _split(P, [(c - w * 0.05, a), (c + w * 0.05, a)])
    elif len(boxes) == 2:
        # touching shoes: the boxes overlap and each holds the other's edge. A point in one box only is that shoe's;
        # a point in both goes to the shoe whose line (centre, axis of its own points) is nearer. (2026-10-21: seeding
        # from the whole boxes put the touching walls into the wrong shoe: axes 20 deg off, an opening lost)
        own = [inb[:, k] & ~inb[:, 1 - k] for k in (0, 1)]
        both = inb[:, 0] & inb[:, 1]
        clouds = [P[o] for o in own]
        if both.any() and min(map(len, clouds)) > 5:
            lines = [(Q[:, :2].mean(0), np.linalg.svd(Q[:, :2] - Q[:, :2].mean(0))[2][0]) for Q in clouds]
            S = P[both, :2]
            cost = np.stack([np.abs((S - c) @ np.array([-a[1], a[0]])) for c, a in lines], 1)
            lab = np.argmin(cost, 1)
            clouds = [np.vstack([clouds[k], P[both][lab == k]]) for k in (0, 1)]
    else:
        clouds = []
    shoes = [s for s in (_shoe(P, cfg) for P in clouds) if s is not None]
    assert len(shoes) == 2, f'need 2 {name}, found {len(shoes)}'
    for s in shoes:
        log_data({'msg': f"[align_shoes] shoe at {np.round(s['c'], 3).tolist()} axis {np.round(s['axis'], 2).tolist()} "
                         f"opening {None if s['opening'] is None else np.round(s['opening'], 3).tolist()} z {s['open_z']}"})
    return shoes


def _gap(a, b):
    """Edge-to-edge gap [m] from shoe a to shoe b ACROSS the shoes (perpendicular to their mean axis), and that
    direction (a -> b). (2026-10-21: measured along the line between the centres, a pair side by side but 7 cm apart
    lengthwise read -1 cm, "touching".)"""
    ub = b['axis'] if np.dot(a['axis'], b['axis']) > 0 else -b['axis']
    m = (a['axis'] + ub) / (np.linalg.norm(a['axis'] + ub) + 1e-9)
    n = np.array([-m[1], m[0]])
    n = n if (b['c'] - a['c']) @ n > 0 else -n
    return float(np.percentile(b['body'][:, :2] @ n, 3) - np.percentile(a['body'][:, :2] @ n, 97)), n


def _side_target(a, b, cfg):
    """Where a's centre goes to lie beside b: `gap` from b across b's axis, level with b along it."""
    n = np.array([-b['axis'][1], b['axis'][0]])
    n = n if (b['c'] - a['c']) @ n > 0 else -n                     # a -> b
    half_b = float(b['c'] @ n - np.percentile(b['body'][:, :2] @ n, 3))
    half_a = float(np.percentile(a['body'][:, :2] @ n, 97) - a['c'] @ n)
    return b['c'] - n * (half_b + half_a + cfg['gap']), n


def _heading(a, b):
    """Signed angle [rad] turning shoe a's long axis onto b's (axes are lines: within +-90 deg)."""
    ub = b['axis'] if np.dot(a['axis'], b['axis']) > 0 else -b['axis']
    return math.atan2(a['axis'][0] * ub[1] - a['axis'][1] * ub[0], float(np.dot(a['axis'], ub)))


def _push(node, a, b, dist, d, cfg, at=0.0):
    """Push shoe a `dist` m along `d`, low. at=0: open fingers spread across the push, through its centre (it slides);
    at=u: the closed gripper on its side away from b, `u` m along the shoe from the centre (it turns)."""
    # contact: a's side away from b, measured where the push lands (at=u: the shoe there, +-1.5 cm along it — the toe
    # is narrower than the middle; with the whole shoe's width the closed fingers came down too far out, 2026-10-21)
    q, pts = a['c'] + a['axis'] * at, a['body'][:, :2]
    if at:
        near = pts[np.abs((pts - a['c']) @ a['axis'] - at) < 0.015]
        pts = near if len(near) > 5 else pts
    contact = q + d * (float(np.percentile(pts @ d, 3)) - q @ d)
    # back off along the push line until the finger tips (closed: one point; open: spread across the push, ~4.8 cm
    # each side) clear both shoes by finger_clear
    shoes_xy = np.vstack([a['body'][:, :2], b['body'][:, :2]])
    across = np.array([-d[1], d[0]])
    tips = [np.zeros(2)] if at else [across * 0.048, -across * 0.048]
    back = 0.0
    while back < 0.15 and min(float(np.min(np.linalg.norm(shoes_xy - (contact - d * back + t), axis=1))) for t in tips) < cfg['finger_clear']:
        back += 0.005
    start, end = contact - d * back, contact + d * dist
    log_data({'msg': f'[align_shoes] push {dist:.3f} m along {np.round(d, 2).tolist()}' + (f', {at:+.2f} m off centre' if at else '')})
    path = [(start[0], start[1], cfg['push_z'] + 0.12), (start[0], start[1], cfg['push_z']), (end[0], end[1], cfg['push_z'])]
    bad = _reachable(node, path, drz=0.0 if at else _turn_angle(node, across))
    assert bad is None, (f'align_shoes: the push {("start", "start", "end")[bad]} {np.round(path[bad], 3).tolist()} is out of the '
                         f'arm\'s reach (shoes at x {a["c"][0]:.2f} / {b["c"][0]:.2f} m): put them closer to the robot')
    # ret = grip(node=node, inputs='close' if at else 'open', wait=True); assert ret['isdone'], f'{ret}'
    # _go_above(node, (start[0], start[1], cfg['push_z'] + 0.12), drz=0.0 if at else _turn_angle(node, across))
    
    _together(lambda: grip(node=node, inputs='close' if at else 'open', wait=True),
              lambda: _go_above(node, (start[0], start[1], cfg['push_z'] + 0.12), drz=0.0 if at else _turn_angle(node, across)))
    for kw in (dict(z=cfg['push_z']), dict(x=float(end[0]), y=float(end[1]), z=cfg['push_z'], acc=0.2), dict(dz=0.12)):
        ret = movel(node=node, **kw); assert ret['isdone'], f'push {kw}: {ret}'
    ret = movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True); assert ret['isdone'], f'{ret}'


def _pinch(a, b, cfg):
    """A finger in each opening, `finger_in` from its middle toward the walls between them, below the opening's seen
    depth -> (centre, direction a->b, open width, z) or None. (2026-10-21: fingers at the edge of the low area and
    1 cm above its depth closed on the collar padding, 9.6 cm high there, and held nothing.)"""
    if a['opening'] is None or b['opening'] is None:
        return None
    d = (b['opening'] - a['opening']) / (np.linalg.norm(b['opening'] - a['opening']) + 1e-9)
    pa, pb = a['opening'] + d * cfg['finger_in'], b['opening'] - d * cfg['finger_in']
    z = max(min(a['open_z'], b['open_z']) + cfg['pinch_above'], cfg['min_tip_z'])
    return 0.5 * (pa + pb), d, float(np.linalg.norm(pb - pa)) + cfg['finger_margin'], z


def _relook(node, name, cfg, d0):
    """Look again; the pushed shoe (a) stays on its side of the pair along the first push direction d0."""
    x, y = _look(node, name, cfg)
    return (x, y) if x['c'] @ d0 < y['c'] @ d0 else (y, x)


@exception_handler
def align_shoes(node, **params) -> dict:
    """Push the pair together, pinch both shoes at once, carry them to the place in `inputs` and put them down.

    inputs: 'black shoes>>entrance' (shoes>>destination) or 'black shoes' (no destination: put back down where they
    are, aligned, toes away from the robot; the base does not move). The shoes are always named. Options: the SHOES keys (max_pushes, gap,
    lift, pinch_above, ...)."""
    cfg = dict(SHOES, **{k: params.pop(k) for k in list(params) if k in SHOES})
    inputs = (params.pop('inputs', None) or '').strip()
    # 'shoes>>place' or 'shoes' (no destination); the shoes are always named
    name, dest = (inputs.split('>>', 1) if '>>' in inputs else (inputs, ''))
    name, dest = name.strip(), dest.strip()
    assert name, f'align_shoes: which shoes? inputs {inputs!r} (e.g. align_shoes::black shoes or black shoes>>entrance)'
    assert not dest or get_env_specs(dest, ENV), f'align_shoes: destination {dest!r} is not a known location'
    name = name[:-1] if name.endswith('shoes') else name    # singular: "black shoes" boxed the pair as one (sim)

    ret = movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True); assert ret['isdone'], f'{ret}'
    ret = lift(node=node, inputs=cfg['lift'], wait=True); assert ret['isdone'], f'{ret}'
    a, b = _look(node, name, cfg)
    for _ in range(2):          # too far for the arm, or so near the heels are hidden: drive the base, look again
        cx = 0.5 * float(a['c'][0] + b['c'][0])
        near = min(float(a['body'][:, 0].min()), float(b['body'][:, 0].min()))
        if cfg['pair_x'][0] <= cx <= cfg['pair_x'][1] and near >= cfg['view_min_x']:
            break
        drive = cx - 0.5 * sum(cfg['pair_x']) if near >= cfg['view_min_x'] else min(cx - 0.5 * sum(cfg['pair_x']), near - cfg['view_min_x'] - 0.05)
        assert abs(drive) <= cfg['max_drive'], f'align_shoes: the shoes are {cx:.2f} m ahead, too far off to drive to ({drive:+.2f} m)'
        log_data({'msg': f'[align_shoes] pair at x {cx:.2f} (nearest {near:.2f}): drive {drive:+.2f} m'})
        ret = forward(node=node, inputs=drive); assert ret['isdone'], f'forward {drive:+.2f}: {ret}'
        a, b = _look(node, name, cfg)
    near = min(float(a['body'][:, 0].min()), float(b['body'][:, 0].min()))
    assert near >= cfg['view_min_x'], (f'align_shoes: the shoes are too close to the robot (nearest point x {near:.2f} m): '
                                       f'their heels are hidden by the gripper in the camera\'s view')
    if dest:    # push the shoe farther from the destination (ENV loc, map frame -> base frame)
        rp = mobile_pose(node=node)['pose']; loc = get_env_specs(dest, ENV)['loc']; yaw = math.radians(rp[-1])
        dx, dy = loc['x'] - rp[0], loc['y'] - rp[1]
        dest_b = np.array([math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy])
        if np.linalg.norm(a['c'] - dest_b) < np.linalg.norm(b['c'] - dest_b):
            a, b = b, a
    elif abs(a['c'][1]) < abs(b['c'][1]):              # else the one farther to the side, toward the middle
        a, b = b, a
    gap, d0 = _gap(a, b)
    phi = math.degrees(_heading(a, b))
    log_data({'msg': f'[align_shoes] gap {gap:.3f} m, headings differ {phi:+.0f} deg, push the shoe at {np.round(a["c"], 2).tolist()}'})
    assert gap - cfg['gap'] <= cfg['max_push'], f'align_shoes: the shoes are {gap:.2f} m apart, too far to push'
    assert abs(phi) <= cfg['max_turn_deg'], f'align_shoes: the shoes point {phi:.0f} deg apart, too much to push'

    pushes = 0
    while True:
        gap, d = _gap(a, b); phi = _heading(a, b); fit = _pinch(a, b, cfg)
        target, n = _side_target(a, b, cfg)
        move_ = target - a['c']; along = float(move_ @ b['axis'])
        state = (f'gap {gap:.3f} m, headings {math.degrees(phi):+.1f} deg, offset along {along:+.3f} m, '
                 + (f'fingers {fit[2]:.3f} m apart' if fit else 'opening not seen'))
        log_data({'msg': f'[align_shoes] {state}'})
        # a turn push needs a push to its place after it (2026-10-21: also when touching — refusing it there left a pair
        # side by side 36 deg apart neither turned nor pinched)
        can_turn = pushes < cfg['max_pushes'] - 1
        if gap <= cfg['gap_ok'] and fit and fit[2] <= cfg['max_open'] and \
                abs(math.degrees(phi)) <= (cfg['align_deg'] if can_turn else cfg['pinch_deg']):
            if pushes >= min(cfg['min_pushes'], cfg['max_pushes']):
                break
            log_data({'msg': f"[align_shoes] looks ready after {pushes} push(es): pushed again (min_pushes {cfg['min_pushes']})"})
        if pushes >= cfg['max_pushes']:
            return {'isdone': False, 'msg': f'align_shoes: after {pushes} pushes: {state}'}
        if abs(math.degrees(phi)) > cfg['align_deg'] and can_turn:
            # the force toward b at u along the axis turns a by u * (axis x d): the side that turns it the right way
            u = math.copysign(cfg['turn_offset'], phi * (a['axis'][0] * d[1] - a['axis'][1] * d[0]))
            _push(node, a, b, float(np.clip(abs(u) * math.tan(abs(phi)), 0.005, cfg['max_turn_push']))
                  + (cfg['first_push_extra'] if pushes == 0 else 0.0), d, cfg, at=u)
        else:
            # to its place beside b: across and along at once; the extra only across (pressing the walls together)
            if gap <= cfg['near_gap']:
                # already close (2026-10-30: the +5 cm extra on a close pair pushed the other shoe askew): to the spot,
                # never back away from b, and only near_press past it
                m = b['axis'] * along + n * (max(float(move_ @ n), 0.0) + cfg['near_press'])
            else:
                extra = cfg['last_push_extra'] if pushes == cfg['max_pushes'] - 1 else cfg['first_push_extra'] if pushes == 0 else 0.0
                m = move_ + n * extra
            if np.linalg.norm(move_) < 0.01 and pushes >= cfg['min_pushes']:   # in place already: a push would only slide both
                why = (f'the openings are {fit[2]:.3f} m apart (fingers open {cfg["max_open"]})' if fit and fit[2] > cfg['max_open'] else
                       'an opening is not seen' if not fit else
                       f'they point {math.degrees(phi):+.0f} deg apart (pinched up to {cfg["pinch_deg"]:.0f})')
                return {'isdone': False, 'msg': f'align_shoes: the shoes lie side by side but {why}: {state}'}
            _push(node, a, b, float(np.linalg.norm(m)), m / np.linalg.norm(m), cfg)
        pushes += 1
        a, b = _relook(node, name, cfg, d0)

    for attempt in range(2):                            # lifted, the wrist depth near the fingers is the test
        centre, d, open_w, z = _pinch(a, b, cfg)
        path = [(centre[0], centre[1], z + 0.12), (centre[0], centre[1], z)]
        bad = _reachable(node, path, drz=_turn_angle(node, d))
        assert bad is None, f'align_shoes: the pinch point {np.round(path[bad], 3).tolist()} is out of the arm\'s reach: put the shoes closer'
        # ret = grip(node=node, inputs=open_w, wait=True); assert ret['isdone'], f'{ret}'
        # ang = _go_above(node, (centre[0], centre[1], z + 0.12), drz=_turn_angle(node, d))
        
        ang = _together(lambda: grip(node=node, inputs=open_w, wait=True),
                        lambda: _go_above(node, (centre[0], centre[1], z + 0.12), drz=_turn_angle(node, d)))[-1]

        log_data({'msg': f'[align_shoes] pinch {attempt + 1}: at {np.round(centre, 3).tolist()} z {z:.3f}, open {open_w:.3f}, wrist {ang:+.0f} deg'})
        ret = movel(node=node, z=z, acc=0.2); assert ret['isdone'], f'pinch down to z {z:.3f}: {ret}'
        ret = grip(node=node, inputs='close', wait=True); assert ret['isdone'], f'{ret}'
        ret = movel(node=node, dz=cfg['lift_dz'], acc=0.2); assert ret['isdone'], f'lift the pair: {ret}'
        held = grasp_succeed(node=node, crop_roi=cfg['grasp_roi'])
        log_data({'msg': f"[align_shoes] grasp check: {'held' if held.get('isdone') else 'missed'}"
                         f" (near {held.get('frac_near', 0):.0%})"})
        if held.get('isdone'):
            break
        log_data({'msg': '[align_shoes] missed: look again'})
        # ret = grip(node=node, inputs='open', wait=True); assert ret['isdone'], f'{ret}'
        # ret = movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True); assert ret['isdone'], f'{ret}'
        _together(lambda: grip(node=node, inputs='open', wait=True),
                  lambda: movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True))
        a, b = _relook(node, name, cfg, d0)
        assert _pinch(a, b, cfg), 'align_shoes: opening not seen after the missed pinch'
    else:
        return {'isdone': False, 'msg': 'align_shoes: missed the pinch twice'}

    # back to HOME_JOINTS (a joint move: the pair turns with the wrist, so the shoes come round straight ahead)
    ret = movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True); assert ret['isdone'], f'home with the pair: {ret}'
    if dest:
        ret = move(node=node, inputs=dest, keep_arm=True, wait=True); assert ret['isdone'], f'move {dest}: {ret}'
    # put it down straight below: along the tool (pointing down) to the height it was picked at
    down = float(arm_pose(node=node)['pose'][2]) - (z + cfg['place_above'])
    log_data({'msg': f'[align_shoes] put down: movet dz {down:.3f}'})
    ret = movet(node=node, dz=down, acc=0.2, wait=True); assert ret['isdone'], f'put down: {ret}'
    ret = grip(node=node, inputs='open', wait=True); assert ret['isdone'], f'{ret}'
    ret = movet(node=node, dz=-down, acc=0.2, wait=True); assert ret['isdone'], f'{ret}'
    ret = movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True); assert ret['isdone'], f'{ret}'
    return {'isdone': True, 'shoes': name, 'destination': dest, 'pushes': pushes,
            'pinch': {'centre': np.round(centre, 3).tolist(), 'z': round(z, 3), 'open': round(open_w, 3)}}
