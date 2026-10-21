

from robot_agent.skills import log_data
from robot_agent.state import current
from robot_agent.utils import exception_handler
from kcare_robot.skills.recognition import find_grasp, detect
from kcare_robot.skills.lift import lift, dlift
from kcare_robot.skills.arm import movet, movej, arm_exception_handler, movel, movelf, arm_joints, arm_pose
from kcare_robot.skills._pick_helpers import grasp_pose_from_ins, fix_angle
import numpy as np
from kcare_robot.skills.pointcloud import get3d_arm
from kcare_robot.skills.grip import grip
from kcare_robot.skills.mobile import move, mobile_pose
from kcare_robot.utils import get_env_specs
from robot_agent.skill_configs import ENV
import math
import time

HOME_JOINTS = [0, 0, 0, 20, 0, 20, -180]     # ready_shoes 의 최초 홈 자세
APPROACH_DZ = 0.2    # movel 어프로치: 신발 표면 z 위로 띄우는 높이 (m)
GRASP_DZ    = 0.2    # 집기: movet 로 내려갔다(+) 올라오는(-) 거리 (m)
PLACE_DZ    = 0.2    # 놓기: movet 로 내려갔다(+) 올라오는(-) 거리 (m)


@exception_handler
def align_shoes_pick(node, **params) -> dict:
    """신발 2짝을 손목 카메라로 찾아, 오른쪽 신발을 집어 왼쪽 신발 오른쪽 옆에 정렬해 놓는다.

    base_footprint 기준 y- 쪽(로봇 오른쪽) 신발이 옮길 대상(src),
    y+ 쪽(로봇 왼쪽) 신발이 정렬 기준(dst). 놓는 위치는 dst 에서 신발 길이 방향에 수직(오른쪽)으로 `place_offset`(m)."""
    inputs = params.pop('inputs', 'black shoes')          # 문자열 입력 (기본값)
    max_instances = params.pop('max_instances', 2)
    approach_dz = params.pop('approach_dz', APPROACH_DZ)
    grasp_dz = params.pop('grasp_dz', GRASP_DZ)
    place_dz = params.pop('place_dz', PLACE_DZ)
    rotate = params.pop('rotate', True)
    place_offset = params.pop('place_offset', 0.15)
    # grasp-gd 의 jaw 방향(rz)을 유지해야 정렬 방향으로 회전할 수 있다 (False 면 rz=0)
    params.setdefault('keep_orientation', True)

    ready_shoes(node)
    ret = find_grasp(node=node, inputs=inputs, max_instances=max_instances, **params)
    assert ret['isdone'], f'{ret}'
    ins = ret['ins'][inputs]
    shoes = ins.get('instances', [ins])

    # 각 grasp 의 base_footprint 3D 좌표 (grasppose 는 tool 기준이므로 따로 계산)
    xyz = [grasp_base_xyz(node, s['box']) for s in shoes]
    for s, p in zip(shoes, xyz):
        s['base_xyz'] = p
    shoes = [s for s in shoes if s['base_xyz'] is not None]
    assert len(shoes) == 2, f'need 2 {inputs}, got {len(shoes)} (xyz: {xyz})'

    # y 오름차순: [0] = 오른쪽(y-) → 옮길 신발, [1] = 왼쪽(y+) → 기준 신발
    right, left = sorted(shoes, key=lambda s: s['base_xyz'][1])

    # 왼쪽 신발의 폭 방향(= jaw 선, 신발 길이 방향에 수직) 을 base xy 단위벡터로.
    # 카메라가 감지 자세일 때 계산해야 하므로 이동 전에 구한다.
    side_dir = jaw_dir_base(node, left['box'], left['grasppose'][3])
    if side_dir is None:
        side_dir = [0.0, -1.0]
        log_data({'msg': '[align_shoes] jaw dir 3D 실패 → base y- 로 대체'})

    out = {}
    for side, s in (('right', right), ('left', left)):
        out[side] = {
            'base_xyz':  s['base_xyz'],                       # [x, y, z] m, base_footprint
            'grasppose': s['grasppose'],                      # [dx, dy, dz, rz, width] tool 기준
            'fine':      grasp_pose_from_ins(s, None),        # (dx, dy, dz, angle, width, dpull) or None
            'score':     s.get('grasp_score'),
        }
        log_data({'msg': f"[align_shoes] {side} {inputs}: base_xyz={np.round(s['base_xyz'], 3).tolist()} "
                         f"grasppose={np.round(s['grasppose'], 3).tolist()} fine={out[side]['fine']}"})

    # 오른쪽(src) 신발 위로 어프로치 (base_footprint 절대좌표 x/y/z)
    ret = approach_above(node, right['base_xyz'], dz=approach_dz)
    assert ret['isdone'], f'{ret}'
    approach_xyz = ret['target']

    # 오른쪽 신발의 grasp 방향에 맞춰 tool z 축 회전 (movet, tool 기준 상대)
    angle = fix_angle(right['grasppose'][3])
    if rotate:
        log_data({'msg': f'[align_shoes] movet rz={angle:+.1f}'})
        ret = movet(node=node, rz=angle, wait=True)
        assert ret['isdone'], f'{ret}'

    # 집기: 열기 → dz 하강 → close → dz 복귀 → 홈 자세
    ret = grasp_and_return(node, dz=grasp_dz)
    assert ret['isdone'], f'{ret}'

    # 놓기: 홈 자세에서 왼쪽 신발 옆(y-, place_offset) 위로 movel 어프로치
    # 신발 길이 방향에 수직(side_dir, 오른쪽을 향함)으로 place_offset 만큼 떨어진 곳
    lx, ly, lz = left['base_xyz']
    px, py = lx + place_offset * side_dir[0], ly + place_offset * side_dir[1]
    log_data({'msg': f'[align_shoes] place side_dir={np.round(side_dir, 3).tolist()} offset={place_offset}'})
    ret = approach_above(node, [px, py, lz], dz=approach_dz)
    assert ret['isdone'], f'{ret}'
    place_xyz = ret['target']

    # 왼쪽 신발의 grasp 방향만큼 tool z 축 회전 → 집은 신발이 왼쪽 신발과 같은 방향
    # (홈 자세 = 감지 자세이므로 두 각도 모두 같은 기준에서 잰 값)
    place_angle = fix_angle(left['grasppose'][3])
    if rotate:
        log_data({'msg': f'[align_shoes] place movet rz={place_angle:+.1f}'})
        ret = movet(node=node, rz=place_angle, wait=True)
        assert ret['isdone'], f'{ret}'

    # dz 하강 → grip open → dz 복귀 → 홈 자세
    ret = place_and_return(node, dz=place_dz)
    assert ret['isdone'], f'{ret}'

    return {'isdone': True, 'inputs': inputs,
            'src': out['right'], 'dst': out['left'], **out,
            'approach_xyz': approach_xyz, 'wrist_angle': angle,
            'place_xyz': place_xyz, 'place_angle': place_angle, 'side_dir': side_dir}


def jaw_dir_base(node, box, rz, n=9):
    """grasp jaw 선(이미지)을 base xy 방향 단위벡터로. 오른쪽(y-)을 향하도록 부호를 맞춘다.

    entry 의 `box` 는 jaw 선 양끝의 min/max 라 어느 대각선인지 잃어버리므로,
    rz(이미지 축 기준, (-90, 90]) 부호로 복원한다: rz>=0 → (x0,y0)-(x1,y1), rz<0 → (x0,y1)-(x1,y0).
    양끝 접점은 바닥에 걸릴 수 있어 가운데 60% 구간 점들을 3D 로 올려 xy 주성분으로 방향을 잡는다."""
    x0, y0, x1, y1 = box
    a, b = ((x0, y0), (x1, y1)) if rz >= 0 else ((x0, y1), (x1, y0))
    ts = np.linspace(0.2, 0.8, n)
    pts = [[int(a[0] + t * (b[0] - a[0])), int(a[1] + t * (b[1] - a[1]))] for t in ts]

    P = np.asarray(get3d_arm(node=node, points=pts)['pose'])[:, :2]   # base xy
    P = P[~np.isnan(P).any(axis=-1)]
    if len(P) < 2:
        return None
    _, _, vt = np.linalg.svd(P - P.mean(axis=0))
    d = vt[0] / np.linalg.norm(vt[0])
    if d[1] > 0:                      # 오른쪽 = base y-
        d = -d
    return d.tolist()


def place_and_return(node, dz=PLACE_DZ):
    """tool 기준 movet 로 dz 하강 → grip open → dz 만큼 다시 올라온 뒤 홈 자세로 movej."""
    log_data({'msg': f'[align_shoes] place movet dz=+{dz:.3f}'})
    ret = movet(node=node, dz=dz, acc=0.25, wait=True)
    assert ret['isdone'], f'movet down: {ret}'

    ret = grip(node=node, inputs='open', wait=True)
    assert ret['isdone'], f'grip open: {ret}'

    log_data({'msg': f'[align_shoes] place movet dz=-{dz:.3f}'})
    ret = movet(node=node, dz=-dz, acc=0.25, wait=True)
    assert ret['isdone'], f'movet up: {ret}'

    return movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True)


def grasp_and_return(node, dz=GRASP_DZ):
    """tool 기준 movet 로 dz 하강 → grip close → dz 만큼 다시 올라온 뒤 홈 자세로 movej.
    단계마다 완료(isdone)를 확인하고 다음으로 넘어간다."""
    log_data({'msg': f'[align_shoes] movet dz=+{dz:.3f}'})
    ret = movet(node=node, dz=dz, acc=0.25, wait=True)
    assert ret['isdone'], f'movet down: {ret}'

    ret = grip(node=node, inputs='close', wait=True)
    assert ret['isdone'], f'grip close: {ret}'

    log_data({'msg': f'[align_shoes] movet dz=-{dz:.3f}'})
    ret = movet(node=node, dz=-dz, acc=0.25, wait=True)
    assert ret['isdone'], f'movet up: {ret}'

    return movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True)


def approach_above(node, xyz, dz=APPROACH_DZ):
    """base_footprint 절대좌표 (x, y, z+dz) 로 movel. 회전 인자는 주지 않아 현재 자세 유지."""
    x, y, z = [float(v) for v in xyz[:3]]
    target = [x, y, z + dz]
    log_data({'msg': f'[align_shoes] approach movel to base xyz={np.round(target, 3).tolist()}'})
    ret = movel(node=node, x=target[0], y=target[1], z=target[2])
    ret['target'] = target
    return ret


def ready_shoes(node):
    movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True)
    lift(node=node, inputs=0.57, wait=True)
    ret = grip(node=node, inputs='open', wait=True)
    assert ret['isdone'], f'grip open: {ret}'
    


def grasp_base_xyz(node, box, grid=5, min_half=10):
    """grasp jaw box 중심 주변 grid×grid 점 → base_footprint xyz (median, m).
    jaw box 는 선분이라 한 축이 얇을 수 있어 최소 반폭 `min_half` px 를 둔다."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw = max((x1 - x0) * 0.3, min_half)
    hh = max((y1 - y0) * 0.3, min_half)
    pts = [[int(gx), int(gy)]
           for gx in np.linspace(cx - hw, cx + hw, grid)
           for gy in np.linspace(cy - hh, cy + hh, grid)]

    P = np.asarray(get3d_arm(node=node, points=pts)['pose'])   # base_footprint, (N, 3)
    P = P[~np.isnan(P).any(axis=-1)]
    return np.median(P, axis=0).tolist() if len(P) else None


# ── align_shoes v2 (2026-10-21): push the pair together, pinch both, carry them to `inputs`, put them down ────
#
#   align_shoes::black shoes>>entrance        (or align_shoes::entrance — the shoe name then defaults)
#
#   1. look (wrist camera, top-down at HOME_JOINTS / lift 0.57): each shoe's 3D points from its detection box →
#      centre, long axis, toe direction, and its COLLAR (the opening: points clearly lower than the upper, the
#      insole seen through it);
#   2. push the shoe farther from the destination toward the other one, closed gripper low, through its centre
#      (an off-centre push turns it), until the gap is ~1 cm; look again (push once more if needed);
#   3. pinch the pair: one finger in each collar next to the touching side walls, close (the two walls squeezed
#      between the fingers), lift;
#   4. move to the destination and put the pair down on the floor in front of the robot, toes pointing away from
#      it (`toe_deg` turns that), open, retreat.
#   Too far apart / too different in heading for a push: the old pick-and-place (align_shoes_pick) brings one shoe
#   next to the other first.

SHOES = dict(gap=0.010, push_z=0.035, push_back=0.03, max_push=0.30, max_turn_deg=30, collar_drop=0.025,
             finger_in=0.0, finger_margin=0.012, collar_half=0.032, tip_above_insole=0.012, lift_dz=0.12, max_open=0.1,
             place_xy=[0.45, 0.0], toe_deg=0.0, min_points=30, look_lift=0.80, work_lift=0.57,
             min_z=0.02,
             gap_first=0.05,      # the first push stops 5 cm short (in contact, a turn push pivots on the other shoe); then the heading is corrected, then a short final push
             align_deg=5.0, turn_offset=0.08, max_turn_push=0.04,   # off-centre point push that turns the pushed shoe
             min_hold_w=0.015,    # fingers closed past this (sim: empty 0.6-1.0 cm, the pair 3.2-3.9 cm) = nothing between them (missed pinch): retry once
             carry_x=0.35, carry_lift=0.85)   # carry pose: tcp pulled back to x 0.35, lift up; move keeps the arm      # depth pixels on a shoe's outline get in-between depths: below 2 cm they widen it by ~2-5 cm


def _rot(rx, ry, rz):
    """arm_pose (roll, pitch, yaw) [deg] -> 3x3, R = Rz Ry Rx (robot_agent.utils.deg2quaternion)."""
    a, b, c = (math.radians(v) for v in (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _jaw_axis(node):
    """Current gripper jaw (tool y) direction, base xy, unit."""
    x, y, z, rx, ry, rz = arm_pose(node=node)["pose"]
    v = _rot(rx, ry, rz)[:, 1][:2]
    return v / (np.linalg.norm(v) + 1e-9)


def _turn_to(node, want_xy, jaw_symmetric=True):
    """Turn the wrist about the base z so the jaw axis lies along `want_xy` (either way round if symmetric)."""
    j = _jaw_axis(node)
    ang = math.degrees(math.atan2(j[0] * want_xy[1] - j[1] * want_xy[0], float(np.dot(j, want_xy))))
    if jaw_symmetric:
        ang = (ang + 90) % 180 - 90
    if abs(ang) > 0.5:
        ret = movel(node=node, drz=ang)
        assert ret['isdone'], f'{ret}'
    return ang


def _box_points(node, box, cfg, stride=6):
    x0, y0, x1, y1 = [int(v) for v in box]
    pts = [[u, v] for v in range(y0, y1 + 1, stride) for u in range(x0, x1 + 1, stride)]
    P = np.asarray(get3d_arm(node=node, points=pts)['pose'])
    P = P[~np.isnan(P).any(axis=-1)]
    return P[P[:, 2] > SHOES['min_z']]                      # not the floor, nor the mixed floor/shoe edge pixels


def _split_by_lines(U, seeds, iters=12, half_len=0.14):
    """Two shoes from one cloud, seeded with where each one should be (centre, long axis): assign every point to the
    nearer shoe 'line' (distance across + beyond the half length), refit centre and axis, repeat."""
    xy = U[:, :2]
    lines = [(np.asarray(c, float), np.asarray(a, float) / np.linalg.norm(a)) for c, a in seeds]
    lab = np.zeros(len(U), int)
    for _ in range(iters):
        cost = []
        for c, a in lines:
            r = xy - c; u = r @ a; v = r @ np.array([-a[1], a[0]])
            cost.append(np.abs(v) + np.maximum(0.0, np.abs(u) - half_len))
        lab = np.argmin(np.stack(cost, 1), axis=1)
        new = []
        for k in (0, 1):
            q = xy[lab == k]
            if len(q) < 5:
                return U[lab == 0], U[lab == 1]
            c = q.mean(0); _, _, vt = np.linalg.svd(q - c); a = vt[0] if np.dot(vt[0], lines[k][1]) > 0 else -vt[0]
            new.append((c, a))
        lines = new
    return U[lab == 0], U[lab == 1]


def _split_side_by_side(U, bin_w=0.005):
    """Two parallel shoes side by side as one cloud: the long axis = the first principal axis; cut across it at the
    least-populated bin around the middle (the gap / the touching walls)."""
    xy = U[:, :2]; c = xy.mean(0); _, _, vt = np.linalg.svd(xy - c)
    v = (xy - c) @ vt[1]
    lo, hi = np.percentile(v, 25), np.percentile(v, 75)
    bins = np.arange(lo, hi + bin_w, bin_w)
    if len(bins) < 3:
        cut = 0.0
    else:
        h, e = np.histogram(v, bins=bins)
        k = int(np.argmin(h)); cut = 0.5 * (e[k] + e[k + 1])
    return U[v < cut], U[v >= cut]


def _split_two(P, iters=20):
    """Two shoes boxed as one (they touch): 2-means on xy, seeded along the main axis of the points."""
    xy = P[:, :2]; c = xy.mean(0); _, _, vt = np.linalg.svd(xy - c)
    t = (xy - c) @ vt[0]
    cen = np.array([xy[t < 0].mean(0), xy[t >= 0].mean(0)])
    for _ in range(iters):
        lab = np.argmin(((xy[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)
        cen = np.array([xy[lab == k].mean(0) for k in (0, 1)])
    return [P[lab == 0], P[lab == 1]]


def _shoe_geometry(node, box, cfg, stride=6, P=None):
    """3D points of one detected shoe (base frame) -> dict(c, axis, toe, collar, collar_z, top, pts) or None."""
    if P is not None:
        return _geom_from_points(P, cfg)
    x0, y0, x1, y1 = [int(v) for v in box]
    pts = [[u, v] for v in range(y0, y1 + 1, stride) for u in range(x0, x1 + 1, stride)]
    P = np.asarray(get3d_arm(node=node, points=pts)['pose'])
    n_all = len(P)
    P = P[~np.isnan(P).any(axis=-1)]
    n_valid = len(P)
    P = P[P[:, 2] > SHOES['min_z']]                         # not the floor, nor the mixed floor/shoe edge pixels
    print(f'[align_shoes] box {[x0, y0, x1, y1]}: {n_all} px, {n_valid} with depth, {len(P)} above the floor', flush=True)
    return _geom_from_points(P, cfg)


def _geom_from_points(P, cfg):
    if len(P) < cfg['min_points']:
        return None
    _xy = P[:, :2]; _c = _xy.mean(0); _, _, _vt = np.linalg.svd(_xy - _c)
    print(f'[align_shoes] shoe pts {len(P)} centre {np.round(_c, 3).tolist()} len {np.ptp(_xy @ _vt[0]):.3f} '
          f'width {np.ptp(_xy @ _vt[1]):.3f} z {np.round(np.percentile(P[:, 2], [5, 50, 95]), 3).tolist()}', flush=True)
    top = float(np.percentile(P[:, 2], 90))
    xy = P[:, :2]; c = xy.mean(0)
    _, _, vt = np.linalg.svd(xy - c); axis = vt[0] / np.linalg.norm(vt[0])
    low = P[P[:, 2] < top - cfg['collar_drop']]             # the insole seen through the opening
    # keep the low points away from the outline (the sole rim): within 70 % of the half widths
    u_all, v_all = (xy - c) @ axis, (xy - c) @ np.array([-axis[1], axis[0]])
    if len(low):
        ul, vl = (low[:, :2] - c) @ axis, (low[:, :2] - c) @ np.array([-axis[1], axis[0]])
        # the opening is around the centre line: outline pixels are low too (in-between depths), so a fixed half width
        keep = (np.abs(vl) < cfg['collar_half']) & (np.abs(ul) < 0.85 * np.percentile(np.abs(u_all), 97))
        low = low[keep]
    if len(low) < 5:
        return dict(c=c, axis=axis, toe=None, collar=None, collar_z=None, top=top, pts=P)
    collar = low[:, :2].mean(0)
    toe = (c - collar) / (np.linalg.norm(c - collar) + 1e-9)    # the toe cap is the closed end
    toe = axis if np.dot(axis, toe) > 0 else -axis
    return dict(c=c, axis=axis, toe=toe, collar=collar, collar_pts=low, collar_z=float(np.median(low[:, 2])), top=top, pts=P)


def _contains(A, B, m=4):
    return A[0] - m <= B[0] and A[1] - m <= B[1] and A[2] + m >= B[2] and A[3] + m >= B[3]


def _look(node, inputs, cfg, prior=None):
    # singular for the detector: "black shoes" boxed the whole pair as one object (sim, 2026-10-21)
    name = inputs[:-1] if inputs.endswith('shoes') else inputs
    ret = lift(node=node, inputs=cfg['look_lift'], wait=True); assert ret['isdone'], f'{ret}'   # both shoes in view,
    for k in range(3):                                      # the detector now and then returns nothing on a good view
        ret = detect(node=node, inputs=name, camera='arm', max_instances=3)
        if ret['isdone']:
            break
        time.sleep(1.0)
    assert ret['isdone'], f'detect {name}: {ret}'
    ins = ret['ins'][name]
    boxes = [s['box'] for s in ins.get('instances', [ins])]
    boxes = [B for B in boxes if not any(B is not A and _contains(B, A) for A in boxes)]   # drop a pair box
    print(f'[align_shoes] {name} boxes {[[int(v) for v in B] for B in boxes]}', flush=True)
    if len(boxes) == 1:                                     # touching shoes boxed as one: split the 3D points
        P = _box_points(node, boxes[0], cfg)
        span = np.ptp(P[:, :2] @ np.linalg.svd(P[:, :2] - P[:, :2].mean(0))[2][1]) if len(P) > 10 else 0
        shoes = [g for g in (_geom_from_points(Q, cfg) for Q in _split_side_by_side(P)) if g is not None] if span > 0.12 else []
    else:
        # touching shoes: their boxes overlap and each box also holds the other's edge -> re-split the union by
        # 2-means from the two box centroids (side by side, each point goes to its own shoe)
        P1, P2 = _box_points(node, boxes[0], cfg), _box_points(node, boxes[1], cfg)
        A, Bx = boxes[0], boxes[1]
        overlap = min(A[2], Bx[2]) > max(A[0], Bx[0]) and min(A[3], Bx[3]) > max(A[1], Bx[1])
        if overlap and len(P1) and len(P2):
            # side by side and touching (after the push): cut the union across, at the emptiest line between them
            U = np.unique(np.round(np.vstack([P1, P2]), 4), axis=0)
            P1, P2 = _split_by_lines(U, prior) if prior else _split_side_by_side(U)
        shoes = [g for g in (_geom_from_points(Q, cfg) for Q in (P1, P2)) if g is not None]
    ret = lift(node=node, inputs=cfg['work_lift'], wait=True); assert ret['isdone'], f'{ret}'
    assert len(shoes) == 2, f'need 2 {inputs}, found {len(shoes)}'
    return shoes


def _gap(a, b):
    """Edge-to-edge gap [m] between shoe a and b along the line a -> b, and that direction."""
    d = b['c'] - a['c']; d = d / (np.linalg.norm(d) + 1e-9)
    return float(np.percentile(b['pts'][:, :2] @ d, 3) - np.percentile(a['pts'][:, :2] @ d, 97)), d


def _push(node, a, b, cfg, target=None, dist=None):
    """Push shoe a toward b (open fingers spread along it, low, through its centre) until the gap is `target`
    (or by `dist` m)."""
    gap, d = _gap(a, b)
    dist = gap - (cfg['gap'] if target is None else target) if dist is None else dist
    if dist <= 0.005:
        return 0.0
    assert dist <= cfg['max_push'], f'shoes {gap:.2f} m apart: too far to push'
    back = float(np.percentile(a['pts'][:, :2] @ d, 97) - a['c'] @ d)   # a's half extent along d
    start = a['c'] - d * (back * 2 + cfg['push_back'])            # behind a's far side, on the line through its centre
    log_data({'msg': f'[align_shoes] push {dist:.3f} m along {np.round(d, 2).tolist()} (gap {gap:.3f})'})
    # push with BOTH finger tips, open and spread along the shoe: two contacts 10 cm apart translate it; one point
    # (closed gripper) turned it 10-20 deg in the sim
    ret = grip(node=node, inputs='open', wait=True); assert ret['isdone'], f'{ret}'
    _turn_to(node, a['axis'])
    ret = movel(node=node, x=float(start[0]), y=float(start[1]), z=cfg['push_z'] + 0.12); assert ret['isdone'], f'{ret}'
    ret = movel(node=node, z=cfg['push_z']); assert ret['isdone'], f'{ret}'
    end = start + d * (dist + cfg['push_back'] + back)
    p0 = arm_pose(node=node)['pose']
    ret = movel(node=node, x=float(end[0]), y=float(end[1]), z=cfg['push_z'], acc=0.2); assert ret['isdone'], f'{ret}'
    p1 = arm_pose(node=node)['pose']
    print(f'[align_shoes] push dist {dist:.3f} start {np.round(start, 3).tolist()} end {np.round(end, 3).tolist()} '
          f'tcp before {p0[:3]} after {p1[:3]}', flush=True)
    ret = movel(node=node, dz=0.12); assert ret['isdone'], f'{ret}'
    ret = movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True); assert ret['isdone'], f'{ret}'
    return dist


def _signed_turn(a, b):
    """Signed angle [rad] that turns shoe a's long axis onto b's (axes are lines: within +-90 deg)."""
    ub = b['axis'] if np.dot(a['axis'], b['axis']) > 0 else -b['axis']
    return math.atan2(a['axis'][0] * ub[1] - a['axis'][1] * ub[0], float(np.dot(a['axis'], ub)))


def _turn_push(node, a, b, phi, cfg, flip=False):
    """Turn shoe a by about `phi` with a closed-gripper point push on its outer side, off its centre: the force
    toward b at r = u * axis gives the torque u * (axis x d), so u takes the sign that turns it the right way."""
    _, d = _gap(a, b)
    axis = a['axis']
    cross = axis[0] * d[1] - axis[1] * d[0]
    u = math.copysign(cfg['turn_offset'], phi * cross) * (-1 if flip else 1)
    half_w = float(np.percentile(a['pts'][:, :2] @ d, 97) - a['c'] @ d)
    contact = a['c'] + axis * u - d * half_w
    dist = float(np.clip(abs(u) * math.tan(abs(phi)), 0.005, cfg['max_turn_push']))
    start = contact - d * cfg['push_back']
    print(f'[align_shoes] turn push {math.degrees(phi):+.1f} deg: at u {u:+.2f}, {dist:.3f} m', flush=True)
    ret = grip(node=node, inputs='close', wait=True); assert ret['isdone'], f'{ret}'
    ret = movel(node=node, x=float(start[0]), y=float(start[1]), z=cfg['push_z'] + 0.12); assert ret['isdone'], f'{ret}'
    ret = movel(node=node, z=cfg['push_z']); assert ret['isdone'], f'{ret}'
    end = contact + d * dist
    ret = movel(node=node, x=float(end[0]), y=float(end[1]), z=cfg['push_z'], acc=0.15); assert ret['isdone'], f'{ret}'
    ret = movel(node=node, dz=0.12); assert ret['isdone'], f'{ret}'
    ret = movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True); assert ret['isdone'], f'{ret}'
    ret = grip(node=node, inputs='open', wait=True); assert ret['isdone'], f'{ret}'
    return dist


def _pinch_points(a, b, cfg):
    """One finger point inside each collar, next to the touching walls, at the same place along the shoes."""
    assert a['collar'] is not None and b['collar'] is not None, 'collar (shoe opening) not seen'
    d = b['collar'] - a['collar']; d = d / (np.linalg.norm(d) + 1e-9)
    axis = a['axis'] if np.dot(a['axis'], b['axis']) > 0 else -a['axis']
    ua, ub = a['collar_pts'][:, :2] @ axis, b['collar_pts'][:, :2] @ axis
    lo, hi = max(ua.min(), ub.min()), min(ua.max(), ub.max())
    assert hi > lo, 'the two openings do not overlap along the shoes'
    um = 0.5 * (lo + hi)
    sel_a = a['collar_pts'][np.abs(ua - um) < 0.03][:, :2]; sel_b = b['collar_pts'][np.abs(ub - um) < 0.03][:, :2]
    pa = sel_a[np.argmax(sel_a @ d)] - d * cfg['finger_in']        # a's collar edge toward b, a little inside
    pb = sel_b[np.argmin(sel_b @ d)] + d * cfg['finger_in']
    open_w = float(np.linalg.norm(pb - pa)) + cfg['finger_margin']
    z = min(a['collar_z'], b['collar_z']) + cfg['tip_above_insole']
    return 0.5 * (pa + pb), d, open_w, z


@exception_handler
def align_shoes(node, **params) -> dict:
    """Push the pair together, pinch both shoes at once, carry them to the place in `inputs` and put them down.

    inputs: 'black shoes>>entrance' (shoes>>destination) or 'entrance'. Options: the SHOES keys (gap, push_z, lift_dz,
    place_xy [m, base frame at the destination], toe_deg [toes relative to straight away from the robot], ...)."""
    cfg = dict(SHOES, **{k: params.pop(k) for k in list(params) if k in SHOES})
    inputs = params.pop('inputs', None) or ''
    shoes_name, dest = (inputs.split('>>', 1) + [None])[:2] if '>>' in inputs else ('black shoes', inputs)
    shoes_name, dest = shoes_name.strip() or 'black shoes', (dest or '').strip()
    assert dest and get_env_specs(dest, ENV), f'align_shoes: destination {dest!r} is not a known location'

    ready_shoes(node)
    a, b = _look(node, shoes_name, cfg)
    # the shoe farther from the destination is pushed (destination = ENV loc, map frame -> base frame)
    rp = mobile_pose(node=node)['pose']; loc = get_env_specs(dest, ENV)['loc']
    yaw = math.radians(rp[-1])                                   # mobile_pose: degrees
    dxy = np.array([loc['x'] - rp[0], loc['y'] - rp[1]])
    dest_b = np.array([math.cos(yaw) * dxy[0] + math.sin(yaw) * dxy[1], -math.sin(yaw) * dxy[0] + math.cos(yaw) * dxy[1]])
    a, b = (a, b) if np.linalg.norm(a['c'] - dest_b) >= np.linalg.norm(b['c'] - dest_b) else (b, a)
    turn = abs(math.degrees(math.acos(min(1.0, abs(float(np.dot(a['axis'], b['axis'])))))))
    gap, _ = _gap(a, b)
    log_data({'msg': f'[align_shoes] gap {gap:.3f} m, headings differ {turn:.0f} deg, push the shoe at {np.round(a["c"], 2).tolist()}'})
    if turn > cfg['max_turn_deg'] or gap - cfg['gap'] > cfg['max_push']:
        log_data({'msg': '[align_shoes] too far apart / too different: pick-and-place one shoe next to the other first'})
        ret = align_shoes_pick(node=node, inputs=shoes_name)
        assert ret['isdone'], f'{ret}'
        a, b = _look(node, shoes_name, cfg)
    def relook(pushed_new_c, pushed_axis, static):
        x, y = _look(node, shoes_name, cfg, prior=[(pushed_new_c, pushed_axis), (static['c'], static['axis'])])
        return (x, y) if np.linalg.norm(x['c'] - static['c']) > np.linalg.norm(y['c'] - static['c']) else (y, x)

    # 1. push most of the way (3 cm short), 2. turn the pushed shoe parallel, 3. a short final push to ~1 cm
    pushed, turned = 0.0, 0.0
    gap0, d0 = _gap(a, b)
    p = _push(node, a, b, cfg, target=cfg['gap_first']); pushed += p
    a2, b2 = relook(a['c'] + d0 * p, a['axis'], b)
    phi = _signed_turn(a2, b2)
    print(f'[align_shoes] after the first push: gap {_gap(a2, b2)[0]:.3f}, headings differ {math.degrees(phi):+.1f} deg', flush=True)
    flip = False
    for k in range(2):                  # closed loop: a turn that made it worse is tried from the other end next
        if abs(math.degrees(phi)) <= cfg['align_deg']:
            break
        turned += _turn_push(node, a2, b2, phi, cfg, flip=flip)
        a2, b2 = relook(a2['c'], b2['axis'], b2)
        new = _signed_turn(a2, b2)
        print(f'[align_shoes] after turn {k + 1}: headings differ {math.degrees(new):+.1f} deg', flush=True)
        if abs(new) > abs(phi):
            flip = not flip
        phi = new
    g1, d1 = _gap(a2, b2)
    if g1 > cfg['gap'] + 0.005:
        p = _push(node, a2, b2, cfg); pushed += p
        a2, b2 = relook(a2['c'] + d1 * p, a2['axis'], b2)
    print(f'[align_shoes] ready to pinch: gap {_gap(a2, b2)[0]:.3f}, headings differ {math.degrees(_signed_turn(a2, b2)):+.1f} deg', flush=True)

    # pinch both; a pinch whose fingers close past min_hold_w held nothing: open, look again, one more try
    for k in range(2):                  # the gap measure can be off by a few cm: the finger spread is the real test
        centre, d, open_w, z = _pinch_points(a2, b2, cfg)
        if open_w <= cfg['max_open']:
            break
        extra = open_w - 0.08
        print(f'[align_shoes] fingers would need {open_w:.3f} m: push {extra:.3f} m more', flush=True)
        g, dd = _gap(a2, b2)
        p = _push(node, a2, b2, cfg, dist=extra); pushed += p
        a2, b2 = relook(a2['c'] + dd * p, a2['axis'], b2)
    for attempt in range(2):
        centre, d, open_w, z = _pinch_points(a2, b2, cfg)
        assert open_w <= cfg['max_open'], f'fingers would need {open_w:.3f} m open'
        toe = a2['toe'] if a2['toe'] is not None else a2['axis']
        ret = grip(node=node, inputs=open_w, wait=True); assert ret['isdone'], f'{ret}'
        ang = _turn_to(node, d)
        jaw_grasp = _jaw_axis(node)
        print(f'[align_shoes] pinch {attempt + 1}: turned {ang:+.0f} deg, at {np.round(centre, 3).tolist()} z {z:.3f} open {open_w:.3f}', flush=True)
        ret = movel(node=node, x=float(centre[0]), y=float(centre[1]), z=z + 0.12); assert ret['isdone'], f'pinch above: {ret}'
        ret = movel(node=node, z=z, acc=0.2); assert ret['isdone'], f'pinch down to z {z:.3f}: {ret}'
        ret = grip(node=node, inputs='close', wait=True); assert ret['isdone'], f'{ret}'
        # the close returns before the fingers settle (sim: 3.5 cm reported on an empty gripper that ends at 0.6 cm):
        # wait, close again, and read the width the fingers really stopped at
        time.sleep(0.8)
        ret = grip(node=node, inputs='close', wait=True); assert ret['isdone'], f'{ret}'
        w = ret.get('position')
        print(f'[align_shoes] fingers closed to {w}', flush=True)
        if w is None or float(w) >= cfg['min_hold_w']:
            break
        log_data({'msg': f'[align_shoes] missed (fingers closed to {float(w):.3f} m): look again'})
        ret = grip(node=node, inputs='open', wait=True); assert ret['isdone'], f'{ret}'
        ret = movel(node=node, dz=0.12); assert ret['isdone'], f'{ret}'
        ret = movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True); assert ret['isdone'], f'{ret}'
        a2, b2 = relook(a2['c'], a2['axis'], b2)
    else:
        return {'isdone': False, 'msg': 'align_shoes: missed the pinch twice'}
    ret = movel(node=node, dz=cfg['lift_dz'], acc=0.2); assert ret['isdone'], f'{ret}'
    # carry pose: pull the pair back toward the robot, lift up; move then drives without folding the arm
    ret = movel(node=node, x=cfg['carry_x'], y=0.0); assert ret['isdone'], f'carry pose: {ret}'
    ret = lift(node=node, inputs=cfg['carry_lift'], wait=True); assert ret['isdone'], f'{ret}'

    # carry to the destination, put the pair down in front of the robot, toes away from it (+ toe_deg)
    ret = move(node=node, inputs=dest, keep_arm=True, wait=True); assert ret['isdone'], f'move {dest}: {ret}'
    ret = lift(node=node, inputs=cfg['work_lift'], wait=True); assert ret['isdone'], f'{ret}'
    px, py = cfg['place_xy']
    ret = movel(node=node, x=float(px), y=float(py)); assert ret['isdone'], f'{ret}'
    want = np.array([math.cos(math.radians(cfg['toe_deg'])), math.sin(math.radians(cfg['toe_deg']))])
    # the toe turns with the wrist: toe now = R(jaw_now vs jaw_at_grasp) * toe_at_grasp
    j_now = _jaw_axis(node)
    rot = math.atan2(jaw_grasp[0] * j_now[1] - jaw_grasp[1] * j_now[0], float(np.dot(jaw_grasp, j_now)))
    toe_now = np.array([math.cos(rot) * toe[0] - math.sin(rot) * toe[1], math.sin(rot) * toe[0] + math.cos(rot) * toe[1]])
    ang = math.degrees(math.atan2(toe_now[0] * want[1] - toe_now[1] * want[0], float(np.dot(toe_now, want))))
    if abs(ang) > 0.5:
        ret = movel(node=node, drz=ang); assert ret['isdone'], f'{ret}'
    ret = movel(node=node, z=z + 0.005, acc=0.2); assert ret['isdone'], f'{ret}'
    ret = grip(node=node, inputs='open', wait=True); assert ret['isdone'], f'{ret}'
    ret = movel(node=node, dz=0.15); assert ret['isdone'], f'{ret}'
    ret = movej(node=node, inputs=HOME_JOINTS, speed=0.5, acc=0.3, wait=True); assert ret['isdone'], f'{ret}'
    return {'isdone': True, 'shoes': shoes_name, 'destination': dest, 'pushed': round(pushed, 3), 'turn_push': round(turned, 3),
            'pinch': {'centre': np.round(centre, 3).tolist(), 'z': round(z, 3), 'open': round(open_w, 3)}}
