

from robot_agent.skills import log_data
from robot_agent.state import current
from robot_agent.utils import exception_handler
from kcare_robot.skills.recognition import find_grasp, detect
from kcare_robot.skills.lift import lift, dlift
from kcare_robot.skills.arm import movet, movej, arm_exception_handler, movel, movelf, arm_joints
from kcare_robot.skills._pick_helpers import grasp_pose_from_ins, fix_angle
import numpy as np
from kcare_robot.skills.pointcloud import get3d_arm
from kcare_robot.skills.grip import grip

HOME_JOINTS = [0, 0, 0, 20, 0, 20, -180]     # ready_shoes 의 최초 홈 자세
APPROACH_DZ = 0.2    # movel 어프로치: 신발 표면 z 위로 띄우는 높이 (m)
GRASP_DZ    = 0.21    # 집기: movet 로 내려갔다(+) 올라오는(-) 거리 (m)
PLACE_DZ    = 0.15    # 놓기: movet 로 내려갔다(+) 올라오는(-) 거리 (m)


@exception_handler
def align_shoes(node, **params) -> dict:
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
