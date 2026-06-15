"""Helpers for recognition.py. Pure refactor of the utility functions out of
the skill module; behaviour matches the original 1:1.

`recognition.py` keeps the registered skills (`find`, `find_arm`,
`get_side_pose_3d`, ...) plus the two orchestrators (`_detect_objects`,
`_detect_grasps`) and pulls everything else from here. This module must NOT
import `recognition` back (no import cycle)."""

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np, threading

from pyconnect.utils import run_parallel
from pyinterfaces.utils import (
    show_box_on_rgb, Ixy2xyz, show_line_on_rgb,
    get_mask_locs_with_stride, calc_normalvector,
)

from robot_agent.skill_configs import (
    ARM_CONFIGS, MOBILE_CONFIGS,
    STANDING_OBJ_NAMES, LYING_OBJ_NAMES, HAVING_HANDLE_OBJ_NAMES,
)
from robot_agent.state import current
from robot_agent.skills import log_data
from kcare_robot.skills.calibrattion import Head2BaseCalibration
from kcare_robot.skills.head import head_state as get_head_state
from kcare_robot.skills.pointcloud import get3d
from kcare_robot.skills.arm import get_wrist_angle
import cv2


@dataclass
class CameraData:
    rgb: np.ndarray
    depth: np.ndarray
    cam_params: dict
    head_state: dict


# ── Camera fetchers ──────────────────────────────────────────────────────────
def _require(value, agent_name: str):
    if value is None:
        raise Exception(f"{agent_name} returned None")
    return value


def _fetch_head(node) -> CameraData:
    """head_cam — used by `find`."""
    a = node.agents
    rgb, depth, cam_params, head_state = run_parallel(funcs=[
        lambda: a['head_rgb'].get(),
        lambda: a['head_depth'].get(),
        lambda: a['head_cam_params'].get(),
        lambda: get_head_state(node=node),
    ])
    _require(rgb,        'head_rgb')
    _require(depth,      'head_depth')
    _require(cam_params, 'head_cam_params')
    return CameraData(rgb=rgb['im'], depth=depth['im'],
                      cam_params=cam_params['cam_params'], head_state=head_state)


def _fetch_arm(node) -> CameraData:
    """wrist_cam — used by `find_arm`."""
    a = node.agents
    rgb, depth, cam_params, head_state = run_parallel(funcs=[
        lambda: a['arm_rgb'].get(),
        lambda: a['arm_depth'].get(),
        lambda: a['arm_cam_params'].get(),
        lambda: get_head_state(node=node),
    ])
    _require(rgb,        'arm_rgb')
    _require(depth,      'arm_depth')
    _require(cam_params, 'arm_cam_params')
    return CameraData(rgb=rgb['im'], depth=depth['im'],
                      cam_params=cam_params['cam_params'], head_state=head_state)


CAMERA_FETCHERS: dict[str, Callable] = {
    'head': _fetch_head,
    'arm':  _fetch_arm,
}


def fetch_camera_data(node, camera: str) -> CameraData:
    fetcher = CAMERA_FETCHERS.get('head' if 'head' in camera else 'arm')
    return fetcher(node)


# ── Visualisation / logging ──────────────────────────────────────────────────
def publish_image(node, rgb):
    node.agents['screen_log'].send({'rgb': rgb})


def log_result_data(node, data):
    node.agents['screen_log'].log_msg(data)


def save_detection_dataset(rgb=None, depth=None, results=None, annotated=None, tag=''):
    """If a per-run capture dir is active (UI "log data" on), persist vision
    detection inputs/outputs: rgb(png), depth(npy), results(json), annotated(png).
    Best-effort — never raises into the skill."""
    try:
        from robot_agent.skills import dataset_dir
        d = dataset_dir()
    except Exception:
        d = None
    if not d:
        return
    try:
        import os, json, time
        os.makedirs(d, exist_ok=True)
        ts = time.strftime('%H%M%S') + f"_{int(time.time() * 1000) % 1000:03d}"
        base = os.path.join(d, f"{ts}_{tag}" if tag else ts)
        if rgb is not None or annotated is not None:
            import cv2
        if rgb is not None:
            cv2.imwrite(base + '_rgb.png', cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR))
        if depth is not None:
            np.save(base + '_depth.npy', np.asarray(depth))
        if annotated is not None:
            a = np.asarray(annotated)
            if a.ndim == 3 and a.shape[2] == 3:
                a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
            cv2.imwrite(base + '_annotated.png', a)
        if results is not None:
            with open(base + '_results.json', 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=1,
                          default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    except Exception as e:
        print(f'[recognition] dataset save failed: {e}')


def _log_annotated(res, rgb, target_grasp=None, target_box=None, **kwargs):
    """Draw VisionServe result with the SDK and push it to the UI camera panel."""
    try:
        annotated = res.visualize(rgb, target_grasp=target_grasp, target_box=target_box)
        log_data({'log_image': np.asarray(annotated)})
        save_detection_dataset(annotated=np.asarray(annotated), tag='annotated')
    except Exception as e:   # SDK draw needs Pillow — fall back to the raw frame
        log_data({'log_image': rgb, 'viz_error': str(e)})


# ── VisionServe SDK access ───────────────────────────────────────────────────
def _vs_client():
    """The 'visionserve' SDK Client registered in the Connection panel."""
    c = current().dm.get_client('visionserve')
    if c is None:
        raise Exception("client 'visionserve' not registered — add it in the Connection panel")
    return c


def _vs_postprocess():
    """Lazy import so the module stays importable without the SDK installed."""
    from visionserve import select_target_object, select_target_grasp
    return select_target_object, select_target_grasp


def _prompt(obj_names) -> str:
    """GroundingDINO / grasp-gd text prompt, e.g. ['cup','bottle'] → 'cup. bottle.'."""
    return ' '.join(f'{n.strip()}.' for n in obj_names if n.strip())


# ── Legacy TCP detector (still used by get_side_pose_3d's fastsam) ────────────
def call_detector(params: dict):
    vlms = current().dm.get_client('vlms')
    if vlms is None:
        raise Exception("TCP connect 'vlms' not registered — add it in the Connection panel")
    return vlms.send(params)


# ── Workspace helpers ────────────────────────────────────────────────────────
def is_inside_workspace_box(x, y, z, workspace):
    """Vectorised workspace membership: bool scalar or bool array. Arm reach is
    extended by MOBILE_CONFIGS['max_shift'] along x for the base nudge."""
    try:
        mforward = MOBILE_CONFIGS.get('max_shift', 0) or 0
        if not isinstance(x, (np.ndarray, list, tuple)):
            return (
                workspace['x'][0] - mforward <= x <= workspace['x'][1] + mforward and
                workspace['y'][0] <= np.abs(y) <= workspace['y'][1] and
                workspace['z'][0] <= z <= workspace['z'][1]
            )
        out = np.ones_like(x, dtype=bool)
        out = np.bitwise_and(out, workspace['x'][0] - mforward <= x)
        out = np.bitwise_and(out, x <= workspace['x'][1] + mforward)
        out = np.bitwise_and(out, workspace['y'][0] <= np.abs(y))
        out = np.bitwise_and(out, np.abs(y) <= workspace['y'][1])
        out = np.bitwise_and(out, workspace['z'][0] <= z)
        out = np.bitwise_and(out, z <= workspace['z'][1])
        return out
    except Exception:
        return False


# ── Depth / 3D helpers ───────────────────────────────────────────────────────
def _sample_depth(depth, x, y, win: int = 7) -> float:
    """Median of valid (>0) depth in a small window around pixel (x, y)."""
    x, y = int(x), int(y)
    h, w = depth.shape[:2]
    x0, y0 = max(0, x - win), max(0, y - win)
    x1, y1 = min(w, x + win), min(h, y + win)
    sub = depth[y0:y1, x0:x1]
    valid = sub[sub > 0]
    return float(np.median(valid)) if valid.size else 0.0


def _box_depths(depth, box) -> dict:
    x0, y0, x1, y1 = [int(v) for v in box]
    sub = depth[y0:y1, x0:x1]
    valid = sub[sub > 0]
    if valid.size == 0:
        return {'obj_min': None, 'obj_median': None, 'bound': None}
    return {'obj_min': float(valid.min()), 'obj_median': float(np.median(valid)), 'bound': None}

def _get_bound_depth(depth, mask):
    mask = (mask > 0).astype(np.uint8)

    dilated = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    boundary = (dilated > 0) & (mask == 0)

    boundary_depths = depth[boundary]

    if boundary_depths.size == 0:
        return None

    return np.median(boundary_depths)

# ── Object classification helpers ────────────────────────────────────────────
def is_lying(obj_name: str, normal_yz, horizon_yz) -> bool:
    for el in STANDING_OBJ_NAMES:
        if el in obj_name:
            return False
    for el in LYING_OBJ_NAMES:
        if el in obj_name:
            return True
    return abs(np.dot(normal_yz, horizon_yz)) > 0.5


def mass_percentages(obj_name: str, mask, box, *, lying: bool):
    if lying:
        return 50, 50
    if not any(el in obj_name for el in HAVING_HANDLE_OBJ_NAMES):
        return 50, 50
    x0, _, x1, _ = box
    xc = (x0 + x1) // 2
    left = round(np.sum(mask[:, x0:xc]) / np.sum(mask) * 100, 2)
    return left, 100 - left


def _box_islying(obj_name: str, box, depth, cam_params, horizon_yz) -> bool:
    """Estimate islying from the surface normal of the valid depth pixels in `box`."""
    if any(el in obj_name for el in STANDING_OBJ_NAMES):
        return False
    if any(el in obj_name for el in LYING_OBJ_NAMES):
        return True
    try:
        x0, y0, x1, y1 = [int(v) for v in box]
        sub = depth[y0:y1, x0:x1]
        ys, xs = np.where(sub > 0)
        if len(ys) < 10:
            return False
        Iy = (ys + y0).astype('float32')
        Ix = (xs + x0).astype('float32')
        Z = depth[(ys + y0, xs + x0)].astype('float32')
        X3, Y3, Z3 = Ixy2xyz(Ix=Ix, Iy=Iy, Z=Z, cam_params=cam_params)
        nv = calc_normalvector(np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=-1))
        nv_yz = nv[1:] / np.linalg.norm(nv[1:])
        return is_lying(obj_name, nv_yz, horizon_yz)
    except Exception:
        return False


# ── Grasp pixel-line → 3D pose (units match pick.py's contract) ──────────────
def _line_to_grasppose(node, line, *, cam_params, depths: dict,
                       keep_orientation: bool, rgb_out=None):
    """Convert a pixel jaw-line (x0,y0,x1,y1) at `depths['obj_median']` into a
    3D grasp pose ``[x, y, depth_final, rz, width]`` (x/y/depth in metres after
    /1000 + the configured wrist offsets; rz in degrees; width in mm).

    This is the exact lift the previous `estimate_grasp` used — only the source
    of the pixel line changed (grasp-gd contacts instead of mask2grasps)."""
    ix0, iy0, ix1, iy1 = [int(v) for v in line]
    if rgb_out is not None:
        rgb_out = show_line_on_rgb(rgb=rgb_out, line=(ix0, iy0, ix1, iy1), thick=3)
        log_data({'log_image': rgb_out})

    rx = get_wrist_angle(node=node)
    to_pick_lying_obj = rx > 40
    v0 = depths['obj_median']
    v1 = depths.get('bound')

    alpha = 0.9
    depth_tune = 60 if v1 is None else 0
    if v1 is None:
        depth_value = v0 + depth_tune
    else:
        depth_value = v0 + min(((1 - alpha) * v0 + alpha * v1) - v0 + depth_tune, 60)

    x0, y0, z0 = Ixy2xyz(Ix=ix0, Iy=iy0, Z=depth_value, cam_params=cam_params)
    x1, y1, z1 = Ixy2xyz(Ix=ix1, Iy=iy1, Z=depth_value, cam_params=cam_params)
    x = (x0 + x1) / 2 / 1000. + ARM_CONFIGS['wrist_cam_offset'][0]
    y = (y0 + y1) / 2 / 1000. + ARM_CONFIGS['wrist_cam_offset'][1]

    width = np.linalg.norm([x0 - x1, y0 - y1, z0 - z1])  # mm
    rz = np.arctan2(y1 - y0, x1 - x0) * 180 / np.pi
    if not keep_orientation:
        rz = rz if to_pick_lying_obj else 0

    depth_final = depth_value / 1000. - ARM_CONFIGS['wrist_tool_length']  # m
    return [float(x), float(y), float(depth_final), float(rz), float(width)], float(depth_final)


# ── Object detection helpers ──────────────────────────────────────────────────
def _bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def _foreground_mask(res, w, h, depth):
    """Union of grounded-sam object masks ∧ valid depth → uint8 fg mask (or None)."""
    masks = getattr(res, 'masks', None) or []
    if not masks:
        return None
    fg = np.zeros((h, w), dtype=bool)
    for m in masks:
        try:
            fg |= m.to_ndarray(width=w, height=h).astype(bool)
        except Exception:
            pass
    if not fg.any():
        return None
    return np.bitwise_and(fg, depth > 0).astype('uint8')


def _compute_side_pose(node, box, side, fg_mask, rgb, head_state):
    """Free spot to the `side` of `box` (head cam) → base-frame pose via get3d.
    Returns None when no spot is found or it's outside the arm workspace."""
    head_rz = head_state.get('current_rz', 0) if head_state else 0
    side_box = make_side_box(box=box, side=side, fg_mask=fg_mask,
                             node=node, rgb=rgb, head_angle=90 - abs(head_rz))
    if side_box is None:
        return None
    x0, y0, x1, y1 = side_box
    xc, yc = (x0 + x1) // 2, (y0 + y1) // 2
    pose = list(get3d(node=node, x=xc, y=yc)['pose'])
    log_data({'side_box': [int(x0), int(y0), int(x1), int(y1)], 'side_pose': pose})
    if not is_inside_workspace_box(*pose[:3], workspace=ARM_CONFIGS['range']):
        return None
    return pose

def _get_placepose(placepose, target_height, robot_mode, islying=False):
    """Resolve a placepose dict + target height into a [x, y, z, wrist_angle] list.

    `islying` is forwarded by callers that have detected the object's posture
    (mirrors carerobotapp's signature) so future overrides can lean shallower
    for lying objects. Currently unused in the body but accepted for parity.
    """
    assert placepose is not None or target_height is not None, \
        f'placepose: {placepose}, target height: {target_height}'
    # x, y, z, dx, dy = ARM_CONFIGS['base_x'], 0.7, target_height, 0, 0
    # x, y, z, dx, dy, dz = ARM_CONFIGS['base_x']-MOBILE_CONFIGS['dshift'], 0.7, target_height, 0, 0, 0
    x, y, z, dx, dy, dz = ARM_CONFIGS['base_x']-MOBILE_CONFIGS['dshift'], 0.7, target_height, 0, 0, 0
    # wrist_angle = 15 if z > 0.45 else 30 if z > 0.2 else 45
    if placepose is not None:
        x, dx = placepose.get('x', x), placepose.get('dx', dx)
        y, dy = abs(placepose.get('y', y)), placepose.get('dy', dy)
        z, dz = placepose.get('z', z), placepose.get('dz', dz)
        # wrist_angle = placepose.get('wrist_angle', wrist_angle)

    x, y, z = x+dx, y+dy, z+dz
    y = y if robot_mode == 'right' else -y
    return [x, y, z]



# ── Input parsing ────────────────────────────────────────────────────────────
def _parse_obj_names(inp, obj_names) -> list[str]:
    if obj_names is not None:
        return obj_names
    if inp is None:
        raise Exception('Either "inputs" or "obj_names" required')
    return [el.strip().lower() for el in inp.split(',') if el.strip()]


# ── Side-box geometry (used by get_side_pose_3d + _compute_side_pose) ─────────
_SIDE_SIGNS = {'left': [-1, -1], 'right': [1, 1], 'front': [-1, 1], 'rear': [1, -1]}


def generate_angle_mask(w, h, x, y, theta_deg, side='beside'):
    """Half-plane mask split by a line through (x, y) at angle theta_deg."""
    assert side in ['left', 'right', 'front', 'rear', 'beside']
    if side == 'beside':
        return np.bitwise_not(generate_angle_mask(w, h, x, y, theta_deg, side='rear'))

    theta = np.deg2rad(45 + theta_deg)
    cost, sint = np.cos(theta), np.sin(theta)
    n0, n1 = (sint, -cost), (cost, sint)

    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    v0 = n0[0] * (X - x) + n0[1] * (Y - y)
    v1 = n1[0] * (X - x) + n1[1] * (Y - y)

    sign0, sign1 = _SIDE_SIGNS[side]
    mask = np.bitwise_and((sign0 * v0) > 0, (sign1 * v1) > 0)
    return mask.astype(np.uint8)


def visualize_mask(rgb, mask):
    import cv2
    overlay = rgb.copy()
    overlay[mask == 0] = (0, 0, 255)
    return cv2.addWeighted(overlay, 0.4, rgb, 0.6, 0)


def make_side_box(fg_mask, box=None, pose_3d=None, depth=None, calib_func=None,
                  head_angle=0, side='beside', area_diameter=120,
                  node=None, rgb=None, search_local=True):
    """Find a free patch next to `box` (or near `pose_3d`) in pixel space."""
    import cv2
    assert side in ['left', 'right', 'front', 'beside']
    h, w = fg_mask.shape[:2]

    if box is None:
        xc, yc = w // 2, int(0.3 * h)
        rx, ry = 10, 10
    else:
        x0, y0, x1, y1 = [int(el) for el in box]
        rx, ry = (x1 - x0) // 2, (y1 - y0) // 2
        xc, yc = (x0 + x1) // 2, (y0 + y1) // 2

    r = 0.9 * np.linalg.norm((rx, ry))
    ra = area_diameter // 2

    if search_local:
        bg_mask = np.zeros_like(fg_mask)
        rr = int(r + 2.5 * ra)
        xx0, yy0 = max(0, xc - rr), max(0, yc - rr)
        xx1, yy1 = min(xc + rr, w), min(yc + rr, h)
        bg_mask[yy0:yy1, xx0:xx1] = (fg_mask < 1)[yy0:yy1, xx0:xx1]
    else:
        bg_mask = (fg_mask < 1).astype('uint8')

    image_angle = -head_angle
    bg_mask = np.bitwise_and(bg_mask, generate_angle_mask(w, h, xc, yc, image_angle, side=side))

    kernel = np.zeros((2 * ra + 1, 2 * ra + 1), dtype=np.uint8)
    bg_mask = cv2.erode(bg_mask, cv2.circle(kernel, (ra, ra), ra, 1, -1))

    cost, sint = np.cos(image_angle * np.pi / 180), np.sin(image_angle * np.pi / 180)
    if box is not None:
        Y, X = np.where(bg_mask > 0)
        if len(Y) == 0:
            return None
        dline = np.abs((X - xc) * sint - (Y - yc) * cost)
        dpoint = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2)
        argmin = np.lexsort((dpoint, dline))[0]
    else:
        assert pose_3d is not None, 'pose_3d is None'
        Y, X = get_mask_locs_with_stride(mask=bg_mask, stride=3)
        if len(Y) == 0:
            return None
        poses_3d = calib_func(X, Y, depth[(Y, X)])
        dpoint = np.linalg.norm(
            np.array(poses_3d).reshape(-1, 3) - np.array(pose_3d[:3]).reshape(-1, 3), axis=-1)
        argmin = int(np.argmin(dpoint))

    xr, yr = X[argmin], Y[argmin]
    x0_, y0_, x1_, y1_ = max(0, xr - ra), max(0, yr - ra), min(xr + ra, w), min(yr + ra, h)

    if node is not None and rgb is not None:
        def _preview():
            rgb_out = visualize_mask(rgb, bg_mask)
            rgb_out = show_box_on_rgb(rgb_out, (x0_, y0_, x1_, y1_), thick=2)
            log_data({'log_image': rgb_out})
        threading.Thread(target=_preview, daemon=True).start()

    return x0_, y0_, x1_, y1_


# ── Head→base calibration ─────────────────────────────────────────────────────
_calib_singleton: Optional[Head2BaseCalibration] = None
def _get_calib() -> Head2BaseCalibration:
    global _calib_singleton
    if _calib_singleton is None:
        _calib_singleton = Head2BaseCalibration()
    return _calib_singleton


def _head_calib_func_factory(robot_mode, head_state):
    def f(x, y, **kwargs):
        node = kwargs.get('node')
        return get3d(node=node, x=x, y=y)['pose']
    return f
