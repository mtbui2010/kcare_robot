"""Helpers for recognition.py. Pure refactor of the utility functions out of
the skill module; behaviour matches the original 1:1.

`recognition.py` keeps the registered skills (`find`, `find_grasp`,
`get_side_pose_3d`, ...) plus the two orchestrators (`_detect_objects`,
`_detect_grasps`) and pulls everything else from here. This module must NOT
import `recognition` back (no import cycle)."""

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np, threading, base64, json, urllib.request

from pyconnect.utils import run_parallel
from visionserve.utils import (
    show_box_on_rgb, Ixy2xyz, show_line_on_rgb,
    get_mask_locs_with_stride, calc_normalvector,
)

from robot_agent.skill_configs import (
    FIND_CONFIGS, ARM_CONFIGS, MOBILE_CONFIGS,
    STANDING_OBJ_NAMES, LYING_OBJ_NAMES, HAVING_HANDLE_OBJ_NAMES,
)
from robot_agent.state import current
from robot_agent.skills import log_data
from kcare_robot.skills.calibrattion import Head2BaseCalibration
from kcare_robot.skills.head import head_state as get_head_state
from kcare_robot.skills.pointcloud import get3d
from kcare_robot.skills.arm import get_wrist_angle, arm_pose
from robot_agent.utils import deg2quaternion
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
    """wrist_cam — used by `find_grasp`."""
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
# def is_lying(obj_name: str, normal_yz, horizon_yz) -> bool:
#     for el in STANDING_OBJ_NAMES:
#         if el in obj_name:
#             return False
#     for el in LYING_OBJ_NAMES:
#         if el in obj_name:
#             return True
#     return abs(np.dot(normal_yz, horizon_yz)) > 0.5


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
    # if any(el in obj_name for el in STANDING_OBJ_NAMES):
    #     return False
    # if any(el in obj_name for el in LYING_OBJ_NAMES):
    #     return True
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

        dot = np.dot(nv_yz, horizon_yz)

        return abs(dot) > 0.5
    except Exception:
        return False


# Decision constants for `_box_islying_pca`.
_PCA_VERTICAL_DOT = 0.5   # |long_axis·gravity| above this ⇒ axis is vertical ⇒ standing
_PCA_ELONGATION   = 1.3   # √(λ0/λ1) below this ⇒ shape not elongated ⇒ use extent ratio
_PCA_MAD_K        = 3.0   # drop pixels whose Z deviates > k·MAD from the median depth


def _gravity_in_cam(node, *, use_head: bool, horizon_yz) -> np.ndarray:
    """Down-direction (gravity) expressed in the camera frame, as a 3-vector.

    The head camera only pans (about the world-vertical) and tilts, so gravity
    stays in its y-z plane and `[0, *horizon_yz]` is exact. The wrist/arm camera,
    however, is rolled/flipped by the arm joints when grasping (e.g. a lying
    object sets `rx=-180, rz=±90`), so a pitch-only `horizon_yz` points the wrong
    way. For the arm we therefore rotate base-frame gravity into the camera using
    the *full* tool orientation:

        g_tool = R(rx,ry,rz)ᵀ · [0,0,-1]      (base-down expressed in the tool)
        g_cam  = [g_tool_y, g_tool_x, g_tool_z]   (fixed tool→cam mount: swap x,y)

    The x↔y swap is recovered from the existing pitch-only convention
    (`horizon_yz = [-cos(90+ry), -sin(90+ry)]`), which this reproduces exactly
    when rx=rz=0.
    """
    if use_head:
        return np.array([0.0, float(horizon_yz[0]), float(horizon_yz[1])], dtype='float32')
    rx, ry, rz = arm_pose(node=node)['pose'][3:]
    qx, qy, qz, qw = deg2quaternion(rx, ry, rz)
    # g_tool = Rᵀ·[0,0,-1] = -(third row of R)
    g_tool = -np.array([
        2 * (qx * qz - qy * qw),
        2 * (qy * qz + qx * qw),
        1 - 2 * (qx * qx + qy * qy),
    ], dtype='float32')
    return np.array([g_tool[1], g_tool[0], g_tool[2]], dtype='float32')


def _box_islying_pca(obj_name: str, box, depth, cam_params, gravity_cam, mask=None) -> bool:
    """Noise-robust islying estimate — drop-in for `_box_islying`.

    `_box_islying` fits a single plane and reads its surface normal, i.e. the
    *thinnest* (worst-conditioned) direction of the point set — exactly the axis
    that depth noise + flying pixels corrupt most, which is why it flips a lot.

    This version instead asks the well-conditioned question "is the object's long
    axis vertical or horizontal?":

      1. object pixels — from the grounded-sam segmentation `mask` when given
         (clean: excludes the supporting table), else the whole bbox + a depth
         foreground filter (which can leak the table around a thin object).
      2. MAD outlier rejection — remove flying pixels at depth discontinuities.
      3. PCA principal axis    — the largest-eigenvalue direction (best-conditioned)
         is the object's long axis; ∥ gravity ⇒ standing, ⟂ gravity ⇒ lying.
      4. extent-ratio tie-break — for blobby (non-elongated) shapes the principal
         axis is ill-defined, so compare vertical vs horizontal 3D extent instead.

    `gravity_cam` is the world-down direction as a 3-vector in the camera frame
    (see `_gravity_in_cam`), valid for any camera roll/flip — this is what makes
    it correct for the wrist camera, not just the head.
    Returns False on any failure, matching `_box_islying`.
    """
    try:
        if mask is not None:
            # Object segmentation pixels (no table contamination).
            ys, xs = np.where((mask > 0) & (depth > 0))
        else:
            x0, y0, x1, y1 = [int(v) for v in box]
            sub = depth[y0:y1, x0:x1]
            yy, xx = np.where(sub > 0)
            ys, xs = yy + y0, xx + x0
        if len(ys) < 10:
            return False
        Z = depth[ys, xs].astype('float32')

        # 1+2) depth foreground filter + MAD outlier rejection (mask already
        # isolates the object; this just drops flying pixels / depth holes).
        zc = float(np.median(Z))
        mad = float(np.median(np.abs(Z - zc))) + 1e-6
        keep = (np.abs(Z - zc) < _PCA_MAD_K * mad) & (Z < zc + 3.0 * mad)
        if keep.sum() < 10:
            return False
        ys, xs, Z = ys[keep], xs[keep], Z[keep]

        Iy = ys.astype('float32')
        Ix = xs.astype('float32')
        X3, Y3, Z3 = Ixy2xyz(Ix=Ix, Iy=Iy, Z=Z, cam_params=cam_params)
        pts = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=-1)

        # gravity (world-down) in the camera frame.
        g = np.asarray(gravity_cam, dtype='float32')
        g = g / (np.linalg.norm(g) + 1e-9)

        # 3) PCA — eigh returns ascending eigenvalues; take the largest as long axis.
        centered = pts - pts.mean(axis=0)
        cov = centered.T @ centered / len(centered)
        evals, evecs = np.linalg.eigh(cov)
        evals = evals[::-1]
        evecs = evecs[:, ::-1]
        principal = evecs[:, 0]
        elongation = float(np.sqrt(evals[0] / (evals[1] + 1e-9)))

        if elongation >= _PCA_ELONGATION:
            return abs(float(np.dot(principal, g))) < _PCA_VERTICAL_DOT

        # 4) tie-break: vertical extent vs horizontal extent in the gravity frame.
        vert = centered @ g
        h_v = float(vert.max() - vert.min())
        horiz = centered - np.outer(vert, g)
        h_h = float(np.linalg.norm(horiz, axis=1).max()) * 2.0
        return h_h > h_v
    except Exception:
        return False


# ── Cross-camera islying consensus ───────────────────────────────────────────
def _arm_horizon_yz(node) -> np.ndarray:
    """World-down in the wrist camera's (y,z) plane, from the arm pitch."""
    cam_angle_rad = (90 + arm_pose(node=node)['pose'][-2]) * np.pi / 180
    return np.array([-np.cos(cam_angle_rad), -np.sin(cam_angle_rad)])


def _object_det_select_configs(overrides: dict | None = None):
    """(det_configs, select_configs) for grounding-dino object detection."""
    configs = FIND_CONFIGS['detect_object']
    det = {k: v for k, v in configs.items()
           if k in ('model', 'min_size', 'max_size', 'text_threshold', 'box_threshold')}
    select = {k: v for k, v in configs.items()
              if k in ('near_point', 'target_distance', 'distance_sigma')}
    if overrides:
        for k in ('text_threshold', 'box_threshold'):
            det[k] = overrides.get(k, det.get(k))
    return det, select


def _cam_gravity(node, cam, use_head) -> np.ndarray:
    """World-down 3-vector in the camera frame, from the head tilt / wrist pitch."""
    cam_angle = abs(cam.head_state['current_ry']) if use_head else get_wrist_angle(node=node)
    cam_angle_rad = cam_angle * np.pi / 180
    horizon_yz = np.array([-np.cos(cam_angle_rad), -np.sin(cam_angle_rad)])
    return _gravity_in_cam(node, use_head=use_head, horizon_yz=horizon_yz)


def _none_res(camera):
    """Empty per-camera islying result (no frame / no detection)."""
    return {'camera': camera, 'vote': None, 'rgb': None, 'box': None}


def _mask_for_target(res, target, w, h):
    """The grounded-sam mask whose bbox best overlaps `target` → bool ndarray,
    or None (no masks → caller falls back to the bbox+depth foreground)."""
    masks = getattr(res, 'masks', None) or []
    if not masks:
        return None
    tx, ty, tw, th = target.bbox

    def iou(b):
        bx, by, bw, bh = b
        x1, y1 = max(tx, bx), max(ty, by)
        x2, y2 = min(tx + tw, bx + bw), min(ty + th, by + bh)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = tw * th + bw * bh - inter
        return inter / union if union > 0 else 0.0

    best = max(masks, key=lambda m: iou(m.bbox))
    if iou(best.bbox) <= 0:
        return None
    try:
        return best.to_ndarray(width=w, height=h)
    except Exception:
        return None


def _predict_detect(rgb, prompt, det_configs):
    """Object detection predict with a reliability fallback: if the configured
    model (e.g. grounded-sam, which can OOM the server's SAM decoder) fails,
    retry with grounding-dino (box only, no mask → islying uses the bbox)."""
    try:
        return _vs_client().predict(image=rgb, prompt=prompt, **det_configs)
    except Exception:
        if det_configs.get('model') in (None, 'grounding-dino'):
            raise
        return _vs_client().predict(image=rgb, prompt=prompt, **{**det_configs, 'model': 'grounding-dino'})


def _islying_one_cam(node, name, camera, det_configs, select_configs, *, min_conf=0.0):
    """islying for `name` from a single camera. Returns a result dict
    ``{camera, vote, rgb, box}``; `vote` is None when the camera cannot detect
    the object (out of FOV / no detection) — caller treats None as a non-vote,
    not as `standing`. `rgb`/`box` are kept for the debug visualisation."""
    out = _none_res(camera)
    try:
        select_target_object, _ = _vs_postprocess()
        cam = fetch_camera_data(node, camera)
        rgb, depth, cam_params = cam.rgb, cam.depth, cam.cam_params
        out['rgb'] = rgb
        h, w = rgb.shape[:2]

        res = _predict_detect(rgb, f'{name.strip()}.', det_configs)
        if min_conf > 0:
            res = res.filter_by_conf(min_conf=min_conf)
        if len(res.detections) == 0:
            return out
        target = select_target_object(res, cls=name, image_size=(w, h),
                                      depth_result=depth, intrinsics=cam_params, **select_configs)
        if target is None:
            return out

        box = _bbox_to_xyxy(target.bbox)
        out['box'] = box
        mask = _mask_for_target(res, target, w, h)   # grounded-sam mask (None → bbox fallback)
        gravity_cam = _cam_gravity(node, cam, 'head' in camera)
        out['vote'] = _box_islying_pca(name, box, depth, cam_params, gravity_cam, mask=mask)
        return out
    except Exception:
        return out


# ── islying debug visualisation ──────────────────────────────────────────────
_LY_COLOR = (255, 0, 0)   # lying    → red   (RGB)
_ST_COLOR = (0, 180, 0)   # standing → green


_GRASP_COLOR = (255, 0, 0)     # red (RGB)


def _vlabel(vote) -> str:
    return 'LYING' if vote else 'STAND' if vote is not None else 'n/a'


def _grasp_label(grasp) -> str:
    """Short 'w<width> <angle>deg' label from a grasp dict, or '' if unavailable."""
    gp = (grasp or {}).get('grasppose')
    if not gp or len(gp) < 5:
        return ''
    return f"w{gp[4]:.0f} {gp[3]:+.0f}deg"


def _draw_grasp(im, line, label=''):
    """Draw a parallel-jaw grasp on `im`: the contact line + perpendicular finger
    ticks at each end (the jaws) + an optional width/angle label."""
    gx0, gy0, gx1, gy1 = [int(v) for v in line]
    cv2.line(im, (gx0, gy0), (gx1, gy1), _GRASP_COLOR, 2)
    dx, dy = gx1 - gx0, gy1 - gy0
    norm = (dx * dx + dy * dy) ** 0.5 or 1.0
    px, py = -dy / norm, dx / norm          # unit perpendicular → finger direction
    t = 12
    for cx, cy in ((gx0, gy0), (gx1, gy1)):
        cv2.line(im, (int(cx - px * t), int(cy - py * t)),
                 (int(cx + px * t), int(cy + py * t)), _GRASP_COLOR, 2)
        cv2.circle(im, (cx, cy), 3, _GRASP_COLOR, -1)
    if label:
        cv2.putText(im, label, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(im, label, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _GRASP_COLOR, 2)


def _compose_islying_vis(head_res, arm_res, fused, vlm=None, vlm_view=None):
    """Side-by-side [head | arm (| vlm-crop)] image: each camera's selected box
    (red=lying, green=standing) + the per-camera, vlm and fused islying in the
    TOP-RIGHT corner. The VLM crop panel (what the VLM actually judged) is only
    appended when the VLM was consulted (cameras disagreed).

    Panels are normalised to a common height (the tallest frame, kept as-is) and
    concatenated horizontally."""
    panels = [r for r in (head_res, arm_res) if r.get('rgb') is not None]
    if not panels:
        return None
    target_h = max(int(r['rgb'].shape[0]) for r in panels)
    imgs = []
    for res in panels:
        im = np.ascontiguousarray(res['rgb'][:, :, :3]).copy()
        box, vote = res.get('box'), res.get('vote')
        if box is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(im, (x0, y0), (x1, y1), _LY_COLOR if vote else _ST_COLOR, 3)
        if res.get('grasp_line') is not None:        # grasppose on the arm panel
            _draw_grasp(im, res['grasp_line'], res.get('grasp_label', ''))
        h, w = im.shape[:2]
        if h != target_h:
            im = cv2.resize(im, (max(1, int(w * target_h / h)), target_h))
        # Camera label at the TOP-left (the bottom-left holds the UI "log_image" badge).
        tag = f"{res['camera']}: {_vlabel(vote)}"
        cv2.putText(im, tag, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(im, tag, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        imgs.append(im)

    # VLM panel — the exact crop the VLM judged, with a vote-coloured border.
    # Only when the VLM returned an actual vote (bool); skip for 'timeout'.
    if isinstance(vlm, bool) and vlm_view is not None:
        vrgb, vbox = vlm_view
        cx0, cy0, cx1, cy1 = _vlm_crop_box(vrgb, vbox)
        crop = np.ascontiguousarray(vrgb[cy0:cy1, cx0:cx1, :3]).copy()
        if crop.size:
            crop = cv2.copyMakeBorder(crop, 8, 8, 8, 8, cv2.BORDER_CONSTANT,
                                      value=(_LY_COLOR if vlm else _ST_COLOR))
            ch, cw = crop.shape[:2]
            crop = cv2.resize(crop, (max(1, int(cw * target_h / ch)), target_h))
            tag = f"vlm: {_vlabel(vlm)}"
            cv2.putText(crop, tag, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
            cv2.putText(crop, tag, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            imgs.append(crop)
    comp = np.concatenate(imgs, axis=1)

    lines = [f"head: {_vlabel(head_res['vote'])}",
             f"arm:  {_vlabel(arm_res['vote'])}"]
    if vlm is not None:
        lines.append(f"vlm:  {vlm if isinstance(vlm, str) else _vlabel(vlm)}")
    lines.append(f"FUSED: {_vlabel(fused)}")
    # Bottom-right corner (top-right is hidden by the UI Save/Clear buttons).
    x = comp.shape[1] - 260
    y0 = comp.shape[0] - 16 - (len(lines) - 1) * 32
    for i, t in enumerate(lines):
        y = y0 + i * 32
        cv2.putText(comp, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 6)
        cv2.putText(comp, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return comp


def _emit_islying_vis(node, name, head_res, arm_res, fused, vlm=None, vlm_view=None):
    """Push the side-by-side islying debug image to the execution panel (log_image)."""
    try:
        comp = _compose_islying_vis(head_res, arm_res, fused, vlm=vlm, vlm_view=vlm_view)
        if comp is not None:
            log_data({'log_image': comp})
    except Exception:
        pass


def _compose_multi_islying_vis(head_rgb, arm_rgb, panels):
    """Single shared [head | arm] image with EVERY object's box (coloured by its
    fused islying), the per-object grasp on the arm panel, and a per-object vote
    legend (head/arm/vlm/fused) in the bottom-right. Used when `find` runs on
    multiple objects so the log_image shows all of them, not just the last."""
    if not panels:
        return None
    frames = [im for im in (head_rgb, arm_rgb) if im is not None]
    if not frames:
        return None
    target_h = max(int(im.shape[0]) for im in frames)

    def draw(rgb, kind):
        im = np.ascontiguousarray(rgb[:, :, :3]).copy()
        for p in panels:
            box = p.get(f'{kind}_box')
            if box is not None:
                x0, y0, x1, y1 = [int(v) for v in box]
                cv2.rectangle(im, (x0, y0), (x1, y1), _LY_COLOR if p['fused'] else _ST_COLOR, 3)
                lbl = f"{p['name']}:{_vlabel(p['fused'])}"
                cv2.putText(im, lbl, (x0, max(22, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(im, lbl, (x0, max(22, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if kind == 'arm' and p.get('grasp_line') is not None:
                _draw_grasp(im, p['grasp_line'], p.get('grasp_label', ''))
        h, w = im.shape[:2]
        if h != target_h:
            im = cv2.resize(im, (max(1, int(w * target_h / h)), target_h))
        cv2.putText(im, kind, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(im, kind, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return im

    cols = [draw(im, k) for im, k in ((head_rgb, 'head'), (arm_rgb, 'arm')) if im is not None]
    comp = np.concatenate(cols, axis=1)

    def short(v):
        if isinstance(v, str):
            return 'TO' if v == 'timeout' else v[:2].upper()
        return 'na' if v is None else ('LY' if v else 'ST')
    lines = [f"{p['name']} h:{short(p.get('head_vote'))} a:{short(p.get('arm_vote'))} "
             f"v:{short(p.get('vlm'))} ->{short(p['fused'])}" for p in panels]
    x = comp.shape[1] - 430
    y0 = comp.shape[0] - 14 - (len(lines) - 1) * 28
    for i, t in enumerate(lines):
        y = y0 + i * 28
        cv2.putText(comp, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5)
        cv2.putText(comp, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return comp


def _emit_multi_islying_vis(node, head_rgb, arm_rgb, panels):
    """Compose + push the shared multi-object islying/grasp debug image."""
    try:
        comp = _compose_multi_islying_vis(head_rgb, arm_rgb, panels)
        if comp is not None:
            log_data({'log_image': comp})
    except Exception:
        pass


# ── VLM tie-breaker (local Ollama vision model) ──────────────────────────────
_VLM_ENABLED = True
_VLM_URL     = 'http://192.168.1.11:11434/api/chat'
_VLM_MODEL   = 'qwen2.5vl:3B'
_VLM_WAIT    = 3.0     # seconds the consensus waits for the VLM (on disagreement)
_VLM_HTTP_TO = 30.0    # the request itself runs longer (warms the model for next time)
# NOTE: this exact wording (the "Decide its orientation:" framing + the trailing
# "reason" field) is what makes the small 3B model answer reliably — shorter
# prompts or dropping "reason" flip its answers. Keep it verbatim.
_VLM_PROMPT  = ('The image shows a {obj}. Decide its orientation: is it LYING DOWN on its side '
                '(its long axis horizontal / fallen over) or STANDING UPRIGHT (its long axis '
                'vertical, resting on its base)? Respond ONLY JSON: '
                '{"islying": true_or_false, "reason": "<short>"}.')


def _vlm_crop_box(rgb, box):
    """Padded crop box for the VLM (proportional context). A *tight* crop of a
    flat lying object reads as 'standing' to the model; surrounding table context
    fixes it. Used for both the VLM query and its debug panel (kept consistent)."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    pad = max(28, int(0.35 * max(x1 - x0, y1 - y0)))
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def _vlm_islying(rgb, box, obj_name):
    """Ask the local VLM whether the boxed object is lying. None on any failure."""
    try:
        cx0, cy0, cx1, cy1 = _vlm_crop_box(rgb, box)
        crop = rgb[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            return None
        crop_bgr = cv2.cvtColor(np.ascontiguousarray(crop[:, :, :3]), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode('.jpg', crop_bgr)
        if not ok:
            return None
        b64 = base64.b64encode(buf).decode()
        body = json.dumps({
            'model': _VLM_MODEL, 'stream': False, 'format': 'json', 'keep_alive': '10m',
            'options': {'temperature': 0},
            'messages': [{'role': 'user', 'content': _VLM_PROMPT.replace('{obj}', obj_name.strip()),
                          'images': [b64]}],
        }).encode()
        req = urllib.request.Request(_VLM_URL, data=body, headers={'Content-Type': 'application/json'})
        r = json.load(urllib.request.urlopen(req, timeout=_VLM_HTTP_TO))
        v = json.loads(r['message']['content']).get('islying')
        return None if v is None else bool(v)
    except Exception:
        return None


def _islying_consensus(node, name, *, min_conf=0.0, seed=None, emit_vis=True, disagree_trust='arm') -> bool:
    """Fuse the lying/standing belief across head + arm cameras, with a VLM tiebreak.

      • both agree       → that value (VLM launched but ignored — no wait)
      • only one detects → use it (the other is out of FOV)
      • disagree         → wait up to `_VLM_WAIT`s for the VLM and take the majority
                           of (head, arm, vlm); VLM unavailable → trust the arm.

    The VLM runs speculatively in parallel from the start (on the `seed` view), so
    in the common agreeing case it adds ZERO wall-clock — we never await it. A
    single detection round (no retry). `seed = (camera, value, rgb, box)` reuses
    the primary detection as that camera's vote (only the other camera is run)."""
    det_configs, select_configs = _object_det_select_configs()

    def run_cam(cam):
        return _islying_one_cam(node, name, cam, det_configs, select_configs, min_conf=min_conf)

    # Fire-and-forget VLM (daemon thread): only awaited if the cameras disagree.
    vlm_holder = {'vote': None}
    vlm_done = threading.Event()
    if _VLM_ENABLED and seed is not None:
        srgb, sbox = seed[2], seed[3]
        def _vlm_task():
            try: vlm_holder['vote'] = _vlm_islying(srgb, sbox, name)
            finally: vlm_done.set()
        threading.Thread(target=_vlm_task, daemon=True).start()
    else:
        vlm_done.set()

    # Single round: head + arm (seed reuses the primary camera's vote).
    if seed is not None:
        scam, sval, srgb, sbox = seed
        seed_res = {'camera': scam, 'vote': sval, 'rgb': srgb, 'box': sbox}
        other_res = run_cam('arm' if scam == 'head' else 'head')
        head_res, arm_res = (seed_res, other_res) if scam == 'head' else (other_res, seed_res)
    else:
        head_res, arm_res = run_parallel(funcs=[lambda: run_cam('head'), lambda: run_cam('arm')])

    hv, av = head_res['vote'], arm_res['vote']
    votes = [v for v in (hv, av) if v is not None]
    vlm_vote = None
    vlm_disp = None        # None=not consulted, bool=vote, 'timeout'=consulted but no reply
    if not votes:
        fused = False
    elif len(votes) == 1:
        fused = votes[0]
    elif hv == av:
        fused = hv                                   # agree → ignore VLM (no wait)
    else:
        vlm_done.wait(timeout=_VLM_WAIT)             # disagree → consult the VLM
        vlm_vote = vlm_holder['vote']
        if vlm_vote is not None:
            fused = (int(bool(hv)) + int(bool(av)) + int(bool(vlm_vote))) >= 2   # majority of 3
            vlm_disp = vlm_vote
        else:
            fused = bool(av) if disagree_trust == 'arm' else bool(hv)             # VLM down → trust chosen cam
            vlm_disp = 'timeout' if (_VLM_ENABLED and seed is not None) else None

    if emit_vis:
        vlm_view = (seed[2], seed[3]) if (seed is not None and vlm_vote is not None) else None
        _emit_islying_vis(node, name, head_res, arm_res, fused, vlm=vlm_disp, vlm_view=vlm_view)
    return fused


def _reconcile_grasp_islying(node, name, islying_arm, islying_kw, _missing, *, seed=None, disagree_trust='arm') -> bool:
    """`_detect_grasps` islying decision: trust the caller's value when it
    matches the arm's own estimate, otherwise fall back to a full cross-camera
    consensus. With no caller value, go straight to consensus. `seed` is the arm
    panel ``(camera, vote, rgb, box)`` reused for the consensus / debug viz."""
    if islying_kw is not _missing and bool(islying_kw) == bool(islying_arm):
        # agree → trust it, but still show the arm panel for debugging.
        arm_res = ({'camera': seed[0], 'vote': seed[1], 'rgb': seed[2], 'box': seed[3]}
                   if seed is not None else _none_res('arm'))
        _emit_islying_vis(node, name, _none_res('head'), arm_res, bool(islying_arm))
        return bool(islying_arm)
    return _islying_consensus(node, name, seed=seed, disagree_trust=disagree_trust)


# ── Parallel head-detect ∥ arm-grasp ∥ VLM (find's fused path) ────────────────
def _spawn_vlm(rgb, box, obj_name):
    """Launch the VLM islying query in a daemon thread → (holder, done_event)."""
    holder = {'vote': None}
    done = threading.Event()
    if not _VLM_ENABLED:
        done.set()
        return holder, done

    def task():
        try:
            holder['vote'] = _vlm_islying(rgb, box, obj_name)
        finally:
            done.set()
    threading.Thread(target=task, daemon=True).start()
    return holder, done


def _arm_grasp_branch(node, name, *, keep_orientation=False, cam=None):
    """Wrist-camera grasp detection (grasp-gd) → {grasp, vote, rgb, box, camera}
    or None when the arm can't see the object (best-effort during `find`).
    Pass a pre-fetched `cam` to share one arm frame across multiple objects."""
    try:
        select_target_object, select_target_grasp = _vs_postprocess()
        if cam is None:
            cam = _fetch_arm(node)
        rgb, depth, cam_params = cam.rgb, cam.depth, cam.cam_params
        h, w = rgb.shape[:2]
        configs = FIND_CONFIGS['detect_grasp']
        det_configs    = {k: v for k, v in configs.items() if k in ('model', 'min_size', 'max_size', 'text_threshold', 'box_theshold')}
        select_configs = {k: v for k, v in configs.items() if k in ('near_point', 'target_distance', 'distance_sigma')}
        grasp_configs  = {k: v for k, v in configs.items() if k in ('gripper_min', 'gripper_max')}
        res = _vs_client().predict(image=rgb, prompt=_prompt([name]), **det_configs)
        if len(res.detections) == 0:
            return None
        obj = select_target_object(res, cls=name, image_size=(w, h),
                                   depth_result=depth, intrinsics=cam_params, **select_configs)
        if obj is None:
            return None
        x0, y0, dx, dy = [int(el) for el in obj.bbox]
        arm_box = [x0, y0, x0 + dx, y0 + dy]
        grasp, obj_mask = _grasp_from_box(node, obj, rgb, depth, cam_params, grasp_configs,
                                          keep_orientation=keep_orientation,
                                          select_target_grasp=select_target_grasp)
        gravity = _gravity_in_cam(node, use_head=False, horizon_yz=_arm_horizon_yz(node))
        vote = _box_islying_pca(name, arm_box, depth, cam_params, gravity, mask=obj_mask)
        return {'grasp': grasp, 'vote': vote, 'rgb': rgb, 'box': arm_box, 'camera': 'arm'}
    except Exception:
        return None


def _fused_head_arm(node, name, *, rgb, depth, cam_params, cam, use_head,
                    det_configs, select_configs, min_conf, keep_orientation=False,
                    emit_vis=True, disagree_trust='arm', arm_cam=None):
    """Run HEAD detection ∥ ARM grasp-gd ∥ speculative VLM concurrently, then fuse
    islying. Returns the merged per-object dict (with `grasp` from the arm when
    visible) — including a `panel` for shared multi-object visualisation — or None
    when HEAD detection fails. Pass `arm_cam` to share one arm frame across objects;
    `emit_vis=False` suppresses the per-object log_image (caller composes instead)."""
    h, w = rgb.shape[:2]
    select_target_object, _ = _vs_postprocess()
    holders = {'head': None, 'arm': None}
    vlm = {'holder': {'vote': None}, 'done': threading.Event(), 'started': False}

    def head_branch():
        try:
            res = _predict_detect(rgb, f'{name.strip()}.', det_configs)
            if min_conf > 0:
                res = res.filter_by_conf(min_conf=min_conf)
            if len(res.detections) == 0:
                return
            target = select_target_object(res, cls=name, image_size=(w, h),
                                          depth_result=depth, intrinsics=cam_params, **select_configs)
            if target is None:
                return
            box = _bbox_to_xyxy(target.bbox)
            # speculative VLM as soon as the head box is known (overlaps the arm branch).
            vlm['holder'], vlm['done'] = _spawn_vlm(rgb, box, name)
            vlm['started'] = True
            mask = _mask_for_target(res, target, w, h)
            islying_head = _box_islying_pca(name, box, depth, cam_params,
                                            _cam_gravity(node, cam, use_head), mask=mask)
            box_depths = _box_depths(depth, box)
            pose = _object_pose_3d(node, box, box_depths, depth, cam_params, use_head=use_head)
            holders['head'] = {'res': res, 'target': target, 'box': box, 'pose': pose,
                               'box_depths': box_depths, 'vote': islying_head}
        except Exception:
            holders['head'] = None

    def arm_branch():
        holders['arm'] = _arm_grasp_branch(node, name, keep_orientation=keep_orientation, cam=arm_cam)

    th = threading.Thread(target=head_branch)
    ta = threading.Thread(target=arm_branch)
    th.start(); ta.start(); th.join(); ta.join()
    if not vlm['started']:
        vlm['done'].set()

    head = holders['head']
    if head is None:
        return None
    arm = holders['arm']
    hv = head['vote']
    av = arm['vote'] if arm else None

    vlm_vote = None
    vlm_disp = None        # None=not consulted, bool=vote, 'timeout'=consulted but no reply
    if av is None or av == hv:
        islying = hv if hv is not None else (av if av is not None else False)
    else:
        vlm['done'].wait(timeout=_VLM_WAIT)
        vlm_vote = vlm['holder']['vote']
        if vlm_vote is not None:
            islying = (int(bool(hv)) + int(bool(av)) + int(bool(vlm_vote))) >= 2   # majority of 3
            vlm_disp = vlm_vote
        else:
            islying = bool(av) if disagree_trust == 'arm' else bool(hv)            # VLM down → trust chosen cam
            vlm_disp = 'timeout' if vlm['started'] else None

    if emit_vis:
        head_panel = {'camera': 'head', 'vote': hv, 'rgb': rgb, 'box': head['box']}
        arm_panel = ({'camera': 'arm', 'vote': av, 'rgb': arm['rgb'], 'box': arm['box'],
                      'grasp_line': arm['grasp'].get('line'), 'grasp_label': _grasp_label(arm['grasp'])}
                     if arm else _none_res('arm'))
        vlm_view = (rgb, head['box']) if vlm_vote is not None else None
        _emit_islying_vis(node, name, head_panel, arm_panel, islying, vlm=vlm_disp, vlm_view=vlm_view)

    head['islying'] = islying
    head['grasp'] = arm['grasp'] if arm else None
    # Per-object annotations for the shared multi-object composite (drawn on the
    # single head + arm frames by `_compose_multi_islying_vis`).
    head['panel'] = {
        'name': name, 'fused': islying,
        'head_box': head['box'], 'head_vote': hv,
        'arm_box': arm['box'] if arm else None, 'arm_vote': av,
        'grasp_line': arm['grasp'].get('line') if arm else None,
        'grasp_label': _grasp_label(arm['grasp']) if arm else '',
        'vlm': vlm_disp,
    }
    return head


# ── Object pose / approach / grasp builders (pulled out of recognition.py) ────
def _object_pose_3d(node, box, box_depths, depth, cam_params, *, use_head):
    """Box centre → 3D pose. Head: base frame via get3d. Wrist: camera frame (m)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if use_head:
        return get3d(node=node, points=[[cx, cy, box_depths['obj_median']]])['pose'][0, :].tolist()
    Z = _sample_depth(depth, cx, cy)
    px, py, pz = Ixy2xyz(Ix=cx, Iy=cy, Z=Z, cam_params=cam_params)
    return [float(px) / 1000., float(py) / 1000., float(pz) / 1000.]


def _build_approach_pose(pose, islying, robot_mode) -> dict:
    """Approach geometry for a found object (lying vs standing branch)."""
    ap = list(pose)
    lift_to = ap[-1] + 0.1
    if islying:
        ap[-1] += 0.2
        ap[1] -= 0.1 if robot_mode == 'right' else -0.1
        ap += [-180., 0., 90. if robot_mode == 'right' else -90.]
        approach_lying = True
    else:
        ap[-1] += 0.1
        ry, dd = 30, 0.25
        sint, cost = np.sin(ry * np.pi / 180), np.cos(ry * np.pi / 180)
        ap[0] += dd * sint
        ap[1] += (-dd if robot_mode == 'right' else dd) * cost
        ap += [-180, -75, 100 + ry if robot_mode == 'right' else -100 - ry]
        approach_lying = False
    mforward = ap[0] - np.clip(ap[0], 0.15, 0.4)
    mforward = 0 if abs(mforward) < 0.1 else mforward
    ap[0] -= mforward
    approach_pose = {k: v for k, v in zip(['x', 'y', 'z', 'rx', 'ry', 'rz'], ap)}
    return {
        'approach_pose':  approach_pose,
        'lift_to':        lift_to,
        'mforward':       mforward,
        'base_rotate':    60 if robot_mode == 'right' else -60,
        'approach_lying': approach_lying,
    }


def _grasp_from_box(node, obj, rgb, depth, cam_params, grasp_configs, *,
                    keep_orientation, select_target_grasp):
    """grasp-gd on `obj.bbox` → ``(grasp_fields, obj_mask)``. `obj_mask` is the
    grasped object's segmentation (for islying); None if unavailable."""
    h, w = rgb.shape[:2]
    gs = _vs_client().predict("grasp", rgb, box=obj.bbox, **grasp_configs)
    g, garg = select_target_grasp(gs.grasps, return_index=True)
    _log_annotated(gs, rgb, target_grasp=g)
    assert g is not None, "failed to select target grasp"

    obj_mask = None
    try:
        obj_mask = gs.masks[garg].to_ndarray(width=w, height=h)
    except Exception:
        obj_mask = None
    depth_bound = _get_bound_depth(depth, obj_mask) if obj_mask is not None else None

    (gx0, gy0), (gx1, gy1) = g.contacts()
    depth_value = _sample_depth(depth, g.x, g.y)
    grasppose, depth_final = _line_to_grasppose(
        node, [gx0, gy0, gx1, gy1], cam_params=cam_params,
        depths={'obj_median': depth_value, 'bound': depth_bound},
        keep_orientation=keep_orientation)
    box = [float(min(gx0, gx1)), float(min(gy0, gy1)),
           float(max(gx0, gx1)), float(max(gy0, gy1))]
    return {
        'duration_ms': gs.duration_ms,
        'device':      gs.device,
        'grasppose':   grasppose,
        'pose_3d':     [grasppose[0], grasppose[1], depth_final],
        'box':         box,
        'line':        [float(gx0), float(gy0), float(gx1), float(gy1)],   # jaw line (pixels)
        'score':       float(getattr(g, 'quality', 0.0)),
        'depths':      {'obj_median': depth_value, 'bound': None, 'final': depth_final},
    }, obj_mask


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
