"""Helpers for recognition.py. Pure refactor of the utility functions out of
the skill module; behaviour matches the original 1:1.

`recognition.py` keeps the registered skills (`find`, `find_grasp`,
`get_side_pose_3d`, ...) plus the single detection orchestrator
(`_detect_objects`) and pulls everything else from here. This module must NOT
import `recognition` back (no import cycle)."""

from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np, threading, json, time, cv2

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
from kcare_robot.skills.pointcloud import get3d, get3d_arm
from kcare_robot.skills.arm import get_wrist_angle, arm_pose
from robot_agent.utils import deg2quaternion
import cv2

def _normalize_orientation(theta):
    return 90.0 - (90.0 - theta) % 180.0


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
    """Fetch data from the head camera used by `find`."""
    a = node.agents

    rgb, depth, cam_params, head_state = run_parallel(funcs=[
        lambda: a["head_rgb"].get(),
        lambda: a["head_depth"].get(),
        lambda: a["head_cam_params"].get(),
        lambda: get_head_state(node=node),
    ])

    _require(rgb, "head_rgb")
    _require(depth, "head_depth")
    _require(cam_params, "head_cam_params")


    if rgb["im"].shape[:2] != depth["im"].shape[:2]:
        depth["im"] = cv2.resize(depth["im"], dsize=rgb["im"].shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

    return CameraData(rgb=rgb["im"], depth=depth["im"],  cam_params=cam_params["cam_params"], head_state=head_state)


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


def _stream_detection_dataset(rgb=None, depth=None, results=None, annotated=None, tag=''):
    """Push the vision inputs/outputs to the dashboard through ``log_data`` so
    the browser can save them (frontend log target — backend writes nothing).

    rgb / annotated → base64 PNG; depth → zlib-compressed uint16 LE + shape (the
    dashboard already decodes this shape for its depth stream); results → JSON.
    Best-effort — never raises into the skill."""
    try:
        import base64, zlib, cv2, time
        payload = {'tag': tag or '', 'ts': time.time()}

        def _png_b64(img, rgb_order=True):
            a = np.asarray(img)
            if a.ndim == 3 and a.shape[2] == 3 and rgb_order:
                a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode('.png', a)
            return base64.b64encode(buf.tobytes()).decode() if ok else None

        if rgb is not None:
            payload['rgb'] = _png_b64(rgb)
        if annotated is not None:
            payload['annotated'] = _png_b64(annotated)
        if depth is not None:
            d16 = np.asarray(depth)
            payload['depth_h'], payload['depth_w'] = int(d16.shape[0]), int(d16.shape[1])
            payload['depth'] = base64.b64encode(
                zlib.compress(np.ascontiguousarray(d16.astype('<u2')).tobytes())).decode()
        if results is not None:
            payload['results'] = results
        log_data({'dataset': payload})
    except Exception as e:
        print(f'[recognition] dataset stream failed: {e}')


def save_detection_dataset(rgb=None, depth=None, results=None, annotated=None, tag=''):
    """Persist vision detection inputs/outputs for the active log target:

    - backend target (a per-run capture dir is set) → write files to disk
      (rgb.png / depth.npy / results.json / annotated.png);
    - frontend target (streaming on) → push them over the agent WebSocket so
      the dashboard saves them under its own ``robotapp_logs`` folder.

    Best-effort — never raises into the skill."""
    try:
        from robot_agent.skills import stream_dataset
        if stream_dataset():
            _stream_detection_dataset(rgb=rgb, depth=depth, results=results,
                                      annotated=annotated, tag=tag)
    except Exception:
        pass
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

    dilated_large = cv2.dilate(mask, np.ones((25, 25), np.uint8))
    dilated_small = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    boundary = (dilated_large > 0) & (dilated_small == 0) & (depth > 0)

    boundary_depths = depth[boundary ]

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


# Decision constants for the islying estimator (`_islying_estimate`).
_PCA_VERTICAL_DOT   = 0.5    # |long_axis·gravity| above this ⇒ axis vertical ⇒ standing
_PCA_ELONGATION     = 1.3    # √(λ0/λ1) below ⇒ shape not elongated (legacy/extent fallback)
_PCA_MAD_K          = 3.0    # drop pixels whose Z deviates > k·MAD from the median depth
_PCA_LONG_LYING_DEG = 50.0   # angle(long-axis, gravity) above ⇒ long-axis cue says lying
_PCA_NORM_LYING_DEG = 40.0   # angle(normal,   gravity) below ⇒ normal cue says lying
_PCA_ANISO_MIN      = 1.15   # eigen-ratio below ⇒ that axis ill-conditioned (cue weight→0)
_PCA_SCORE_MARGIN   = 0.34   # |fused score| below ⇒ cues weak/disagree ⇒ defer to prior/extent


def _gravity_in_cam(node, *, use_head: bool, horizon_yz) -> np.ndarray:
    """Down-direction (gravity) expressed in the camera frame, as a 3-vector.

    The head camera only pans (about the world-vertical) and tilts, so gravity
    stays in its y-z plane and `[0, *horizon_yz]` is exact. The wrist/arm camera,
    however, is rolled/flipped by the arm joints when grasping (e.g. a lying
    object sets `rx=-180, rz=±90`), so a pitch-only `horizon_yz` points the wrong
    way. For the arm we therefore rotate base-frame gravity into the camera using
    the *full* tool orientation:

        g_tool = R(rx,ry,rz)ᵀ · [0,0,-1]            (base-down expressed in the tool)
        g_cam  = [-g_tool_y, g_tool_x, -g_tool_z]   (fixed tool→cam mount)

    The mount maps tool→camera. It was validated empirically against the object
    surface-normal across 2 arm poses × 5 objects: with this transform a flat
    object lying on a horizontal table has its measured surface normal within
    ~2–5° of `g_cam` (∥ gravity, as physics requires); the earlier
    `[g_tool_y, g_tool_x, g_tool_z]` swap was ~43° off (X and Z signs flipped),
    which made every lying object misclassify as standing.
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
    return np.array([-g_tool[1], g_tool[0], -g_tool[2]], dtype='float32')


def _class_prior(obj_name: str) -> int:
    """Object-class orientation prior from the configured name lists:
    +1 ⇒ usually lying, -1 ⇒ usually standing, 0 ⇒ unknown. Used only as a
    tie-break for near-isotropic shapes (e.g. a squat cup) where the geometry
    cannot tell lying from standing."""
    n = (obj_name or '').lower()
    if any(el in n for el in LYING_OBJ_NAMES):
        return 1
    if any(el in n for el in STANDING_OBJ_NAMES):
        return -1
    return 0


def _islying_estimate(obj_name: str, box, depth, cam_params, gravity_cam, mask=None):
    """Robust lying/standing estimate from a single (arm) view.

    Returns ``(lying: bool, confidence: float)`` with confidence in [0, 1].

    Fuses two COMPLEMENTARY gravity-relative PCA cues, each weighted by how
    well-conditioned its own axis is, and defers to the object-class prior (then
    the 3D extent ratio) only when the geometry is ambiguous:

      • long-axis·gravity — largest-eigenvalue axis ∥ gravity ⇒ standing,
        ⟂ gravity ⇒ lying. Reliable when the shape is ELONGATED.
      • normal·gravity    — smallest-eigenvalue axis (dominant-surface normal)
        ∥ gravity ⇒ lying (flat object face-up), ⟂ ⇒ standing. Reliable when the
        shape is PLANAR.
      • weights `w_long ∝ elongation`, `w_norm ∝ planarity`: every object has at
        least one well-conditioned axis, so the trustworthy cue dominates.
      • class prior / extent ratio — only when both cues are weak (near-isotropic)
        or they disagree (|score| small).

    Validated on 2 arm poses × 5 objects (water/juice bottle, cup, phone, remote):
    each angle cue alone separated lying vs standing with a ~77° margin once the
    `gravity_cam` mount was corrected (see `_gravity_in_cam`). Requires a CORRECT
    `gravity_cam`; returns (False, 0.0) on any failure.
    """
    try:
        if mask is not None:
            ys, xs = np.where((mask > 0) & (depth > 0))   # clean object pixels (no table)
        else:
            x0, y0, x1, y1 = [int(v) for v in box]
            sub = depth[y0:y1, x0:x1]
            yy, xx = np.where(sub > 0)
            ys, xs = yy + y0, xx + x0
        if len(ys) < 10:
            return False, 0.0
        Z = depth[ys, xs].astype('float32')

        # depth foreground filter + MAD outlier rejection (drop flying pixels).
        zc = float(np.median(Z))
        mad = float(np.median(np.abs(Z - zc))) + 1e-6
        keep = (np.abs(Z - zc) < _PCA_MAD_K * mad) & (Z < zc + 3.0 * mad)
        if keep.sum() < 10:
            return False, 0.0
        ys, xs, Z = ys[keep], xs[keep], Z[keep]

        X3, Y3, Z3 = Ixy2xyz(Ix=xs.astype('float32'), Iy=ys.astype('float32'),
                             Z=Z, cam_params=cam_params)
        pts = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=-1)

        g = np.asarray(gravity_cam, dtype='float32')
        g = g / (np.linalg.norm(g) + 1e-9)

        centered = pts - pts.mean(axis=0)
        cov = centered.T @ centered / len(centered)
        evals, evecs = np.linalg.eigh(cov)           # ascending
        evals = evals[::-1]; evecs = evecs[:, ::-1]  # → descending (λ0≥λ1≥λ2)
        long_axis = evecs[:, 0]                       # largest variance
        normal    = evecs[:, 2]                       # smallest variance (surface normal)
        elong = float(np.sqrt(evals[0] / (evals[1] + 1e-9)))   # how elongated
        flat  = float(np.sqrt(evals[1] / (evals[2] + 1e-9)))   # how planar

        # gravity-frame 3D extents (for the extent-ratio last resort).
        vert = centered @ g
        h_v = float(vert.max() - vert.min())
        horiz = centered - np.outer(vert, g)
        h_h = float(np.linalg.norm(horiz, axis=1).max()) * 2.0

        prior = _class_prior(obj_name)

        # Two cues vote: +1 ⇒ lying, -1 ⇒ standing.
        deg = 180.0 / np.pi
        th_long = float(np.arccos(min(1.0, abs(float(np.dot(long_axis, g)))))) * deg
        th_norm = float(np.arccos(min(1.0, abs(float(np.dot(normal, g)))))) * deg
        s_long = 1.0 if th_long > _PCA_LONG_LYING_DEG else -1.0
        s_norm = 1.0 if th_norm < _PCA_NORM_LYING_DEG else -1.0
        w_long = max(0.0, elong - _PCA_ANISO_MIN)
        w_norm = max(0.0, flat - _PCA_ANISO_MIN)
        wsum = w_long + w_norm

        if wsum < 1e-6:
            # Near-isotropic blob — geometry is blind. Trust the class prior, else
            # the extent ratio (low confidence either way).
            if prior != 0:
                return prior > 0, 0.5
            return (h_h > h_v), 0.3

        score = (w_long * s_long + w_norm * s_norm) / wsum     # in [-1, 1]
        if abs(score) < _PCA_SCORE_MARGIN:
            # Cues weak / disagree → class prior breaks the tie, else extent ratio.
            if prior != 0:
                return prior > 0, 0.4
            return (h_h > h_v), 0.35
        return score > 0, float(min(1.0, abs(score)))
    except Exception:
        return False, 0.0


def _box_islying_pca(obj_name: str, box, depth, cam_params, gravity_cam, mask=None) -> bool:
    """Backward-compatible bool wrapper around `_islying_estimate` (drops the
    confidence). Existing callers expect a plain bool."""
    if obj_name in FIND_CONFIGS['lying_obj_names']:
        return True
    if obj_name in FIND_CONFIGS['standing_obj_names']:
        return False
    return _islying_estimate(obj_name, box, depth, cam_params, gravity_cam, mask=mask)[0]


def _object_det_select_configs(overrides: dict | None = None, *, cfg_key: str = 'detect_object'):
    """(det_configs, select_configs) for grounding-dino object detection.

    `cfg_key` chooses the profile: 'detect_object' for plain detection (head, or a
    coarse wrist locate) and 'detect_grasp' for the wrist grasp path (a closer
    `target_distance`, so the near object is selected)."""
    configs = FIND_CONFIGS[cfg_key]
    det = {k: v for k, v in configs.items()
           if k in ('model', 'min_size', 'max_size', 'text_threshold', 'box_threshold')}
    select = {k: v for k, v in configs.items()
              if k in ('near_point', 'target_distance', 'distance_sigma')}
    if overrides:
        for k in ('text_threshold', 'box_threshold'):
            det[k] = overrides.get(k, det.get(k))
    return det, select


def _grasp_configs() -> dict:
    """Gripper width bounds for grasp-gd, from the `detect_grasp` profile."""
    configs = FIND_CONFIGS['detect_grasp']
    return {k: v for k, v in configs.items() if k in ('gripper_min', 'gripper_max')}


def _cam_gravity(node, cam, use_head) -> np.ndarray:
    """World-down 3-vector in the camera frame, from the head tilt / wrist pitch."""
    cam_angle = abs(cam.head_state['current_ry']) if use_head else get_wrist_angle(node=node)
    cam_angle_rad = cam_angle * np.pi / 180
    horizon_yz = np.array([-np.cos(cam_angle_rad), -np.sin(cam_angle_rad)])
    return _gravity_in_cam(node, use_head=use_head, horizon_yz=horizon_yz)


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
    # try:
    return _vs_client().predict(image=rgb, prompt=prompt, **det_configs)
    # except Exception as e:
    #     if det_configs.get('model') in (None, 'grounding-dino'):
    #         raise
    #     return _vs_client().predict(image=rgb, prompt=prompt, **{**det_configs, 'model': 'grounding-dino'})


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


def _compose_detection_vis(rgb, camera, panels):
    """One annotated frame for a single camera's detection: every object's box
    (red=lying, green=standing) with a short name + score, the grasp overlay on
    the wrist camera, and a per-object islying legend in the bottom-right."""
    if not panels:
        return None
    im = np.ascontiguousarray(rgb[:, :, :3]).copy()
    for p in panels:
        box = p.get('box')
        if box is not None:
            x0, y0, x1, y1 = [int(v) for v in box]
            cv2.rectangle(im, (x0, y0), (x1, y1), _LY_COLOR if p['vote'] else _ST_COLOR, 3)
            lbl = f"{p['name'][:2]} {p.get('score', 0.0):.2f}"   # short name + score (box colour = islying)
            cv2.putText(im, lbl, (x0, max(22, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(im, lbl, (x0, max(22, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if p.get('grasp_line') is not None:        # grasppose, wrist camera only
            _draw_grasp(im, p['grasp_line'], p.get('grasp_label', ''))

    cv2.putText(im, camera, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(im, camera, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    lines = [f"{p['name']} {_vlabel(p['vote'])}" for p in panels]
    x = im.shape[1] - 260
    y0 = im.shape[0] - 14 - (len(lines) - 1) * 28
    for i, t in enumerate(lines):
        y = y0 + i * 28
        cv2.putText(im, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5)
        cv2.putText(im, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return im


def _emit_detection_vis(node, rgb, camera, panels):
    """Compose + push the single-camera detection debug image (log_image).

    Returns the composed frame so callers can also persist it (dataset capture)
    without re-drawing; ``None`` when nothing was drawn."""
    try:
        comp = _compose_detection_vis(rgb, camera, panels)
        if comp is not None:
            log_data({'log_image': comp})
        return comp
    except Exception:
        return None


# ── Object pose / approach / grasp builders (pulled out of recognition.py) ────
def _object_pose_3d(node, box, box_depths, depth, cam_params, *, use_head):
    """Box centre → 3D pose. Head: base frame via get3d. Wrist: camera frame (m)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    func = get3d if use_head else get3d_arm
    return func(node=node, points=[[cx, cy, box_depths['obj_median']]])['pose'][0, :].tolist()
    # Z = _sample_depth(depth, cx, cy)
    # px, py, pz = Ixy2xyz(Ix=cx, Iy=cy, Z=Z, cam_params=cam_params)
    # return [float(px) / 1000., float(py) / 1000., float(pz) / 1000.]


def _build_approach_pose(pose, islying, robot_mode) -> dict:
    """Approach geometry for a found object (lying vs standing branch)."""
    ap = list(pose)
    lift_to = ap[-1] + 0.05
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
    mforward = ap[0] - np.clip(ap[0], 0.15, 0.7 if islying else 0.4)
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

    # pose_3d = get3d_arm(node=node,points=[[(gx0+gx1)/2,(gy0+gy1)/2]])['pose'].tolist()[0]
    pose_3d = grasppose[:3]
    return {
        'duration_ms': gs.duration_ms,
        'device':      gs.device,
        'grasppose':   grasppose,
        'pose_3d':     pose_3d,
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
        depth_value = v0 + min(((1 - alpha) * v0 + alpha * v1) - v0 + depth_tune, 50)

    x0, y0, z0 = Ixy2xyz(Ix=ix0, Iy=iy0, Z=depth_value, cam_params=cam_params)
    x1, y1, z1 = Ixy2xyz(Ix=ix1, Iy=iy1, Z=depth_value, cam_params=cam_params)
    x = (x0 + x1) / 2 / 1000. + ARM_CONFIGS['wrist_cam_offset'][0]
    y = (y0 + y1) / 2 / 1000. + ARM_CONFIGS['wrist_cam_offset'][1]

    width = np.linalg.norm([x0 - x1, y0 - y1, z0 - z1])  # mm
    rz = np.arctan2(y1 - y0, x1 - x0) * 180 / np.pi
    if not keep_orientation:
        rz = rz if to_pick_lying_obj else 0
    rz = _normalize_orientation(rz)

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
