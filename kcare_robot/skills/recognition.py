from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np, time, cv2, os, threading

from pyconnect.utils import dict2str, data_info, Timer, run_parallel, update_dict
from pyinterfaces.utils import (
    visualize_pred, show_box_on_rgb, Ixy2xyz, show_line_on_rgb,
    get_valid_depth_locs, get_mask_locs_with_stride, calc_normalvector,
)

from robot_agent.skill_configs import (
    FIND_CONFIGS, ARM_CONFIGS, MOBILE_CONFIGS, LIFT_CONFIGS, ENV,
    STANDING_OBJ_NAMES, LYING_OBJ_NAMES, HAVING_HANDLE_OBJ_NAMES,
)
from robot_agent.state import current
from robot_agent.utils import exception_handler, describe_object
from robot_agent.skills import log_data
from kcare_robot.skills.calibrattion import Head2BaseCalibration
from kcare_robot.skills.head import moveh, get_robot_mode, head_state as get_head_state
from kcare_robot.skills.mobile import move
from kcare_robot.skills.pointcloud import get3d
from kcare_robot.skills.arm import get_wrist_angle, arm_pose


# ── Agent names (centralised so renaming is one-touch) ───────────────────────

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


def _fetch_wrist_raw(node) -> CameraData:
    """Combined 'wrist_raw' agent returns {rgb, depth, cam_params} in one shot."""
    a = node.agents
    data, head_state = run_parallel(funcs=[
        lambda: a['wrist_raw'].get(),
        lambda: get_head_state(node=node),
    ])
    _require(data, 'wrist_raw')
    return CameraData(
        rgb=data['rgb'], depth=data['depth'],
        cam_params=data['cam_params'], head_state=head_state,
    )


CAMERA_FETCHERS: dict[str, Callable] = {
    'head':       _fetch_head,
    'arm':        _fetch_arm,
    'wrist_raw':  _fetch_wrist_raw,
}


def fetch_camera_data(node, camera: str) -> CameraData:
    fetcher = CAMERA_FETCHERS.get(camera)
    if fetcher is None:
        raise Exception(f"Unknown camera '{camera}'. Available: {list(CAMERA_FETCHERS)}")
    return fetcher(node)


# ── Visualisation / logging ──────────────────────────────────────────────────
def publish_image(node, rgb):
    node.agents['screen_log'].send({'rgb': rgb})


def log_result_data(node, data):
    node.agents['screen_log'].log_msg(data)


# ── Workspace + scoring helpers ──────────────────────────────────────────────
def is_inside_workspace_box(x, y, z, workspace):
    """Vectorised workspace membership: returns bool (scalar) or bool array.

    The arm reach is extended by MOBILE_CONFIGS['max_shift'] along x to account
    for the base nudging forward/back before the arm motion. Matches the
    carerobotapp pattern.
    """
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


def calc_obj_distance(obj_loc_3d, use_head_cam=True):
    """Distance from robot reach origin. For head-cam, "reachable" x/z are
    clipped into arm range so a point already within reach scores distance 0
    along that axis."""
    X = np.array(obj_loc_3d).reshape(-1, 3)
    if use_head_cam:
        mforward = MOBILE_CONFIGS.get('max_shift', 0) or 0
        x_range = (ARM_CONFIGS['range']['x'][0] - mforward,
                   ARM_CONFIGS['range']['x'][1] + mforward)
        z_range = LIFT_CONFIGS['range']
        X[:, 0] -= np.clip(X[:, 0], x_range[0], x_range[1])
        X[:, 2] -= np.clip(X[:, 2], z_range[0], z_range[1])
    return np.linalg.norm(X, axis=-1)


def calc_distance_score(distance, target_distance=None):
    """Gaussian score: 1 at target_distance, decays with σ²=3e4."""
    target = 0 if target_distance is None else np.array(distance)
    return np.exp(-(np.asarray(distance) - target) ** 2 / 9e4)


# ── Image overlays ───────────────────────────────────────────────────────────
def visualize_depth(rgb, depth, box, margin=25):
    """Paste a JET-colourmap of the depth around `box` into the bottom-right
    of `rgb`, with the box outlined."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = [int(el) for el in box]
    xm, ym = np.clip(x0 - margin, 0, w), np.clip(y0 - margin, 0, h)
    xM, yM = np.clip(x1 + margin, 0, w), np.clip(y1 + margin, 0, h)

    depth_roi   = depth[int(ym):int(yM), int(xm):int(xM)]
    depth_norm  = cv2.normalize(depth_roi, None, 0, 255, cv2.NORM_MINMAX)
    depth_8u    = depth_norm.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_JET)
    depth_color = cv2.rectangle(depth_color, (x0 - xm, y0 - ym), (x1 - xm, y1 - ym), (0, 255, 0), 2)

    # rgb_out = cv2.rectangle(rgb, (xm, ym), (xM, yM), (0, 0, 0), 1)
    rgb_out = rgb.copy()
    rgb_out[h - (yM - ym):, w - (xM - xm):, :] = depth_color
    return rgb_out


def visualize_mask(rgb, mask):
    """Tint the `mask==0` (background) region of `rgb` red at 40% alpha."""
    overlay = rgb.copy()
    overlay[mask == 0] = (0, 0, 255)
    return cv2.addWeighted(overlay, 0.4, rgb, 0.6, 0)


# ── Side-box helpers (used by side-pose flow) ────────────────────────────────
_SIDE_SIGNS = {'left': [-1, -1], 'right': [1, 1], 'front': [-1, 1], 'rear': [1, -1]}


def generate_angle_mask(w, h, x, y, theta_deg, side='beside'):
    """Half-plane mask split by a line through (x, y) at angle theta_deg.

    `side` picks which quadrant of the rotated frame is kept. 'beside' is the
    complement of 'rear'.
    """
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


def make_side_box(fg_mask, box=None, pose_3d=None, depth=None, calib_func=None,
                  head_angle=0, side='beside', area_diameter=120,
                  node=None, rgb=None, search_local=True):
    """Find a free patch next to `box` (or near `pose_3d`) in pixel space.

    Returns a (x0, y0, x1, y1) pixel box for the candidate side region, or
    None if no valid spot is found. When `node` and `rgb` are passed, an
    annotated preview is published to the UI in a background thread.
    """
    assert side in ['left', 'right', 'front', 'beside']
    h, w = fg_mask.shape[:2]

    if box is None:
        xc, yc = w // 2, int(0.3 * h)
        rx, ry = 10, 10
    else:
        x0, y0, x1, y1 = [int(el) for el in box]
        rx, ry = (x1 - x0) // 2, (y1 - y0) // 2
        xc, yc = (x0 + x1) // 2, (y0 + y1) // 2

    r  = 0.9 * np.linalg.norm((rx, ry))
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
        dline  = np.abs((X - xc) * sint - (Y - yc) * cost)
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


# ── Detector (TCP call via dm) ───────────────────────────────────────────────
def call_detector(params: dict):
    vlms = current().dm.get_client('vlms')
    if vlms is None:
        raise Exception("TCP connect 'vlms' not registered — add it in the Connection panel")
    return vlms.send(params)


# ── 3D feature extraction ────────────────────────────────────────────────────
_calib_singleton: Optional[Head2BaseCalibration] = None
def _get_calib() -> Head2BaseCalibration:
    global _calib_singleton
    if _calib_singleton is None:
        _calib_singleton = Head2BaseCalibration()
    return _calib_singleton


def attach_3d_features(node, ins, depth, cam_params, head_state, camera: str, robot_mode):
    """Enrich `ins` in place with normal_vectors / depths / locs_3d / poses_3d."""
    ins.normal_vectors = ins.get_cluster_normalvectors(
        depth=depth, cam_params=cam_params, weights=[[0,0],[1,1],[1,1]])
    ins.min_depths    = ins.get_cluster_depths(depth=depth, mode='min')
    ins.median_depths = ins.get_cluster_depths(depth=depth, mode='median')
    ins.locs_3d       = ins.get_cluster_locs_3d(
        depth=depth, cam_params=cam_params, cluster_depths=ins.median_depths)
    try:
        ins.bound_depths  = ins.get_cluster_depths(depth=depth, mode='max', bound_pixels=True)
    except:
        ins.bound_depths= None


    if 'head' in camera:
        # calib = _get_calib()
        # calib_func = lambda xc, yc, obj_depth: calib.convert_head_to_base_point(
        #     robot_mode, xc, yc, obj_depth,
        #     head_state['current_ry'], head_state['current_rz'])[:3]
        # ins.poses_3d = ins.get_cluster_calibrated_3d(
        #     depth=depth, calib_func=calib_func, cluster_depths=ins.median_depths)
        
        ins.pose_3d= {lb:get3d(node=node, x=v.centers[v.target_ind][0], y=v.centers[v.target_ind][1])['pose'] 
                      for lb, v in ins.label_clusters.items()}


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


# ── Per-object output builder ────────────────────────────────────────────────
def build_obj_info(ins, obj_name: str, horizon_yz, *, camera: str) -> dict:
    v = np.array(ins.normal_vectors[obj_name])
    v_yz = v[1:]
    v_yz = v_yz / np.linalg.norm(v_yz)

    cluster_ins = ins. label_clusters[obj_name]

    box = cluster_ins.boxes[cluster_ins.target_ind, :]
    lying = is_lying(obj_name, v_yz, horizon_yz)

    info = {
        'loc_3d':           ins.locs_3d[obj_name],
        'box':              box,
        'score':            float(cluster_ins.scores[cluster_ins.target_ind]),
        'normal_vector_yz': v_yz,
        'horizon_plan_yz':  horizon_yz,
        'islying':          lying,
        'depths': {
            'obj_min':    ins.min_depths[obj_name],
            'obj_median': ins.median_depths[obj_name],
            'bound':      None if ins.bound_depths is None else ins.bound_depths[obj_name],
        },
    }
    try:
        info['mass_percents'] = mass_percentages(
            obj_name, ins.label_clusters[obj_name].masks[0], box, lying=lying)
    except Exception:
        info['mass_percents'] = [50, 50]

    if 'head' in camera:
        info['pose_3d'] = ins.pose_3d[obj_name]

    return info


# ── Grasp pose estimation ────────────────────────────────────────────────────
def estimate_grasp(node, cluster_ins, *, target_ind: int = 0, rgb_out,
                   cam_params, depths: dict, keep_orientation: bool,
                   deeper: bool = False, deep_ratio: float | None = None,
                   detect_params: dict):
    """Run mask2grasps detector and convert pixel line → 3D grasp pose.

    Units: x/y in metres (after `/1000`), depth_final in metres, width in mm,
    rz in degrees. Matches the contract `pick.py` consumes.

    Args:
        cluster_ins: pyinterfaces cluster for ONE label (use ``ins.label_clusters[name]``).
        target_ind:  which instance inside that cluster to grasp. Default 0
                     for backwards-compat; the new detect flow passes the
                     scored target index.
        deep_ratio:  blend between ``obj_median`` and ``bound``: depth used is
                     v0 + ((1-α)·v0 + α·v1 - v0). When None, falls back to
                     α=0.99 if ``deeper`` else 0.6 (legacy behaviour).
                     Mirrors carerobotapp's ``deep_ratio``.

    Returns:
        ([x, y, depth_final, rz, width], depth_final)
    """
    gg = call_detector({
        **detect_params,
        'mask':     cluster_ins.masks[target_ind],
        'detector': 'mask2grasps',
    })
    ix0, iy0, ix1, iy1 = gg[0, :4].astype('int')
    if rgb_out is not None:
        rgb_out = show_line_on_rgb(rgb=rgb_out, line=(ix0, iy0, ix1, iy1), thick=3)
        log_data({'log_image': rgb_out})

    # depth (z) — pixel-space line lies at this depth
    rx = get_wrist_angle(node=node)
    to_pick_lying_obj = rx > 40
    v0 = depths['obj_median']
    v1 = depths.get('bound')

    alpha = deep_ratio if deep_ratio is not None else (0.99 if deeper else 0.6)
    depth_tune = 20 if v1 is None else 0
    if v1 is None:
        depth_value = v0 + depth_tune
    else:
        depth_value = v0 + min(((1 - alpha) * v0 + alpha * v1) - v0 + depth_tune, 90)

    # x, y from pixel endpoints (Ixy2xyz returns mm, convert to m for arm)
    x0, y0, z0 = Ixy2xyz(Ix=ix0, Iy=iy0, Z=depth_value, cam_params=cam_params)
    x1, y1, z1 = Ixy2xyz(Ix=ix1, Iy=iy1, Z=depth_value, cam_params=cam_params)
    x = (x0 + x1) / 2 / 1000. + ARM_CONFIGS['wrist_cam_offset'][0]
    y = (y0 + y1) / 2 / 1000. + ARM_CONFIGS['wrist_cam_offset'][1]

    width = np.linalg.norm([x0 - x1, y0 - y1, z0 - z1])  # mm
    rz = np.arctan2(y1 - y0, x1 - x0) * 180 / np.pi
    if not keep_orientation:
        rz = rz if to_pick_lying_obj else 0

    depth_final = depth_value / 1000. - ARM_CONFIGS['wrist_tool_length']  # m
    return [x, y, depth_final, rz, width], depth_final


# ── Input parsing ────────────────────────────────────────────────────────────
def _parse_obj_names(inp, obj_names) -> list[str]:
    if obj_names is not None:
        return obj_names
    if inp is None:
        raise Exception('Either "inputs" or "obj_names" required')
    return [el.strip().lower() for el in inp.split(',')]


def _pop_detect_params(kwargs: dict, detector: str) -> dict:
    params = {k: kwargs.pop(k, v) for k, v in FIND_CONFIGS['params'].items()}
    params['detector'] = detector
    return params


# ── Main entry points ────────────────────────────────────────────────────────
# ── detect: per-camera calibration ───────────────────────────────────────────
def _head_calib_func_factory(robot_mode, head_state):
    """Closure that maps (xc, yc, obj_depth) → base-frame xyz via Head2BaseCalibration."""
    # calib = _get_calib()
    # def f(xc, yc, obj_depth, **_):
    #     return calib.convert_head_to_base_point(
    #         robot_mode, xc, yc, obj_depth,
    #         head_state['current_ry'], head_state['current_rz'])
    def f(xc, yc, **kwargs):
        node = kwargs.get('node')
        return get3d(node=node, x=xc, y=yc)['pose']
    return f


def _wrist_calib_func(x, y, **kwargs):
    """Wrist-camera path uses raw image-plane → camera-frame mapping
    (no head-base calibration). Returns a (N, 3) array in mm."""
    x, y, z =  np.stack(Ixy2xyz(Ix=x, Iy=y, Z=kwargs['obj_depth'], 
                            cam_params=kwargs['cam_params']), axis=-1)
    return x/1000., y/1000., z/1000.


def _resolve_camera_setup(node, camera, cam, target_distance):
    """Pick the right calib_func + workspace based on camera kind. Returns
    (calib_func, workspace, target_distance) — target_distance falls back to
    FIND_CONFIGS['target_distance'] for the head camera."""
    use_head = 'head' in camera
    if use_head:
        robot_mode_init = get_robot_mode(node=node)
        calib_func = _head_calib_func_factory(robot_mode_init, cam.head_state)
        workspace = ARM_CONFIGS['range']
        if target_distance is None:
            target_distance = FIND_CONFIGS.get('target_distance')
    elif camera == 'wrist_raw':
        calib_func = _wrist_calib_func
        workspace = ARM_CONFIGS.get('tool_range', ARM_CONFIGS['range'])
    else:
        # 'arm' or any other separate-agent camera — same calibration as wrist.
        calib_func = _wrist_calib_func
        workspace = ARM_CONFIGS.get('tool_range', ARM_CONFIGS['range'])
    return calib_func, workspace, target_distance


# ── detect: per-target info builder (post target_ind selection) ──────────────
def _build_target_info(node, *, cluster_ins, cam, valid_locs, obj_depths, poses_3d,
                       obj_name, target_ind, horizon_yz, camera, robot_mode,
                       calib_func, workspace, side, sam_ret, sam_thread,
                       do_estimate_grasp, keep_orientation, deeper, deep_ratio,
                       detect_params, rgb_out_main):
    """Compute normal/islying/side/grasp/mass for ONE target instance.

    Output dict matches kcare_robot's existing build_obj_info contract (so
    pick.py / approach.py keep working) PLUS optional 'side_pose'.
    """
    rgb_out = rgb_out_main.copy() if rgb_out_main is not None else None
    box       = cluster_ins.boxes[target_ind]
    obj_depth = obj_depths[target_ind]
    pose      = poses_3d[target_ind]
    pose_list = pose.tolist() if hasattr(pose, 'tolist') else list(pose)

    # normal vector + islying (per-target)
    Iy, Ix = valid_locs[target_ind]
    Z      = cam.depth[(Iy, Ix)].astype('float32')
    X3, Y3, Z3 = Ixy2xyz(Ix=Ix.astype('float32'), Iy=Iy.astype('float32'),
                         Z=Z, cam_params=cam.cam_params)
    nv = calc_normalvector(np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=-1))
    nv_yz = nv[1:] / np.linalg.norm(nv[1:])
    islying = is_lying(obj_name, nv_yz, horizon_yz)

    # depths
    obj_min = float(np.min(cam.depth[(Iy, Ix)])) if len(Iy) else None
    try:
        bound_loc = cluster_ins.get_valid_depth_locs(
            depth=cam.depth, bound_pixels=True, target_ind=target_ind)
        bound_depth = None if len(bound_loc[0]) == 0 else float(np.median(cam.depth[bound_loc]))
    except:
        bound_loc  = None
        bound_depth = None
    

    # mass split
    try:
        mass_pct = mass_percentages(obj_name, cluster_ins.masks[target_ind], box, lying=islying)
    except Exception:
        mass_pct = [50, 50]

    info = {
        'loc_3d':            pose_list,
        # pose_3d is set for ALL cameras so downstream code (fine_move's nudge,
        # pushing, etc.) can rely on it. For head camera it's the base-frame
        # pose (Head2BaseCalibration); for arm/wrist_raw it's camera-frame.
        'pose_3d':           pose_list,
        'box':               box,
        'score':             float(cluster_ins.scores[target_ind]),
        'normal_vector_yz':  nv_yz,
        'horizon_plan_yz':   horizon_yz,
        'islying':           islying,
        'depths': {
            'obj_min':    obj_min,
            'obj_median': obj_depth,
            'bound':      bound_depth,
        },
        'mass_percents':     mass_pct,
    }

    log_data({'normal_yz': nv_yz, 'horizon_plan_yz': horizon_yz,
              'islying':   islying, 'target_ind': target_ind})

    # side pose
    if side is not None:
        if sam_thread is not None:
            sam_thread.join()
        fg_mask = sam_ret.get('fg_mask')
        if fg_mask is not None:
            head_rz = cam.head_state.get('current_rz', 0) if cam.head_state else 0
            side_box = make_side_box(
                box=box, side=side, node=node, rgb=rgb_out, fg_mask=fg_mask,
                head_angle=90 - abs(head_rz),
            )
            if side_box is not None:
                x0, y0, x1, y1 = side_box
                xc, yc = (x0 + x1) // 2, (y0 + y1) // 2
                depth_crop = cam.depth[y0:y1, x0:x1]
                side_depth = float(np.median(depth_crop[get_valid_depth_locs(depth_crop)]))
                side_pose = calib_func(xc, yc, side_depth, cam_params=cam.cam_params,
                                       robot_mode=robot_mode, head_state=cam.head_state)
                # side_pose = get3d(node=node, x=xc, y=yc)['pose']
                
                side_pose_list = side_pose.tolist() if hasattr(side_pose, 'tolist') else list(side_pose)
                if rgb_out is not None:
                    rgb_out = show_box_on_rgb(rgb=rgb_out, box=side_box, thick=2)
                log_data({'side_pose': side_pose_list, 'side_depth': side_depth,
                          'xc_yc': [int(xc), int(yc)]})
                if (is_inside_workspace_box(*np.asarray(side_pose_list)[:3], workspace=workspace)
                        and abs(side_depth - obj_depth) < 0.15):
                    info['side_pose'] = side_pose_list

    # grasp pose
    if do_estimate_grasp:
        gp, df = estimate_grasp(
            node, cluster_ins, target_ind=target_ind, rgb_out=rgb_out,
            cam_params=cam.cam_params, depths=info['depths'],
            keep_orientation=keep_orientation, deeper=deeper, deep_ratio=deep_ratio,
            detect_params=detect_params,
        )
        info['grasppose'] = gp
        info['depths']['final'] = df

    if rgb_out is not None:
        log_data({'log_image': rgb_out})

    return info


def _detect_one(node, *, ins, cam, obj_name, camera, workspace, calib_func,
                target_distance, horizon_yz, robot_mode, side, sam_ret, sam_thread,
                do_estimate_grasp, keep_orientation, deeper, deep_ratio,
                detect_params, return_multiple):
    """Filter cluster by valid depth + workspace, score by (det_score × distance),
    pick best target, build per-target info. Returns dict (or list of dicts
    when return_multiple)."""
    cluster_ins = ins.label_clusters[obj_name]
    assert len(cluster_ins) > 0, f'No {obj_name} detected'

    # 1. drop instances with no valid depth locs
    valid_locs_all = [
        cluster_ins.get_valid_depth_locs(depth=cam.depth, bound_pixels=False, target_ind=i)
        for i in range(len(cluster_ins))
    ]
    inds = [i for i, loc in enumerate(valid_locs_all) if len(loc[0]) > 0]
    assert len(inds) > 0, f'Removed all {obj_name}: no valid depth locs'
    cluster_ins = cluster_ins.select(inds=inds, issorted=True)
    valid_locs  = [valid_locs_all[i] for i in inds]
    obj_depths  = [float(np.median(cam.depth[loc])) for loc in valid_locs]

    # 2. poses_3d for every surviving instance
    centers = cluster_ins.centers
    Xc = [centers[(i, 0)] for i in range(len(cluster_ins))]
    Yc = [centers[(i, 1)] for i in range(len(cluster_ins))]
    # poses_arr = np.array(calib_func(Xc, Yc, obj_depths,
    #                                 cam_params=cam.cam_params,
    #                                 robot_mode=robot_mode,
    #                                 head_state=cam.head_state)).reshape(-1, 3)
    

    

    # 3. workspace filter
    # inside = is_inside_workspace_box(
    #     x=poses_arr[:, 0], y=poses_arr[:, 1], z=poses_arr[:, 2], workspace=workspace)
    # inds = [i for i, ok in enumerate(np.atleast_1d(inside)) if ok]
    # assert len(inds) > 0, f'pose_3d: {poses_arr.tolist()} out of workspace'
    cluster_ins = cluster_ins.select(inds=inds, issorted=True)
    valid_locs  = [valid_locs[i] for i in inds]
    obj_depths  = [obj_depths[i] for i in inds]
    # poses_3d    = [poses_arr[i] for i in inds]

    poses_3d = [calib_func(node=node, x=x, y=y, obj_depth=dd, cam_params=cam.cam_params) for x,y,dd in zip(Xc, Yc, obj_depths)]

    # 4. score by (det_score × distance_score) and pick target
    distances = calc_obj_distance(poses_3d, use_head_cam='head' in camera)
    d_scores  = calc_distance_score(distances, target_distance)
    scores    = np.multiply(cluster_ins.scores.flatten(), d_scores)
    cluster_ins.target_ind = int(np.argmax(scores))
    log_data({
        'obj_scores':      cluster_ins.scores.flatten().tolist(),
        'distance_scores': np.asarray(d_scores).tolist(),
        'scores':          np.asarray(scores).tolist(),
        'target_ind':      cluster_ins.target_ind,
    })

    # 5. target-only visualization
    box = cluster_ins.boxes[cluster_ins.target_ind]
    rgb_out = visualize_pred(rgb=cam.rgb, pred=cluster_ins, show_steps=True)
    rgb_out = visualize_depth(rgb_out, cam.depth, box)
    log_data({'log_image': rgb_out})

    # 6. build per-target info
    def _est(ti):
        return _build_target_info(
            node, cluster_ins=cluster_ins, cam=cam, valid_locs=valid_locs,
            obj_depths=obj_depths, poses_3d=poses_3d, obj_name=obj_name,
            target_ind=ti, horizon_yz=horizon_yz, camera=camera,
            robot_mode=robot_mode, calib_func=calib_func, workspace=workspace,
            side=side, sam_ret=sam_ret, sam_thread=sam_thread,
            do_estimate_grasp=do_estimate_grasp, keep_orientation=keep_orientation,
            deeper=deeper, deep_ratio=deep_ratio, detect_params=detect_params,
            rgb_out_main=rgb_out,
        )

    if return_multiple:
        return [_est(i) for i in range(len(cluster_ins))]
    return _est(cluster_ins.target_ind)


@exception_handler
def detect(node, **kwargs):
    """Detect objects with the 'vlms' TCP detector and extract 3D features.

    Required (one of):
        inputs='obj1,obj2,...'  OR  obj_names=['obj1','obj2',...]

    Optional:
        camera             — 'head' / 'arm' / 'wrist_raw' (default FIND_CONFIGS['camera'])
        detector           — detector name                 (default FIND_CONFIGS['detector'])
        side               — 'left' / 'right' / 'front' / 'beside' — also compute side_pose
        target_distance    — distance score peak (head-cam only); None ⇒ FIND_CONFIGS['target_distance']
        estimate_grasp     — also compute grasp pose           (default False)
        keep_orientation   — preserve detected rz              (default False)
        deeper             — deeper depth bias                 (default False)
        deep_ratio         — explicit blend [0..1] between obj_median and bound
                             (overrides ``deeper`` if provided)
        return_multiple    — return list of per-instance infos instead of just target
        keep_closed_objects— reserved for parity (no-op here)
        Remaining kwargs are forwarded to the detector.

    Returns ``{'isdone': bool, 'ins': {obj_name: info_dict_or_list, ...}}``.
    """
    obj_names           = _parse_obj_names(kwargs.pop('inputs', None), kwargs.pop('obj_names', None))
    do_estimate_grasp   = kwargs.pop('estimate_grasp', False)
    keep_orientation    = kwargs.pop('keep_orientation', False)
    side                = kwargs.pop('side', None)
    deeper              = kwargs.pop('deeper', False)
    deep_ratio          = kwargs.pop('deep_ratio', None)
    return_multiple     = kwargs.pop('return_multiple', False)
    target_distance     = kwargs.pop('target_distance', None)
    kwargs.pop('keep_closed_objects', False)  # parity / reserved
    camera              = kwargs.pop('camera',   FIND_CONFIGS['camera'])
    detector            = kwargs.pop('detector', FIND_CONFIGS['detector'])
    detect_params       = _pop_detect_params(kwargs, detector)

    cam = fetch_camera_data(node, camera)
    calib_func, workspace, target_distance = _resolve_camera_setup(node, camera, cam, target_distance)
    robot_mode = get_robot_mode(node=node)

    # 1. main detection
    ins = call_detector({
        **detect_params,
        **kwargs,
        'rgb':     cam.rgb,
        'caption': '. '.join(obj_names),
    })
    if ins is None:
        raise Exception(f'detected ins: {data_info(ins)} ...')

    # 2. publish raw detection preview
    rgb_out_raw = visualize_pred(rgb=cam.rgb, pred=ins, show_steps=True)
    log_data({'log_image': rgb_out_raw})

    # 3. fastsam in parallel iff side is requested
    sam_ret: dict = {}
    sam_thread = None
    if side is not None:
        def _sam_func():
            sam_ins = call_detector({
                'rgb': cam.rgb, 'detector': 'fastsam', 'caption': 'obj',
            })
            if sam_ins is not None:
                sam_ret['fg_mask'] = np.bitwise_and(sam_ins.mask_all > 0, cam.depth > 0).astype('uint8')
        sam_thread = threading.Thread(target=_sam_func, daemon=True)
        sam_thread.start()

    # 4. horizon vector (used for islying)
    cam_angle = (abs(cam.head_state['current_ry']) if 'head' in camera
                 else get_wrist_angle(node=node))
    cam_angle_rad = cam_angle * np.pi / 180
    horizon_yz = np.array([-np.cos(cam_angle_rad), -np.sin(cam_angle_rad)])

    # 5. per-obj processing
    out: dict = {}
    for obj_name in obj_names:
        out[obj_name] = _detect_one(
            node, ins=ins, cam=cam, obj_name=obj_name, camera=camera,
            workspace=workspace, calib_func=calib_func,
            target_distance=target_distance, horizon_yz=horizon_yz,
            robot_mode=robot_mode, side=side, sam_ret=sam_ret, sam_thread=sam_thread,
            do_estimate_grasp=do_estimate_grasp, keep_orientation=keep_orientation,
            deeper=deeper, deep_ratio=deep_ratio,
            detect_params=detect_params, return_multiple=return_multiple,
        )

    return {'isdone': len(ins) > 0, 'ins': out}


@exception_handler
def find_once(**kwargs):
    node = kwargs.pop('node', None)
    loc  = kwargs.get('loc', None)
    if loc is not None:
        ret = move(node=node, inputs=loc)
        if not ret['isdone']:
            return ret
    view = kwargs.pop('view', 'down')
    moveh(node=node, ry=view)
    time.sleep(1)

    ret = detect(node=node, **kwargs)
    moveh(node=node, ry='straight')
    return ret


@exception_handler
def find(node, **kwargs):
    loc_name = kwargs.pop('inputs', None)
    if loc_name is None:
        raise Exception(f'inputs :{loc_name}')
    splits = loc_name.split('@')
    caption, loc = (splits[0], '@'.join(splits[1:])) if len(splits) >= 2 else (loc_name, None)

    obj_names = [el.strip() for el in caption.split(',') if el.strip()]
    kwargs['obj_names'] = obj_names

    views = kwargs.pop('views', ['down'])
    once  = kwargs.pop('once', True)

    def run(loc, views):
        ret = {'isdone': False}
        for view in views:
            ret = find_once(**kwargs, node=node, view=view, loc=loc)
            if ret['isdone']:
                return ret
        return ret

    ret = run(loc, views)
    if ret['isdone'] or once:
        return ret

    locs = list(ENV.keys())
    print(f'Recoginition failed. Try to find in {locs}')
    for loc in locs:
        ret = run(loc, views=views)
        if ret['isdone']:
            return ret

    raise Exception(f'find {obj_names} failed ...')


@exception_handler
def find_arm(node, **kwargs):
    kwargs.setdefault('detector',       'groundedsam')
    kwargs.setdefault('camera',         'arm')
    kwargs.setdefault('estimate_grasp', True)

    to_find_handle = 'handle' in kwargs.get('inputs', '')
    kwargs.setdefault('dmin', 20)
    kwargs.setdefault('dmax', 200 if to_find_handle else 400)

    if to_find_handle:
        kwargs.setdefault('text_threshold', 0.1)
        kwargs.setdefault('box_threshold',  0.1)

    return detect(node=node, **kwargs)


@exception_handler
def grasp_succeed_v1(node, **kwargs):
    inp = kwargs.get('inputs', 'item')
    kwargs['inputs']   = inp
    kwargs.setdefault('detector', 'groundingdino')
    kwargs.setdefault('camera',   'rs_raw')
    kwargs.setdefault('dmin', 50)
    kwargs.setdefault('dmax', 300)
    kwargs.setdefault('crop_roi', [400, 350, 1000, 600])

    ret = detect(node=node, **kwargs)
    if not ret['isdone']:
        return ret

    obj_depth = ret['ins'][inp]['depths']['obj_median']
    ret['isdone'] = -100 < obj_depth < 20
    return ret


@exception_handler
def grasp_succeed(node, **kwargs):
    x0, y0, x1, y1 = kwargs.get('crop_roi', [448, 333, 464, 347])
    cam_data = _fetch_arm(node=node)
    rgb, depth = cam_data.rgb, cam_data.depth

    rgb_out = show_box_on_rgb(rgb=rgb, box=[x0, y0, x1, y1], thick=2)
    log_data({'log_image': rgb_out})

    depth_roi = depth[y0:y1, x0:x1]
    obj_depth = np.median(depth_roi[depth_roi > 0])/1000. - ARM_CONFIGS['wrist_tool_length']
    return {'isdone': -0.250 < obj_depth < 0.020, 'obj_depth': obj_depth}


# ── Side-pose skill ──────────────────────────────────────────────────────────
@exception_handler
def get_side_pose_3d(node, **kwargs):
    """Find a free pose to the side of an arbitrary scene from the head camera.

    Runs fastsam on the head-camera frame, then `make_side_box` near the
    requested side of `pose_3d` (defaulting to a workspace-edge point), and
    returns the resolved 3D pose in the base frame.

    Optional kwargs:
        side    — 'left' / 'right' / 'front' / 'beside' (default 'beside')
        view    — head pitch view to take the shot from ('down', 'straight', 'up'); default 'down'
        pose_3d — reference pose [x, y, z (, rx?)] in mm; default depends on robot_mode

    Returns ``{'isdone': True, 'pose_3d': [x, y, z, ...]}``. If no free side
    spot is found or the candidate is outside the arm workspace, falls back
    to the input ``pose_3d``.
    """
    side    = kwargs.pop('side',    'beside')
    view    = kwargs.pop('view',    'down')
    pose_3d_in = kwargs.pop('pose_3d', None)

    moveh(node=node, inputs=view, wait=True)
    time.sleep(1)
    cam = _fetch_head(node)
    robot_mode = get_robot_mode(node=node)
    moveh(node=node, inputs='straight', wait=False)

    sam_ins = call_detector({
        'rgb':     cam.rgb,
        'detector': 'fastsam',
        'caption': 'obj',
        'max_ratio':     0.3,
        'min_mass':      500,
        'box_threshold': 0.4,
    })
    if sam_ins is None:
        raise Exception('Sam returned None')

    pose_3d = pose_3d_in if pose_3d_in is not None \
        else [150, 600 if robot_mode == 'right' else -600, 400]
    calib_func = _head_calib_func_factory(robot_mode, cam.head_state)

    fg_mask = np.bitwise_and(sam_ins.mask_all > 0, cam.depth > 0).astype('uint8')
    side_box = make_side_box(
        pose_3d=pose_3d, fg_mask=fg_mask, depth=cam.depth, calib_func=calib_func,
        node=node, rgb=cam.rgb,
        head_angle=90 - abs(cam.head_state['current_rz']),
        search_local=False, side=side,
    )

    if side_box is None:
        return {'isdone': True, 'pose_3d': pose_3d}

    x0, y0, x1, y1 = side_box
    xc, yc = (x0 + x1) // 2, (y0 + y1) // 2
    depth_crop = cam.depth[y0:y1, x0:x1]
    side_depth = float(np.median(depth_crop[get_valid_depth_locs(depth_crop)]))
    side_pose = calib_func(xc, yc, side_depth)
    side_pose_list = list(side_pose) if not hasattr(side_pose, 'tolist') else side_pose.tolist()
    # preserve original z + any rx hint from the input
    side_pose_list[2] = pose_3d[2]
    if len(pose_3d) > 3:
        side_pose_list = side_pose_list + [pose_3d[3]]

    log_data({'xc_yc_depth': [int(xc), int(yc), side_depth],
              'side_pose':   side_pose_list})

    if not is_inside_workspace_box(*side_pose_list[:3], workspace=ARM_CONFIGS['range']):
        return {'isdone': True, 'pose_3d': pose_3d}

    rgb_out = show_box_on_rgb(cam.rgb, side_box, thick=2)
    log_data({'log_image': rgb_out})

    return {'isdone': True, 'pose_3d': side_pose_list}


if __name__ == '__main__':
    pass
