from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np, time, cv2, os

from pyconnect.utils import dict2str, data_info, Timer, run_parallel, update_dict
from pyinterfaces.utils import visualize_pred, show_box_on_rgb, Ixy2xyz, show_line_on_rgb

from robot_agent.skill_configs import FIND_CONFIGS, ARM_CONFIGS, ENV
from robot_agent.skill_configs import STANDING_OBJ_NAMES, LYING_OBJ_NAMES, HAVING_HANDLE_OBJ_NAMES
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


CAMERA_FETCHERS: dict[str, Callable] = {
    'head':      _fetch_head,
    'arm': _fetch_arm,
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

    box = ins.label_clusters[obj_name].boxes[0, :]
    lying = is_lying(obj_name, v_yz, horizon_yz)

    info = {
        'loc_3d':           ins.locs_3d[obj_name],
        'box':              box,
        'score':            float(ins.label_clusters[obj_name].scores[0]),
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
def estimate_grasp(node, ins, obj_name: str, rgb_out, cam_params, depths: dict,
                   *, keep_orientation: bool, deeper: bool, detect_params: dict):
    """Run mask2grasps detector and convert pixel line → 3D grasp pose.

    Returns:
        ([x, y, z, rz, width], depth_final)
    """
    gg = call_detector({
        **detect_params,
        'mask':     ins.label_clusters[obj_name].masks[0],
        'detector': 'mask2grasps',
    })
    ix0, iy0, ix1, iy1 = gg[0, :4].astype('int')
    rgb_out = show_line_on_rgb(rgb=rgb_out, line=(ix0, iy0, ix1, iy1), thick=3)
    log_data({'log_image': rgb_out})
    # publish_image(node=node, rgb=rgb_out)
    # log_result_data(node=node, data={'grasp': rgb_out})
    

    # depth (z)
    rx = get_wrist_angle(node=node)
    to_pick_lying_obj = rx > 40
    v0, v1 = depths['obj_median'], depths['bound']
    alpha = 0.99 if deeper else 0.6
    depth_value = v0 + min(((1 - alpha) * v0 + alpha * v1) - v0, 90)

    # x, y from pixel endpoints
    x0, y0, z0 = Ixy2xyz(Ix=ix0, Iy=iy0, Z=depth_value, cam_params=cam_params)
    x1, y1, z1 = Ixy2xyz(Ix=ix1, Iy=iy1, Z=depth_value, cam_params=cam_params)
    x = (x0 + x1) / 2 /1000. + ARM_CONFIGS['wrist_cam_offset'][0]
    y = (y0 + y1) / 2 /1000. + ARM_CONFIGS['wrist_cam_offset'][1]

    # rz, width
    width = np.linalg.norm([x0 - x1, y0 - y1, z0 - z1])
    rz = np.arctan2(y1 - y0, x1 - x0) * 180 / np.pi
    if not keep_orientation:
        rz = rz if to_pick_lying_obj else 0

    depth_final = depth_value/1000. - ARM_CONFIGS['wrist_tool_length']
    # return [x, y, depth_final, rz, width], depth_final
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
@exception_handler
def detect(node, **kwargs):
    """Detect objects with the 'vlms' TCP detector and extract 3D features.

    Required (one of): inputs='obj1,obj2,...'  OR  obj_names=['obj1','obj2',...]

    Optional:
        camera           — 'head' or 'arm'        (default FIND_CONFIGS['camera'])
        detector         — detector name                 (default FIND_CONFIGS['detector'])
        estimate_grasp   — also compute grasp pose       (default False)
        keep_orientation — preserve detected rz          (default False)
        deeper           — deeper depth bias             (default False)
        Remaining kwargs are forwarded to the detector.
    """
    obj_names           = _parse_obj_names(kwargs.pop('inputs', None), kwargs.pop('obj_names', None))
    do_estimate_grasp   = kwargs.pop('estimate_grasp', False)
    keep_orientation    = kwargs.pop('keep_orientation', False)
    deeper              = kwargs.pop('deeper', False)
    camera              = kwargs.pop('camera',   FIND_CONFIGS['camera'])
    detector            = kwargs.pop('detector', FIND_CONFIGS['detector'])
    detect_params       = _pop_detect_params(kwargs, detector)

    cam = fetch_camera_data(node, camera)

    ins = call_detector({
        **detect_params,
        **kwargs,
        'rgb':     cam.rgb,
        'caption': '. '.join(obj_names),
    })
    if ins is None:
        raise Exception(f'detected ins: {data_info(ins)} ...')

    rgb_out = visualize_pred(rgb=cam.rgb, pred=ins)
    log_data({'log_image': rgb_out})

    # for obj_name in obj_names:
    #     x, y = ins.label_clusters[obj_name].centers[0,:]
    #     pose3d = get3d(node=node, x=x, y=y)
    #     # log_data({'pose3d': pose3d})    

    attach_3d_features(node, ins, cam.depth, cam.cam_params, cam.head_state,
                       camera, get_robot_mode(node=node))

    wrist_angle_rad = get_wrist_angle(node=node)
    horizon_yz = np.array([-np.cos(wrist_angle_rad*np.pi/180.), -np.sin(wrist_angle_rad*np.pi/180.)])

    out = {}
    for obj_name in obj_names:
        info = build_obj_info(ins, obj_name, horizon_yz, camera=camera)
        if do_estimate_grasp:
            info['grasppose'], info['depths']['final'] = estimate_grasp(
                node, ins, obj_name, rgb_out, cam.cam_params, info['depths'],
                keep_orientation=keep_orientation, deeper=deeper,
                detect_params=detect_params,
            )
        out[obj_name] = info

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


if __name__ == '__main__':
    pass
