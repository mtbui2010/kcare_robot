"""recognition — simplified detector skills built on the VisionServe SDK.

Two entry points, each bound to one camera + one model:

    find(...)      head camera  → 'grounding-dino'  → object 3D pose (base frame, via get3d)
    find_arm(...)  wrist camera → 'grasp-gd'        → grasp pose      (camera frame, via Ixy2xyz)

Both return the same contract the rest of the stack already consumes::

    {'isdone': bool, 'ins': {obj_name: {loc_3d, pose_3d, box, score,
                                        islying, mass_percents, depths,
                                        grasppose?}, ...}}

The VisionServe client is the SDK ``Client`` registered as the 'visionserve'
connection (see robot_agent device_manager). The legacy side-pose flow
(``get_side_pose_3d``) still uses the 'vlms' TCP detector for fastsam.

The utility/helper functions live in `_recognition_helpers.py`; this module
keeps the registered skills plus the two orchestrators (`_detect_objects`,
`_detect_grasps`).
"""

import numpy as np, time, copy, cv2

from pyinterfaces.utils import show_box_on_rgb, Ixy2xyz, get_valid_depth_locs
from visionserve.postprocess import get_depth_at_detection, select_target_object, select_target_grasp

from robot_agent.skill_configs import FIND_CONFIGS, ARM_CONFIGS, ENV
from robot_agent.utils import exception_handler, get_env_specs
from robot_agent.skills import log_data
from kcare_robot.skills.head import moveh, get_robot_mode
from kcare_robot.skills.mobile import move
from kcare_robot.skills.pointcloud import get3d
from kcare_robot.skills.arm import get_wrist_angle

from kcare_robot.skills._recognition_helpers import (
    fetch_camera_data, _fetch_head, _fetch_arm, _get_bound_depth,
    _vs_client, _vs_postprocess, _prompt, _log_annotated, _get_placepose,
    _foreground_mask, _bbox_to_xyxy, _sample_depth, _box_islying, _box_depths,
    _compute_side_pose, _line_to_grasppose, _parse_obj_names,
    call_detector, make_side_box, _head_calib_func_factory, is_inside_workspace_box,
    save_detection_dataset,
)

def _detect_nearest(node, pose, **kwargs) -> dict:
    stride = kwargs.pop('stride', FIND_CONFIGS['params']['stride'])
    cam = fetch_camera_data(node, 'head')
    h, w =  cam.rgb.shape[:2]
    fx,fy, cx, cy = cam.cam_params

    params = {k:kwargs.pop(k, v) for k,v in FIND_CONFIGS['params'].items() if k in ['roi','method', 'dilate']}

    res = _vs_client().predict('background', cam.rgb, depth=cam.depth, bg_max_area=60, **params)
    _log_annotated(res,  cam.rgb)

    if len(res.masks)==0:
        return pose

    bg_mask = (res.masks[0].to_ndarray(width=w, height=h)>0).astype('uint8')

    # bg_mask = cv2.erode(bg_mask, np.ones((21, 21), 'uint8'))
    sampled_mask = bg_mask[::stride, ::stride]
    Iy, Ix = np.where(sampled_mask)

    if len(Ix) == 0:
        return pose

    Ix = Ix * stride
    Iy = Iy * stride
    points = [[x, y] for x,y in zip(Ix, Iy)]

    ret = get3d(node=node, points=points)
    assert ret['isdone'], f'{ret}'
    
    P = ret['pose']
    inds = ~np.isnan(P).any(axis=-1)
    P = P[inds, ...]

    dist2 = np.sum((P - np.array(pose).reshape(1,3)) ** 2, axis=1)
    argmin = np.argmin(dist2)

    x, y =(int(np.array(Ix)[inds][argmin]), int(np.array(Iy)[inds][argmin]))
    annotated = np.asarray(res.visualize(cam.rgb))
    log_data({'log_image': cv2.drawMarker(annotated, (x,y), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 20, 2)})
    
    return P[argmin].tolist()
    
    

# ── Core: object detection (grounding-dino) ──────────────────────────────────
def _detect_objects(node, obj_names, **kwargs) -> dict:
    """grounding-dino on `camera`. Head → 3D via get3d (base frame); wrist →
    3D via Ixy2xyz (camera frame, metres)."""
    simple_return = kwargs.pop('simple_return', False)
    target_distance = kwargs.pop('target_distance', 0.9)
    camera = kwargs.pop('camera', 'head')
    robot_mode = get_robot_mode(node=node)
    select_target_object, _ = _vs_postprocess()
    use_head = 'head' in camera

    cam = fetch_camera_data(node, camera)
    rgb, depth, cam_params = cam.rgb, cam.depth, cam.cam_params
    h, w = rgb.shape[:2]

    max_size = kwargs.pop('max_size', FIND_CONFIGS['params'].get('max_ratio', 0.6) * 100)
    min_conf = kwargs.pop('min_conf', 0.1 if any('handle' in n for n in obj_names) else 0.0)
    side = kwargs.pop('side', None)
    text_threshold = kwargs.pop('text_threshold', 0.15)

    # A side approach needs object masks → grounded-sam (boxes + masks in one
    # call); plain find uses grounding-dino (boxes only).
    want_side = side is not None and use_head
    model = 'grounded-sam' if want_side else 'grounding-dino'
    res = _vs_client().predict(model, rgb, prompt=_prompt(obj_names), max_size=max_size, text_threshold=text_threshold)
    if min_conf > 0:
        res = res.filter_by_conf(min_conf=min_conf)
    # log_data(**res)
    _log_annotated(res, rgb)

    fg_mask = _foreground_mask(res, w, h, depth) if want_side else None

    cam_angle = abs(cam.head_state['current_ry']) if use_head else get_wrist_angle(node=node)
    cam_angle_rad = cam_angle * np.pi / 180
    horizon_yz = np.array([-np.cos(cam_angle_rad), -np.sin(cam_angle_rad)])

    out: dict = {}
    for name in obj_names:
        target = select_target_object(
            res, cls=name, near_point='center', image_size=(w, h), 
            target_distance=target_distance, depth_result=depth, intrinsics=cam_params,
            weights={'quality': 0.5, 'distance': 0.3, 'near': 0.2})
        if target is None:
            continue
        _log_annotated(res, rgb, target_box=target)
        box = _bbox_to_xyxy(target.bbox)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

        if use_head:
            # pose = list(get3d(node=node, x=cx, y=cy)['pose'])
            # pose = list(get3d(node=node, x=int(cx), y=int(cy))['pose'])
            pose = get3d(node=node, points = [[cx, cy]])['pose'][0, :].tolist()
        else:
            Z = _sample_depth(depth, cx, cy)
            px, py, pz = Ixy2xyz(Ix=cx, Iy=cy, Z=Z, cam_params=cam_params)
            pose = [float(px) / 1000., float(py) / 1000., float(pz) / 1000.]

        islying = _box_islying(name, box, depth, cam_params, horizon_yz)
        
        if simple_return:
            out[name] = {'pose': pose,'islying':islying}
            continue

        approach_pose = copy.deepcopy(pose)
        lift_to = approach_pose[-1] + 0.1
        if islying:
            approach_pose[-1] += 0.2
            approach_pose[1] -= 0.1 if robot_mode=='right' else -0.1
            approach_pose += [-180., 0., 90. if robot_mode=='right' else -90.]
            approach_lying=True
        else:
            approach_pose[-1] += 0.1
            ry, dd = 30, 0.25
            sint, cost = np.sin(ry*np.pi/180), np.cos(ry*np.pi/180)
            approach_pose[0] += dd*sint
            approach_pose[1] += (-dd if robot_mode=='right' else dd)*cost
            approach_pose += [-180, -75, 100+ry if robot_mode=='right' else -100-ry]
            approach_lying=False
        mforward = approach_pose[0] - np.clip(approach_pose[0], 0.15, 0.4)
        mforward = 0 if abs(mforward)<0.1 else mforward
        approach_pose[0] -= mforward

        approach_pose = {k:v for k,v in zip(['x', 'y', 'z', 'rx', 'ry', 'rz'], approach_pose)}

        base_rotate = 90 if robot_mode=='right' else -90
        out[name] = {
            'duration_ms':   res.duration_ms,
            'device':        res.device,
            # 'loc_3d':        pose,
            'pose_3d':       pose,
            'approach_pose': approach_pose,
            'lift_to':      lift_to,
            'mforward':      mforward,
            'base_rotate':   base_rotate,
            'approach_lying':   approach_lying,
            'box':           box,
            'score':         float(getattr(target, 'conf', 0.0)),
            'islying':       islying,
            'mass_percents': [50, 50],
            'depths':        _box_depths(depth, box),
        }
        if fg_mask is not None:
            sp = _compute_side_pose(node, box, side, fg_mask, rgb, cam.head_state)
            if sp is not None:
                out[name]['side_pose'] = sp
    save_detection_dataset(rgb=rgb, depth=depth, results=out, tag=camera)
    return {'isdone': len(out) > 0, 'ins': out}


# ── Core: grasp detection (grasp-gd) ─────────────────────────────────────────
def _detect_grasps(node, obj_names, **kwargs: dict) -> dict:
    """grasp-gd on the wrist camera → per-object 3D grasp pose."""
    camera = kwargs.pop('camera', 'arm')
    _, select_target_grasp = _vs_postprocess()

    cam = _fetch_arm(node)
    rgb, depth, cam_params = cam.rgb, cam.depth, cam.cam_params
    h, w = rgb.shape[:2]

    gmin = kwargs.pop('gripper_min', None)
    gmax = kwargs.pop('gripper_max', None)
    text_threshold = kwargs.pop('text_threshold', 0.15)
    keep_orientation = kwargs.pop('keep_orientation', False)


    res = _vs_client().predict(
        'grounding-dino', rgb, prompt=_prompt(obj_names), text_threshold=text_threshold)
    _log_annotated(res, rgb)
    

    out: dict = {}
    # target_for_draw = None
    for name in obj_names:
        obj = select_target_object(res, cls=name, near_point="center", image_size=(w, h), 
            target_distance=0.3, depth_result=depth, intrinsics=cam_params,
            weights={'quality': 0.5, 'distance': 0.3, 'near': 0.2}) 
        gs  = _vs_client().predict("grasp", rgb, box=obj.bbox, gripper_min=gmin, gripper_max=gmax) 
        g, garg = select_target_grasp(gs.grasps, return_index=True)
        _log_annotated(gs, rgb, target_grasp=g)
        if g is None:
            continue
        # target_for_draw = g
        (gx0, gy0), (gx1, gy1) = g.contacts()
        depth_value = _sample_depth(depth, g.x, g.y)
        depth_bound = _get_bound_depth(depth, gs.masks[garg].to_ndarray(width=w, height=h))
        grasppose, depth_final = _line_to_grasppose(
            node, [gx0, gy0, gx1, gy1], cam_params=cam_params,
            depths={'obj_median': depth_value, 'bound': depth_bound},
            keep_orientation=keep_orientation)

        box = [float(min(gx0, gx1)), float(min(gy0, gy1)),
               float(max(gx0, gx1)), float(max(gy0, gy1))]
        out[name] = {
            'duration_ms':   gs.duration_ms,
            'device':        gs.device,
            'grasppose':     grasppose,
            # 'loc_3d':        [grasppose[0], grasppose[1], depth_final],
            'pose_3d':       [grasppose[0], grasppose[1], depth_final],
            'box':           box,
            'score':         float(getattr(g, 'quality', 0.0)),
            'islying':       False,
            'mass_percents': [50, 50],
            'depths':        {'obj_median': depth_value, 'bound': None, 'final': depth_final},
        }

    # _log_annotated(res, rgb, target_grasp=target_for_draw)
    save_detection_dataset(rgb=rgb, depth=depth, results=out, tag='grasp')
    return {'isdone': len(out) > 0, 'ins': out}


# ── Public skills ─────────────────────────────────────────────────────────────
@exception_handler
def detect(node, **kwargs):
    """Object detection with grounding-dino (camera defaults to FIND_CONFIGS).

    Required (one of): inputs='obj1,obj2,...'  OR  obj_names=[...].
    Returns ``{'isdone', 'ins': {name: {...}}}``."""
    obj_names = _parse_obj_names(kwargs.pop('inputs', None), kwargs.pop('obj_names', None))
    camera = kwargs.pop('camera', FIND_CONFIGS['camera'])
    return _detect_objects(node, obj_names, camera=camera, kwargs=kwargs)


@exception_handler
def find_once(node, **kwargs):
    loc = kwargs.get('loc', None)
    camera = kwargs.pop('camera', 'head')
    if loc is not None:
        ret = move(node=node, inputs=loc)
        if not ret['isdone']:
            return ret
    if 'head' in camera:
        view = kwargs.pop('view', 'down')
        moveh(node=node, ry=view)
        time.sleep(1)

    obj_names = _parse_obj_names(kwargs.get('inputs', None), kwargs.pop('obj_names', None))
    ret = _detect_objects(node, obj_names, camera=camera, **kwargs)
    moveh(node=node, ry='straight')
    return ret


@exception_handler
def find(node, **kwargs):
    """Find objects from the head camera (grounding-dino). `inputs` is
    'cup,bottle@table' (objects, optional '@location'). Retries over `views`
    and, if `once=False`, over every ENV location."""
    loc_name = kwargs.pop('inputs', None)
    if loc_name is None:
        raise Exception(f'inputs :{loc_name}')
    splits = loc_name.split('@')
    caption, loc = (splits[0], '@'.join(splits[1:])) if len(splits) >= 2 else (loc_name, None)

    obj_names = [el.strip() for el in caption.split(',') if el.strip()]
    kwargs['obj_names'] = obj_names

    views = kwargs.pop('views', ['down'])
    once = kwargs.pop('once', True)

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
    print(f'Recognition failed. Try to find in {locs}')
    for loc in locs:
        ret = run(loc, views=views)
        if ret['isdone']:
            return ret

    raise Exception(f'find {obj_names} failed ...')


@exception_handler
def find_arm(node, **kwargs):
    """From the wrist camera:
      - estimate_grasp=True (default) → grasp-gd grasp pose (`grasppose`).
      - estimate_grasp=False          → grounding-dino object location (`loc_3d`)."""
    obj_names = _parse_obj_names(kwargs.get('inputs', None), kwargs.pop('obj_names', None))
    estimate_grasp = kwargs.pop('estimate_grasp', True)

    if not estimate_grasp:
        kwargs.pop('detector', None)
        return _detect_objects(node, obj_names, **kwargs)

    to_find_handle = any('handle' in n for n in obj_names)
    kwargs.setdefault('gripper_min', 20)
    kwargs.setdefault('gripper_max', 200 if to_find_handle else 400)
    kwargs.pop('detector', None)
    return _detect_grasps(node, obj_names, **kwargs)


@exception_handler
def grasp_succeed(node, **kwargs):
    # x0, y0, x1, y1 = kwargs.get('crop_roi', [448, 333, 464, 347])
    x0, y0, x1, y1 = kwargs.get('crop_roi', [448, 359, 464, 377]) #
    cam_data = _fetch_arm(node=node)
    rgb, depth = cam_data.rgb, cam_data.depth

    rgb_out = show_box_on_rgb(rgb=rgb, box=[x0, y0, x1, y1], thick=2)
    log_data({'log_image': rgb_out})

    depth_roi = depth[y0:y1, x0:x1]
    obj_depth = np.median(depth_roi[depth_roi > 0]) / 1000. - ARM_CONFIGS['wrist_tool_length']
    return {'isdone': -0.250 < obj_depth < 0.020, 'obj_depth': obj_depth}


# ── Side-pose flow (legacy: fastsam via 'vlms' TCP) ──────────────────────────
@exception_handler
def get_side_pose_3d(node, **kwargs):
    """Find a free pose to the side of a scene from the head camera (fastsam +
    make_side_box). Returns ``{'isdone': True, 'pose_3d': [...]}``."""
    side = kwargs.pop('side', 'beside')
    view = kwargs.pop('view', 'down')
    pose_3d_in = kwargs.pop('pose_3d', None)

    moveh(node=node, inputs=view, wait=True)
    time.sleep(1)
    cam = _fetch_head(node)
    robot_mode = get_robot_mode(node=node)
    moveh(node=node, inputs='straight', wait=False)

    sam_ins = call_detector({
        'rgb':           cam.rgb,
        'detector':      'fastsam',
        'caption':       'obj',
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
    side_pose_list[2] = pose_3d[2]
    if len(pose_3d) > 3:
        side_pose_list = side_pose_list + [pose_3d[3]]

    log_data({'xc_yc_depth': [int(xc), int(yc), side_depth], 'side_pose': side_pose_list})

    if not is_inside_workspace_box(*side_pose_list[:3], workspace=ARM_CONFIGS['range']):
        return {'isdone': True, 'pose_3d': pose_3d}

    rgb_out = show_box_on_rgb(cam.rgb, side_box, thick=2)
    log_data({'log_image': rgb_out})

    return {'isdone': True, 'pose_3d': side_pose_list}

@exception_handler
def find_place(node, **kwargs) -> dict:
    
    point3d = kwargs.pop('point3d', None)
    inputs = kwargs.pop('inputs', None)
    islying = kwargs.pop('islying', False)
    placeside = kwargs.pop('placeside', None)
    
    assert inputs is not None or point3d is not None
    robot_mode = get_robot_mode(node=node)

    place_beside = True
    # get pose
    if point3d is None:
        # branch 1:
        env = get_env_specs(inputs, ENV)
        if len(env)==0: 
            target_obj = inputs.split('@')[0]
            ret = find(node=node, inputs=inputs, simple_return=True)
            point3d = ret['ins'][target_obj]['pose']
            place_beside = placeside is not None
        # branch 2:
        else:
            # env = get_env_specs(env, ENV)
            target_height = env.get('height', 0.75)
            robot_mode = env['default_mode']
            placepose = dict(env.get('placepose', None) or {})
            placepose.update(kwargs.pop('placepose', {}) or {})
            point3d = _get_placepose(
                placepose or None, target_height, robot_mode,
                islying,
            )

    # avoid obstacle
    if place_beside:
        point3d = _detect_nearest(node=node, pose=point3d)
        
        

    lift_to = point3d[2] + 0.1
    point3d[2] += 0.05      #add 5cm height to place
    

    dz_up = 0.1
    approach_pose = copy.deepcopy(point3d)
    approach_pose[-1] +=dz_up
    
    if islying:
        dapproach = 0.1
        approach_pose[-1] += dapproach
        approach_pose[1] -= 0.1 if robot_mode=='right' else -0.1
        approach_pose += [-180., 0., 90. if robot_mode=='right' else -90.]
        approach_lying=True
    else:
        ry, dapproach = 30, 0.25
        sint, cost = np.sin(ry*np.pi/180), np.cos(ry*np.pi/180)
        approach_pose[0] += dapproach*sint
        approach_pose[1] += (-dapproach if robot_mode=='right' else dapproach)*cost
        approach_pose[2] += dapproach*np.sin(15*np.pi/180)
        approach_pose += [-180, -75, 100+ry if robot_mode=='right' else -100-ry]
        approach_lying=False
    mforward = approach_pose[0] - np.clip(approach_pose[0], 0.15, 0.55)
    mforward = 0 if abs(mforward)<0.1 else mforward
    approach_pose[0] -= mforward

    approach_pose = {k:v for k,v in zip(['x', 'y', 'z', 'rx', 'ry', 'rz'], approach_pose)}

    base_rotate = 60 if robot_mode=='right' else -60
    return {
        'isdone': True,
        'islying':       islying,
        'pose_3d':       point3d,
        'approach_pose': approach_pose,
        'lift_to':       lift_to,
        'mforward':      mforward,
        'base_rotate':   base_rotate,
        'approach_lying':   approach_lying,
        'dapproach':    dapproach,
        'dz_up':   dz_up,        
    }
    

if __name__ == '__main__':
    pass


