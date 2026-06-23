"""recognition — simplified detector skills built on the VisionServe SDK.

Two entry points, each bound to one camera + one model:

    find(...)      head camera  → 'grounding-dino'  → object 3D pose (base frame, via get3d)
    find_grasp(...)  wrist camera → 'grasp-gd'        → grasp pose      (camera frame, via Ixy2xyz)

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

from visionserve.utils import show_box_on_rgb, get_valid_depth_locs
from visionserve.postprocess import select_target_object, select_target_grasp

from robot_agent.skill_configs import FIND_CONFIGS, ARM_CONFIGS, ENV
from robot_agent.utils import exception_handler, get_env_specs, run_parallel_check
from robot_agent.skills import log_data
from kcare_robot.skills.head import moveh, get_robot_mode
from kcare_robot.skills.mobile import move
from kcare_robot.skills.pointcloud import get3d
from kcare_robot.skills.arm import movej

from kcare_robot.skills._recognition_helpers import (
    fetch_camera_data, _fetch_head, _fetch_arm,
    _vs_client, _prompt, _log_annotated, _get_placepose,
    _bbox_to_xyxy, _box_islying_pca, _gravity_in_cam, _box_depths,
    _arm_horizon_yz, _cam_gravity, _object_det_select_configs, _islying_consensus, _reconcile_grasp_islying,
    _mask_for_target, _emit_islying_vis, _none_res, _predict_detect, _fused_head_arm,
    _object_pose_3d, _build_approach_pose, _grasp_from_box, _grasp_label,
    _parse_obj_names,
    call_detector, make_side_box, _head_calib_func_factory, is_inside_workspace_box,
    save_detection_dataset,
)

def _detect_nearest(node, pose, **kwargs) -> dict:
    # ret = moveh(node=node, inputs="down")
    # assert ret['isdone'], f'{ret}'
    # time.sleep(0.5)

    cam = fetch_camera_data(node, 'head')
    h, w =  cam.rgb.shape[:2]
    fx,fy, cx, cy = cam.cam_params

    configs = FIND_CONFIGS['detect_background']
    if not configs['use']:
        return pose
    det_configs = {k:v for k,v in configs.items() if k in ['model', 'roi', 'method', 'dilate', 'method', 'bg_min_size', 'bg_min_size']}
    stride = configs['stride']

    res = _vs_client().predict(image=cam.rgb, depth=cam.depth, **det_configs)
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


# ── Core: grasp detection (grasp-gd) ─────────────────────────────────────────
_MISSING = object()


def _detect_grasps(node, obj_names, **kwargs: dict) -> dict:
    """grasp-gd on the wrist camera → per-object 3D grasp pose.

    `islying` is reconciled per `_reconcile_grasp_islying`: the arm camera always
    produces its own estimate; a caller-supplied `islying` (from the prior `find`)
    is trusted only when it agrees, otherwise a cross-camera consensus runs.
    """
    kwargs.pop('camera', 'arm')
    keep_orientation = kwargs.pop('keep_orientation', False)
    islying_kw = kwargs.pop('islying', _MISSING)
    fuse = FIND_CONFIGS.get('fuse_islying', False)   # False → arm camera only (no head cross-check)
    disagree_trust = FIND_CONFIGS.get('disagree_trust', 'arm')   # cam to trust if VLM unavailable on disagreement

    cam = _fetch_arm(node)
    rgb, depth, cam_params = cam.rgb, cam.depth, cam.cam_params
    h, w = rgb.shape[:2]

    configs = FIND_CONFIGS['detect_grasp']
    det_configs    = {k:v for k,v in configs.items() if k in ['model','min_size', 'max_size', 'text_threshold', 'box_theshold']}
    select_configs = {k:v for k,v in configs.items() if k in ['near_point', 'target_distance', 'distance_sigma']}
    grasp_configs  = {k:v for k,v in configs.items() if k in ['gripper_min', 'gripper_max']}

    res = _vs_client().predict(image=rgb, prompt=_prompt(obj_names), **det_configs)
    _log_annotated(res, rgb)
    assert len(res.detections) > 0, f'No object detected'

    gravity_cam = _gravity_in_cam(node, use_head=False,
                                  horizon_yz=_arm_horizon_yz(node))

    out: dict = {}
    for name in obj_names:
        obj = select_target_object(res, cls=name, image_size=(w, h),
            depth_result=depth, intrinsics=cam_params, **select_configs)
        assert obj is not None, f"failed to select target object"

        x0, y0, dx, dy = [int(el) for el in obj.bbox]
        arm_box = [x0, y0, x0+dx, y0+dy]

        # Grasp first (emits its own annotated log_image + the object mask), then
        # islying last so the islying debug view is the visible log_image.
        grasp, obj_mask = _grasp_from_box(node, obj, rgb, depth, cam_params, grasp_configs,
                                          keep_orientation=keep_orientation,
                                          select_target_grasp=select_target_grasp)
        islying_arm = _box_islying_pca(name, arm_box, depth, cam_params, gravity_cam, mask=obj_mask)

        if fuse:
            islying = _reconcile_grasp_islying(node, name, islying_arm, islying_kw, _MISSING,
                                               seed=('arm', islying_arm, rgb, arm_box),
                                               disagree_trust=disagree_trust)
        else:
            islying = islying_arm
            _emit_islying_vis(node, name, _none_res('head'),
                              {'camera': 'arm', 'vote': islying_arm, 'rgb': rgb, 'box': arm_box,
                               'grasp_line': grasp.get('line'), 'grasp_label': _grasp_label(grasp)},
                              islying)
        out[name] = {**grasp, 'islying': islying, 'mass_percents': [50, 50]}

    save_detection_dataset(rgb=rgb, depth=depth, results=out, tag='grasp')
    return {'isdone': len(out) > 0, 'ins': out}


# ── Core: object detection (grounding-dino) ──────────────────────────────────
def _detect_objects(node, obj_names, **kwargs) -> dict:
    """grounding-dino on `camera`. Head → 3D via get3d (base frame); wrist →
    3D via Ixy2xyz (camera frame, metres)."""
    simple_return = kwargs.pop('simple_return', False)
    camera = kwargs.pop('camera', 'head')
    fuse = kwargs.get('fuse_islying', FIND_CONFIGS['detect_object']['fuse_islying'])   # False → `camera` only (no cross-check)
    disagree_trust = kwargs.get('disagree_trust', FIND_CONFIGS['detect_object']['disagree_trust'])   # cam to trust if VLM unavailable on disagreement
    robot_mode = get_robot_mode(node=node)
    use_head = 'head' in camera

    cam = fetch_camera_data(node, camera)
    rgb, depth, cam_params = cam.rgb, cam.depth, cam.cam_params
    h, w = rgb.shape[:2]

    min_conf = kwargs.pop('min_conf', 0.1 if any('handle' in n for n in obj_names) else 0.0)
    det_configs, select_configs = _object_det_select_configs(kwargs)

    out: dict = {}
    for name in obj_names:
        grasp = None
        if fuse and use_head:
            # HEAD detection ∥ ARM grasp-gd ∥ speculative VLM, run concurrently.
            # The arm branch also yields a `grasppose` (best-effort, None if the
            # wrist camera can't see the object yet).
            fr = _fused_head_arm(node, name, rgb=rgb, depth=depth, cam_params=cam_params, cam=cam,
                                 use_head=use_head, det_configs=det_configs, select_configs=select_configs,
                                 min_conf=min_conf, disagree_trust=disagree_trust)
            assert fr is not None, f'No object detected'
            res, target, box = fr['res'], fr['target'], fr['box']
            pose, box_depths, islying = fr['pose'], fr['box_depths'], fr['islying']
            grasp = fr.get('grasp')
        else:
            # Single-camera (or arm-primary) path with grounded-sam fallback.
            res = _predict_detect(rgb, f'{name.strip()}.', det_configs)
            if min_conf > 0:
                res = res.filter_by_conf(min_conf=min_conf)
            assert len(res.detections) > 0, f'No object detected'
            target = select_target_object(
                res, cls=name, image_size=(w, h), depth_result=depth, intrinsics=cam_params, **select_configs)
            assert target is not None, f'Failed to select target object'

            box = _bbox_to_xyxy(target.bbox)
            box_depths = _box_depths(depth, box)
            pose = _object_pose_3d(node, box, box_depths, depth, cam_params, use_head=use_head)

            pmask = _mask_for_target(res, target, w, h)
            islying_primary = _box_islying_pca(name, box, depth, cam_params,
                                               _cam_gravity(node, cam, use_head), mask=pmask)
            pcam = 'head' if use_head else 'arm'
            if fuse:
                islying = _islying_consensus(node, name, min_conf=min_conf,
                                             seed=(pcam, islying_primary, rgb, box),
                                             disagree_trust=disagree_trust)
            else:
                islying = islying_primary
                pres = {'camera': pcam, 'vote': islying_primary, 'rgb': rgb, 'box': box}
                _emit_islying_vis(node, name, pres if pcam == 'head' else _none_res('head'),
                                  pres if pcam == 'arm' else _none_res('arm'), islying)

        if simple_return:
            entry = {'pose': pose, 'islying': islying}
            if grasp is not None:
                entry['grasppose'] = grasp['grasppose']
            out[name] = entry
            continue

        approach = _build_approach_pose(pose, islying, robot_mode)
        entry = {
            'duration_ms':   res.duration_ms,
            'device':        res.device,
            'pose_3d':       pose,
            **approach,
            'box':           box,
            'score':         float(getattr(target, 'conf', 0.0)),
            'islying':       islying,
            'mass_percents': [50, 50],
            'depths':        box_depths,
        }
        if grasp is not None:
            entry['grasppose']   = grasp['grasppose']
            entry['grasp_score'] = grasp['score']
        out[name] = entry
    # log_image is emitted per-object by the islying consensus (both-camera debug view).
    save_detection_dataset(rgb=rgb, depth=depth, results=out, tag=camera)
    return {'isdone': len(out) > 0, 'ins': out}




# ── Public skills ─────────────────────────────────────────────────────────────
@exception_handler
def detect(node, **kwargs):
    """Object detection with grounding-dino (camera defaults to FIND_CONFIGS).

    Required (one of): inputs='obj1,obj2,...'  OR  obj_names=[...].
    Returns ``{'isdone', 'ins': {name: {...}}}``."""
    obj_names = _parse_obj_names(kwargs.pop('inputs', None), kwargs.pop('obj_names', None))
    camera = kwargs.pop('camera', FIND_CONFIGS['camera'])
    return _detect_objects(node, obj_names, camera=camera, **kwargs)


@exception_handler
def find_once(node, **kwargs):
    fuse_islying = kwargs.get('fuse_islying', FIND_CONFIGS['detect_object']['fuse_islying'])
    loc = kwargs.get('loc', None)
    camera = kwargs.pop('camera', 'head')
    if loc is not None:
        ret = move(node=node, inputs=loc)
        if not ret['isdone']:
            return ret
    if 'head' in camera:
        view = kwargs.pop('view', 'down')
        ret = run_parallel_check(funcs=[
            lambda: (moveh(node=node, ry=view),time.sleep(1))[0],
            lambda: movej(node=node, inputs='pre_pick') if fuse_islying else {'isdone': True} 
        ])
        # moveh(node=node, ry=view)
        # time.sleep(1)

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
    splits = loc_name.split('|')
    loc_name, num_trials = splits if len(splits)==2 else (splits[0], 1)
    num_trials = int(num_trials)


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
    
    for i in range(num_trials):
        ret = run(loc, views)
        if ret['isdone']:
            return ret
    if once:
        return ret


    locs = list(ENV.keys())
    print(f'Recognition failed. Try to find in {locs}')
    for loc in locs:
        ret = run(loc, views=views)
        if ret['isdone']:
            return ret

    raise Exception(f'find {obj_names} failed ...')


@exception_handler
def find_grasp(node, **kwargs):
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
    """Presence check: a grasp succeeded if a meaningful chunk of the wrist-cam
    ROI is occupied by something NEAR the gripper (closer than the tool tip +
    `margin`). More robust than a single median-depth window — transparent /
    depth-holed objects still pass as long as ≥`min_frac` of the ROI reads near,
    and the far empty background never does."""
    x0, y0, x1, y1 = kwargs.get('crop_roi', FIND_CONFIGS['grasp_check_roi'])
    margin   = kwargs.get('grasp_check_margin',   FIND_CONFIGS.get('grasp_check_margin', 0.06))   # m beyond tip
    min_frac = kwargs.get('grasp_check_min_frac', FIND_CONFIGS.get('grasp_check_min_frac', 0.30))
    cam_data = _fetch_arm(node=node)
    rgb, depth = cam_data.rgb, cam_data.depth

    depth_roi = depth[y0:y1, x0:x1]
    valid = depth_roi[depth_roi > 0]
    tip_mm = ARM_CONFIGS['wrist_tool_length'] * 1000.
    near_mm = tip_mm + margin * 1000.
    frac_near = float((valid < near_mm).sum()) / max(depth_roi.size, 1)   # ROI fraction "near"
    obj_depth = float(np.median(valid)) / 1000. - ARM_CONFIGS['wrist_tool_length'] if valid.size else None
    isdone = bool(frac_near > min_frac)

    # Debug overlay: ROI box + presence stats.
    rgb_out = show_box_on_rgb(rgb=rgb, box=[x0, y0, x1, y1], thick=2)
    tag = (f"grasp {'OK' if isdone else 'FAIL'}  near={frac_near:.0%}"
           + (f"  d={obj_depth:.3f}m" if obj_depth is not None else "  no-depth"))
    org = (x0, max(22, y0 - 8))
    cv2.putText(rgb_out, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(rgb_out, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0) if isdone else (255, 0, 0), 2)
    log_data({'log_image': rgb_out})

    return {'isdone': isdone, 'frac_near': frac_near, 'obj_depth': obj_depth}


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


