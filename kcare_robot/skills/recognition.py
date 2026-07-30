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
keeps the registered skills plus the single detection orchestrator
(`_detect_objects`), which serves both cameras (the wrist camera additionally
runs grasp-gd for a `grasppose`).
"""

import numpy as np, time, copy, cv2, threading

from visionserve.utils import show_box_on_rgb, get_valid_depth_locs
from visionserve.postprocess import select_target_object, select_target_grasp

from robot_agent.skill_configs import FIND_CONFIGS, ARM_CONFIGS, ENV
from robot_agent.utils import exception_handler, get_env_specs, run_parallel_check
from robot_agent.skills import log_data
from kcare_robot.skills.head import moveh, get_robot_mode
from kcare_robot.skills.mobile import move
from kcare_robot.skills.pointcloud import get3d
from kcare_robot.skills.arm import movej, movet, movel

from kcare_robot.skills._recognition_helpers import (
    fetch_camera_data, _fetch_head, _fetch_arm,
    _vs_client, _get_placepose,
    _bbox_to_xyxy, _box_islying_pca, _box_depths,
    _cam_gravity, _object_det_select_configs, _grasp_configs,
    _mask_for_target, _emit_detection_vis, _predict_detect,
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
    stride       = configs['stride']
    band         = configs.get('place_height_band', 0.05)            # ±m around the target support height
    avoid_model  = configs.get('avoid_model', 'grounded-sam')   # open-vocab + mask detector for objects to avoid
    avoid_prompt = configs.get('avoid_prompt', 'object.')            # class-agnostic: any object, flat or tall
    avoid_conf   = configs.get('avoid_conf', 0.25)
    avoid_pad    = configs.get('avoid_pad', 8)                       # enlarge each object box (px) before excluding
    avoid_max    = configs.get('avoid_max_area', 0.15)              # ignore boxes larger than this fraction of the image: "object." also boxes the empty TABLE/couch itself — those are the SURFACE, not an obstacle

    # Two cheap, complementary filters so a place point never lands on an object:
    #   1) detect objects (open-vocab "object.") and exclude their footprint —
    #      catches FLAT objects (phone/card) a height test can't, ~0.3s. When the
    #      detector returns SEGMENTATION masks (e.g. rfdetr-gdino-sam-etri) the
    #      tight per-object mask is used instead of the bbox, so round/irregular
    #      objects don't blank out the surrounding table; else fall back to boxes.
    #   2) height band around the target support height `pose[2]` — rejects the
    #      floor/couch (lower) and any TALL object the detector missed (higher);
    #      free, since get3d is run anyway.
    objects = np.zeros((h, w), bool)
    od = None
    max_px = avoid_max * h * w        # a footprint bigger than this is the SURFACE (empty table/couch), not an obstacle
    try:
        od = _vs_client().predict(model=avoid_model, image=cam.rgb, prompt=avoid_prompt,
                                  box_threshold=avoid_conf, text_threshold=0.2)
        masks = od.masks or []
        if masks:
            for m in masks:
                mm = m.to_ndarray(width=w, height=h) > 0
                if mm.sum() > max_px:              # whole-surface mask → not an obstacle
                    continue
                if avoid_pad:
                    mm = cv2.dilate(mm.astype('uint8'), np.ones((avoid_pad, avoid_pad), 'uint8')).astype(bool)
                objects |= mm
        else:
            for d in od.detections:
                bx, by, bw, bh = [int(v) for v in d.bbox]
                if bw * bh > max_px:               # whole-surface box → not an obstacle
                    continue
                x0, y0 = max(0, bx - avoid_pad), max(0, by - avoid_pad)
                x1, y1 = min(w, bx + bw + avoid_pad), min(h, by + bh + avoid_pad)
                objects[y0:y1, x0:x1] = True
    except Exception:
        pass

    margin_cells = configs.get('place_margin_cells', 2)   # erode the placeable mask by N grid cells (≈ N*stride px) → clearance from objects AND table edges

    # Sample a full-frame grid; keep cells that miss every object box.
    gyg, gxg = np.mgrid[0:h:stride, 0:w:stride]
    gh, gw = gyg.shape
    gxf, gyf = gxg.ravel(), gyg.ravel()
    free = ~objects[gyf, gxf]
    if not free.any():
        return pose
    points = [[int(x), int(y)] for x, y in zip(gxf[free], gyf[free])]

    ret = get3d(node=node, points=points)
    assert ret['isdone'], f'{ret}'

    # Map per-free-point 3D back onto the full grid; mark cells on the support plane.
    P_full = np.full((gxf.size, 3), np.nan)
    P_full[free] = np.asarray(ret['pose'])
    valid = ~np.isnan(P_full).any(axis=-1)
    on_plane = valid & (np.abs(P_full[:, 2] - pose[2]) < band)

    place_grid = on_plane.reshape(gh, gw).astype('uint8')

    # Other surfaces (a couch cushion, a shelf) can sit at nearly the same height
    # as the target table, so the raw height-band can span several disconnected
    # regions. Keep only the connected blob that is both a real surface (enough
    # cells) and NEAREST the robot base — the actual target support is always the
    # close one; a couch a metre further back is not. Falls back to the full band
    # if there's only one component (or none pass the size filter).
    min_comp = configs.get('place_min_component_cells', 6)
    num_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(place_grid, connectivity=8)
    if num_cc > 2:   # background (0) + more than one candidate component
        P_grid = P_full.reshape(gh, gw, 3)
        best_label, best_dist = None, None
        for lbl in range(1, num_cc):
            if stats_cc[lbl, cv2.CC_STAT_AREA] < min_comp:
                continue
            comp = labels_cc == lbl
            pts3 = P_grid[comp]
            pv = ~np.isnan(pts3).any(axis=-1)
            if not pv.any():
                continue
            dist = float(np.nanmean(np.linalg.norm(pts3[pv][:, :2], axis=-1)))   # planar distance from robot base
            if best_dist is None or dist < best_dist:
                best_dist, best_label = dist, lbl
        if best_label is not None:
            place_grid = (labels_cc == best_label).astype('uint8')

    if margin_cells > 0:
        k = 2 * margin_cells + 1
        eroded = cv2.erode(place_grid, np.ones((k, k), 'uint8'))
        if eroded.any():
            place_grid = eroded               # keep clearance; fall back if it empties out
    sel = place_grid.ravel().astype(bool)
    if not sel.any():
        sel = valid                           # last resort: any valid free point
        if not sel.any():
            return pose

    cand_xy = np.stack([gxf, gyf], axis=1)[sel]
    cand_P = P_full[sel]
    weights = np.array([0.2, 1.0, 1.0])
    dist2 = np.sum(weights * (cand_P - np.array(pose).reshape(1, 3)) ** 2, axis=1)
    argmin = int(np.argmin(dist2))
    x, y = int(cand_xy[argmin, 0]), int(cand_xy[argmin, 1])

    # Visualize: detected objects (boxes) + placeable mask (green) + chosen spot (marker).
    annotated = np.asarray(od.visualize(cam.rgb)).copy() if od is not None else np.asarray(cam.rgb).copy()
    place_full = cv2.resize(place_grid * 255, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    overlay = np.zeros_like(annotated); overlay[place_full] = (0, 200, 0)
    annotated = cv2.addWeighted(annotated, 0.75, overlay, 0.25, 0)
    cv2.drawMarker(annotated, (x, y), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 24, 3)
    log_data({'log_image': annotated})

    return cand_P[argmin].tolist()


# ── Core: object detection (grounding-dino) + wrist grasp (grasp-gd) ─────────
def _detect_objects(node, obj_names, **kwargs) -> dict:
    """Detect `obj_names` on one camera and return a per-object entry.

    Both cameras run the same grounding-dino detector; the camera selects the
    geometry, and the wrist additionally runs grasp-gd:

      head → object pose in the BASE frame (`get3d`) + an approach pose.
      arm  → object pose in the CAMERA frame (`Ixy2xyz`, metres) AND — unless
             ``estimate_grasp=False`` — a grasp-gd ``grasppose`` for the same box.

    `islying` is the single-camera PCA estimate (no cross-camera fusion / VLM).

    Returns ``{'isdone', 'ins': {name: {pose_3d, approach_pose, lift_to, mforward,
    base_rotate, approach_lying, box, score, islying, mass_percents, depths,
    grasppose?, grasp_score?}}}`` (or ``{pose, islying, grasppose?}`` when
    ``simple_return``)."""
    simple_return    = kwargs.pop('simple_return', False)
    camera           = kwargs.pop('camera', 'arm')
    estimate_grasp   = kwargs.pop('estimate_grasp', False)
    keep_orientation = kwargs.pop('keep_orientation', False)
    num_trials = kwargs.pop('num_trials', 3)
    use_head   = 'head' in camera
    with_grasp = (not use_head) and estimate_grasp     # grasp-gd: wrist camera only
    robot_mode = get_robot_mode(node=node)

    cam = fetch_camera_data(node, camera)
    rgb, depth, cam_params = cam.rgb, cam.depth, cam.cam_params
    # h, w = rgb.shape[:2]
    gravity_cam = _cam_gravity(node, cam, use_head)

    min_conf = kwargs.pop('min_conf', 0.1 if any('handle' in n for n in obj_names) else 0.0)
    # Grasp selection uses the closer `detect_grasp` profile; plain detection the
    # default `detect_object` profile.
    cfg_key = 'detect_grasp' if with_grasp else 'detect_object'
    det_configs, select_configs = _object_det_select_configs(kwargs, cfg_key=cfg_key)
    if 'box_threshold' in kwargs:
        det_configs['box_threshold'] = kwargs['box_threshold']
    grasp_configs = _grasp_configs() if with_grasp else None

    out: dict = {}
    panels: list = []
    for name in obj_names:
        for _ in range(num_trials):
            cam = fetch_camera_data(node, camera)
            rgb, depth, cam_params = cam.rgb, cam.depth, cam.cam_params
            h, w = rgb.shape[:2]
            
            res = _predict_detect(rgb, f'{name.strip()}.', det_configs)
            if min_conf > 0:
                res = res.filter_by_conf(min_conf=min_conf)
            if len(res.detections) != 0:
                break
        if len(res.detections) == 0:
            continue
        target = select_target_object(
            res, cls=name, image_size=(w, h), depth_result=depth, 
            intrinsics=cam_params, **select_configs)
        if target is None:
            continue

        det_box = _bbox_to_xyxy(target.bbox)        # object bbox → islying + viz

        # Wrist camera: grasp-gd on the box also yields the grasppose, a base-frame
        # pose_3d (get3d_arm) and the grasped-object mask (a better islying source
        # than the bbox).
        grasp, obj_mask = (None, None)
        if with_grasp:
            grasp, obj_mask = _grasp_from_box(node, target, rgb, depth, cam_params, grasp_configs,
                                              keep_orientation=keep_orientation,
                                              select_target_grasp=select_target_grasp)
        mask = obj_mask if obj_mask is not None else _mask_for_target(res, target, w, h)
        islying = _box_islying_pca(name, det_box, depth, cam_params, gravity_cam, mask=mask)

        panels.append({
            'name': name, 'box': det_box, 'vote': islying,
            'score': float(getattr(target, 'conf', 0.0)),
            'grasp_line':  grasp.get('line') if grasp else None,
            'grasp_label': _grasp_label(grasp) if grasp else '',
        })

        # Entry geometry: the grasp (pose_3d in the BASE frame, jaw box, grasp
        # quality) when grasp-gd ran; else the detection box (head → base frame via
        # get3d, wrist coarse → camera frame via Ixy2xyz).
        if grasp is not None:
            pose, ebox, escore, box_depths = grasp['pose_3d'], grasp['box'], grasp['score'], grasp['depths']
        else:
            box_depths = _box_depths(depth, det_box)
            pose   = _object_pose_3d(node, det_box, box_depths, depth, cam_params, use_head=use_head)
            ebox, escore = det_box, float(getattr(target, 'conf', 0.0))

        if simple_return:
            entry = {'pose': pose, 'islying': islying}
            if grasp is not None:
                entry['grasppose'] = grasp['grasppose']
            out[name] = entry
            continue

        entry = {
            'duration_ms':   res.duration_ms,
            'device':        res.device,
            'pose_3d':       pose,
            **({} if with_grasp else _build_approach_pose(pose, islying, robot_mode)),
            'box':           ebox,
            'score':         escore,
            'islying':       islying,
            'mass_percents': [50, 50],
            'depths':        box_depths,
        }
        if grasp is not None:
            entry['grasppose']   = grasp['grasppose']
            entry['grasp_score'] = grasp['score']
        out[name] = entry

    # One annotated frame for this camera with every object's box + grasp.
    _emit_detection_vis(node, rgb, camera, panels)
    save_detection_dataset(rgb=rgb, depth=depth, results=out, tag=camera)
    return {'isdone': len(out) > 0, 'ins': out}




# ── Public skills ─────────────────────────────────────────────────────────────
@exception_handler
def detect(node, **kwargs):
    """Object detection with grounding-dino (camera defaults to FIND_CONFIGS).

    Required (one of): inputs='obj1,obj2,...'  OR  obj_names=[...].
    Returns ``{'isdone', 'ins': {name: {...}}}``."""
    obj_names = _parse_obj_names(kwargs.pop('inputs', None), kwargs.pop('obj_names', None))
    camera = kwargs.pop('camera', 'arm')
    return _detect_objects(node, obj_names, camera=camera, **kwargs)


@exception_handler
def find_once(node, **kwargs):
    """Detect on the head and wrist cameras concurrently, then merge per object,
    preferring the wrist (arm) result whenever it sees the object.

    The head tilts to `view` (down) while the arm moves to `pre_pick`, so both
    cameras frame the workspace; one detection request then goes to each camera.
    An arm entry carries a grasp-gd `grasppose`, so a downstream pick prefers it;
    the head entry (base-frame pose + approach) is the fallback used only for
    objects the wrist camera missed."""

    robot_mode = get_robot_mode(node=node)
  
    move_arm = kwargs.pop("move_arm", True)
    cameras = kwargs.pop("cameras", "arm, head")
    cameras = [el.strip().lower() for el in cameras.split(',')]
    priority_cam, secondary_cam = (cameras[0], None) if len(cameras)==1 else cameras[:2]

    loc = kwargs.get('loc', None)
    if loc is not None:
        ret = move(node=node, inputs=loc)
        if not ret['isdone']:
            return ret

    # Aim both cameras at the workspace (head down ∥ arm to pre_pick) before detect.
    view = kwargs.pop('view', 'down')

    obj_names = _parse_obj_names(kwargs.get('inputs', None), kwargs.pop('obj_names', None))
    kwargs.pop('camera', None)        # camera is fixed per branch below

    results = {}
    def detect_on(camera):
        # A camera that fails to detect (or whose grasp estimation raises) must not
        # sink the other camera — degrade to an empty result instead.
        try:
            if camera=='head':
                ret = moveh(node=node, inputs=view)
                assert ret['isdone'], f'{ret}'
                time.sleep(1)
            else: 
                if move_arm:
                    ret = movej(node=node, inputs='give')
                    assert ret['isdone'], f'{ret}'
                
            results[camera] = _detect_objects(node, obj_names, camera=camera, **kwargs)
            if camera=='head':
                ret = moveh(node=node, ry='straight')
                assert ret['isdone'], f'{ret}'
        except Exception as e:
            results[camera] = {'isdone': False, 'ins': {}}

    if secondary_cam is not None:
        detect_head_thread = threading.Thread(target=detect_on, args=(secondary_cam,), daemon=True)
        detect_head_thread.start()

    detect_on(priority_cam)
    ret = results[priority_cam]
    if not ret['isdone'] and secondary_cam is not None:
        detect_head_thread.join()
        ret = results[secondary_cam]

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
    """Wrist-camera detection of `inputs` (object name):
      - estimate_grasp=True (default) → grasp-gd grasp pose (`grasppose`).
      - estimate_grasp=False          → object location only (no grasp).
    Grasp gripper bounds come from the `detect_grasp` config profile."""
    obj_names = _parse_obj_names(kwargs.get('inputs', None), kwargs.pop('obj_names', None))
    kwargs['estimate_grasp'] = True
    kwargs.pop('detector', None)
    return _detect_objects(node, obj_names, camera='arm', **kwargs)


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
    # frac_near = float((valid < near_mm).sum()) / max(depth_roi.size, 1)   # ROI fraction "near"
    frac_near = float((valid < near_mm).sum()) / max((0.1*depth_roi.size+ 0.9*valid.size), 1)   # ROI fraction "near"
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
            splits = inputs.split('@')
            dest_obj, dest_loc = (splits[0], '@'.join(splits[1:])) if len(splits)>1 else (inputs, None)
            env = get_env_specs(dest_loc, ENV)
        else:
            dest_obj, dest_loc = None, inputs
        #
        if len(env)>0:
            target_height = env.get('height', 0.75)
            robot_mode = env['default_mode']
            placepose = dict(env.get('placepose', None) or {})
            placepose.update(kwargs.pop('placepose', {}) or {})
            point3d = _get_placepose(
                placepose or None, target_height, robot_mode,
                islying,
            )
        #
        if dest_obj is not None: 
            ret = find(node=node, inputs=dest_obj, cameras="head", simple_return=True)
            p3 = ret['ins'][dest_obj]['pose']
            if point3d is None:
                point3d = p3
            else:
                point3d[:2] = p3[:2]
            place_beside = placeside is not None


    # avoid obstacle
    if place_beside and 'trash' not in inputs and 'sink' not in inputs:
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

    # calib tune
    dx = FIND_CONFIGS['detect_background']['calib_tune']['dx']
    dy = FIND_CONFIGS['detect_background']['calib_tune']['dy']
    approach_pose['x'] += dx  if robot_mode=='right' else -dx
    approach_pose['y'] += dy  if robot_mode=='right' else -dy
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