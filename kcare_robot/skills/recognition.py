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

import json
import numpy as np, time, copy, cv2, threading

from visionserve.utils import show_box_on_rgb, get_valid_depth_locs
from visionserve.postprocess import select_target_object, select_target_grasp

from robot_agent.skill_configs import FIND_CONFIGS, ARM_CONFIGS, ENV
from robot_agent.utils import exception_handler, run_parallel_check
from kcare_robot.utils import get_env_specs
from robot_agent.skills import log_data
from kcare_robot.skills.head import moveh, get_robot_mode
from kcare_robot.skills.mobile import move
from kcare_robot.skills.pointcloud import get3d
from kcare_robot.skills.arm import movej, movet, movel

from kcare_robot.skills import _products
from kcare_robot.skills._recognition_helpers import (
    reset_log_image,
    fetch_camera_data, _fetch_head, _fetch_arm,
    _vs_client, _get_placepose,
    _bbox_to_xyxy, _box_islying_pca, _box_depths,
    _cam_gravity, _object_det_select_configs, _grasp_configs,
    _mask_for_target, _emit_detection_vis, _predict_detect,
    _object_pose_3d, _build_approach_pose, _grasp_from_box, _grasp_label,
    _parse_obj_names,
    call_detector, make_side_box, _head_calib_func_factory, is_inside_workspace_box,
    save_detection_dataset, _normalize_orientation
)

def _support_masks(node, masks, target_z, max_px, min_px, tol, flat, samples=40) -> set:
    """Indices of the masks (between min_px and max_px pixels) that are the
    support surface: flat and at the target height — measured with one get3d
    call over a few sampled pixels of each."""
    pts, owner = [], []
    rng = np.random.default_rng(0)
    for k, mm in enumerate(masks):
        n = int(mm.sum())
        if n < min_px or n > max_px:
            continue
        ys, xs = np.nonzero(mm)
        for i in rng.choice(n, min(samples, n), replace=False):
            pts.append([int(xs[i]), int(ys[i])])
            owner.append(k)
    if not pts:
        return set()
    ret = get3d(node=node, points=pts)
    if not ret.get('isdone', True):
        return set()
    z = np.asarray(ret['pose'], dtype=float)[:, 2]
    owner = np.asarray(owner)
    out = set()
    for k in set(owner.tolist()):
        zk = z[(owner == k) & ~np.isnan(z)]
        if len(zk) < 5:
            continue
        z50, z90 = np.percentile(zk, [50, 90])
        if abs(z50 - target_z) <= tol and z90 - z50 <= flat:
            out.add(k)
    if out:
        log_data({'msg': f'find_place: {len(out)} detected mask(s) at the target height {target_z:.2f} m '
                         'kept as the support surface, not obstacles'})
    return out


def _detect_nearest(node, pose, **kwargs) -> dict:
    ret = moveh(node=node, ry=-40)
    assert ret['isdone'], f'{ret}'
    time.sleep(0.5)

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
    # A smaller mask can still be the support itself: on find_place::sofa the
    # living-room table's mask was 10.0 % of the frame, right on avoid_max_area,
    # so the table was an "obstacle" in one frame and a surface in the next, and
    # the robot placed on the sofa. A mask that is big enough (surface_min_area,
    # so a phone / card lying on the table stays an obstacle), FLAT
    # (z90 - z50 <= surface_flat) and AT the target height (|z50 - target| <=
    # surface_tol) is the surface, whatever its size.
    surf_min     = configs.get('surface_min_area', 0.02)
    surf_tol     = configs.get('surface_tol', 0.03)
    surf_flat    = configs.get('surface_flat', 0.04)

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
            mms = [m.to_ndarray(width=w, height=h) > 0 for m in masks]
            surfaces = _support_masks(node, mms, pose[2], max_px, surf_min * h * w, surf_tol, surf_flat)
            for k, mm in enumerate(mms):
                if mm.sum() > max_px or k in surfaces:   # whole-surface mask / the support itself → not an obstacle
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

    # Clearance in METRES, not grid cells: a cell is a place point only if every edge of the support blob —
    # the table's own edge and the outline of each object cut out of it — is at least the object's radius
    # plus a margin away (the grid erosion was in pixels, so the far edge got only a few cm, and an empty
    # erosion fell back to the edge itself). Nothing left -> None: find_place fails instead of dropping
    # the object over an edge.
    need = float(kwargs.get('obj_radius') or configs.get('place_obj_radius', 0.05)) + float(configs.get('place_edge_margin', 0.03))
    P_grid = P_full.reshape(gh, gw, 3)
    blob = place_grid.astype(bool)
    edge = blob & ~cv2.erode(place_grid, np.ones((3, 3), 'uint8'), borderType=cv2.BORDER_REPLICATE).astype(bool)
    edge[[0, -1], :] = False; edge[:, [0, -1]] = False     # the image border is not the table's edge: it goes on
    edge_xy = P_grid[edge][:, :2]
    edge_xy = edge_xy[~np.isnan(edge_xy).any(axis=-1)]
    ok = blob.copy()
    if edge_xy.size:
        cells = np.argwhere(blob)
        cxy = P_grid[blob][:, :2]
        d = np.sqrt(((cxy[:, None, :] - edge_xy[None, :, :]) ** 2).sum(-1)).min(axis=1)
        ok[cells[:, 0], cells[:, 1]] = d >= need
    elif margin_cells > 0:                   # no 3D for the edge: the old grid erosion
        k = 2 * margin_cells + 1
        ok = cv2.erode(place_grid, np.ones((k, k), 'uint8')).astype(bool)
    place_grid = ok.astype('uint8')
    sel = place_grid.ravel().astype(bool)
    if not sel.any():
        log_data({'msg': f'find_place: no spot {need:.2f} m clear of the surface edges and objects'})
        return None

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
def _label_owner(cls: str, prompts: list):
    """Which prompt a detection's label belongs to, for a multi-label request:
    the label itself, else the longest prompt it contains or is part of
    ("white cup" → "cup"); None when it matches none."""
    c = ' '.join(str(cls or '').lower().split())
    if c in prompts:
        return c
    hits = [p for p in prompts if p and (p in c or (c and c in p))]
    return max(hits, key=len) if hits else None


# find::main_body=True (find_place's destination). A pot's box takes its handle: on find_place::steel pot
# (2026-10-14, head down, 27 frames) the top box held the handle in every frame for steel pot / stainless
# steel pot / cooking pot / saucepan / pot (metal bowl 4/15, at 0.43), and the box centre was 6.8-7.2 cm
# off the pot's centre. A handle is thin next to the body: opening the mask with a disk of BODY_OPEN_K x
# the largest inscribed radius cut it off in 27 / 27 frames (0 handle pixels left). The pose is then taken
# from the box of what is left, the same way as from a detection box.
BODY_OPEN_K = 0.5


def _main_body(mask):
    """The object's main body in `mask` (bool, same size): thin parts (a handle, a spout, a strap) cut off by an
    opening with a disk of BODY_OPEN_K x the largest inscribed radius; None when the mask is missing or tiny."""
    if mask is None:
        return None
    full = np.asarray(mask) > 0
    vv, uu = np.nonzero(full)
    if not len(uu):
        return None
    y0, y1, x0, x1 = vv.min(), vv.max() + 1, uu.min(), uu.max() + 1        # work on the mask's own box (~1 ms vs 0.2 s)
    m = np.pad(full[y0:y1, x0:x1].astype(np.uint8), 1)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    R = float(dt.max())
    if R < 5:
        return None
    rad = max(2, int(BODY_OPEN_K * R))
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1)))
    _, lab = cv2.connectedComponents(opened)
    v0, u0 = np.unravel_index(np.argmax(dt), dt.shape)
    if lab[v0, u0] == 0:
        return None
    body = np.zeros_like(full)
    body[y0:y1, x0:x1] = (lab == lab[v0, u0])[1:-1, 1:-1]
    return body


def _object_entry(node, name, target, res, rgb, depth, cam_params, w, h, with_grasp, grasp_configs,
                  keep_orientation, gravity_cam, use_head, robot_mode, simple_return, panels, tag, main_body=False):
    """One detected object's entry (pose, approach, grasp …) — and its panel."""
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
        'name': tag, 'box': det_box, 'vote': islying,
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
        pose_box = det_box
        if main_body:
            body = _main_body(mask)
            if body is None:
                log_data({'msg': f'find {tag}: no mask for main_body, detection box used'})
            else:
                vv, uu = np.nonzero(body)
                pose_box = [int(uu.min()), int(vv.min()), int(uu.max()) + 1, int(vv.max()) + 1]
                log_data({'msg': f'find {tag}: main body box {pose_box} (detection box {[int(v) for v in det_box]}, '
                                 f'{int((np.asarray(mask) > 0).sum() - body.sum())} px cut off)'})
        box_depths = _box_depths(depth, pose_box)
        pose   = _object_pose_3d(node, pose_box, box_depths, depth, cam_params, use_head=use_head)
        if use_head:
            # head camera calibration error: ARM_CONFIGS.calib_offset [dx, dy, dz] (m, base frame)
            pose = [float(p) + float(o) for p, o in zip(pose, ARM_CONFIGS.get('calib_offset') or [0, 0, 0])]
        ebox, escore = det_box, float(getattr(target, 'conf', 0.0))

    if simple_return:
        entry = {'pose': pose, 'islying': islying}
        if grasp is not None:
            entry['grasppose'] = grasp['grasppose']
        return entry

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
    return entry


# The wrist camera sees the robot's own gripper along the image bottom. On
# find::black bag at the dressroom (2026-10-13) the detector boxed the black
# gripper base (conf 0.38–0.46, median depth 114 mm) and gave the bag itself
# 0.28–0.38, below box_threshold 0.35 in 2 of 3 frames; with the gripper painted
# grey the bag scored 0.69–0.71 (beige / blue bags unchanged). So the gripper is
# hidden from the detector (_hide_gripper) and any box still on it is dropped
# (_robot_body). "The gripper" = depth nearer than the tool tip
# (wrist_tool_length + FIND_CONFIGS.self_depth_margin, 0.02 m).

def _gripper_near_mm() -> float:
    return (ARM_CONFIGS['wrist_tool_length'] + FIND_CONFIGS.get('self_depth_margin', 0.02)) * 1000.


def _hide_gripper(rgb, depth):
    """`rgb` with the pixels nearer than the tool tip painted grey — what the
    detector is shown; geometry and grasping keep the real image."""
    if depth is None:
        return rgb
    near = ((depth > 0) & (depth < _gripper_near_mm())).astype(np.uint8)
    if near.shape[:2] != rgb.shape[:2]:
        near = cv2.resize(near, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    if not near.any():
        return rgb
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # depth holes on the gripper
    near = cv2.dilate(near, np.ones((9, 9), np.uint8)) > 0
    out = rgb.copy()
    out[near] = 128
    return out


def _robot_body(res, depth, rgb_shape) -> set:
    """Indices of wrist-camera detections that are the robot's own gripper:
    at least `self_min_frac` of the box's depth nearer than the tool tip."""
    if depth is None:
        return set()
    near_mm = _gripper_near_mm()
    min_frac = FIND_CONFIGS.get('self_min_frac', 0.5)
    dh, dw = depth.shape[:2]
    sx, sy = dw / rgb_shape[1], dh / rgb_shape[0]
    out = set()
    for k, d in enumerate(res.detections):
        x, y, w, h = (float(v) for v in d.bbox)
        roi = depth[max(0, int(y * sy)):int((y + h) * sy), max(0, int(x * sx)):int((x + w) * sx)]
        valid = roi[roi > 0]
        if valid.size >= 20 and float((valid < near_mm).mean()) >= min_frac:
            out.add(k)
            log_data({'msg': f"{getattr(d, 'cls', '?')} {float(getattr(d, 'conf', 0)):.2f} at "
                             f"[{x:.0f}, {y:.0f}, {w:.0f}, {h:.0f}] is the gripper "
                             f"({100 * float((valid < near_mm).mean()):.0f}% nearer than {near_mm:.0f} mm) — skipped"})
    return out


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
    ``simple_return``).

    ``main_body=True``: the pose comes from the box of the object's main body (handle / thin parts cut off
    its mask, see _main_body) instead of the detection box.

    All names go to the detector in ONE request per trial ("cup. bottle.").
    ``max_instances`` (1 | N | 'all', or per name {name: …, '*': …}): with more than 1, each name's entry is
    its best object plus ``instances`` (up to N entries, best first) and
    ``count``."""
    simple_return    = kwargs.pop('simple_return', False)
    main_body        = bool(kwargs.pop('main_body', False))
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
    # verify=True: a VLM looks at the crop of each detection scoring below `verify_above` ("is this a cup?")
    # and drops the ones it says are not (the detector "finds" absent objects at 0.4-0.6).
    verify       = str(kwargs.pop('verify', False)).strip().lower() in ('1', 'true', 'yes', 'on')
    verify_above = float(kwargs.pop('verify_above', 0.75))
    # others='beige bag, blue bag': what the asked object must be told apart from (ask's other options). They go
    # into the same detector request and each box goes to the name it matches best; a box taken by one of them is
    # not the asked object. Asked for "black bag" alone the detector boxed the beige bag (0.45 vs 0.39-0.42: it
    # scores "a bag", hardly the colour); with all three names the black bag got 0.79-0.86 (2026-10-15).
    others = _other_names(kwargs.pop('others', None), obj_names)
    # others_strict False (find_grasp, the wrist close-up once pick is in front of the object): a box another
    # name claims is only a last resort, kept when no other box of the asked name is left. Up close the colour
    # is read worse: pick::black bag lost the black bag to "blue bag" (0.43-0.45 vs 0.35-0.39, same box) in
    # every frame of both grasp trials, with find's own choice already made (2026-10-15).
    others_strict = str(kwargs.pop('others_strict', True)).strip().lower() in ('1', 'true', 'yes', 'on')
    if others:
        log_data({'msg': f"find {', '.join(obj_names)}: told apart from {', '.join(others)}"})
    # Grasp selection uses the closer `detect_grasp` profile; plain detection the
    # default `detect_object` profile.
    cfg_key = 'detect_grasp' if with_grasp else 'detect_object'
    det_configs, select_configs = _object_det_select_configs(kwargs, cfg_key=cfg_key)
    if 'box_threshold' in kwargs:
        det_configs['box_threshold'] = kwargs['box_threshold']
    grasp_configs = _grasp_configs() if with_grasp else None

    # max_instances: 1 (default) → the best object per name, as before; N or
    # 'all' → up to N (every) object, best first, in entry['instances'].
    # A dict sets it per name: {'foot': 2, 'spoon': 1} ('*' = the others).
    max_inst = kwargs.pop('max_instances', 1)

    def _limit(v):
        return None if str(v).strip().lower() == 'all' else max(1, int(v))
    if isinstance(max_inst, dict):
        limits = {n: _limit(max_inst.get(n, max_inst.get('*', 1))) for n in obj_names}
    else:
        limits = {n: _limit(max_inst) for n in obj_names}

    # What to ask the detector for each name: a product (신라면 …) is detected
    # as its generic class and told apart afterwards; others by their name.
    plans = {}
    for name in obj_names:
        product = _products.lookup(name)
        prompt, cfg = name, det_configs
        if product is not None:
            prompt = product[1].get('class', name)
            cfg = {**det_configs, **{k: product[2][k] for k in ('box_threshold', 'text_threshold')
                                     if k in product[2]}}
        plans[name] = (prompt.strip().lower(), cfg, product)

    # ONE detector request per trial with every label still missing
    # ("cup. bottle. noodle package."), instead of one request per object;
    # the most permissive thresholds go to the request, each name then keeps
    # only detections above its own box_threshold.
    found = {}                      # name → (result with only its detections, rgb, depth, cam_params)
    for _ in range(num_trials):
        missing = [n for n in obj_names if n not in found]
        if not missing:
            break
        cam = fetch_camera_data(node, camera)
        rgb, depth, cam_params = cam.rgb, cam.depth, cam.cam_params
        prompts = list(dict.fromkeys([plans[n][0] for n in missing] + others))
        req_cfg = dict(det_configs)
        for k in ('box_threshold', 'text_threshold'):
            vals = [plans[n][1][k] for n in missing if plans[n][1].get(k) is not None]
            if vals:
                req_cfg[k] = min(vals)
        det_rgb = rgb if use_head else _hide_gripper(rgb, depth)
        res_all = _predict_detect(det_rgb, ' '.join(f'{p}.' for p in prompts), req_cfg)
        if min_conf > 0:
            res_all = res_all.filter_by_conf(min_conf=min_conf)
        owner = [_label_owner(getattr(d, 'cls', ''), prompts) for d in res_all.detections]
        body = set() if use_head else _robot_body(res_all, depth, rgb.shape)
        taken = _taken_by_others(res_all, owner, set(others)) if others else {}
        reports = {}                 # (prompt, floor) → classify() of those packs, shared by products
        for n in missing:
            prompt, cfg, product = plans[n]
            floor = float(cfg.get('box_threshold') or 0.0)
            idx = [k for k, d in enumerate(res_all.detections)
                   if owner[k] == prompt and float(getattr(d, 'conf', 0.0)) >= floor and k not in body]
            mine = [k for k in idx if k not in taken]
            for k in idx:
                if k in taken:
                    c = float(getattr(res_all.detections[k], 'conf', 0.0))
                    log_data({'msg': f'{prompt} {c:.2f}: the same box is {taken[k][0]} {taken[k][1]:.2f} — '
                                     + ('dropped' if others_strict or mine else 'kept, nothing else is a ' + prompt)})
            if mine or others_strict:
                idx = mine
            if verify and product is None and idx:
                idx = _vlm_filter(rgb, res_all, idx, n, verify_above)
            if not idx:
                continue
            res = _products.subset(res_all, idx, n)          # relabelled to the name asked
            if product is not None:
                key = (prompt, floor)
                if key not in reports:
                    reports[key] = _products.classify(_vs_client(), rgb, res, product[2])
                res, report = _products.match(_vs_client(), rgb, res, product[0], product[2],
                                              label=n, report=reports[key])
                log_data({'msg': f'{n} = {product[0]}: ' + ', '.join(
                    f"{r['best']} {r['sim']:.2f}{'' if r['sure'] else '?'}"
                    + (f" (masked {r['masked']['best']} {r['masked']['sim']:.2f}/{r['masked']['margin']:.3f})"
                       if 'best' in r.get('masked', {}) else '') for r in report)
                    + ('' if res.detections else ' — not sure which pack, not picking')})
            if res.detections:
                found[n] = (res, rgb, depth, cam_params)

    out: dict = {}
    panels: list = []
    for name in obj_names:
        if name not in found:
            continue
        res, rgb, depth, cam_params = found[name]
        h, w = rgb.shape[:2]
        entries, remaining = [], res
        max_inst = limits[name]
        while max_inst is None or len(entries) < max_inst:
            target, k = select_target_object(
                remaining, cls=name, image_size=(w, h), depth_result=depth,
                intrinsics=cam_params, return_index=True, **select_configs)
            if target is None:
                break
            entry = _object_entry(node, name, target, remaining, rgb, depth, cam_params, w, h,
                                  with_grasp, grasp_configs, keep_orientation, gravity_cam,
                                  use_head, robot_mode, simple_return, panels,
                                  tag=name if not entries else f'{name}#{len(entries) + 1}', main_body=main_body)
            entries.append(entry)
            rest = [x for x in range(len(remaining.detections)) if x != k]
            if not rest:
                break
            remaining = _products.subset(remaining, rest, name)
        if not entries:
            continue
        out[name] = dict(entries[0])
        if max_inst != 1:
            out[name]['instances'] = entries
            out[name]['count'] = len(entries)

    # One annotated frame for this camera with every object's box + grasp —
    # captured alongside the raw inputs so a logged case shows what was detected.
    vis = _emit_detection_vis(node, rgb, camera, panels)
    save_detection_dataset(rgb=rgb, depth=depth, results=out, annotated=vis, tag=camera)
    ret = {'isdone': len(out) > 0, 'ins': out}
    if 'head' in camera:
        ret['log_image'] = vis
    return ret




def _other_names(others, obj_names) -> list:
    """`others` ('a, b' or a list; ask's 'said=real' / 'said->real' pairs give their real side) as detector
    prompts, without the asked names themselves — so every option can be passed whatever the answer was."""
    if not others:
        return []
    items = others if isinstance(others, (list, tuple)) else str(others).split(',')
    asked = {str(n).strip().lower() for n in obj_names}
    out = []
    for it in items:
        it = str(it)
        for sep in ('->', '='):
            if sep in it:
                it = it.split(sep)[-1]
        it = ' '.join(it.strip().lower().split())
        if it and it not in asked and it not in out:
            out.append(it)
    return out


def _taken_by_others(res, owner, others, iou_min=0.5) -> dict:
    """{index: (other name, its score)} for the detections one of `others` claims better: the detector boxes ONE
    object once per name (the blue bag alone in view: blue 0.60, beige 0.50, black 0.45, same box), so a box
    overlapping (IoU >= `iou_min`) a box of a different name that one of `others` owns with a higher score is
    not the asked object."""
    dets = res.detections
    boxes = [_bbox_to_xyxy(d.bbox) for d in dets]
    conf = [float(getattr(d, 'conf', 0.0)) for d in dets]

    def iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0
    out = {}
    for k in range(len(dets)):
        if owner[k] in others:
            continue
        for j in range(len(dets)):
            if owner[j] in others and conf[j] > conf[k] and iou(boxes[k], boxes[j]) >= iou_min:
                out[k] = (owner[j], conf[j])
                break
    return out


def _vlm_filter(rgb, res, idx, name, verify_above, max_checks=3):
    """The detections of `idx` a VLM agrees are a `name` (crop + yes / no). Above `verify_above` a detection
    is kept unchecked; at most `max_checks` are checked (best first), the rest are dropped. On a VLM error the
    detections are kept as they were (a broken check must not hide real objects)."""
    import time as _t
    dets = res.detections
    idx = sorted(idx, key=lambda k: -float(getattr(dets[k], 'conf', 0.0)))
    keep, notes, t0 = [], [], _t.time()
    try:
        from kcare_robot.skills import hri as _qa       # the VLM helpers (qa.py was folded into hri.py on 2026-10-13)
        cfg = _qa._qa_config({})
    except Exception as e:
        log_data({'msg': f'find verify: no VLM config ({e}), not verifying'})
        return idx
    h, w = rgb.shape[:2]
    for n_checked, k in enumerate(idx):
        conf = float(getattr(dets[k], 'conf', 0.0))
        if conf >= verify_above:
            keep.append(k); notes.append(f'{conf:.2f} kept'); continue
        if n_checked >= max_checks:
            break
        x0, y0, x1, y1 = _bbox_to_xyxy(dets[k].bbox)
        px, py = max(16, 0.3 * (x1 - x0)), max(16, 0.3 * (y1 - y0))
        crop = rgb[int(max(0, y0 - py)):int(min(h, y1 + py)), int(max(0, x0 - px)):int(min(w, x1 + px))]
        if crop.size == 0:
            continue
        if min(crop.shape[:2]) < 224:
            f = 224 / min(crop.shape[:2])
            crop = cv2.resize(crop, (int(crop.shape[1] * f), int(crop.shape[0] * f)), interpolation=cv2.INTER_CUBIC)
        body = {'model': cfg['model'], 'stream': False, 'think': False, 'keep_alive': cfg['keep_alive'],
                'format': {'type': 'object', 'properties': {'answer': {'type': 'string', 'enum': ['yes', 'no']}},
                           'required': ['answer']},
                'options': {'temperature': 0.0, 'num_predict': 16},
                'messages': [{'role': 'user', 'images': [_qa._b64jpeg(crop, 448)],
                              'content': f'Is the main object in the middle of this photo a {name}? '
                                         f'Answer yes or no.'}]}
        try:
            ans = json.loads(_qa._post_chat(cfg, body).json()['message']['content']).get('answer', 'no')
        except Exception as e:
            log_data({'msg': f'find verify: VLM error ({e}), keeping the detections'})
            return idx
        notes.append(f'{conf:.2f} {ans}')
        if ans == 'yes':
            keep.append(k)
    log_data({'msg': f'find verify {name}: ' + ', '.join(notes) + f' ({_t.time() - t0:.1f} s, {_qa._vlm_label(cfg)})'})
    return keep


# ── Public skills ─────────────────────────────────────────────────────────────
@exception_handler
def detect(node, **kwargs):
    """Object detection with grounding-dino (camera defaults to FIND_CONFIGS).

    Required (one of): inputs='obj1,obj2,...'  OR  obj_names=[...].
    Returns ``{'isdone', 'ins': {name: {...}}}``."""
    obj_names = _parse_obj_names(kwargs.pop('inputs', None), kwargs.pop('obj_names', None))
    camera = kwargs.pop('camera', 'arm')
    reset_log_image(camera)
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
    reset_log_image(kwargs.get('cameras', 'arm'))

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
        log_data(ret)
        # if 'vis' in ret:
        #     log_data({'log_image': ret['vis']})

    return ret


@exception_handler
def find(node, **kwargs):
    """Find objects from the head camera (grounding-dino). `inputs` is
    'cup,bottle@table' (objects, optional '@location'). Retries over `views`
    and, if `once=False`, over every ENV location.

    Telling look-alikes apart (black / beige / blue bag), both off by default:
      others='beige bag, blue bag' — the other things in view, detected in the same request; a box one of them
          matches better is not the asked one. No extra time. The asked name may be in the list too.
      verify=True — a VLM checks each box scoring below verify_above (0.75; 1 = every box, at most 3) on its
          crop ("is this a black bag?"), ~0.2-0.8 s per box."""
    reset_log_image('head')
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
    reset_log_image('arm')
    kwargs['estimate_grasp'] = True
    kwargs.pop('detector', None)
    kwargs.setdefault('others_strict', False)       # pick is already in front of the one find chose
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
    reset_log_image('head')
    point3d = kwargs.pop('point3d', None)
    inputs = kwargs.pop('inputs', None)
    islying = kwargs.pop('islying', False)
    placeside = kwargs.pop('placeside', None)
    dest_pose = kwargs.pop('dest_pose', None)     # the destination object's pose, already found (pnp): no head find
    
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
            placepose['dy'] = placepose.get('dy', 0) + (0.03 if islying else -0.03)
            point3d = _get_placepose(
                placepose or None, target_height, robot_mode,
                islying,
            )
        #
        if dest_obj is not None: 
            if dest_pose is not None:
                p3 = list(dest_pose)
            else:
                ret = find(node=node, inputs=dest_obj, cameras="head", simple_return=True, main_body=True)
                if dest_obj not in (ret.get('ins') or {}):
                    raise Exception(f'find_place: {dest_obj} not found')
                p3 = ret['ins'][dest_obj]['pose']
            if point3d is None:
                point3d = p3
            else:
                point3d[:2] = p3[:2]
            place_beside = placeside is not None


    # avoid obstacle
    if place_beside and 'trash' not in inputs and 'sink' not in inputs and 'laundry' not in inputs:
        point3d = _detect_nearest(node=node, pose=point3d, obj_radius=kwargs.get('obj_radius'))
        if point3d is None:
            raise Exception(f'find_place: no free spot on {inputs} clear of its edges and of the objects on it')
        
        

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
    mforward = approach_pose[0] - np.clip(approach_pose[0], 0.15, 0.35)
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