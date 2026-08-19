"""Skill: real2sim_capture — sweep the wrist camera around an object and
capture RGB-D + camera poses for real-to-sim 3D reconstruction.

Flow:
  1. ``approach_pick`` does the hard part: it locates the object with the HEAD
     camera, drives the mobile base so the object lands at the arm's working
     distance, sets the lift, folds/unfolds the arm and finishes at a valid
     approach pose near the object. It stops short of grasping (``fine_move``
     in ``pick`` is what closes the gripper), so it is exactly the right
     starting state for a camera sweep.
  2. Measure the object FROM THE GRIPPER with ``find_grasp`` (wrist camera) and
     lift that offset into the base frame. ``approach_pick``'s own ``pose_3d``
     predates its base move, and the head camera only agrees to ~1.4 cm.
  3. Sweep an arc in the **x-z plane** (a great circle about the Y axis through
     the object), executed as ``movel`` deltas relative to the approach pose.
     x is the arm's widest axis, measured on the real robot at 0.76 m of travel
     versus 0.58 (y) and 0.62 (z), so this is where the parallax is.
  4. Walk waypoint to waypoint: each ``movel`` delta is measured against the
     arm's ACTUAL current pose, so no detour back to the approach pose is
     needed and nothing accumulates. Wrist deltas go through ``fix_angle`` so
     the last joint never turns more than 90 deg.
  5. Put the robot back: drive the base back by ``mforward`` and fold the arm,
     the same wind-down ``pick`` performs.

Safety: every waypoint is clipped to a floor at ``table_clearance`` above the
object's own z. Downward is the one direction that hits the table — a reach
probe that stepped 0.24 m down ended with the arm unable to plan its way out.

Reconstruction input per shot: ``*_rgb.png``, ``*_depth.npy`` and
``*_results.json`` holding ``tool_pose_base`` ([x,y,z,rx°,ry°,rz°] in
``base_footprint``) plus ``cam_params`` ([fx, fy, cx, cy]); ``poses.json``
collects every shot in one file.

Usage:
    real2sim_capture inputs=phone
    real2sim_capture inputs=phone dry_run=True        # validate, move nothing
    real2sim_capture inputs=phone radius=0.3 n_arc=13
    real2sim_capture inputs=phone approach=False      # sweep from where it is
"""

import json
import os
import time

import numpy as np

from robot_agent.skills import dataset_dir, log_data, set_dataset_dir, stream_dataset
from robot_agent.utils import exception_handler

from kcare_robot.skills.arm import arm_pose, movej, movel
from kcare_robot.skills.mobile import forward, mobile_pose
from kcare_robot.skills.pick import approach_pick
from kcare_robot.skills._pick_helpers import fix_angle
from kcare_robot.skills.recognition import find, find_grasp
from kcare_robot.skills._recognition_helpers import (
    fetch_camera_data, save_detection_dataset,
)

RADIUS = 0.25            # camera → object distance (m)
# Arc measured from straight above the object in the x-z plane. Negative swings
# the camera toward the robot (-x), positive away from it (+x). Measured on the
# real robot from the approach_pick pose: -60..+12 planned, +24 and beyond did
# not — the arm cannot lean past the object away from itself. Every run reports
# `reachable_arc_deg`, so re-tune this per setup.
ARC_DEG = (-60.0, 30.0)
N_ARC = 20                # shots per arc
N_SLICE = 1              # y-offsets for extra parallax (1 = single plane)
SLICE_SPAN = 0.10        # total y spread across the slices (m)
TABLE_CLEARANCE = 0.10   # never take the camera below object_z + this (m)
SETTLE = 0.4             # s to damp vibration before a shot
SPEED = 0.2              # movel velocity scale for the sweep

# Relative movel travel measured from the `approach_lying` pose. x and z hit the
# probe's 0.40 m cap without failing, so those are lower bounds. Used only to
# drop obviously-out-of-range waypoints before asking MoveIt.
REACH = {'x': [-0.15, 0.61], 'y': [0.09, 0.68], 'z': [0.68, 1.31]}


# ── Geometry ─────────────────────────────────────────────────────────────────

def _rot_y(deg: float) -> np.ndarray:
    """Rotation about the base Y axis."""
    t = np.deg2rad(float(deg))
    c, s_ = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])


def aim_along_arc(R_ref: np.ndarray, arc_deg: float) -> np.ndarray:
    """Orientation for a waypoint `arc_deg` along the arc, from the reference.

    The arc IS a rotation about the base Y axis through the object, so turning
    the whole tool frame by the same angle keeps the camera pointed at the
    object while changing exactly one thing.

    The obvious alternative — rebuilding the frame with a look-at whose up
    vector is world +Z — is degenerate directly above the object: there `z`
    becomes parallel to up, `cross(up, z)` flips sign, and the frame snaps 180°
    about the view axis. Measured on the real arc that showed up as drz jumping
    from 0 to +180 the moment the sweep crossed the top, spinning the last wrist
    joint half a turn mid-run. This formulation has no such singularity.
    """
    return _rot_y(arc_deg) @ R_ref


def rpy_xyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → fixed-axis roll-pitch-yaw (X then Y then Z), radians.
    Inverse of the Rz@Ry@Rx convention used by ``deg2quaternion`` in movel."""
    pitch = float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(R[2, 0]) < 1.0 - 1e-9:
        roll = float(np.arctan2(R[2, 1], R[2, 2]))
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    else:                                     # gimbal lock
        roll = float(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def _wrap180(a: float) -> float:
    """Shortest signed representation of an angle delta, in degrees."""
    return (float(a) + 180.0) % 360.0 - 180.0


def orbit_poses(centre, R_ref: np.ndarray, radius: float = RADIUS,
                arc_deg=ARC_DEG, n_arc: int = N_ARC, n_slice: int = N_SLICE,
                slice_span: float = SLICE_SPAN,
                table_clearance: float = TABLE_CLEARANCE,
                reach: dict | None = None):
    """Absolute waypoints ``(position, R_base_tool, arc_deg)`` on an x-z arc.

    The arc is a great circle about the Y axis through *centre*: 0° is straight
    above the object, negative angles swing the camera toward the robot (-x) and
    positive away from it (+x). x is the arm's widest axis, so this is the plane
    that actually buys parallax.

    Optional y-slices repeat the arc at a few y offsets; alternate slices run in
    reverse so the wrist never unwinds a whole sweep between them.

    Waypoints below ``centre_z + table_clearance`` are dropped — that is the
    table, and downward is the one direction where a failed plan can leave the
    arm stuck.
    """
    centre = np.asarray(centre, dtype=float).reshape(3)
    reach = REACH if reach is None else reach
    z_floor = centre[2] + table_clearance

    if n_slice > 1:
        ys = np.linspace(-slice_span / 2.0, slice_span / 2.0, int(n_slice))
    else:
        ys = np.array([0.0])

    poses = []
    for si, dy in enumerate(ys):
        angles = np.deg2rad(np.linspace(arc_deg[0], arc_deg[1], int(n_arc)))
        if si % 2:                            # serpentine
            angles = angles[::-1]
        for a in angles:
            p = centre + np.array([radius * np.sin(a), float(dy), radius * np.cos(a)])
            if p[2] < z_floor:                # table guard
                continue
            if not all(reach[ax][0] <= p[i] <= reach[ax][1]
                       for i, ax in enumerate('xyz')):
                continue
            deg = float(np.rad2deg(a))
            poses.append((p, aim_along_arc(R_ref, deg), deg))
    return poses


def _rot_base_tool(rpy_deg) -> np.ndarray:
    """R_base_tool from a tool roll-pitch-yaw in degrees (Rz @ Ry @ Rx, the
    convention ``deg2quaternion`` and therefore ``movel`` use)."""
    rx, ry, rz = np.deg2rad(np.asarray(rpy_deg, dtype=float))
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _find_centre(node, obj_name: str) -> np.ndarray:
    """BASE-frame centre of `obj_name`, measured from the GRIPPER.

    ``find_grasp`` (wrist camera) reports ``pose_3d`` as an offset from the
    gripper, not a base-frame point — that is the number to use once
    ``approach_pick`` has parked the gripper next to the object.

    Its axes are swapped with respect to the tool frame exactly the way
    ``movet`` swaps them (``dx=-dy, dy=dx``). Measured on the real robot, that
    convention lands 1.4 cm from the head camera's independent base-frame
    answer, while every other permutation is off by 14 cm or more:

        wrist [0.0108, -0.1089, 0.1977] @ tool [0.4418, 0.4241, 0.9402]
          -> (-y, x, z) = [0.4526, 0.5330, 0.7425]
             head cam   = [0.4437, 0.5228, 0.7389]
    """
    ret = find_grasp(node=node, inputs=obj_name)
    assert ret['isdone'], f'find_grasp failed: {ret}'
    entry = ret['ins'][obj_name.strip()]
    w = np.asarray(entry['pose_3d'], dtype=float).reshape(3)
    assert not np.isnan(w).any(), f'find_grasp returned NaN pose_3d for {obj_name!r}'

    P = arm_pose(node=node)
    assert P['isdone'], f'arm_pose failed: {P}'
    P = P['pose']
    return np.asarray(P[:3], dtype=float) + _rot_base_tool(P[3:6]) @ np.array([-w[1], w[0], w[2]])


def _find_centre_head(node, obj_name: str) -> np.ndarray:
    """Base-frame centre from the HEAD camera — independent cross-check."""
    ret = find(node=node, inputs=obj_name)
    assert ret['isdone'], f'find failed: {ret}'
    return np.asarray(ret['ins'][obj_name.strip()]['pose_3d'], dtype=float).reshape(3)


# ── Skill ────────────────────────────────────────────────────────────────────

@exception_handler
def real2sim_capture(node, **params) -> dict:
    """Approach the object, sweep the wrist camera over an x-z arc, capture.

    Params:
        inputs (str):   object name (passed to approach_pick / find).
        approach (bool): run approach_pick first (default True). False sweeps
            from wherever the arm is — only useful mid-debug.
        radius (float): camera distance from the object, m (default 0.25).
        arc_deg (list): [from°, to°] about the Y axis, 0 = straight above
            (default ±60).
        n_arc (int):    shots per arc (default 11).
        n_slice (int):  y offsets for extra parallax (default 1).
        slice_span (float): total y spread across slices, m (default 0.10).
        table_clearance (float): floor above the object's z, m (default 0.10).
        settle (float): seconds to wait before each shot (default 0.4).
        speed (float):  movel velocity scale (default 0.2).
        dry_run (bool): compute + validate, command nothing.
        restore (bool): drive the base back and fold the arm afterwards
            (default True, mirrors what `pick` does).
        out_dir (str):  capture directory (default a timestamped folder under
            the agent log dir).

    Unreachable waypoints are skipped, not fatal. The result carries
    ``reachable_arc_deg`` and a per-waypoint ``failed`` list so the arc can be
    tuned for a given object.
    """
    obj_name = params.pop('inputs', None)
    approach = bool(params.pop('approach', True))
    radius = float(params.pop('radius', RADIUS))
    arc_deg = params.pop('arc_deg', ARC_DEG)
    n_arc = int(params.pop('n_arc', N_ARC))
    n_slice = int(params.pop('n_slice', N_SLICE))
    slice_span = float(params.pop('slice_span', SLICE_SPAN))
    table_clearance = float(params.pop('table_clearance', TABLE_CLEARANCE))
    settle = float(params.pop('settle', SETTLE))
    speed = float(params.pop('speed', SPEED))
    dry_run = bool(params.pop('dry_run', False))
    restore = bool(params.pop('restore', True))
    out_dir = params.pop('out_dir', None)
    assert obj_name, 'need inputs=<object name>'

    mforward = 0.0
    if approach and not dry_run:
        log_data({'msg': f'real2sim: approach_pick::{obj_name}'})
        ap = approach_pick(node=node, inputs=obj_name)
        if not (isinstance(ap, dict) and ap.get('isdone')):
            return {'isdone': False, 'msg': f'approach_pick failed: {ap}'}
        mforward = float(ap.get('mforward', 0.0) or 0.0)

    # Measure the object from the gripper now that approach_pick has parked it
    # alongside — approach_pick's own pose_3d predates the base move, and the
    # head camera's answer is a coarser cross-check.
    centre = _find_centre(node, obj_name) if approach and not dry_run \
        else _find_centre_head(node, obj_name)

    # The approach pose supplies the reference orientation every waypoint is
    # rotated from, so the wrist only ever swings along the arc.
    p0 = arm_pose(node=node)
    assert p0['isdone'], f'arm_pose failed: {p0}'
    P0 = list(p0['pose'])
    R_ref = _rot_base_tool(P0[3:6])

    poses = orbit_poses(centre, R_ref, radius, arc_deg=arc_deg, n_arc=n_arc,
                        n_slice=n_slice, slice_span=slice_span,
                        table_clearance=table_clearance)
    assert poses, ('orbit produced no waypoints — every pose was below the table '
                   'floor or outside the measured reach; try a smaller radius')

    log_data({'msg': f'real2sim: {len(poses)} waypoints on an x-z arc around '
                     f'[{centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f}]'})

    if dry_run:
        P = np.array([p for p, _, _ in poses])
        span = {ax: [round(float(P[:, i].min()), 4), round(float(P[:, i].max()), 4)]
                for i, ax in enumerate('xyz')}
        return {'isdone': True, 'dry_run': True, 'n_waypoints': len(poses),
                'span': span, 'centre': centre.tolist(), 'radius': radius,
                'z_floor': round(float(centre[2] + table_clearance), 4),
                'poses': [[round(float(v), 4) for v in p] for p, _, _ in poses],
                'angles_deg': [round(a, 1) for _, _, a in poses]}

    # This skill exists to produce files, so it must not depend on the
    # dashboard's log-target toggles: `save_detection_dataset` is a no-op unless
    # UnifiedAgent set a dataset dir (log_mode='backend') or turned on streaming
    # (log_mode='frontend'), neither of which happens over the CLI or Python API.
    # Open our own session dir only when nothing is already active.
    prev_dir = dataset_dir()
    owns_dir = prev_dir is None and not stream_dataset()
    if out_dir is None:
        from robot_agent.connect.paths import log_dir as _log_dir
        stamp = time.strftime('%Y%m%d-%H%M%S')
        name = obj_name.strip().replace('/', '_').replace(' ', '_')
        out_dir = str(_log_dir('real2sim', f'{stamp}_{name}'))
    else:
        os.makedirs(out_dir, exist_ok=True)
    if owns_dir:
        set_dataset_dir(out_dir)
    log_data({'msg': f'real2sim: writing captures to {out_dir}'})

    base_world = mobile_pose(node=node).get('pose')
    log_data({'msg': f'real2sim: reference pose {[round(v, 3) for v in P0]}'})

    t0 = time.time()
    shots, skipped = 0, 0
    reached_deg: list[float] = []
    failed: list[dict] = []
    shot_records: list[dict] = []

    for i, (p, R, ang) in enumerate(poses):
        rx, ry, rz = np.rad2deg(rpy_xyz(R))
        # Delta from where the arm actually IS, so the sweep walks waypoint to
        # waypoint without detouring through the approach pose every time — half
        # the motion, and no drift either, since the real pose is re-read here
        # rather than assumed.
        cur = arm_pose(node=node)
        if not cur.get('isdone'):
            skipped += 1
            continue
        C = cur['pose']
        # fix_angle keeps each wrist delta inside (-90, 90]: a 180 deg roll of
        # the camera is the same view upside down, never worth turning for, and
        # it is what strains the last joint's cabling.
        d = {'dx': float(p[0] - C[0]), 'dy': float(p[1] - C[1]),
             'dz': float(p[2] - C[2]),
             'drx': float(fix_angle(_wrap180(rx - C[3]))),
             'dry': float(fix_angle(_wrap180(ry - C[4]))),
             'drz': float(fix_angle(_wrap180(rz - C[5])))}

        ret = movel(node=node, speed=speed, **d)
        if not (isinstance(ret, dict) and ret.get('isdone')):
            skipped += 1
            failed.append({'index': i, 'arc_deg': round(ang, 1),
                           'xyz': [round(float(v), 4) for v in p],
                           'delta': {k: round(v, 4) for k, v in d.items()},
                           'msg': (ret or {}).get('msg', '') if isinstance(ret, dict) else str(ret)})
            log_data({'msg': f'real2sim: arc {ang:+.0f}deg unreachable ({i + 1}/{len(poses)})'})
            continue
        reached_deg.append(round(ang, 1))

        time.sleep(settle)                    # blur ruins the very features
        cam = fetch_camera_data(node, 'arm')  # this sweep exists to collect
        tool_pose = arm_pose(node=node)['pose']

        record = {
            'index': shots,
            'waypoint_index': i,
            'arc_deg': round(ang, 2),
            'centre': centre.tolist(),
            'radius': radius,
            'reference_pose_base': P0,
            'delta_from_previous': {k: round(v, 4) for k, v in d.items()},
            'target_xyz': [float(v) for v in p],
            'target_rpy_deg': [float(rx), float(ry), float(rz)],
            'tool_pose_base': tool_pose,   # [x,y,z,rx°,ry°,rz°] base_footprint
            'base_pose_world': base_world,  # [x, y, rz deg] in the map frame
            'cam_params': list(cam.cam_params),  # [fx, fy, cx, cy]
            'ts': time.time(),
        }
        shot_records.append(record)
        save_detection_dataset(rgb=cam.rgb, depth=cam.depth,
                               tag=f'r2s_{shots:03d}', results=record)
        shots += 1
        log_data({'msg': f'real2sim: shot {shots} ({i + 1}/{len(poses)})'})

    try:
        with open(os.path.join(out_dir, 'poses.json'), 'w') as f:
            json.dump({'object': obj_name, 'radius': radius,
                       'centre_base': centre.tolist(),
                       'reference_pose_base': P0,
                       'base_pose_world': base_world,
                       'n_shots': len(shot_records), 'shots': shot_records}, f, indent=2)
    except Exception as e:
        log_data({'msg': f'real2sim: could not write poses.json: {e}'})

    if owns_dir:
        set_dataset_dir(prev_dir)

    # Wind down exactly like `pick`: give → fold, and give the base back the
    # ground approach_pick took. Leaving the arm extended over the table is how
    # the next command clips something.
    if restore:
        log_data({'msg': 'real2sim: restoring arm and base'})
        movej(node=node, inputs='give')
        movej(node=node, inputs='fold', wait=True)
        if abs(mforward) > 0.02:
            forward(node=node, inputs=-mforward)

    dt = time.time() - t0
    reach_span = [min(reached_deg), max(reached_deg)] if reached_deg else None
    log_data({'msg': f'real2sim: done — {shots} shots, {skipped} unreachable, '
                     f'reachable arc {reach_span}, {dt:.0f}s'})
    return {'isdone': shots > 0, 'shots': shots, 'skipped': skipped,
            'n_waypoints': len(poses), 'centre': centre.tolist(),
            'radius': radius, 'duration_s': round(dt, 1), 'out_dir': out_dir,
            'reference_pose_base': P0, 'mforward_returned': round(mforward, 4),
            'reached_deg': reached_deg,      # arc angles that planned
            'reachable_arc_deg': reach_span,  # -> use as arc_deg next run
            'failed': failed}
