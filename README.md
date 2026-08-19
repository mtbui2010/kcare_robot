# kcare_robot — Assistive Mobile Manipulator

> **23 production skills** on a 6-DOF cobot with a RealSense D405 wrist camera
> and Femto Bolt head stereo. Open-vocabulary grasping, drawer manipulation,
> Nav2-driven mobile base, head-to-base calibrated 3D perception. Three ways
> to drive it: web dashboard, CLI, or Python API.

A reference implementation for the
[`robot_agent`](../robot_agent) runtime, controllable from the
[`robotapp`](../robotapp) dashboard
([live: robot.aistations.org](https://robot.aistations.org)).

---

## The robot

| Subsystem | Hardware | ROS2 interface |
|---|---|---|
| **Manipulator** | 6-DOF KAAIR cobot arm | `/kaair_worker/arm_moveJ`, `/arm_moveT` (actions) |
| **End-effector** | Two-finger gripper + suction | `/body/tool_controller/gripper_cmd` |
| **Wrist camera** | Intel RealSense **D405** | `/hand/d405/color/image_raw/compressed`, `/depth/image_rect_raw` |
| **Head cameras** | Orbbec **Femto Bolt** RGB-D | `/femto/color/...`, `/femto/depth/.../compressedDepth` |
| **Mobile base** | 2-wheel diff-drive, LiDAR | Nav2 `/navigate_to_pose` |
| **Lift** | Vertical linear actuator | `/kaair_worker/lift_move` |
| **Head** | 2-DOF pan-tilt | `/kaair_worker/head_move` (rz, ry) |
| **Proprioception** | Joint states, tool pose, mobile odom | `/joint_states`, `/robot_pose/...` |
| **Perception backend** | TCP VLM service (GroundingDINO / GroundedSAM / mask2grasps) | `tcp://192.168.1.11:8805` |

All device registrations live in
[`kcare_robot/data/connections.json`](kcare_robot/data/connections.json) and
are managed at runtime via `robot_agent`'s `POST /connects` API.

---

## The 23 skills

Declared in
[`kcare_robot/configs/skills_config.py`](kcare_robot/configs/skills_config.py),
implemented in [`kcare_robot/skills/`](kcare_robot/skills/).

| Group | Skills |
|---|---|
| **Perception** | `find`, `detect`, `find_arm`, `grasp_succeed`, `get3d`, `inform` |
| **Manipulation** | `pick`, `pick_no_sound`, `pick_card`, `fine_move`, `place`, `placeat`, `placep`, `open_drawer`, `close_drawer`, `collect_card`, `return_card`, `stack`, `wipe` |
| **Arm motion** | `arm_joints`, `arm_pose`, `movel`, `movej`, `movet`, `movelf` |
| **Mobile base** | `move`, `forward`, `turn`, `rotate`, `moveb`, `mobile_pose` |
| **Head / lift / gripper** | `moveh`, `head_state`, `lift`, `lift_state`, `dlift`, `grip` |
| **Interaction** | `select_response`, `llm` |

Every skill follows the contract:

```python
def skill(node, **params) -> dict:
    return {'isdone': bool, 'msg': str, ...}   # planner-readable
```

Wrapping (`auto_wrap_skills`) injects the ROS node on first call so Python-API
users can write `find('apple')` without touching rclpy.

---

## What's interesting under the hood

### Open-vocabulary 3D perception

[`skills/recognition.py`](kcare_robot/skills/recognition.py) — 415 lines — runs
the full pipeline:

1. **Fetch** RGB-D from wrist or head camera (D405 or Femto Bolt)
2. **Detect** via TCP to the VLM service (`GroundingDINO` for text queries,
   `GroundedSAM` for masks)
3. **Lift to 3D** — `attach_3d_features()` reconstructs per-cluster normals,
   min/median/max depth, 3D centroids via inverse projection `Ixy2xyz()`
4. **Classify pose** — detects lying objects from normal-vector dispersion;
   estimates mass-center percentages for handle-equipped items
5. **Grasp** — `mask2grasps` returns 2D pixel endpoints; the skill lifts them
   to a 6-DOF grasp pose using depth + camera intrinsics + wrist-offset
   geometry

### Head-to-base calibration

[`skills/calibrattion.py`](kcare_robot/skills/calibrattion.py) ships a
`Head2BaseCalibration` class with:
- Intrinsic camera parameters (fx, fy, ppx, ppy) per stream
- 4×4 link-to-base and base-to-lift transforms
- Per-mode (front / left / right) error-linear corrections

This is what makes "the apple your wrist camera sees" turn into "an XYZ in the
base frame the arm can actually move to."

### Closed-loop grasping with self-correction

[`skills/pick.py`](kcare_robot/skills/pick.py) — 422 lines — orchestrates the
full pick:

1. `find_arm()` — wrist-camera detection
2. `fine_move()` — wrist-guided approach with **up to 2 self-correction trials**
   if the object drifts out of frame
3. `grip()` — close gripper
4. `grasp_succeed()` — verify by re-imaging the gripper ROI and checking depth
   in a ±0.27 m window

Place is the mirror: `placeat()` / `placep()` plus retraction choreography.
Drawer skills detect the handle as a separate class and run open/close as
a constrained Cartesian movement.

### Parallel actuator coordination

```python
# Common pattern: lift + arm + head move simultaneously
run_parallel_check([
    ('lift', {'height': 0.4}),
    ('movej', {'joints': ARM_PRE_PICK}),
    ('moveh', {'rz': -30, 'ry': 20}),
])
```

`run_parallel_check()` (from `robot_agent.connect.parallel`) fires ROS actions in parallel and
waits for all to converge before continuing — drops a typical pick from
~7 s sequential to ~3 s.

### Persistent symbolic world state

The robot keeps a small **belief** about itself — `arrived` (where it is),
`found` (+ `found_pose`), `holding`, `opened`, `on` — that survives across plan
runs and (selectively) across a restart. Running skills feed it through the
`grace_namemap.apply_skill_effect(world, skill, params, result, node)` hook,
keyed on the kcare skill name:

| Skill | World effect |
|---|---|
| `find` / `find_arm` / `find_once` | set `found`; stash `found_pose` (`loc_3d`/`pose_3d`/`grasppose` from `ins[name]`, stamped with the base pose at detection) |
| `pick` / `grasp` | `holding` ← prior `found`; clear `found`/`found_pose`; set `holding_since` + `holding_pose` (grasp `pick` returns) |
| `placeat` / `place` / `put` / `putin` / `give` | clear `holding` |
| `open_drawer` / `open` · `close_drawer` / `close` | add / remove in `opened` |
| `move` | `arrived` — set by sensor reconcile (localization), not the hook |

Effects apply only on `result['isdone']`. **Only `arrived` is sensor-derived**;
the rest are beliefs (there is no gripper width/force sensor — `holding_since`
timestamps the grasp belief for staleness). `found_pose` is the detection-time
**base-frame** geometry and is flagged stale once the robot moves, so it is
display-only. The state is shown and editable in the dashboard "Robot State"
panel via `GET`/`PUT /agent/world`.

### Three ways to drive the same skills

| Mode | Use case | Latency |
|---|---|---|
| **UI / HTTP** — `make run` | dashboard, multi-user, REST clients | ~10 ms / call |
| **CLI** — `kcare_robot pick::apple` | scripting, demos, CI | ~3 s first call (bootstrap), <100 ms after |
| **Python API** — `from kcare_robot.skills.pick import pick` | notebooks, tests | same |

All three share `robot_agent.runtime.bootstrap()`.
**Do not run two modes against the same physical robot simultaneously** — no
arbitration layer.

---

## Quick start

```bash
make install           # creates conda env 'kcare' + installs ALL dependencies
make doctor            # pre-flight: ROS 2, rosinterfaces, 49 skill imports
make run               # uvicorn kcare_robot.main:app --port 8001
                       # auto-sources /opt/ros/humble/setup.bash
```

`make install` is idempotent — re-run it any time to pick up new dependencies.

It starts with **`make check-deps`**, which verifies every sibling repo it is
about to editable-install is present in `../` *and* is a buildable project
(has `pyproject.toml`/`setup.py`) — so a missing checkout fails immediately,
before an env is created or anything is downloaded:

```
  [ OK ] robot_agent  ../robot_agent
  [ OK ] pyplanner    ../pyplanner
  [ -- ] visionserve  from PyPI (no local checkout)
```

The "is it buildable" half matters: a directory that exists but whose package
source is gitignored installs *successfully* as an empty module, and `pip list`
will happily show it. `make install` finishes by importing every non-ROS
dependency for the same reason.

The env is created with **the same CPython version ROS was built for** (probed
from `/opt/ros/$ROS_DISTRO/lib/python3.*`), because `rclpy` is a compiled
extension and will not import under a mismatched interpreter.

**numpy is pinned to 1.26.4** for that same ABI reason: ROS 2's `rclpy` and
`cv_bridge` are built against numpy 1.x and fail at import under 2.x. That in
turn caps `opencv-python` below 5 (5.x requires `numpy>=2`), so the resolved
set is numpy 1.26.4 + opencv-python 4.11.0.86.

What it does and does not install:

| | |
|---|---|
| conda env + native libs | python, `portaudio` (TTS playback), `ffmpeg` (mp3 decode) |
| editable, from sibling repos | `robot_agent`, `pyplanner` (GRACE planner) |
| from PyPI | `visionserve`, numpy, opencv, scipy + every `robot_agent` extra (transports, LLM backends, TTS) |
| **not installed** — you provide | **ROS 2** (`rclpy`, `cv_bridge`, `sensor_msgs`) and the **`rosinterfaces`** ament package (`colcon build`, then source the workspace) |

Useful overrides:

### Installing into an existing env

By default `make install` targets an env named **`kcare`** and creates it if it
is not there. To install into an env you already have:

```bash
make install CONDA_ENV=myenv                  # by name (still created if absent)
make install CONDA_ENV=myenv USE_EXISTING=1   # by name, must already exist
make install ENV_PREFIX=/path/to/env USE_EXISTING=1        # conda env by path
```

### Installing into the python that is active right now

`USE_CURRENT=1` skips conda entirely and targets whatever `python3` is first on
your PATH — an activated conda env, a venv, pyenv, anything:

```bash
conda activate myenv          # or: source .venv/bin/activate
make install USE_CURRENT=1
make run     USE_CURRENT=1    # run/cli/doctor take the same flag

make install PYTHON=/opt/py310/bin/python     # exact interpreter (implies USE_CURRENT)
```

Behaviour in this mode:

- **Nothing is ever created** — the interpreter must already exist.
- **conda base and the system python are refused** unless you add `FORCE=1`;
  forgetting to activate is exactly how ~40 packages end up somewhere painful.
- `portaudio`/`ffmpeg` are conda-installed only when the target *is* a conda
  env; for a venv/system python you get an
  `sudo apt install portaudio19-dev ffmpeg` hint instead (TTS stays silent
  without them, everything else works).
- The ROS python-version check still runs and warns on mismatch.
- `make env-recreate` refuses to operate — it only manages conda envs this
  Makefile created.

`USE_EXISTING=1` turns "env not found" into a hard error that lists your envs,
instead of quietly building a new one — worth using whenever you type a name,
since a typo would otherwise cost you a full fresh install. It also blocks
`make env-recreate`, so the env you asked to reuse cannot be deleted by it.

`ENV_PREFIX` takes any path, so prefix envs (`conda create -p ...`) and envs
outside the conda base directory work too; every conda call uses `-p`.

Two things to know when reusing an env:

- **Python version is checked, not changed.** If the env's interpreter differs
  from the one ROS was built for, install prints a warning — `rclpy` is a
  compiled extension and will not import there. `make doctor` confirms it.
- **`portaudio` + `ffmpeg` are conda-installed into it.** To leave the env's
  conda packages alone, pass `CONDA_SYS_LIBS=` (TTS then degrades to silent).

### Running without `make install`

`run`, `cli`, `doctor` and the skill-scaffolding targets do **not** require the
conda env to exist. When it is missing they fall back to the python currently on
PATH and say so:

```
[kcare_robot] conda env 'kcare' not found -- using current python: /path/to/python (/path)
```

That makes the repo usable on a machine whose environment was prepared some
other way. If that python does not have `kcare_robot` installed either, you get
an actionable error instead of a bare `ModuleNotFoundError`. Pass
`USE_CURRENT=1` to make the choice explicit and silence the notice.

`install`, `install-deps` and `env` never fall back — creating and populating
the env is their job.

### Other overrides

```bash
make install PYTHON_VERSION=3.11        # force an interpreter for a new env
make install ROBOT_AGENT_DIR=/path/to/robot_agent    # repos live elsewhere
make install PYPLANNER_DIR=             # skip it (no GRACE planner)
make install VISIONSERVE_DIR=/path/to/src            # local instead of PyPI
make run     ROS_DISTRO=jazzy           # different ROS distro
make check-deps                         # just the sibling-repo preflight
make env-info                           # show every resolved path
make env-recreate                       # wipe the env and rebuild
```

Open <https://robot.aistations.org>, click **Guide**, paste
`http://<robot-host>:8001`, connect — you're driving the robot from a browser.

Or raw HTTP:

```bash
curl -X POST http://localhost:8001/skill/find -d '{"inputs":"apple"}'
curl -X POST http://localhost:8001/skill/pick -d '{"inputs":"apple"}'
```

Or CLI:

```bash
kcare_robot --list
kcare_robot find::apple                                  # inputs=apple
kcare_robot find::apple estimate_grasp=true camera=arm   # mixed args
kcare_robot pick::apple
```

Or Python:

```python
from kcare_robot.skills.recognition import find
from kcare_robot.skills.pick        import pick

ret = find(inputs='apple')           # bootstraps rclpy + devices on first call
if ret['isdone']:
    pick(inputs='apple')
```

---

## Layout

```
kcare_robot/
├── Makefile                  install · run · cli · doctor · terminate
├── pyproject.toml            [project.scripts] kcare_robot = kcare_robot.__main__:cli
└── kcare_robot/
    ├── main.py               create_app('kcare_robot', data_dir=...)
    ├── __main__.py           CLI entry — from robot_agent.cli import main
    ├── configs/
    │   ├── skills_config.py  SKILL_CONFIGS (23 entries)
    │   ├── tasks.py          ARM_CONFIGS, ENV (locations)
    │   └── guide.py          LLM planner guide
    ├── data/
    │   └── connections.json  device registrations (cameras, arms, base, TCP)
    ├── skills/               production implementations
    │   ├── recognition.py    perception + 3D + grasp pose
    │   ├── pick.py           pick / fine_move / drawer / verification
    │   ├── _pick_helpers.py  workspace checks, retraction choreography
    │   ├── approach.py       object-guided arm pre-positioning
    │   ├── mobile.py         Nav2 + parallel lift coordination
    │   ├── grip.py · lift.py · arm.py · head.py · place.py · vlm.py …
    └── template_skills/      three reference patterns (NOT auto-registered)
        ├── grip_node_agent.py    Pattern 1 — connect-layer NodeAgent (90 % of skills)
        ├── grip_pure_ros2.py     Pattern 2 — raw rclpy + custom QoS / feedback
        └── grip_external.py     Pattern 3 — separate process, registered by URL
```

---

## Adding a new skill

```python
# kcare_robot/skills/wave.py
def wave(node, **params) -> dict:
    arm = node.agents['movej']
    arm.send({'joints': [0, -1.2, 1.5, 0, 0.8, 0]})
    return {'isdone': True, 'msg': 'waved'}
```

```python
# kcare_robot/configs/skills_config.py
SKILL_CONFIGS = {
    ...
    'wave': (f'{_PKG}.wave', 'wave'),
}
```

Then `POST /skills/reload` — no robot restart.

For external skills (GPU box, separate language, microservice) register via
HTTP: `POST /skills {"type":"external", "url":"http://gpu:9000/wave"}`.

---

## Debugging

```bash
make doctor                          # env + ROS + import every skill
make doctor ARGS=--verbose           # show OK + FAIL
ROBOT_AGENT_DEBUG_RESPONSE=1 make run   # full tracebacks on errors
ROBOT_AGENT_LOG_LEVEL=DEBUG make run
curl http://localhost:8001/diagnostics/boot | python3 -m json.tool
```

Logs at `robot_agent/logs/robot_agent.log` (rotating 5 × 10 MB).

---

## Related

- [`robot_agent`](../robot_agent) — FastAPI runtime, skill registry, device
  manager, streaming agent
- [`robotapp`](../robotapp) — Next.js 14 ops dashboard
- [`robot_template`](../robot_template) — cookiecutter to bootstrap your own
  robot package on the same contract
