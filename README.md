# kcare_robot

Skill + config package for the **KCare** mobile manipulator. Plugs into
[`robot_agent`](../../robot_agent) as the source of `SKILL_CONFIGS` and concrete
skill implementations.

## Three ways to drive the robot

| Mode | Use case | Needs `make run` |
|---|---|---|
| **UI / HTTP** (`make run`) | Web dashboard, multi-user, REST clients | yes (this IS the server) |
| **CLI** (`kcare_robot find::apple`) | One-off ops, shell scripting | no |
| **Python API** (`from kcare_robot.skills.recognition import find`) | Scripts, notebooks, tests | no |

All three share the same `bootstrap()` core in `robot_agent.runtime`; only the
outer shell differs. **Do not run two modes against the same robot at the same
time** — two clients steering one arm is a safety hazard.

## Install

```bash
make install          # editable-install pyconnect, robot_agent, kcare_robot
```

This also registers the `kcare_robot` console-script (CLI) via
`[project.scripts]` in `pyproject.toml`.

## UI / HTTP mode

```bash
make run              # uvicorn kcare_robot.main:app, port 8001
```

`make run` sources `/opt/ros/humble/setup.bash`. Override with
`ROS_SETUP=/opt/ros/iron/setup.bash make run`. Browse the API at
<http://localhost:8001/docs>.

## CLI mode

```bash
# After sourcing ROS (or via `make cli` which sources for you):
kcare_robot find::apple
kcare_robot find::apple estimate_grasp=true camera=arm
kcare_robot pick::apple
kcare_robot --list                # show all registered skills
kcare_robot --help

# Or via Makefile (auto-sources ROS):
make cli ARGS="find::apple"
make cli ARGS="pick::apple"
```

Argument syntax:
- `name::value` — the part after `::` becomes `inputs=value`.
- `key=value` pairs (space-separated) — coerced to bool / int / float / JSON
  when possible, otherwise kept as string.
- `name` alone — only kwargs, e.g. `find inputs=apple camera=arm`.

Output is JSON on stdout. Exit code 0 if `isdone: true`, 1 otherwise.

## Python API mode

```python
from kcare_robot.skills.recognition import find
from kcare_robot.skills.pick        import pick

ret = find(inputs='apple')       # first call: ~3s bootstrap (rclpy + devices)
if ret['isdone']:
    pick(inputs='apple')         # subsequent calls: reuse the same ROS node

# Short form — first positional becomes `inputs`:
find('apple')
```

How it works: [kcare_robot/skills/__init__.py](kcare_robot/skills/__init__.py)
calls `robot_agent.skills.auto_wrap_skills(SKILL_CONFIGS, pkg='kcare_robot')`
on import. Each public skill function gets wrapped so that:
- if `node` is not provided, `bootstrap('kcare_robot')` runs (idempotent) and
  `state.dm._ros_node` is injected;
- if the first positional argument is a string, it's mapped to `inputs=`.

Skills called via SkillRegistry (UI/CLI) already pass `node=` explicitly so
the wrapper is a no-op for them.

## Folder layout

```
kcare_robot/
├── Makefile
├── README.md                # this file
├── pyproject.toml
└── kcare_robot/
    ├── configs/             # SKILL_CONFIGS + per-skill defaults
    ├── skills/              # production skill implementations
    └── template_skills/     # reference templates (NOT auto-registered)
        ├── grip_pyconnect.py    # Pattern 1
        ├── grip_pure_ros2.py    # Pattern 2
        └── grip_external.py    # Pattern 3
```

## How `robot_agent` discovers skills

On startup, `robot_agent` reads two env vars (auto-loaded from
`robot_agent/robot_agent/.env`):

| Var | Purpose |
|---|---|
| `ROBOT_SKILLS_PKG`  | Python package name (e.g. `kcare_robot`) used for `importlib.import_module(f'{pkg}.configs.skills_config')` |
| `ROBOT_SKILLS_PATH` | Optional `sys.path` fallback if the package is not pip-installed |

`kcare_robot/configs/skills_config.py` exports a flat dict
`SKILL_CONFIGS: dict[str, tuple[module_path, func_name]]` -- this is the entire
contract between `robot_agent` and the skill package.

## Adding a new skill

Three steps, regardless of which pattern you pick.

### Step 1 -- pick a pattern

| Pattern | When to use | Where to look |
|---|---|---|
| **1. pyconnect**  | 90% of skills. Device is already registered via `POST /devices`. | [template_skills/grip_pyconnect.py](kcare_robot/template_skills/grip_pyconnect.py) |
| **2. pure ROS2**  | Need custom QoS, action feedback, timeout, or non-`SendStringData` interfaces but still want shared executor. | [template_skills/grip_pure_ros2.py](kcare_robot/template_skills/grip_pure_ros2.py) |
| **3. external**   | Skill is a separate process / language / host. | [template_skills/grip_external.py](kcare_robot/template_skills/grip_external.py) |

All three are interchangeable from the caller's point of view --
`POST /skill/<name>` dispatches identically.

### Step 2 -- write the function

Function signature is fixed:

```python
def my_skill(node, **params) -> dict:
    ...
    return {'isdone': True, 'msg': 'ok', ...}     # contract
```

The return MUST be a dict; `isdone` is required by the planner. Anything else
is free-form. Skills are imported lazily by `SkillRegistry.execute()`, so heavy
imports at module level are fine but will slow first-call latency.

For Pattern 1, drop the file in `kcare_robot/skills/`. For Pattern 2, same
location. For Pattern 3, run the FastAPI server separately and only register
the URL with `POST /skills`.

### Step 3 -- register it

For Pattern 1 & 2, add an entry to
[`kcare_robot/configs/skills_config.py`](kcare_robot/configs/skills_config.py):

```python
SKILL_CONFIGS = {
    ...
    'my_skill': (f'{_PKG}.my_module', 'my_skill'),
}
```

Then either reload via `POST /skills/reload`, or restart `make run`.

For Pattern 3, register via the API:

```bash
curl -X POST http://localhost:8001/skills \
     -H 'Content-Type: application/json' \
     -d '{"name":"my_skill","type":"external",
          "url":"http://localhost:9000/my_skill",
          "timeout":15}'
```

## Testing a skill

UI / HTTP:
```bash
# list all registered skills
curl http://localhost:8001/skills | python3 -m json.tool

# call one
curl -X POST http://localhost:8001/skill/grip \
     -H 'Content-Type: application/json' \
     -d '{"inputs":"open"}'
```

CLI (no `make run` required):
```bash
kcare_robot --list
kcare_robot grip::open
```

Python:
```python
from kcare_robot.skills.grip import grip
grip(inputs='open')
```

## Devices

Skills that use Pattern 1 or 2 expect specific devices to be registered in
`robot_agent`. Add them via `POST /devices`:

```bash
curl -X POST http://localhost:8001/devices \
     -H 'Content-Type: application/json' \
     -d '{"type":"ros_service",
          "name":"grip",
          "config":{"conn_name":"grip","is_client":true}}'
```

Devices persist in `robot_agent/robot_agent/devices.json` and are reconnected
on next start.

## Debugging

```bash
make doctor                     # pre-flight: env, ROS2, every skill import
make doctor ARGS=--verbose      # list every skill, OK and FAIL alike
```

Live (after `make run`):

```bash
curl http://localhost:8001/diagnostics       | python3 -m json.tool
curl http://localhost:8001/diagnostics/boot  | python3 -m json.tool
```

Set `ROBOT_AGENT_DEBUG_RESPONSE=1` before `make run` to attach full tracebacks
to every `{'isdone': False}` skill response. Set `ROBOT_AGENT_LOG_LEVEL=DEBUG`
for verbose dispatch logs. Both can be combined.

Logs: `robot_agent/logs/robot_agent.log` (rotating, 5 x 10 MB).

## Starting from scratch on a new robot

Use [`../robot_template`](../robot_template) — a cookiecutter scaffold that
emits the same layout (skills, configs, Makefile, `__main__.py`, CLI script)
ready to edit. See its README for prompts.
