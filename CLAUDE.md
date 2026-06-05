# kcare_robot — Claude notes

Concrete skill + config implementation for the KCare mobile manipulator.
Hosted on top of [`robot_agent`](../robot_agent) as a pluggable package.

## Three execution modes — single shared bootstrap

| Mode | Entry | Shell |
|---|---|---|
| UI / HTTP | `make run` → `uvicorn kcare_robot.main:app` | FastAPI (`create_app`) |
| CLI | `kcare_robot find::apple` | `kcare_robot/__main__.py` → `robot_agent.cli.main(robot_pkg='kcare_robot')` |
| Python API | `from kcare_robot.skills.recognition import find` | `@skill_entry` wrapper auto-triggers `bootstrap()` on first call |

All three call `robot_agent.runtime.bootstrap('kcare_robot', ...)`. It is
idempotent (guarded by `_BOOTED` + `threading.Lock`) and creates a single
ROS node per process. One process = one mode. **Do not mix.**

## Key file map

```
kcare_robot/kcare_robot/
├── main.py                 # uvicorn entry: create_app('kcare_robot', config_dir=…/configs)
├── __main__.py             # CLI entry: from robot_agent.cli import main; cli()
├── skills/__init__.py      # calls auto_wrap_skills(SKILL_CONFIGS, pkg='kcare_robot')
├── skills/*.py             # concrete skills (find, pick, place, …)
├── configs/                          # ALL configs live here
│   ├── skills_config.py              # SKILL_CONFIGS dict — the registry contract
│   ├── tasks.py                      # ARM_CONFIGS / ENV / LIFT_CONFIGS overrides
│   ├── guide*.py, word_mapping.py    # prompt/guide content (shared, code)
│   ├── common/                       # shared across all sites
│   │   ├── skills.json               # skill registry
│   │   └── buttons.json              # shortcut buttons
│   ├── locations/<site>/             # per-deployment-site config
│   │   ├── connections.json          # device endpoints (IPs / ROS topics)
│   │   ├── skill_configs_override.json   # global config overrides (HOME_LOC, LLM_SERVERS, …)
│   │   └── .env                      # API keys (gitignored)
│   ├── locations/default/            # fallback site (always present)
│   └── active_location               # name of the active site (gitignored)
└── data/logs/                        # rotating logs (not config)
```

## Locations (per-site config profiles)

The same robot deployed at different sites needs different **connections** and
**global configs**. Each site is a folder under `configs/locations/<name>`; the
active one is `configs/common/active_location` (defaults to `default`).

The dashboard switches sites live via the `robot_agent` API (no restart):

| Endpoint | Action |
|---|---|
| `GET    /config/locations` | list sites + active |
| `POST   /config/locations` | create (`{name, copy_from?}`) |
| `POST   /config/locations/<name>/activate` | hot-switch (reconnect devices) |
| `PUT    /config/locations/<name>` | rename (`{new_name}`) |
| `DELETE /config/locations/<name>` | delete (not `default`/active) |

Switching tears down the current device connections (keeping the shared ROS
node) and reconnects from the new site's `connections.json` + global configs.

## Skill contract

```python
def my_skill(node, **params) -> dict:
    # node = pyconnect.ros.custom_node.CustomNode (spinning).
    # params = kwargs from caller (HTTP body / CLI key=val / Python kwargs).
    return {'isdone': bool, ...}
```

Skills MUST return a dict with `isdone`. `@exception_handler` (in
`robot_agent.utils`) wraps exceptions into `{'isdone': False, 'msg': ...}`
when applied — most skills already use it.

## Inter-skill calls

Skills can call other skills, but always pass `node=` through:

```python
def pick_no_sound(**kwargs):
    node = kwargs.pop('node', None)
    ret = grip(node=node, inputs='open')       # always pass node down
    ret = approach_close(node=node, **kwargs)
```

This is what keeps the Python-API mode safe: the wrapper only auto-injects
node when the caller has not supplied one. Skill-to-skill calls always
supply, so they never re-enter bootstrap.

## When editing skills

- Add a `(module_path, func_name)` entry to
  [skills_config.py](kcare_robot/configs/skills_config.py); the entry is
  what `auto_wrap_skills` iterates, so unlisted private helpers stay
  unwrapped (no auto-bootstrap risk).
- For UI: hit `POST /skills/reload` or restart `make run`.
- For CLI / Python: re-import the module (or restart the process).

## Devices

Cameras, arm, gripper, mobile base, and the `vlms` TCP detector are
registered in `configs/locations/<active-site>/connections.json` and reloaded
by `DeviceManager.load_saved()` during `bootstrap()` (and again on every
location switch via `DeviceManager.reload_from()`). CLI mode blocks until all
devices are (re)connected so the first skill call has them ready; UI mode loads
in a background thread for fast uvicorn startup.

## Debug entry points

```bash
make doctor                           # ROS env + skill-import smoke
make doctor ARGS=--verbose
make cli ARGS="--list"                # list skills without UI
ROBOT_AGENT_DEBUG_RESPONSE=1 make run # full traceback in skill error dicts
ROBOT_AGENT_LOG_LEVEL=DEBUG make run
```

Logs: `kcare_robot/data/logs/kcare_robot.log` (rotating).

## Related

- [robot_agent](../robot_agent) — runtime core (`bootstrap`, `cli`,
  `SkillRegistry`, `DeviceManager`).
- [robot_template](../robot_template) — cookiecutter for new robots; emits
  the same layout (skills/__init__.py auto-wrap, __main__.py CLI shim).
- [robotapp](../robotapp) — Next.js web dashboard that talks to the UI
  mode over HTTP.
