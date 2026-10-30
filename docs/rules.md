# kcare_robot rules (full text)

The short form is in [../CLAUDE.md](../CLAUDE.md); the numbers match. **This repo
runs on the real robot**: keep changes backwards compatible, and check them with
`make test` (offline) before `make doctor` / `make smoke-live` on the robot.

## Repo rules

### 1. Use only robot_agent's public API

No `_name` imports or attributes across the package boundary. Allowed:
`robot_agent.state.current`, `robot_agent.skills` (`log_data`, `spoken`,
`auto_wrap_skills`, dataset helpers), the public names in `robot_agent.utils`,
`robot_agent.skill_configs`, `robot_agent.env_names`,
`robot_agent.core.run_control` (`cancel_requested`, `dialog_*`),
`robot_agent.connect.serde`, `robot_agent.core.shared_json.read_fresh`,
`robot_agent.create_app`, `robot_agent.contract`.

Known violations still to remove (do not add more): `robot_agent.utils._llm`,
`skill_configs._ConfigProxy`, `core.planning.loop._read_robot_xy`,
`dm._get_ros_node` / `dm._ros_node` in `configs/grace_namemap.py`.

Why: robot_agent changes its private code freely; a private import here breaks
the real robot on the next core update.

### 2. Every skill the planner uses has a SkillSpec in the manifest

[configs/manifest.yaml](../kcare_robot/configs/manifest.yaml) holds name, action,
params, pre / effects, verify, interruptible, errors. The skill returns a
`SkillResult` (until migrated: `{'isdone': bool, 'msg': str, ...}`, which
`SkillResult.from_legacy` reads). Changing a skill's params, effects or failure
modes means updating its entry; `tests/test_manifest.py` checks it against
`grace_namemap` and `skills_config`.

Why: the core plans, binds and verifies only from the manifest
([robot_agent contract](../../robot_agent/docs/contract.md)).

### 3. `configs/` holds configuration only

Task runs, world state, memories and logs go to `kcare_robot/data/` or the NAS
and are gitignored (see [notes/runtime_data.md](notes/runtime_data.md)).

Why: runtime data in `configs/` gets committed by accident and mixes robot
history with configuration.

### 4. No `.bak` / `.orig` / dated copies in the repo

Git history is the backup. No tracked `__pycache__` / `*.pyc`.

### 5. CLAUDE.md holds rules only

Experiment logs and notes go to `docs/notes/<topic>_<yyyymmdd>.md`; how things
work goes to [notes/architecture.md](notes/architecture.md) and other `docs/*.md`,
linked from CLAUDE.md and [README.md](README.md).

## Code rules

### 6. One process = one mode

UI (`make run`), CLI (`kcare_robot find::apple`) and Python API
(`from kcare_robot.skills.recognition import find`) all call
`robot_agent.runtime.bootstrap('kcare_robot', ...)`, which creates one ROS node
per process. Do not mix modes in a process.

### 7. Skill contract

`def my_skill(node, **params) -> dict` returning a dict with `isdone`; wrap entry
points with `@exception_handler` (`robot_agent.utils`) so an exception becomes
`{'isdone': False, 'msg': ...}`.

### 8. Inter-skill calls always pass `node=` through

`grip(node=node, ...)`; the auto-wrapper only injects a node, and bootstraps,
when none is given.

Why: a second node per process breaks ROS callbacks (rule 6).

### 9. New skill: register it

Add its `(module_path, func_name)` to
[skills_config.py](../kcare_robot/configs/skills_config.py) (what
`auto_wrap_skills` iterates; unlisted private helpers stay unwrapped), and, if the
planner uses it, to `configs/manifest.yaml`. Then `POST /skills/reload` or restart.

### 10. World state is a belief, not a sensor read

`holding` / `found_pose` / `holding_pose` are hints (re-detect if the base moved;
use `grasp_succeed` to know what is in the gripper). Skills reach it through
`robot_agent.state.current().world`; a skill's effect on it lives in
`grace_namemap.apply_skill_effect`, kept consistent with the manifest's effects.

### 11. Configs: per-site values and secrets

Per-site values go in `configs/locations/<site>/skill_configs_override.json`;
secrets in that site's `.env` (gitignored), never in a tracked file.

### 12. Tests are pure Python

`make test`: no ROS, no network, no running agent. `make smoke-live` is the only
target that talks to `:$(PORT)`. See [testing.md](testing.md).

### 13. Install stops on any failure

`make install` installs robot_agent and embodiplan (`EMBODIPLAN_DIR`, default
`../embodiplan`, pinned `>=0.2,<0.3`) and stops on any failure; do not
reintroduce `|| true` on install steps.

Why: a half-installed robot fails later, on the robot, with a confusing error.
