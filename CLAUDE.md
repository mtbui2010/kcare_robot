# kcare_robot — rules

Concrete skill + config implementation for the KCare (KAAIR) mobile
manipulator, hosted on top of [`robot_agent`](../robot_agent) as a pluggable
package. **This repo runs on the real robot**: keep changes backwards
compatible, and check them with `make test` (offline) before `make doctor` /
`make smoke-live` on the robot.

This file holds rules only. How things work, file map, locations, world state,
devices, debugging: [docs/notes/architecture.md](docs/notes/architecture.md).
Experiment logs and measurements go in `docs/notes/<topic>_<yyyymmdd>.md`.

## Repo rules

1. **Use only robot_agent's public API.** No `_name` imports or attributes
   across the package boundary. Allowed: `robot_agent.state.current`,
   `robot_agent.skills` (`log_data`, `spoken`, `auto_wrap_skills`, dataset
   helpers), the public names in `robot_agent.utils`,
   `robot_agent.skill_configs`, `robot_agent.env_names`,
   `robot_agent.core.run_control` (`cancel_requested`, `dialog_*`),
   `robot_agent.connect.serde`, `robot_agent.core.shared_json.read_fresh`,
   `robot_agent.create_app`, `robot_agent.contract`. Known violations still to
   remove (do not add more): `robot_agent.utils._llm`,
   `skill_configs._ConfigProxy`, `core.planning.loop._read_robot_xy`,
   `dm._get_ros_node` / `dm._ros_node` in `configs/grace_namemap.py`.
2. **Every skill the planner uses has a SkillSpec in
   [configs/manifest.yaml](kcare_robot/configs/manifest.yaml)** (name,
   action, params, pre / effects, verify, interruptible, errors) and returns a
   `SkillResult` (until migrated: `{'isdone': bool, 'msg': str, ...}`, which
   `SkillResult.from_legacy` reads). Changing a skill's params, effects or
   failure modes means updating its entry; `tests/test_manifest.py` checks it
   against `grace_namemap` and `skills_config`.
3. **`configs/` holds configuration only.** Task runs, world state, memories
   and logs go to `kcare_robot/data/` or the NAS and are gitignored (see
   [docs/notes/runtime_data.md](docs/notes/runtime_data.md)).
4. **No `.bak` / `.orig` / dated copies in the repo**; git history is the
   backup. No tracked `__pycache__` / `*.pyc`.
5. **CLAUDE.md holds rules only.** Experiment logs and notes go to
   `docs/notes/`.

## Code rules

- **One process = one mode.** UI (`make run`), CLI (`kcare_robot find::apple`)
  and Python API (`from kcare_robot.skills.recognition import find`) all call
  `robot_agent.runtime.bootstrap('kcare_robot', ...)`, which creates one ROS
  node per process. Do not mix modes in a process.
- **Skill contract:** `def my_skill(node, **params) -> dict` returning a dict
  with `isdone`; wrap entry points with `@exception_handler`
  (`robot_agent.utils`) so an exception becomes `{'isdone': False, 'msg': ...}`.
- **Inter-skill calls always pass `node=` through** (`grip(node=node, ...)`);
  the auto-wrapper only injects a node, and bootstraps, when none is given.
- **New skill:** add its `(module_path, func_name)` to
  [skills_config.py](kcare_robot/configs/skills_config.py) (what
  `auto_wrap_skills` iterates; unlisted private helpers stay unwrapped), and,
  if the planner uses it, to `configs/manifest.yaml`. Then
  `POST /skills/reload` or restart.
- **World state is a belief**, not a sensor read: `holding` / `found_pose` /
  `holding_pose` are hints (re-detect if the base moved; use `grasp_succeed`
  to know what is in the gripper). Skills reach it through
  `robot_agent.state.current().world`; a skill's effect on it lives in
  `grace_namemap.apply_skill_effect`, kept consistent with the manifest's
  effects.
- **Configs:** per-site values go in
  `configs/locations/<site>/skill_configs_override.json`; secrets in that
  site's `.env` (gitignored), never in a tracked file.
- **Tests** (`make test`) are pure Python: no ROS, no network, no running
  agent. `make smoke-live` is the only target that talks to `:$(PORT)`.
- **Install:** `make install` installs robot_agent and embodiplan
  (`EMBODIPLAN_DIR`, default `../embodiplan`, pinned `>=0.2,<0.3`) and stops
  on any failure; do not reintroduce `|| true` on install steps.
