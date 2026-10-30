# Runtime data under configs/ (task_runs, world state, memories, logs)

Rule: `kcare_robot/configs/` holds configuration only; runs, beliefs and logs
belong in `kcare_robot/data/` or on the NAS, and are gitignored.

## Who writes what (checked 2026-10-30)

Nothing in kcare_robot writes these files. They are all written by
robot_agent, relative to the *common* config dir it is given
(`AgentState.common_dir` = `kcare_robot/configs/common/`):

| File | Writer | Default path | Override |
|---|---|---|---|
| closed-loop run records (`RunLogger` JSONL) | `robot_agent/core/planning/loop.py:216` | `common/task_runs/<host>/` | planner option `log_dir` |
| GRACE live memory | `robot_agent/core/planning/loop.py:217` | `common/grace_memory.<host>.jsonl` | planner option `live_path` |
| object memory (assist loop) | `robot_agent/core/planning/assist_loop.py:293` | `common/object_memory.<host>.json` | planner option `memory_path` |
| persisted world belief | `robot_agent/state.py:158` (`AgentState._world_file`) | `common/world_state.<host>.json` | none |
| rotating logs | `robot_agent/runtime.py:73,83` | `kcare_robot/data/logs/<host>/` | `bootstrap(log_dir=)` |

`<host>` is `robot_agent.state.host_id()` (`$ROBOT_AGENT_HOST` or the hostname).

## What was done

All of the above are now in the root `.gitignore`, so new runs never show up in
`git status`.

## What was deliberately NOT done, and why

- **The default paths were not moved to `data/`.** They are not one
  configurable constant in this repo: four separate expressions in robot_agent
  build them from `common_dir`, and `kcare-sim` (`sim/scripts/kcare_sim_app.py`,
  `sim/kcare_sim/tests/live_bridge/make_scratch_root.sh`) relies on the same
  shared-folder layout. Moving them means a robot_agent change (one
  `runtime_data_dir` on `AgentState`, with a fallback read of the old
  `common/` location) and is left to the RobotAdapter / run-manager work.
- **The 69 tracked legacy runs (`configs/common/task_runs/*.jsonl`, the
  pre-`<host>` layout) and `configs/common/world_state.json` were not
  untracked.** `git rm --cached` in a commit deletes the files from every
  other checkout that pulls it, including the robot's. The runs are wanted as
  golden test data (roadmap step 2), and `world_state.json` is no longer read
  (state.py reads `world_state.<host>.json`; `new_robot.py` treats it as
  stale). To untrack them safely: copy `configs/common/task_runs/` to the NAS
  (or to `kcare_robot/data/task_runs_legacy/`) on every machine that has it,
  then `git rm -r --cached kcare_robot/configs/common/task_runs
  kcare_robot/configs/common/world_state.json` and commit.
