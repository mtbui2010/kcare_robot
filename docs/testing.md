# Testing kcare_robot

- `make test`: offline pytest of `tests/` (configs, manifest, namemap; no ROS, no
  network, no running agent). Needs robot_agent and embodiplan importable; from a
  checkout of the sibling repos:
  `PYTHONPATH=$PWD:../robot_agent:../embodiplan python -m pytest -q tests`
- `python -m robot_agent.contract.check_manifest --strict kcare_robot`: the manifest
  against the skills.
- On the robot only, after `make test`: `make doctor` (env, ROS 2, skill imports),
  then `make smoke-live` (lists the skills of the agent running on `:$(PORT)`; the
  only target that talks to the network).
- Whole-system scenarios: [../../scenarios/CLAUDE.md](../../scenarios/CLAUDE.md).
