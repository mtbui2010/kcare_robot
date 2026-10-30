# kcare_robot

Skills and config of the KCare (KAAIR) mobile manipulator, a robot package on top of [robot_agent](../robot_agent/CLAUDE.md).
**Runs on the real robot**: keep changes backwards compatible. Full rules and reasons: [docs/rules.md](docs/rules.md).

## Repo rules

1. Use only robot_agent's public API; add no new `_private` imports (list of allowed names in [rules](docs/rules.md)).
2. Give every skill the planner uses a SkillSpec in `configs/manifest.yaml` and keep it in sync.
3. Keep `configs/` for configuration only; runtime data goes to `kcare_robot/data/` or the NAS ([runtime_data](docs/notes/runtime_data.md)).
4. Commit no `.bak` / `.orig` / dated copies and no `__pycache__` / `*.pyc`.
5. Keep this file short (rules and links only); notes go to `docs/notes/<topic>_<yyyymmdd>.md`, details to `docs/*.md`.

## Code rules

6. Run one mode per process (UI, CLI or Python API), all through `robot_agent.runtime.bootstrap`.
7. Write skills as `def skill(node, **params) -> dict` with `isdone`, wrapped with `@exception_handler`.
8. Pass `node=` through on every inter-skill call.
9. Register a new skill in `configs/skills_config.py` (and the manifest if the planner uses it).
10. Treat world state as a belief; keep `grace_namemap.apply_skill_effect` consistent with the manifest.
11. Put per-site values in `configs/locations/<site>/`; secrets only in that site's gitignored `.env`.
12. Keep tests pure Python; only `make smoke-live` talks to a running agent.
13. Let `make install` stop on any failure; never add `|| true`.

## Test

`make test` (offline), then on the robot `make doctor` / `make smoke-live`. Details: [docs/testing.md](docs/testing.md).

## Docs

- [docs/README.md](docs/README.md): index
- [docs/notes/architecture.md](docs/notes/architecture.md): how it fits together, file map, debugging
- [docs/SKILLS.md](docs/SKILLS.md): skill reference
- Planning specs: [robot_agent/docs](../robot_agent/docs/README.md)
