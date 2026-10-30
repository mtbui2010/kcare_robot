"""Pure-python checks of kcare_robot's config files: no ROS, no network, no
running agent. Run: make test  (or  PYTHONPATH=. python -m pytest -q tests)."""

import json
from pathlib import Path

import pytest

CONFIGS = Path(__file__).resolve().parents[1] / 'kcare_robot' / 'configs'


def _json_files():
    return sorted(p for p in CONFIGS.rglob('*.json') if 'task_runs' not in p.parts)


@pytest.mark.parametrize('path', _json_files(), ids=lambda p: str(p.relative_to(CONFIGS)))
def test_json_config_parses(path):
    json.loads(path.read_text(encoding='utf-8'))


def test_skills_json_matches_skills_config():
    from kcare_robot.configs.skills_config import SKILL_CONFIGS
    entries = json.loads((CONFIGS / 'common' / 'skills.json').read_text(encoding='utf-8'))
    names = [e['name'] for e in entries]
    assert len(names) == len(set(names)), 'duplicate skill names in skills.json'
    for e in entries:
        if e.get('type') != 'internal':
            continue
        assert e['name'] in SKILL_CONFIGS, f"{e['name']} is in skills.json but not in skills_config.py"
        assert (e['module_path'], e['func_name']) == SKILL_CONFIGS[e['name']]


def test_skill_modules_exist():
    from kcare_robot.configs.skills_config import SKILL_CONFIGS
    root = CONFIGS.parent.parent
    for name, (module, func) in SKILL_CONFIGS.items():
        src = root.joinpath(*module.split('.')).with_suffix('.py')
        assert src.is_file(), f'{name}: {src} missing'
        assert f'def {func}(' in src.read_text(encoding='utf-8'), f'{name}: no def {func} in {src.name}'
