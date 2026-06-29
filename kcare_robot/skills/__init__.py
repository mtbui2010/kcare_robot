"""Skill package init for kcare_robot.

Auto-wraps every skill listed in ``kcare_robot.configs.skills_config.SKILL_CONFIGS``
with ``robot_agent.skills.skill_entry``. After this runs::

    from kcare_robot.skills.recognition import find
    ret = find(inputs='apple')      # works without prior bootstrap()
    ret = find('apple')             # first positional treated as `inputs`

Wrapping is lazy in effect: the wrapper only triggers ``bootstrap()`` the
first time a wrapped skill is actually called.

When the same skill is invoked via SkillRegistry (UI / CLI), ``node`` is
already supplied, so the wrapper passes through without re-entering
bootstrap.
"""

from robot_agent.skills import auto_wrap_skills, log_data  # noqa: F401

try:
    from kcare_robot.configs.skills_config import SKILL_CONFIGS
    auto_wrap_skills(SKILL_CONFIGS, pkg='kcare_robot')
except Exception as _e:
    import sys
    print(f'[kcare_robot.skills] auto_wrap_skills skipped: {_e}', file=sys.stderr)
