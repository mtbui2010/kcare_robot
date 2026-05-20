"""CLI entry point for the kcare_robot package.

Exposed via ``[project.scripts]`` in ``pyproject.toml``::

    kcare_robot = "kcare_robot.__main__:cli"

Usage::

    kcare_robot find::apple
    kcare_robot pick::apple
    kcare_robot --list
"""

from robot_agent.cli import main


def cli() -> int:
    return main(robot_pkg='kcare_robot')


if __name__ == '__main__':
    raise SystemExit(cli())
