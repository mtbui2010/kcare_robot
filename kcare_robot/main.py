"""Entry point for the kcare_robot agent.

Run with:

    cd kcare_robot/
    make run                  # default port 8001
    make run PORT=8002        # alternative port (for running multiple robots)

Or directly with uvicorn:

    uvicorn kcare_robot.main:app --host 0.0.0.0 --port 8001 --reload

VSCode debug: point a "Python: Module" launch config at ``uvicorn`` with args
``kcare_robot.main:app --reload --host 0.0.0.0 --port 8001`` to set breakpoints
in both kcare_robot.skills and robot_agent.core.
"""

from pathlib import Path

from robot_agent import create_app

DATA_DIR = Path(__file__).parent / 'data'

app = create_app(robot_pkg='kcare_robot', data_dir=DATA_DIR)
