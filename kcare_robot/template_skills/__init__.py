# Reference skill templates for the 3 patterns supported by robot_agent.
#
# Files in this folder are NOT auto-registered in SKILL_CONFIGS.
# Copy one as a starting point for your own skill, then add an entry to
# kcare_robot/configs/skills_config.py to expose it via POST /skill/{name}.
#
#   grip_node_agent.py  -- Pattern 1: shortest, uses the connect-layer NodeAgent
#   grip_pure_ros2.py   -- Pattern 2: pure rclpy via node.agents['grip'].raw
#   grip_external.py    -- Pattern 3: external HTTP service (FastAPI example)
