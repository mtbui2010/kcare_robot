SHELL := /bin/bash
ROOT  := $(shell pwd)

# robot_agent and pyconnect are expected as sibling repos by default.
# Override on the command line if your layout differs:
#   make install ROBOT_AGENT_DIR=/path/to/robot_agent PYCONNECT_DIR=/path/to/pyconnect
ROBOT_AGENT_DIR ?= $(ROOT)/../robot_agent
PYCONNECT_DIR   ?= $(ROOT)/../pyconnect

ROS_SETUP ?= /opt/ros/humble/setup.bash

# Server port — override on the command line: make run PORT=8002
PORT ?= 8001

.PHONY: install install-deps run cli terminate doctor skill-generic skill-detect run-external test clean help

help:
	@echo "kcare_robot -- robot skills + entry points (UI / CLI / Python API) that use robot_agent"
	@echo ""
	@echo "Targets:"
	@echo "  make install                          Editable-install kcare_robot (+ pyconnect, robot_agent if missing)"
	@echo "  make install-deps                     Install pyconnect + robot_agent only"
	@echo "  make doctor                           Pre-flight check (verifies env, ROS2, skill imports)"
	@echo "  make run                              Start the UI/HTTP agent (PORT=$(PORT))"
	@echo "  make cli ARGS=\"<skill>[::<inputs>] [k=v ...]\""
	@echo "                                        Run one skill via CLI (sources ROS, no server)"
	@echo "                                        e.g.  make cli ARGS=\"find::apple\""
	@echo "                                              make cli ARGS=\"--list\""
	@echo "  make terminate                        Stop the agent listening on PORT=$(PORT)"
	@echo "  make skill-generic SKILL=<name> [FILE=<file>]"
	@echo "                                        Scaffold a generic skill stub (creates/appends file + updates skills_config)"
	@echo "  make skill-detect  SKILL=<name> [FILE=<file>]"
	@echo "                                        Scaffold a detect-style skill (mock _fetch_data + detector client call)"
	@echo "  make run-external                     Start the example external-skill server (template_skills/grip_external.py)"
	@echo "  make test                             Smoke test: list skills via curl"
	@echo "  make clean                            Remove __pycache__ and *.egg-info"

install-deps:
	@if [ -d "$(PYCONNECT_DIR)" ]; then \
		pip install -e $(PYCONNECT_DIR); \
	else \
		echo "[kcare_robot] pyconnect not found at $(PYCONNECT_DIR) -- skipping"; \
	fi
	@if [ -d "$(ROBOT_AGENT_DIR)" ]; then \
		pip install -e $(ROBOT_AGENT_DIR); \
	else \
		echo "[kcare_robot] robot_agent not found at $(ROBOT_AGENT_DIR) -- skipping"; \
	fi

install: install-deps
	pip install -e $(ROOT)
	@echo ""
	@echo "[kcare_robot] Installed. Run with: make run"

run:
	source $(ROS_SETUP) && \
	cd $(ROOT) && \
	uvicorn kcare_robot.main:app --host 0.0.0.0 --port $(PORT) --reload

# Run a single skill from the CLI, sourcing ROS first.
# Pass everything via ARGS so the user controls the arg list verbatim.
#   make cli ARGS="find::apple"
#   make cli ARGS="find::apple estimate_grasp=true camera=arm"
#   make cli ARGS="--list"
cli:
	@source $(ROS_SETUP) && \
	cd $(ROOT) && \
	python3 -m kcare_robot $(ARGS)

# Kill whatever is listening on $(PORT): SIGTERM, wait 3s, then SIGKILL if alive.
terminate:
	@pids=$$(lsof -t -i tcp:$(PORT) -sTCP:LISTEN 2>/dev/null); \
	if [ -z "$$pids" ]; then \
		echo "[kcare_robot] nothing listening on port $(PORT)"; \
		exit 0; \
	fi; \
	echo "[kcare_robot] SIGTERM -> PID(s): $$pids"; \
	kill -TERM $$pids 2>/dev/null || true; \
	for i in 1 2 3; do \
		sleep 1; \
		pids=$$(lsof -t -i tcp:$(PORT) -sTCP:LISTEN 2>/dev/null); \
		[ -z "$$pids" ] && break; \
	done; \
	if [ -n "$$pids" ]; then \
		echo "[kcare_robot] still alive after 3s, SIGKILL -> $$pids"; \
		kill -KILL $$pids 2>/dev/null || true; \
		sleep 1; \
	fi; \
	pids=$$(lsof -t -i tcp:$(PORT) -sTCP:LISTEN 2>/dev/null); \
	if [ -z "$$pids" ]; then \
		echo "[kcare_robot] port $(PORT) is free"; \
	else \
		echo "[kcare_robot] WARNING: port $(PORT) still occupied by: $$pids"; \
		exit 1; \
	fi

doctor:
	@source $(ROS_SETUP) 2>/dev/null || true; \
	python3 -m robot_agent.diagnose kcare_robot $(ROOT)/kcare_robot/data $(ARGS)

# Scaffold a new skill. SKILL is required; FILE defaults to <SKILL>.py.
#   make skill-generic SKILL=wave
#   make skill-generic SKILL=rotate FILE=arm.py
skill-generic:
	@if [ -z "$(SKILL)" ]; then \
		echo "Usage: make skill-generic SKILL=<skill_name> [FILE=<file>]"; \
		exit 2; \
	fi
	python3 -m robot_agent.new_skill kcare_robot $(SKILL) $(FILE) --template generic

# Scaffold a detect-style skill (mocks _fetch_data + detector TCP client call).
#   make skill-detect SKILL=find_apple
#   make skill-detect SKILL=find_apple FILE=detectors.py
skill-detect:
	@if [ -z "$(SKILL)" ]; then \
		echo "Usage: make skill-detect SKILL=<skill_name> [FILE=<file>]"; \
		exit 2; \
	fi
	python3 -m robot_agent.new_skill kcare_robot $(SKILL) $(FILE) --template detect

run-external:
	source $(ROS_SETUP) && \
	cd $(ROOT) && \
	uvicorn kcare_robot.template_skills.grip_external:app --host 0.0.0.0 --port 9000

test:
	@echo "[kcare_robot] Registered skills:"
	@curl -s http://localhost:$(PORT)/skills | python3 -m json.tool | head -40

clean:
	find $(ROOT) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find $(ROOT) -type d -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	@echo "[kcare_robot] cleaned."
