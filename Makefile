SHELL := /bin/bash
ROOT  := $(shell pwd)

# ── Sibling repos (editable-installed by `make install`) ─────────────────────
# Checked by `make check-deps` BEFORE anything is created or downloaded.
# Override on the command line if your layout differs:
#   make install ROBOT_AGENT_DIR=/path/to/robot_agent
# Set one to empty to skip it entirely:
#   make install PYPLANNER_DIR=          # no GRACE planner
ROBOT_AGENT_DIR ?= $(ROOT)/../robot_agent
PYPLANNER_DIR   ?= $(ROOT)/../pyplanner

# `visionserve` comes from PyPI (a normal dependency in pyproject.toml), NOT
# from ../vision_serve/clients/python — that checkout gitignores the package
# source, so an editable install there silently yields an empty module.
# To develop against a local copy instead:
#   make install VISIONSERVE_DIR=/path/to/visionserve/source
VISIONSERVE_DIR ?=

# ── ROS 2 ────────────────────────────────────────────────────────────────────
# rclpy / cv_bridge / sensor_msgs are NOT pip-installable — they come from the
# sourced ROS distro and are compiled against ONE specific CPython version.
# The conda env below is therefore created with exactly that version, so
# `source setup.bash` + the env's python can import rclpy.
ROS_DISTRO ?= humble
ROS_PREFIX ?= /opt/ros/$(ROS_DISTRO)
ROS_SETUP  ?= $(ROS_PREFIX)/setup.bash

# Colcon overlay(s) carrying the robot's OWN interfaces — `rosinterfaces`
# (SendStringData srv/action, used by every generic ROS agent config),
# kaair_msgs, drivers. The distro setup.bash does NOT provide these, so a
# login shell that sources them from .bashrc works while `make` did not.
# Space-separated setup.bash paths; only the ones that exist are sourced.
# Override when your workspace lives elsewhere:
#   make doctor ROS_WS_SETUP=/path/to/ws/install/setup.bash
ROS_WS_SETUP ?= $(wildcard /ros_ws/ros2_ws/install/setup.bash \
                           $(HOME)/ros2_ws/install/setup.bash \
                           $(ROOT)/../ros2_ws/install/setup.bash)
_SOURCE_ROS = source $(ROS_SETUP) 2>/dev/null; $(foreach w,$(ROS_WS_SETUP),source $(w) 2>/dev/null;)

# Probe the distro for the CPython version rclpy was built for (e.g. "3.10").
# Falls back to 3.10 (Humble's default) when ROS is not installed locally.
ROS_PY_VER := $(shell ls -d $(ROS_PREFIX)/lib/python3.* 2>/dev/null | head -1 | sed 's|.*/python||')
PYTHON_VERSION ?= $(if $(ROS_PY_VER),$(ROS_PY_VER),3.10)

# ── Target environment ───────────────────────────────────────────────────────
# `conda activate` does not work in make's non-interactive shell, so every
# target calls the target interpreter by absolute path instead.
#
# Target env, in order of precedence:
#   make install                          -> conda env 'kcare' (created if absent)
#   make install CONDA_ENV=myenv          -> conda env 'myenv' (created if absent)
#   make install USE_EXISTING=1           -> must already exist; never created
#   make install ENV_PREFIX=/path/to/env  -> conda env by path
#   make install USE_CURRENT=1            -> whatever `python3` is active NOW
#                                            (activated conda env, venv, pyenv…)
#   make install PYTHON=/path/bin/python  -> that exact interpreter (implies
#                                            USE_CURRENT=1)
# USE_CURRENT never creates anything; conda base / system python are refused
# unless FORCE=1.
CONDA_ENV    ?= kcare
# 1 = refuse to create anything; fail if the target env does not exist. Use it
# to install into an env you already manage, and as typo protection (a wrong
# CONDA_ENV would otherwise silently build a brand-new env).
USE_EXISTING ?= 0
# 1 = target the python currently on PATH instead of any conda env by name.
USE_CURRENT  ?= 0
# Explicit interpreter path; setting it switches USE_CURRENT on.
PYTHON       ?=
# Confirmation gate for installing into conda base or the system python.
FORCE        ?= 0
# Native libraries the pip wheels dlopen at runtime but do not bundle:
#   portaudio → sounddevice (TTS playback)
#   ffmpeg    → pydub (mp3 decode in text2voice)
# Without them TTS imports fine but fails at first use, which `make doctor`
# reports as "installed but not usable". Set empty to leave a pre-existing env's
# conda packages untouched:  make install USE_EXISTING=1 CONDA_SYS_LIBS=
CONDA_SYS_LIBS ?= portaudio ffmpeg
CONDA_BASE := $(shell conda info --base 2>/dev/null)

ifneq ($(PYTHON),)
  USE_CURRENT := 1
endif

ifeq ($(USE_CURRENT),1)
  # Current-python mode: resolve the interpreter first, derive the prefix from
  # it (sys.prefix — works for conda envs, venvs and system python alike).
  ENV_PY     := $(if $(PYTHON),$(PYTHON),$(shell command -v python3 2>/dev/null))
  ENV_PREFIX := $(shell $(ENV_PY) -c 'import sys; print(sys.prefix)' 2>/dev/null)
else
  # Conda mode: resolve the prefix first, the interpreter lives inside it.
  # Overridable so an env anywhere on disk can be targeted, not just
  # $base/envs/. Every conda call below uses `-p $(ENV_PREFIX)`, which works
  # for both named and prefix envs, so there is a single code path either way.
  ENV_PREFIX ?= $(CONDA_BASE)/envs/$(CONDA_ENV)
  ENV_PY     = $(ENV_PREFIX)/bin/python
endif
ENV_PIP    = $(ENV_PY) -m pip
# mamba resolves much faster when it is available.
CONDA_BIN  := $(shell command -v mamba 2>/dev/null || command -v conda 2>/dev/null)

# ── Run-time interpreter ─────────────────────────────────────────────────────
# run / cli / doctor / scaffolding must not require `make install` to have run
# first: when the target conda env does not exist they fall back to the python
# currently on PATH, so a machine whose environment was prepared some other way
# still works. `install` / `install-deps` / `env` never fall back — creating and
# populating the env is exactly their job.
CURRENT_PY   := $(shell command -v python3 2>/dev/null)
RUN_FALLBACK := $(if $(wildcard $(ENV_PY)),,1)
RUN_PY       := $(if $(RUN_FALLBACK),$(CURRENT_PY),$(ENV_PY))
RUN_PREFIX   := $(shell $(RUN_PY) -c 'import sys; print(sys.prefix)' 2>/dev/null)

# Prefix for anything that needs both ROS and the run-time python.
# The env's bin/ goes on PATH too: calling $(RUN_PY) by absolute path does NOT
# activate the env, so console scripts and the native tools the wheels shell out
# to (ffmpeg for pydub) would otherwise be invisible.
ROS_ENV = $(_SOURCE_ROS) export PATH="$(RUN_PREFIX)/bin:$$PATH"; cd $(ROOT) &&

# Server port — override on the command line: make run PORT=8002
PORT ?= 8001

# Post-install sanity check. A pip install can report success while leaving an
# unimportable package behind (a source tree whose contents are gitignored
# builds into an empty wheel), so `pip list` alone proves nothing. Exported so
# the recipe can pass it to python -c; multi-line heredocs do not survive make's
# one-shell-per-recipe-line execution.
define VERIFY_IMPORTS_PY
import importlib, sys
bad = []
for m in ('numpy', 'cv2', 'scipy', 'visionserve', 'robot_agent.connect', 'kcare_robot'):
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append('    %s: %s: %s' % (m, type(e).__name__, e))
if bad:
    print('[kcare_robot] FAILED -- installed but not importable:')
    print('\n'.join(bad))
    sys.exit(1)
import numpy
print('[kcare_robot] imports OK (numpy %s)' % numpy.__version__)
endef
export VERIFY_IMPORTS_PY

.PHONY: install install-deps check-deps env _env-current env-recreate env-info run cli terminate doctor \
        skill-generic skill-detect skill-external delete-skill rename-skill \
        run-external test clean help _require-env _require-run-py

help:
	@echo "kcare_robot -- robot skills + entry points (UI / CLI / Python API) that use robot_agent"
	@echo ""
	@echo "Setup:"
	@echo "  make install                          Create conda env '$(CONDA_ENV)' + install ALL dependencies"
	@echo "  make install CONDA_ENV=<name>         ... into a differently-named env (created if missing)"
	@echo "  make install USE_EXISTING=1           ... into an env that must ALREADY exist (never creates)"
	@echo "  make install ENV_PREFIX=<path>        ... into a conda env given by path"
	@echo "  make install USE_CURRENT=1            ... into the python that is active RIGHT NOW (conda/venv/pyenv)"
	@echo "  make install PYTHON=<path>            ... into that exact interpreter (implies USE_CURRENT=1)"
	@echo "  make env                              Create/reuse the conda env only (python $(PYTHON_VERSION))"
	@echo "  make env-recreate                     Delete and rebuild the env from scratch"
	@echo "  make env-info                         Show resolved env / ROS / python paths"
	@echo "  make check-deps                       Verify the sibling repos exist in ../ (run by install)"
	@echo "  make install-deps                     Editable-install sibling repos only"
	@echo "  make doctor                           Pre-flight check (verifies env, ROS2, skill imports)"
	@echo ""
	@echo "Run:"
	@echo "  make run                              Start the UI/HTTP agent (PORT=$(PORT))"
	@echo "  make cli ARGS=\"<skill>[::<inputs>] [k=v ...]\""
	@echo "                                        Run one skill via CLI (sources ROS, no server)"
	@echo "                                        e.g.  make cli ARGS=\"find::apple\""
	@echo "                                              make cli ARGS=\"--list\""
	@echo "  make terminate                        Stop the agent listening on PORT=$(PORT)"
	@echo "  make run-external                     Start the example external-skill server (template_skills/grip_external.py)"
	@echo "  make test                             Smoke test: list skills via curl"
	@echo ""
	@echo "Skills:"
	@echo "  make skill-generic SKILL=<name> [FILE=<file>]"
	@echo "                                        Scaffold a generic skill stub (creates/appends file + updates skills_config)"
	@echo "  make skill-detect  SKILL=<name> [FILE=<file>]"
	@echo "                                        Scaffold a detect-style skill (mock _fetch_data + detector client call)"
	@echo "  make skill-external SKILL=<name>"
	@echo "                                        Scaffold a standalone HTTP-service skill (own rclpy node + FastAPI app)"
	@echo "                                        in template_skills/<skill>_external.py"
	@echo "  make delete-skill  SKILL=<name> [YES=1]"
	@echo "                                        Remove a skill from skills_config and from its file."
	@echo "                                        If that file holds only this skill, the whole file is deleted."
	@echo "                                        Prompts for confirmation unless YES=1."
	@echo "  make rename-skill  SKILL=<old> NEW=<new> [YES=1]"
	@echo "                                        Rename a skill (registry key, def, and its file when dedicated)"
	@echo ""
	@echo "  make clean                            Remove __pycache__ and *.egg-info"
	@echo ""
	@echo "Overridables: CONDA_ENV=$(CONDA_ENV)  USE_EXISTING=$(USE_EXISTING)  USE_CURRENT=$(USE_CURRENT)  ROS_DISTRO=$(ROS_DISTRO)"
	@echo "              PYTHON_VERSION=$(PYTHON_VERSION)  PORT=$(PORT)  CONDA_SYS_LIBS=\"$(CONDA_SYS_LIBS)\""

# ── Environment ──────────────────────────────────────────────────────────────

env-info:
	@echo "conda binary    : $(if $(CONDA_BIN),$(CONDA_BIN),NOT FOUND)"
	@echo "conda base      : $(if $(CONDA_BASE),$(CONDA_BASE),NOT FOUND)"
	@echo "mode            : $(if $(filter 1,$(USE_CURRENT)),current python (USE_CURRENT=1),conda env '$(CONDA_ENV)')"
	@echo "env prefix      : $(ENV_PREFIX)"
	@echo "env python      : $(ENV_PY) $(if $(wildcard $(ENV_PY)),(exists),(NOT CREATED))"
	@if [ -x "$(ENV_PY)" ]; then \
		echo "env py version  : $$($(ENV_PY) -c 'import sys; print(sys.version.split()[0])')"; \
	fi
	@echo "USE_EXISTING    : $(USE_EXISTING) $(if $(filter 1,$(USE_EXISTING)),(never create),(create if missing))"
	@echo "run/doctor use  : $(RUN_PY)$(if $(RUN_FALLBACK), (FALLBACK -- env absent),)"
	@echo "target py       : $(PYTHON_VERSION) $(if $(ROS_PY_VER),(probed from $(ROS_PREFIX)),(default — ROS not found locally))"
	@echo "ROS distro      : $(ROS_DISTRO)"
	@echo "ROS setup       : $(ROS_SETUP) $(if $(wildcard $(ROS_SETUP)),(exists),(MISSING))"
	@echo "ROS overlays    : $(if $(ROS_WS_SETUP),$(ROS_WS_SETUP),(none found — rosinterfaces will be missing))"
	@echo "robot_agent     : $(ROBOT_AGENT_DIR) $(if $(wildcard $(ROBOT_AGENT_DIR)),(exists),(MISSING))"
	@echo "pyplanner       : $(PYPLANNER_DIR) $(if $(wildcard $(PYPLANNER_DIR)),(exists),(MISSING))"
	@echo "visionserve     : $(if $(VISIONSERVE_DIR),$(VISIONSERVE_DIR) $(if $(wildcard $(VISIONSERVE_DIR)),(exists),(MISSING)),from PyPI)"

# Verify every sibling repo we editable-install is actually present AND is a
# buildable Python project, before `install` creates an env or downloads
# anything. A directory that exists but has no pyproject.toml/setup.py installs
# "successfully" as an empty package — that is how a gitignored source tree
# silently disabled every vision skill once. Reports all problems at once.
check-deps:
	@missing=0; \
	for spec in "robot_agent:$(ROBOT_AGENT_DIR)" "pyplanner:$(PYPLANNER_DIR)" "visionserve:$(VISIONSERVE_DIR)"; do \
		name=$${spec%%:*}; dir=$${spec#*:}; \
		var=$$(echo "$$name" | tr 'a-z' 'A-Z')_DIR; \
		if [ -z "$$dir" ]; then \
			if [ "$$name" = "visionserve" ]; then \
				printf '  [ -- ] %-12s no local checkout -> will pip install from PyPI\n' "$$name"; \
			else \
				printf '  [ -- ] %-12s skipped (%s is empty)\n' "$$name" "$$var"; \
			fi; \
			continue; \
		fi; \
		if [ ! -d "$$dir" ]; then \
			printf '  [FAIL] %-12s NOT FOUND at %s\n' "$$name" "$$dir"; \
			missing=1; \
		elif [ ! -f "$$dir/pyproject.toml" ] && [ ! -f "$$dir/setup.py" ]; then \
			printf '  [FAIL] %-12s %s has no pyproject.toml/setup.py\n' "$$name" "$$dir"; \
			missing=1; \
		else \
			printf '  [ OK ] %-12s %s\n' "$$name" "$$dir"; \
		fi; \
	done; \
	if [ "$$missing" = "1" ]; then \
		echo ""; \
		echo "[kcare_robot] missing sibling repo(s). Expected them next to this one:"; \
		echo "                 <workspace>/robot_agent"; \
		echo "                 <workspace>/pyplanner"; \
		echo "                 <workspace>/kcare_robot   <- you are here"; \
		echo "               Point at yours:  make install ROBOT_AGENT_DIR=/path/to/robot_agent"; \
		echo "               Or skip one:     make install PYPLANNER_DIR="; \
		exit 1; \
	fi

# Reuse the target env when it exists, otherwise create it — unless
# USE_EXISTING=1, which makes a missing env a hard error instead.
# USE_CURRENT=1 short-circuits into _env-current: nothing is ever created.
env:
ifeq ($(USE_CURRENT),1)
	@$(MAKE) --no-print-directory _env-current
else
	@if [ -z "$(CONDA_BIN)" ]; then \
		echo "[kcare_robot] conda not found on PATH."; \
		echo "               Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"; \
		exit 1; \
	fi
	@if [ -x "$(ENV_PY)" ]; then \
		have=$$($(ENV_PY) -c 'import sys; print("%d.%d" % sys.version_info[:2])'); \
		echo "[kcare_robot] reusing existing env at $(ENV_PREFIX) (python $$have)"; \
		if [ -n "$(ROS_PY_VER)" ] && [ "$$have" != "$(ROS_PY_VER)" ]; then \
			echo ""; \
			echo "  ! python $$have does not match ROS $(ROS_DISTRO), which was built for $(ROS_PY_VER)."; \
			echo "    rclpy is a compiled extension -- it will not import in this env."; \
			echo "    Use an env on python $(ROS_PY_VER), or: make install CONDA_ENV=kcare$(ROS_PY_VER)"; \
			echo ""; \
		fi; \
	elif [ "$(USE_EXISTING)" = "1" ]; then \
		echo "[kcare_robot] USE_EXISTING=1 but no conda env at $(ENV_PREFIX)"; \
		echo "               Existing envs:"; \
		$(CONDA_BIN) env list 2>/dev/null | sed 's/^/                 /'; \
		echo "               Pick one:  make install CONDA_ENV=<name> USE_EXISTING=1"; \
		echo "                          make install ENV_PREFIX=/path/to/env USE_EXISTING=1"; \
		echo "               Or drop USE_EXISTING to create '$(CONDA_ENV)'."; \
		exit 1; \
	else \
		echo "[kcare_robot] creating conda env at $(ENV_PREFIX) with python $(PYTHON_VERSION) ..."; \
		if [ ! -d "$(ROS_PREFIX)" ]; then \
			echo "[kcare_robot] NOTE: $(ROS_PREFIX) not found -- using default python $(PYTHON_VERSION)."; \
			echo "               On the robot, run this where ROS $(ROS_DISTRO) is installed so the"; \
			echo "               env python matches rclpy's build, or pass PYTHON_VERSION=<x.y>."; \
		fi; \
		$(CONDA_BIN) create -y -p "$(ENV_PREFIX)" -c conda-forge python=$(PYTHON_VERSION); \
	fi
	@if [ -n "$(CONDA_SYS_LIBS)" ]; then \
		echo "[kcare_robot] ensuring native libs: $(CONDA_SYS_LIBS)"; \
		$(CONDA_BIN) install -y -p "$(ENV_PREFIX)" -c conda-forge $(CONDA_SYS_LIBS) >/dev/null \
			|| echo "[kcare_robot] WARNING: could not install $(CONDA_SYS_LIBS) -- TTS may be unavailable"; \
	fi
endif

# USE_CURRENT=1 — validate the active interpreter instead of creating anything.
# Refuses conda base and the system python unless FORCE=1: ~40 packages in
# either are painful to undo, and forgetting `conda activate` is exactly how
# you end up there.
_env-current:
	@if [ -z "$(ENV_PY)" ] || [ ! -x "$(ENV_PY)" ]; then \
		echo "[kcare_robot] USE_CURRENT=1 but no usable python found ($(if $(PYTHON),PYTHON=$(PYTHON),python3 not on PATH))"; \
		exit 1; \
	fi
	@have=$$($(ENV_PY) -c 'import sys; print("%d.%d" % sys.version_info[:2])'); \
	echo "[kcare_robot] using current python: $(ENV_PY) (python $$have, prefix $(ENV_PREFIX))"; \
	risky=""; \
	if [ -n "$(CONDA_BASE)" ] && [ "$(ENV_PREFIX)" = "$(CONDA_BASE)" ]; then \
		risky="the conda BASE env"; \
	fi; \
	case "$(ENV_PY)" in /usr/bin/*|/bin/*) risky="the SYSTEM python";; esac; \
	if [ -n "$$risky" ] && [ "$(FORCE)" != "1" ]; then \
		echo ""; \
		echo "  ! $(ENV_PY) is $$risky."; \
		echo "    Installing ~40 packages there is hard to undo. If you really mean it:"; \
		echo "        make install USE_CURRENT=1 FORCE=1"; \
		echo "    Otherwise activate the env you want first:"; \
		echo "        conda activate <env>   (or: source .venv/bin/activate)"; \
		echo "        make install USE_CURRENT=1"; \
		exit 1; \
	fi; \
	if [ -n "$(ROS_PY_VER)" ] && [ "$$have" != "$(ROS_PY_VER)" ]; then \
		echo ""; \
		echo "  ! python $$have does not match ROS $(ROS_DISTRO), which was built for $(ROS_PY_VER)."; \
		echo "    rclpy is a compiled extension -- it will not import in this env."; \
		echo ""; \
	fi
	@if [ -n "$(CONDA_SYS_LIBS)" ]; then \
		if [ -d "$(ENV_PREFIX)/conda-meta" ] && [ -n "$(CONDA_BIN)" ]; then \
			echo "[kcare_robot] conda env detected -- ensuring native libs: $(CONDA_SYS_LIBS)"; \
			$(CONDA_BIN) install -y -p "$(ENV_PREFIX)" -c conda-forge $(CONDA_SYS_LIBS) >/dev/null \
				|| echo "[kcare_robot] WARNING: could not install $(CONDA_SYS_LIBS) -- TTS may be unavailable"; \
		else \
			echo "[kcare_robot] not a conda env -- skipping native libs ($(CONDA_SYS_LIBS))."; \
			echo "               For TTS install them system-wide:  sudo apt install portaudio19-dev ffmpeg"; \
		fi; \
	fi

# Destructive: refuses to touch an env you asked to reuse.
env-recreate:
	@if [ "$(USE_CURRENT)" = "1" ]; then \
		echo "[kcare_robot] env-recreate cannot rebuild the current python ($(ENV_PREFIX))."; \
		echo "               It only manages conda envs created by this Makefile."; \
		exit 1; \
	fi
	@if [ -z "$(CONDA_BIN)" ]; then echo "[kcare_robot] conda not found on PATH."; exit 1; fi
	@if [ "$(USE_EXISTING)" = "1" ]; then \
		echo "[kcare_robot] env-recreate would DELETE $(ENV_PREFIX), but USE_EXISTING=1 asked to reuse it."; \
		echo "               Drop USE_EXISTING to confirm you want it rebuilt."; \
		exit 1; \
	fi
	@echo "[kcare_robot] removing conda env at $(ENV_PREFIX) ..."
	-@$(CONDA_BIN) env remove -y -p "$(ENV_PREFIX)"
	@$(MAKE) --no-print-directory install

# Guard used by every target that needs the env to already exist.
_require-env:
	@if [ ! -x "$(ENV_PY)" ]; then \
		echo "[kcare_robot] no python at $(ENV_PY)"; \
		echo "               Run: make install                (conda env '$(CONDA_ENV)')"; \
		echo "               Or:  make <target> CONDA_ENV=<name> | USE_CURRENT=1"; \
		exit 1; \
	fi

# Guard for the run-time targets: any working python will do. Announces the
# fallback so it is never a surprise which interpreter actually ran.
_require-run-py:
	@if [ -z "$(RUN_PY)" ] || [ ! -x "$(RUN_PY)" ]; then \
		echo "[kcare_robot] no usable python found (conda env '$(CONDA_ENV)' absent and no python3 on PATH)"; \
		echo "               Run: make install"; \
		exit 1; \
	fi
	@if [ -n "$(RUN_FALLBACK)" ]; then \
		echo "[kcare_robot] conda env '$(CONDA_ENV)' not found -- using current python: $(RUN_PY) ($(RUN_PREFIX))"; \
		if ! (cd / && $(RUN_PY) -c 'import robot_agent, kcare_robot') >/dev/null 2>&1; then \
			echo ""; \
			echo "  ! kcare_robot is not installed in this python either."; \
			echo "    Build the default env:        make install"; \
			echo "    Or install into this python:  make install USE_CURRENT=1"; \
			echo "    Or activate the right env first, then re-run with USE_CURRENT=1."; \
			exit 1; \
		fi; \
		echo "               Run 'make install' to build the env, or pass USE_CURRENT=1 to silence this."; \
	fi

# Editable-install the sibling repos. Order matters: pyplanner and robot_agent
# must be importable before kcare_robot's own dependency resolution runs.
# check-deps has already proved each non-empty dir is a buildable project.
# visionserve has no local checkout by default (VISIONSERVE_DIR empty) — it is
# then installed straight from PyPI here, so `make install-deps` alone already
# yields a working import (the later `pip install -e .[all]` keeps it satisfied).
install-deps: check-deps _require-env
	@$(ENV_PIP) install -q --upgrade pip setuptools wheel
	@for d in "$(PYPLANNER_DIR)" "$(ROBOT_AGENT_DIR)" "$(VISIONSERVE_DIR)"; do \
		if [ -z "$$d" ]; then continue; fi; \
		echo "[kcare_robot] pip install -e $$d"; \
		$(ENV_PIP) install -e "$$d" || exit 1; \
	done
	@if [ -z "$(VISIONSERVE_DIR)" ]; then \
		if $(ENV_PY) -c 'import visionserve' >/dev/null 2>&1; then \
			v=$$($(ENV_PY) -c 'import importlib.metadata as md; print(md.version("visionserve"))' 2>/dev/null); \
			echo "[kcare_robot] visionserve $$v already installed -- skipping"; \
		else \
			echo "[kcare_robot] visionserve not found locally -- pip install visionserve (PyPI)"; \
			$(ENV_PIP) install visionserve || exit 1; \
		fi; \
	fi

install: check-deps env
	@$(MAKE) --no-print-directory install-deps
	@echo "[kcare_robot] pip install -e . [all]"
	@$(ENV_PIP) install -e "$(ROOT)[all]"
	@echo ""
	@if [ -d "$(PYPLANNER_DIR)" ]; then \
		echo "[kcare_robot] enabling the GRACE planner extra"; \
		$(ENV_PIP) install -e "$(ROBOT_AGENT_DIR)[grace]" >/dev/null 2>&1 || true; \
	fi
	@echo "[kcare_robot] Installed into $(ENV_PREFIX) ($$($(ENV_PY) -V))."
	@echo ""
	@echo "[kcare_robot] verifying the non-ROS imports actually resolve ..."
	@$(ENV_PY) -c "$$VERIFY_IMPORTS_PY"
	@echo ""
	@if [ ! -f "$(ROS_SETUP)" ]; then \
		echo "  ! ROS 2 not found at $(ROS_SETUP)."; \
		echo "    rclpy / cv_bridge / sensor_msgs come from the ROS distro, not pip."; \
		echo "    Install ROS $(ROS_DISTRO), or point at yours: make run ROS_DISTRO=<distro>"; \
		echo ""; \
	fi
	@echo "  Next:  make doctor      # verifies ROS, rosinterfaces and skill imports"
	@echo "         make run         # starts the agent on port $(PORT)"

# ── Run ──────────────────────────────────────────────────────────────────────

run: _require-run-py
	$(ROS_ENV) $(RUN_PY) -m uvicorn kcare_robot.main:app --host 0.0.0.0 --port $(PORT) --reload

# Run a single skill from the CLI, sourcing ROS first.
# Pass everything via ARGS so the user controls the arg list verbatim.
#   make cli ARGS="find::apple"
#   make cli ARGS="find::apple estimate_grasp=true camera=arm"
#   make cli ARGS="--list"
cli: _require-run-py
	@$(ROS_ENV) $(RUN_PY) -m kcare_robot $(ARGS)

run-external: _require-run-py
	$(ROS_ENV) $(RUN_PY) -m uvicorn kcare_robot.template_skills.grip_external:app --host 0.0.0.0 --port 9000

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

# No data_dir argument: diagnose auto-detects the split config layout
# (configs/locations/<active>). Pass one explicitly via ARGS to override.
doctor: _require-run-py
	@$(ROS_ENV) $(RUN_PY) -m robot_agent.diagnose kcare_robot $(ARGS)

test:
	@echo "[kcare_robot] Registered skills:"
	@curl -s http://localhost:$(PORT)/skills | python3 -m json.tool | head -40

# ── Skill scaffolding ────────────────────────────────────────────────────────

# Scaffold a new skill. SKILL is required; FILE defaults to <SKILL>.py.
#   make skill-generic SKILL=wave
#   make skill-generic SKILL=rotate FILE=arm.py
skill-generic: _require-run-py
	@if [ -z "$(SKILL)" ]; then \
		echo "Usage: make skill-generic SKILL=<skill_name> [FILE=<file>]"; \
		exit 2; \
	fi
	@$(RUN_PY) -m robot_agent.new_skill kcare_robot $(SKILL) $(FILE) --template generic

# Scaffold a detect-style skill (mocks _fetch_data + detector TCP client call).
#   make skill-detect SKILL=find_apple
#   make skill-detect SKILL=find_apple FILE=detectors.py
skill-detect: _require-run-py
	@if [ -z "$(SKILL)" ]; then \
		echo "Usage: make skill-detect SKILL=<skill_name> [FILE=<file>]"; \
		exit 2; \
	fi
	@$(RUN_PY) -m robot_agent.new_skill kcare_robot $(SKILL) $(FILE) --template detect

# Scaffold a standalone HTTP-service skill (own rclpy node + FastAPI app).
# Written to kcare_robot/template_skills/<skill>_external.py — NOT added to
# skills_config (external skills register with robot_agent via URL).
#   make skill-external SKILL=grip
skill-external: _require-run-py
	@if [ -z "$(SKILL)" ]; then \
		echo "Usage: make skill-external SKILL=<skill_name>"; \
		exit 2; \
	fi
	@$(RUN_PY) -m robot_agent.new_skill kcare_robot $(SKILL) --template external

# Remove a skill. Prompts for confirmation; pass YES=1 to skip the prompt.
#   make delete-skill SKILL=wave
#   make delete-skill SKILL=wave YES=1
# If the target file holds only this skill, the entire file is deleted.
# Otherwise only `def <skill>` is spliced out.
delete-skill: _require-run-py
	@if [ -z "$(SKILL)" ]; then \
		echo "Usage: make delete-skill SKILL=<skill_name> [YES=1]"; \
		exit 2; \
	fi
	@if [ "$(YES)" = "1" ]; then \
		$(RUN_PY) -m robot_agent.delete_skill kcare_robot $(SKILL) --yes; \
	else \
		$(RUN_PY) -m robot_agent.delete_skill kcare_robot $(SKILL); \
	fi

# Rename a skill. Renames the registry key (invocation name); also renames the
# `def` + module file when the skill owns a dedicated single-skill file.
#   make rename-skill SKILL=wave NEW=greet
#   make rename-skill SKILL=wave NEW=greet YES=1
rename-skill: _require-run-py
	@if [ -z "$(SKILL)" ] || [ -z "$(NEW)" ]; then \
		echo "Usage: make rename-skill SKILL=<old> NEW=<new> [YES=1]"; \
		exit 2; \
	fi
	@if [ "$(YES)" = "1" ]; then \
		$(RUN_PY) -m robot_agent.rename_skill kcare_robot $(SKILL) $(NEW) --yes; \
	else \
		$(RUN_PY) -m robot_agent.rename_skill kcare_robot $(SKILL) $(NEW); \
	fi

clean:
	find $(ROOT) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find $(ROOT) -type d -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	@echo "[kcare_robot] cleaned."
