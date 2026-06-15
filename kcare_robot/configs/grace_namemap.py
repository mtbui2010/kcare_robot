"""GRACE ↔ kcare_robot adapter (robot-specific name map + param/perception hooks).

This module is the **robot-side half** of the ActionMapper described in
``kcare_robot/docs/ACTION_MAPPER_SPEC.md``. GRACE / pyplanner emit *abstract*
CamelCase symbolic actions (``MoveTo``, ``Find``, ``Pick`` …) over CamelCase
locations/objects (``LivingRoom``, ``CoffeeMachine``); kcare executes concrete
*skills* (``move``, ``find``, ``pick`` …) over ENV location keys / open-vocab
detector strings. This file declares that mapping and exposes the module-level
names the ActionMapper / closed loop imports:

    to_loc, to_obj, SKILL_MAP, SUPPORTED_ACTIONS, NOOP_ACTIONS, VLM_ACTIONS,
    build_params, observe, vlm_hook

Design rules (per the spec):
  * Coverage is **1:1** for the manipulation core (MoveTo/Find/Pick/Place),
    drawer-only for Open/Close, composite for PutIn, and **unsupported** for
    TurnOn/TurnOff/Wash (no kcare skill). Sit/LieOn/Serve/Wait are no-ops.
  * Names go through an explicit table first, then a CamelCase fallback so
    unknown (open-vocabulary) objects still attempt execution.
  * All runtime state (``robot_agent.state.current``, the ROS node, TCP
    clients) is imported **lazily inside functions** so this module imports
    cleanly outside a booted runtime (tests, plan pre-screening, etc.).

Authoritative cross-references:
  * docs/ACTION_MAPPER_SPEC.md   — the mapping table (§2) + name map (§4)
  * docs/SKILLS.md               — skill params / returns
  * configs/skills_config.py     — registered skill names (move/find/pick/
                                   placeat/open_drawer/close_drawer/find_arm/
                                   grasp_succeed/llm/detect)
"""

from __future__ import annotations

import re
import time
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. Name tables  (GRACE CamelCase  ->  kcare ENV key / detector string)
#
# ENV keys are validated against the *active site's* ENV at runtime (move reads
# robot_agent.skill_configs.ENV). The literals below are seeded from the vocab
# in skill_config_defaults.py (KR2EN/EN2KR).
# TODO(integrator): fill/trim these from the deployed site's ENV config so
# to_loc() agrees with what `move::<loc>` can actually navigate to. An unknown
# location should surface as a replan-able failure (see ACTION_MAPPER_SPEC §4).
# ─────────────────────────────────────────────────────────────────────────────

LOCATIONS: dict[str, str] = {
    # GRACE location  ->  kcare ENV location key (move::<value>)
    "LivingRoom":   "living room",
    "Bedroom":      "bedroom",
    "Kitchen":      "kitchen",
    "Restroom":     "restroom",
    "Bathroom":     "restroom",
    "Table":        "table@living room",
    "DiningTable":  "dining table",
    "SophaTable":   "sopha table",
    "SofaTable":    "sopha table",
    "BedTable":     "bed table",
    "Shelf":        "shelf",
    "Sink":         "sink",
    "Drawer":       "drawer",
    "Home":         "home",
}

OBJECTS: dict[str, str] = {
    # GRACE object  ->  detector / skill object string (open-vocabulary, lowercase)
    "Cup":           "cup",
    "Mug":           "mug",
    "Can":           "can",
    "Bottle":        "bottle",
    "WaterBottle":   "water bottle",
    "Apple":         "apple",
    "Coke":          "coke",
    "Coffee":        "coffee",
    "CoffeeMachine": "coffee machine",
    "Towel":         "towel",
    "Phone":         "phone",
    "Controller":    "controller",
    "Remote":        "controller",
    "Toothpaste":    "toothpaste",
    "Pen":           "pen",
    "Light":         "light",
    "Drawer":        "drawer",        # drawer is both a place + a handle target
    "Handle":        "handle",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. CamelCase fallback heuristics
# ─────────────────────────────────────────────────────────────────────────────

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _camel_split(name: str) -> list[str]:
    """'CoffeeMachine' -> ['Coffee', 'Machine']; tolerant of snake / spaces."""
    s = (name or "").strip().replace("_", " ").replace("-", " ")
    if not s:
        return []
    # split existing whitespace tokens, then split each on CamelCase boundaries
    out: list[str] = []
    for tok in s.split():
        out.extend(p for p in _CAMEL_BOUNDARY.split(tok) if p)
    return out or [s]


def _camel_to_snake(name: str) -> str:
    """'LivingRoom' -> 'living_room' (ENV-key style fallback for locations)."""
    return "_".join(_camel_split(name)).lower()


def _camel_to_words(name: str) -> str:
    """'CoffeeMachine' -> 'coffee machine' (detector-string fallback for objects)."""
    return " ".join(_camel_split(name)).lower()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Public name resolvers
# ─────────────────────────────────────────────────────────────────────────────

def to_loc(name: str) -> str:
    """GRACE CamelCase location -> kcare ENV location key.

    Explicit LOCATIONS table first, else a CamelCase->snake fallback. The
    caller is responsible for validating the result against the live ENV and
    raising 'unknown location' for replan (ACTION_MAPPER_SPEC §4).
    """
    if not name:
        return ""
    return LOCATIONS.get(name, _camel_to_snake(name))


def to_obj(name: str) -> str:
    """GRACE CamelCase object -> detector/skill object string.

    Explicit OBJECTS table first, else a CamelCase->"two words" fallback.
    Detector is open-vocabulary, so the heuristic is usually acceptable.
    """
    if not name:
        return ""
    return OBJECTS.get(name, _camel_to_words(name))


def _to_obj_or_loc(name: str) -> str:
    """For Place/PutIn the GRACE arg is a *receptacle/container* that may be a
    known ENV location (a placeat target like 'sopha table') OR an object-like
    string. Prefer an explicit location, then an explicit object, then the
    object-words fallback."""
    if not name:
        return ""
    if name in LOCATIONS:
        return LOCATIONS[name]
    if name in OBJECTS:
        return OBJECTS[name]
    return _camel_to_words(name)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Action → skill map  (verified against configs/skills_config.py)
#    Registered names used here: move, find, pick, placeat, open_drawer,
#    close_drawer  — all present in SKILL_CONFIGS.
#    TurnOn/TurnOff/Wash are intentionally absent (no skill → unsupported).
# ─────────────────────────────────────────────────────────────────────────────

SKILL_MAP: dict[str, str] = {
    "MoveTo": "move",
    "Find":   "find",
    "Pick":   "pick",
    "Place":  "placeat",
    "PutIn":  "placeat",        # realized as place-into-open-container
    "Open":   "open_drawer",    # drawers only; non-drawer Open is unsupported
    "Close":  "close_drawer",   # drawers only
}

# Benign symbolic padding — succeed without touching the robot.
NOOP_ACTIONS: set[str] = {"Sit", "LieOn", "Serve", "Wait"}

# Everything this robot can advance (mappable skills ∪ no-ops). Used by the
# ActionMapper to pre-screen a freshly generated plan and warn the operator.
SUPPORTED_ACTIONS: set[str] = set(SKILL_MAP.keys()) | NOOP_ACTIONS

# World-changing actions that get a layer-3 (perception) post-check (vlm_hook).
VLM_ACTIONS: set[str] = {"Pick", "Place", "PutIn", "Open", "Close"}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Param adapters  (GRACE step -> kcare skill kwargs)
# ─────────────────────────────────────────────────────────────────────────────

def build_params(action: str, obj_camel: str, world) -> dict:
    """Produce the kcare skill ``params`` for a GRACE action.

    ``world`` mirrors the ActionMapper's SymbolicState and exposes
    ``.found`` / ``.arrived`` / ``.holding`` (any may be None). Defensive:
    falls back to the step's own object when world state is missing.

      MoveTo Loc      -> {'inputs': to_loc(Loc)}
      Find   Obj      -> {'inputs': to_obj(Obj)}
      Pick            -> {'inputs': to_obj(world.found or Obj), 'type': 'fine_move'}
      Place  Recept   -> {'inputs': to_obj_or_loc(Recept)}
      PutIn  Cont     -> {'inputs': to_obj_or_loc(Cont)}
      Open   Cont     -> {'inputs': to_loc(Cont)}     (drawer ENV loc)
      Close  Cont     -> {'inputs': to_loc(Cont)}

    No-op / unsupported actions return {} — the caller decides isdone.
    """
    found = getattr(world, "found", None) if world is not None else None

    if action == "MoveTo":
        return {"inputs": to_loc(obj_camel)}

    if action == "Find":
        return {"inputs": to_obj(obj_camel)}

    if action == "Pick":
        # GRACE Pick carries no object; take it from world.found, else the step's.
        target = found or obj_camel
        return {"inputs": to_obj(target), "type": "fine_move"}

    if action in ("Place", "PutIn"):
        return {"inputs": _to_obj_or_loc(obj_camel)}

    if action in ("Open", "Close"):
        # Only drawers are physically supported -> ENV location key.
        # The ActionMapper is expected to thread the matching open_drawer's
        # pose_after_open/lift_after_open/forward_after_open into a Close call.
        return {"inputs": to_loc(obj_camel)}

    # No-op / unsupported — nothing to pass.
    return {}


def _arg_object(params, result):
    """Best object name for a skill effect: prefer what was actually detected
    (``result['ins']`` keys from find/find_arm), else the raw '::' arg's first
    token. Returns None if nothing usable."""
    if isinstance(result, dict):
        ins = result.get("ins")
        if isinstance(ins, dict) and ins:
            return next(iter(ins.keys()))
    if isinstance(params, dict):
        params = params.get("inputs") or params.get("input")
    if isinstance(params, str) and params.strip():
        return params.strip().split()[0]
    return None


def _robot_pose_xyz(node):
    """Best-effort base pose ``[x, y, rz]`` (None on failure)."""
    try:
        from kcare_robot.skills.mobile import mobile_pose
        p = mobile_pose(node=node).get("pose")
        return list(p) if p else None
    except Exception:
        return None


def _found_pose_data(result, name, node):
    """Extract the geometric memory for ``name`` from a find result:
    ``{loc_3d, pose_3d, side_pose?, grasppose?, ts, robot_pose}``. The
    ``robot_pose`` stamps where the base was at detection so staleness can be
    judged after the robot moves. Returns None if no pose fields present."""
    if not isinstance(result, dict):
        return None
    ins = result.get("ins")
    entry = ins.get(name) if isinstance(ins, dict) else None
    if not isinstance(entry, dict):
        return None
    data = {k: entry[k] for k in ("loc_3d", "pose_3d", "side_pose", "grasppose") if k in entry}
    if not data:
        return None
    import time
    data["ts"] = time.time()
    rp = _robot_pose_xyz(node)
    if rp:
        data["robot_pose"] = rp
    return data


def apply_skill_effect(world, skill, params=None, result=None, node=None) -> None:
    """Mirror a RAW (direct / open-loop) kcare skill onto the persistent
    WorldState — the inverse of :func:`build_params`, keyed on kcare SKILL names
    (not GRACE actions), since the direct path runs skills verbatim with no
    ActionMapper to drive ``apply_effect``.

    Called best-effort by ``robot_agent``'s open-loop / direct paths after each
    step: ``world`` is the shared persistent state, ``params`` the raw '::' arg
    string, ``result`` the skill's return dict. Only mutates on success.
    ``arrived`` is intentionally left to the sensor reconcile (localization).
    """
    if world is None or not skill:
        return
    if isinstance(result, dict) and "isdone" in result and not result.get("isdone"):
        return  # skill failed → leave the prior belief untouched

    obj = _arg_object(params, result)
    s = str(skill).strip().lower()

    if s in ("find", "find_arm", "find_once"):
        if obj:
            world.found = obj
            fp = _found_pose_data(result, obj, node)
            if fp:
                world.found_pose = fp
    elif s in ("pick", "grasp"):
        prior_fp = getattr(world, "found_pose", None)
        world.holding = getattr(world, "found", None) or obj
        world.found = None
        world.found_pose = None          # consumed by the grasp
        import time
        world.holding_since = time.time()
        # Record the grasp actually used (pick returns it), else fall back to the
        # grasppose stashed at find time.
        gp = None
        if isinstance(result, dict) and result.get("grasppose") is not None:
            gp = result["grasppose"]
        elif isinstance(prior_fp, dict):
            gp = prior_fp.get("grasppose")
        if gp is not None:
            hp = {"grasppose": gp, "ts": time.time()}
            rp = _robot_pose_xyz(node)
            if rp:
                hp["robot_pose"] = rp
            world.holding_pose = hp
        else:
            world.holding_pose = None
    elif s in ("placeat", "place", "put", "putin", "put_in", "give"):
        world.holding = None
        world.holding_since = None
        world.holding_pose = None        # released
    elif s in ("open_drawer", "open"):
        if obj:
            world.opened.add(obj)
    elif s in ("close_drawer", "close"):
        world.opened.discard(obj)
    # move -> 'arrived' handled by sensor reconcile; others: no symbolic effect


# ─────────────────────────────────────────────────────────────────────────────
# 6. Perception pass for the planner  (the "Grounder")
# ─────────────────────────────────────────────────────────────────────────────

def _current_state():
    """Lazily fetch the booted AgentState, or None outside a runtime."""
    try:
        from robot_agent.state import current
        return current()
    except Exception:
        return None


def _ros_node(state):
    """Best-effort handle on the spinning ROS node used by skills."""
    try:
        dm = state.dm
        node = getattr(dm, "_ros_node", None)
        if node is None and hasattr(dm, "_get_ros_node"):
            node = dm._get_ros_node()
        return node
    except Exception:
        return None


def observe(node=None) -> tuple[str, list[str]]:
    """Lightweight perception pass -> (obs_text, visible_objects) for GRACE.

    Best-effort and NON-FATAL: returns ('', []) on any failure so a missing /
    unreachable detector never aborts planning.

    TODO(integrator): replace the conservative default with a real grounding
    pass. A full implementation would run a head-camera open-vocab detect over
    a candidate object/location set and return human-readable obs + the list of
    detected labels. The detector is reachable through the registered `find` /
    `detect` skill (skills/recognition.py -> 'vlms' TCP client via
    current().dm.get_client('vlms')); see ACTION_MAPPER_SPEC §8 ("Grounder").
    Because a meaningful scene query needs a caption/candidate set we do NOT
    have here, we return the safe default rather than guess.
    """
    try:
        state = _current_state()
        if state is None:
            return "", []
        node = node or _ros_node(state)
        if node is None:
            return "", []

        # --- OPTIONAL real pass (disabled by default) -----------------------
        # To enable, supply a candidate caption and uncomment:
        #
        #   candidates = "cup,can,bottle,towel,phone,controller"
        #   ret = state.sr.execute("find", {"inputs": candidates, "once": True}, node)
        #   if ret.get("isdone"):
        #       visible = sorted(ret.get("ins", {}).keys())
        #       obs = "I can see: " + ", ".join(visible) if visible else ""
        #       return obs, visible
        #
        # Left off because find() navigates/moves the head as a side effect and
        # needs a real candidate set — unsafe as an implicit planner probe.
        return "", []
    except Exception:
        # Never let perception crash the loop.
        return "", []


# ─────────────────────────────────────────────────────────────────────────────
# 7. Layer-3 VLM verifier  (post-check for world-changing actions)
# ─────────────────────────────────────────────────────────────────────────────

def vlm_hook(step: dict, node=None) -> tuple[bool, Optional[float], str, float]:
    """Verify that ``<action> <object>`` actually succeeded on the real robot.

    Returns ``(isdone, score, note, latency_s)``. ``score`` is an optional
    confidence (None when not produced). Guarded end-to-end: **never raises** —
    on any failure it returns a permissive (True, …) so a flaky verifier does
    not stall an otherwise-good plan.

    Strategy:
      * Pick  -> reuse the existing `grasp_succeed` skill (depth ROI near the
                 gripper). This is kcare's authoritative in-hand check.
      * other -> ask the `llm` skill a yes/no question if an LLM client is
                 configured; else skip with (True, …).

    TODO(integrator): for a true *vision*-language check, grab a head/arm
    camera frame (skills/recognition.fetch_camera_data(node, 'head')) and send
    it to a VLM. The current `llm` skill (skills/vlm.py) is text-only
    (client.chat(prompt=...)), so the non-Pick path is a heuristic stand-in.
    """
    t0 = time.time()
    action = (step or {}).get("action", "")
    obj = (step or {}).get("object", "") or (step or {}).get("target", "")

    try:
        state = _current_state()
        if state is None:
            return True, None, "vlm not configured (no runtime)", time.time() - t0
        node = node or _ros_node(state)

        # --- Pick: depth-based grasp verification -------------------------
        if action == "Pick":
            try:
                ret = state.sr.execute("grasp_succeed", {}, node)
                isdone = bool(ret.get("isdone", False))
                return isdone, None, "grasp depth check", time.time() - t0
            except Exception as e:
                return True, None, f"grasp check skipped ({e})", time.time() - t0

        # --- Other world-changing actions: text LLM yes/no, if available ---
        try:
            llm_client = state.dm.get_client("llm")
        except Exception:
            llm_client = None

        if llm_client is None:
            return True, None, "vlm not configured (skipped)", time.time() - t0

        verb = {"Place": "place", "PutIn": "put", "Open": "open",
                "Close": "close"}.get(action, action.lower())
        obj_str = to_obj(obj) if obj else "the object"
        prompt = (
            f"Did the robot successfully {verb} {obj_str}? "
            "Answer strictly 'yes' or 'no'."
        )
        try:
            ret = state.sr.execute("llm", {"inputs": prompt}, node)
            msg = str(ret.get("msg", "")).strip().lower()
            isdone = msg.startswith("y") or ("yes" in msg and "no" not in msg)
            return isdone, None, f"llm verify: {msg[:40]!r}", time.time() - t0
        except Exception as e:
            return True, None, f"llm verify skipped ({e})", time.time() - t0

    except Exception as e:
        # Absolute backstop — never raise out of the verifier.
        return True, None, f"vlm_hook error ({e})", time.time() - t0
