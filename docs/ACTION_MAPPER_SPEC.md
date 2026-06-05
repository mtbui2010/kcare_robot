# ActionMapper — GRACE action ↔ kcare_robot skill mapping (as-built spec)

This documents the glue layer that turns a **GRACE/pyplanner plan** (abstract symbolic
actions) into **kcare_robot skill calls** (`SkillRegistry.execute`), and tracks the
symbolic world state in lock-step so a real skill failure can drive GRACE's `replan()`.

The implementation is split across **two packages**, on purpose:

| Half | Package / module | Role |
|---|---|---|
| **Engine** (robot-agnostic) | `robot_agent/robot_agent/core/planning/mapper.py` + `…/base.py` | `ActionMapper`, `WorldState`, `Planner` Protocol. Hard-codes **no** robot knowledge. |
| **NameMap** (robot-specific) | `kcare_robot/kcare_robot/configs/grace_namemap.py` | The GRACE↔kcare tables, skill map, action sets, param/perception/VLM hooks. |

`ActionMapper(namemap)` takes the robot config module by **duck typing** — it reads
attributes (`SKILL_MAP`, `SUPPORTED_ACTIONS`, …) and calls `to_loc` / `to_obj` off
whatever module is handed in. Swapping robots = swapping the namemap module; the
engine is untouched.

> **Hard constraint (still in force): do NOT modify `pyplanner/` or `paper_grace/`.**
> GRACE stays a black box consumed through its public API (`generate_plan`, `replan`).
> Neither `mapper.py` nor `base.py` imports `pyplanner`; the planner contract is the
> structural `Planner` Protocol in `base.py`, satisfied by GRACE's `BasePlanner`.

Companion: skill semantics in [`SKILLS.md`](SKILLS.md); driver/architecture in
[`CLOSED_LOOP_ARCHITECTURE.md`](CLOSED_LOOP_ARCHITECTURE.md).

---

## 1. GRACE side (recap — read-only, authoritative)

GRACE plan = `list[dict]`. `base.py` aliases this as `PlanStep = dict` and documents
the shape (each step is a **plain dict, not a class**):

```python
{
    "action": str,   # one GRACE action, CamelCase
    "object": str,   # target, CamelCase (Apple, CoffeeMachine, DiningTable)
    "target": str,   # optional, rarely used
    "reason": str,   # optional, ignored by the verifier
}
```

**GRACE action vocabulary** — 14 actions (`pyplanner/base.py` `ROBOT_ACTIONS`):

```
MoveTo, Find, Pick, Place, PutIn, Open, Close,
TurnOn, TurnOff, Wash, Sit, LieOn, Serve, Wait
```

**Preconditions & symbolic effects** (`verifier.py`, for reference):

| GRACE action | arg | precondition (`verify_step`) | state effect |
|---|---|---|---|
| `MoveTo` | location | — | `arrived=loc`; reset `found` unless `found==loc` |
| `Find` | object | — (soft: warn if not visible) | `found=obj` |
| `Pick` | (uses `found`) | `found≠∅` **and** `holding=∅` | `holding=found`, `found=∅` |
| `Place` | receptacle | `holding≠∅` **and** `arrived==receptacle` | `holding=∅` |
| `PutIn` | container | `holding≠∅` **and** (container⇒`container∈opened`) | `holding=∅` |
| `Open` | container | `found==obj` **or** `arrived==obj` | `opened += obj` |
| `Close` | container | found/arrived==obj **and** `obj∈opened` | `opened -= obj` |
| `TurnOn` | appliance | found/arrived==obj | `on += obj` |
| `TurnOff` | appliance | found/arrived==obj **and** `obj∈on` | `on -= obj` |
| `Wash` | item | found/arrived==obj | — |
| `Sit` | furniture | — | — |
| `LieOn` | furniture | — | — |
| `Serve` | — | — | — |
| `Wait` | — | — | — |

`SymbolicState` (GRACE side): `arrived, found, holding: str|None`; `opened, on, visible: set[str]`.
Container rule: `MoveTo C → Open C → Find obj → Pick → Close C`.

**Planner contract** — engine-side, in `base.py`, as a structural `typing.Protocol`
(`@runtime_checkable`). Any backend matching this shape is accepted; GRACE is
instantiated robot-side and never edited:

```python
@runtime_checkable
class Planner(Protocol):
    def generate_plan(
        self, task: str, obs: str, visible_objects: list[str],
    ) -> tuple[list[dict], dict]: ...

    def replan(
        self, task: str, completed: list[dict], failed_step: dict,
        failure_reason: str, obs: str, visible_objects: list[str],
    ) -> tuple[list[dict], dict]: ...
```

Both methods return `(steps, metrics)` where `steps` is a list of PlanStep dicts and
`metrics` is an arbitrary dict (may be `{}`). A typical wiring:

```python
import pyplanner
planner: Planner = pyplanner.get("GRACE", host=..., model=...)
steps, m  = planner.generate_plan(task, obs, visible_objects)
suffix, m = planner.replan(task, completed, failed_step, failure_reason, obs, visible_objects)
```

---

## 2. The mapping table (GRACE → kcare skill)

The table below is the **robot-specific** `SKILL_MAP` + action sets declared in
`kcare_robot/configs/grace_namemap.py`. The engine reads it through the namemap; it
contains no kcare names itself.

`<obj>` / `<loc>` = robot-side name produced by the **NameMap** (§4). `world` =
the `WorldState` the mapper maintains in lock-step with the executed plan.

| GRACE action | `SKILL_MAP` skill | Param adapter (`build_params` / `to_skill`) | Notes |
|---|---|---|---|
| **MoveTo** `Loc` | `move` | `{'inputs': to_loc(Loc)}` | `<loc>` must exist in robot ENV; an unknown one should surface as a replan-able failure. |
| **Find** `Obj` | `find` | `{'inputs': to_obj(Obj)}` | Keep GRACE's explicit MoveTo+Find split; don't collapse into `find::obj@loc`. |
| **Pick** | `pick` (`obj = world.found`) | `{'inputs': to_obj(world.found or Obj), 'type': 'fine_move'}` | GRACE `Pick` carries no object → taken from `world.found`. `pick` self-verifies via `grasp_succeed`. |
| **Place** `Recept` | `placeat` | `{'inputs': to_obj_or_loc(Recept)}` | Robot already `arrived==recept` (GRACE precond). |
| **PutIn** `Container` | `placeat` | `{'inputs': to_obj_or_loc(Cont)}` | Realized as place-into-open-container; requires a prior `Open`. See §5. |
| **Open** `Container` | `open_drawer` | `{'inputs': to_loc(Cont)}` | **Drawers only.** Non-drawer Open → physically unsupported (no generic fridge/cabinet open). |
| **Close** `Container` | `close_drawer` | `{'inputs': to_loc(Cont)}` | **Drawers only.** Pair with the Open of the same container; thread its `pose_after_open/lift_after_open/forward_after_open`. See §5. |
| **TurnOn** `Appl` | **UNSUPPORTED** (no `SKILL_MAP` entry) | — | No appliance-toggle skill → `Unmappable`. §6. |
| **TurnOff** `Appl` | **UNSUPPORTED** | — | §6. |
| **Wash** `Item` | **UNSUPPORTED** | — | §6. |
| **Sit / LieOn / Serve** | **NO-OP** (in `NOOP_ACTIONS`) | — | `to_skill` returns `None`; driver marks done without touching the robot. |
| **Wait** | **NO-OP** | — | `to_skill` returns `None`. |

The robot config encodes this as:

```python
SKILL_MAP = {
    "MoveTo": "move",
    "Find":   "find",
    "Pick":   "pick",
    "Place":  "placeat",
    "PutIn":  "placeat",        # realized as place-into-open-container
    "Open":   "open_drawer",    # drawers only; non-drawer Open is unsupported
    "Close":  "close_drawer",   # drawers only
}
NOOP_ACTIONS:      set[str] = {"Sit", "LieOn", "Serve", "Wait"}
SUPPORTED_ACTIONS: set[str] = set(SKILL_MAP.keys()) | NOOP_ACTIONS
VLM_ACTIONS:       set[str] = {"Pick", "Place", "PutIn", "Open", "Close"}
```

Note `SUPPORTED_ACTIONS` is **derived** (`SKILL_MAP.keys() ∪ NOOP_ACTIONS`), so the
mappable set and the no-op set are the single source of truth — `TurnOn/TurnOff/Wash`
are absent and therefore unsupported by construction.

> Skills like `grip`, `approach_close`, `fine_move`, `movel/j/t`, `lift`, `moveh`,
> `find_arm`, `grasp_succeed` are **sub-skills** invoked *inside* the high-level skills
> above — they are **not** direct GRACE targets. The mapper targets only the coarse
> registered names in `SKILL_MAP` (`move`, `find`, `pick`, `placeat`, `open_drawer`,
> `close_drawer`); kcare's composites drive the primitives.

**Coverage summary:** clean 1:1 for the manipulation core (`MoveTo, Find, Pick, Place`);
partial for `Open/Close` (drawers only); `PutIn` composite; **unsupported** for
`TurnOn/TurnOff/Wash`; **no-op** for `Sit/LieOn/Serve/Wait`. This coverage gap is a key
integration finding.

---

## 3. State tracking (engine-owned `WorldState`, mirrors GRACE)

`WorldState` lives in `robot_agent/core/planning/base.py` (robot-agnostic, no
`pyplanner` import). It is a `@dataclass`:

```python
@dataclass
class WorldState:
    arrived: Optional[str] = None          # location robot is at
    found:   Optional[str] = None          # object located, not yet picked
    holding: Optional[str] = None          # object currently grasped
    opened:  set[str] = field(default_factory=set)   # open containers
    on:      set[str] = field(default_factory=set)   # appliances switched on

    def copy(self) -> "WorldState":  ...   # duplicates the two sets
    def as_text(self) -> str:        ...   # "arrived=- found=cup holding=- opened=[] on=[]"
```

Note this is **five** fields (`arrived/found/holding/opened/on`) — GRACE's `visible`
set is **not** mirrored here.

The driver mutates the state **only from skills that actually succeeded** (`isdone==True`),
via `ActionMapper.apply_effect(step, world)`. That method is the as-built mirror of
`verifier._apply`:

```
MoveTo loc  → world.arrived = obj; if world.found != obj: world.found = None
Find  obj   → world.found  = obj
Pick        → world.holding = world.found; world.found = None
Place rcpt  → world.holding = None
PutIn cont  → world.holding = None
Open  cont  → world.opened.add(obj)        (only if obj is non-empty)
Close cont  → world.opened.discard(obj)
TurnOn  ap  → world.on.add(obj)            (only if obj is non-empty; effect tracked even though action is unsupported)
TurnOff ap  → world.on.discard(obj)
Sit/LieOn/Serve/Wait/Wash → no symbolic effect
```

`copy()` is available for snapshotting state before a tentative step. To recompute
state from a `completed` prefix you may still replay it through `apply_effect`, or
reuse `pyplanner.verifier.simulate(...)` (read-only call — no pyplanner change needed).

---

## 4. NameMap — GRACE CamelCase ↔ robot names (robot-specific)

GRACE emits CamelCase (`CoffeeMachine`, `DiningTable`, `Kitchen`); kcare uses ENV
location keys / open-vocab detector strings. The two-way map lives **in the robot
package** (`kcare_robot/configs/grace_namemap.py`), declared as explicit tables plus a
CamelCase fallback:

```python
LOCATIONS = {     # GRACE location -> kcare ENV key (move::<value>)
    "LivingRoom": "living room", "Bedroom": "bedroom", "Kitchen": "kitchen",
    "Restroom": "restroom", "Bathroom": "restroom",
    "Table": "table@living room", "DiningTable": "dining table",
    "SophaTable": "sopha table", "SofaTable": "sopha table",
    "BedTable": "bed table", "Shelf": "shelf", "Sink": "sink",
    "Drawer": "drawer", "Home": "home",
}
OBJECTS = {       # GRACE object -> detector/skill string (open-vocab, lowercase)
    "Cup": "cup", "Mug": "mug", "Can": "can", "Bottle": "bottle",
    "WaterBottle": "water bottle", "Apple": "apple", "Coke": "coke",
    "Coffee": "coffee", "CoffeeMachine": "coffee machine", "Towel": "towel",
    "Phone": "phone", "Controller": "controller", "Remote": "controller",
    "Toothpaste": "toothpaste", "Pen": "pen", "Light": "light",
    "Drawer": "drawer", "Handle": "handle",
}
```

Resolvers (all robot-side):

```python
def to_loc(name):       # LOCATIONS table, else _camel_to_snake('LivingRoom' -> 'living_room')
def to_obj(name):       # OBJECTS  table, else _camel_to_words('CoffeeMachine' -> 'coffee machine')
def _to_obj_or_loc(n):  # Place/PutIn receptacle: LOCATIONS, then OBJECTS, then words-fallback
```

Rules / as-built behaviour:
- Prefer the explicit table; fall back to a `CamelCase → snake_case` (locations) or
  `CamelCase → "two words"` (objects) heuristic, so unknown/open-vocab names still
  attempt execution. Empty names resolve to `""`.
- `to_loc` does **not** validate against the live ENV — the caller (driver) is
  responsible for validating the result and emitting `failure_reason="unknown location <X>"`
  for replan if `move::<loc>` can't navigate there.
- The `LOCATIONS`/`OBJECTS` literals are seeded from `skill_config_defaults.py`
  (KR2EN/EN2KR vocab) and carry an integrator TODO to trim them to the deployed
  site's ENV so `to_loc()` agrees with what `move` can reach.

> The engine's `to_skill` calls `namemap.to_loc` for `_LOCATION_ACTIONS =
> {"MoveTo", "Place", "PutIn"}` and `namemap.to_obj` otherwise (`_OBJECT_ACTIONS =
> {"Find","Pick","Open","Close","TurnOn","TurnOff","Wash"}`). The richer
> `_to_obj_or_loc` / `'type': 'fine_move'` / drawer-`to_loc` logic shown in the table
> lives in the namemap's own `build_params(action, obj_camel, world)` adapter, which a
> driver can use instead of (or alongside) the engine's simpler `{'inputs': value}`
> param build.

---

## 5. Container realization (PutIn / Open+Find+Pick+Close)

GRACE's container rule decomposes into discrete steps the mapper already handles
individually — **as long as the container is a drawer**:

```
MoveTo Drawer   → move::drawer_loc
Open   Drawer   → open_drawer::drawer_loc          (capture pose_after_open, ...)
Find   Obj      → find / find_arm  (inside drawer; wrist cam → prefer find_arm)
Pick            → pick::obj  ({'type': 'fine_move'})
Close  Drawer   → close_drawer::drawer_loc  (pass the captured open-state)
PutIn  Drawer   → placeat::drawer_loc   (while opened)
```

The `open_drawer` result fields (`pose_after_open`, `lift_after_open`,
`forward_after_open`) **must be threaded** into the paired `close_drawer` call — the
driver should stash them keyed by container name. `build_params` documents this
expectation in its `Open`/`Close` branch but does not itself carry the state across
steps; threading is the driver's job.

---

## 6. Unsupported / no-op policy (as-built)

This is enforced by the **engine** in `ActionMapper`, using the namemap's action sets.

**No-op** (`action in NOOP_ACTIONS`): `to_skill` returns `None`. The driver marks the
step done without touching the robot — benign plan padding (`Sit/LieOn/Serve/Wait`)
never aborts a run.

**Unsupported** — `to_skill` raises `Unmappable` (a custom `Exception` carrying the
offending action/object in its message) when **any** of:
- the step has no `action`;
- `action not in SUPPORTED_ACTIONS` → `"no skill for action <A> on object <O> (not in SUPPORTED_ACTIONS)"`;
- `action` is supported but has no `SKILL_MAP` entry → `"… (no SKILL_MAP entry)"`;
- the action needs an object but none is resolvable (`Pick` with empty `world.found`
  → `"Pick requested but nothing has been found"`; otherwise
  `"action <A> requires an object"`).

The driver converts the `Unmappable` message into a `failure_reason` for
`planner.replan(...)`, or escalates to the operator.

**Pre-screen.** `ActionMapper.screen(steps) -> list[str]` walks a freshly generated
plan and returns one human-readable warning per step whose action is **neither**
supported **nor** a no-op:

```
"step {i+1}: unsupported action {action!r} on object {object!r} -- this robot cannot execute it"
```

so the operator is warned *before* execution. Coverage stays auditable because
`SUPPORTED_ACTIONS` is derived from `SKILL_MAP` + `NOOP_ACTIONS` in one place.

> **Extension path:** adding appliance control later = add a `turn_on`/`turn_off`
> skill in kcare + one row in `SKILL_MAP`; no GRACE change. `SUPPORTED_ACTIONS`
> (and `screen`) update automatically.

---

## 7. Engine API (as-built)

`robot_agent/robot_agent/core/planning/mapper.py`:

```python
class Unmappable(Exception):
    """No executable skill for this GRACE step on this robot."""

# location-like vs object-like arg routing (param building):
_LOCATION_ACTIONS = {"MoveTo", "Place", "PutIn"}                 # -> namemap.to_loc
_OBJECT_ACTIONS   = {"Find", "Pick", "Open", "Close",
                     "TurnOn", "TurnOff", "Wash"}                # -> namemap.to_obj

class ActionMapper:
    def __init__(self, namemap) -> None: ...                     # duck-typed robot config module

    def to_skill(self, step: dict, world: WorldState
                 ) -> Optional[tuple[str, dict]]:
        """GRACE step -> (skill_name, params); None for a NOOP_ACTIONS step.
           Raises Unmappable for unsupported / no-SKILL_MAP / no-object steps.
           params shape: {'inputs': <robot-name>}.  Pick fills obj from world.found."""

    def apply_effect(self, step: dict, world: WorldState) -> None:
        """Mutate world to mirror the GRACE symbolic effect (call after success)."""

    def screen(self, steps: list[dict]) -> list[str]:
        """Pre-flight human warnings for steps this robot can't execute."""
```

The engine reads these attributes off `namemap` (all optional, defaulted defensively):
`SKILL_MAP`, `SUPPORTED_ACTIONS`, `NOOP_ACTIONS`, and the callables `to_loc` / `to_obj`.

Closed-loop driver (pseudocode — the real one is `core/planning/loop.py::ClosedLoop`):

```python
planner = pyplanner.get("GRACE", host=H, model=M)
mapper  = ActionMapper(grace_namemap)
steps, _ = planner.generate_plan(task, obs, visible)
warns = mapper.screen(steps)                            # tell operator up-front
world, completed = WorldState(), []
i = 0
while i < len(steps):
    step = steps[i]
    try:
        mapped = mapper.to_skill(step, world)           # may be None (no-op)
    except Unmappable as e:                             # unsupported -> replan / escalate
        suffix, _ = planner.replan(task, completed, step, str(e), obs2, visible2)
        steps = completed + suffix; i = len(completed); continue
    if mapped is None:                                  # no-op: succeed, no robot call
        mapper.apply_effect(step, world); completed.append(step); i += 1; continue
    name, params = mapped
    ret = skill_registry.execute(name, params, node)    # REAL robot + in-skill verify
    if ret.get("isdone"):
        mapper.apply_effect(step, world); completed.append(step); i += 1
    else:                                                # CHECK failed -> REPLAN
        reason = ret.get("msg") or "skill failed"
        suffix, _ = planner.replan(task, completed, step, reason, obs2, visible2)
        steps = completed + suffix; i = len(completed)
        if exceeded_replan_budget: break
```

**Two/three-level checking** (the design's core): GRACE's `verify_step` is a *pre*-check
(symbolic, 0-token, inside `generate_plan`); kcare's per-skill perception
(`grasp_succeed`, retries) is the *post*-check (ground truth); and an optional **layer-3
VLM** post-check (`VLM_ACTIONS` + `vlm_hook`, §8) re-verifies world-changing actions.
The mapper bridges them: GRACE's `failure_reason` is sourced from the real skill's
`isdone/msg` (or the `Unmappable` message). The layered verifier itself is
`core/planning/verify.py::StepVerifier` (see
[`TRACKING_VERIFY_VOICE.md`](TRACKING_VERIFY_VOICE.md)).

---

## 8. Robot-side hooks: Grounder (`observe`) and layer-3 verifier (`vlm_hook`)

These live in the namemap (`grace_namemap.py`) because they touch robot perception.
Both import all runtime state (`robot_agent.state.current`, the ROS node, TCP clients)
**lazily inside the function** so the module imports cleanly outside a booted runtime
(tests, plan pre-screening).

**`observe(node=None) -> tuple[str, list[str]]`** — the "Grounder": a lightweight
perception pass producing `(obs_text, visible_objects)` for GRACE. As-built it is a
**conservative stub**: best-effort and non-fatal, it returns `("", [])` on any path
(no runtime, no ROS node, or any exception). A real grounding pass (head-camera
open-vocab `find`/`detect` over a candidate set) is sketched but left disabled because
`find()` moves the head and needs a real candidate caption — see the in-file
`TODO(integrator)` and §9.

**`vlm_hook(step, node=None) -> tuple[bool, Optional[float], str, float]`** — the
layer-3 post-check for `VLM_ACTIONS`. Returns `(isdone, score, note, latency_s)`
(`score` is an optional confidence, `None` when not produced). **Never raises** — every
failure path returns a permissive `(True, …)` so a flaky verifier can't stall a good
plan. Strategy:
- `Pick` → reuse the `grasp_succeed` skill (depth ROI near the gripper); kcare's
  authoritative in-hand check.
- other (`Place/PutIn/Open/Close`) → if an `llm` client is configured, ask a strict
  yes/no question (`"Did the robot successfully <verb> <to_obj(obj)>? …"`) via the
  `llm` skill and parse the answer; otherwise skip with `(True, …)`.

The current `llm` skill is text-only, so the non-`Pick` path is a heuristic stand-in;
a true vision-language check (grab a head/arm frame → VLM) is an integrator TODO.

---

## 9. Open items for the integrator

- **ENV ↔ GRACE location table** (`LOCATIONS`) must be trimmed/filled from the active
  site's ENV config so `to_loc()` agrees with what `move::<loc>` can navigate to;
  unknown locations should be validated by the driver into a replan-able failure.
- **`Find` inside vs outside containers**: outside → `find` (head/multi-view); inside
  an open drawer → `find_arm` (wrist). Pick based on whether `world.opened` shows the
  object's container is open. (The engine maps both to the `find` skill via `SKILL_MAP`;
  drawer-aware switching is driver/namemap policy.)
- **`obs` / `visible_objects` for GRACE** must come from real perception — implement
  `observe()` (the Grounder) beyond its current safe stub.
- **Replan budget** (max attempts) and operator escalation on `Unmappable` — the
  driver already bounds this via `max_replans` (default 3).
- **Close-state threading**: stash each `open_drawer` result and feed
  `pose_after_open/lift_after_open/forward_after_open` into the paired `close_drawer`.
- Keep `SKILL_MAP` / `NOOP_ACTIONS` (and thus the derived `SUPPORTED_ACTIONS`) in the
  one namemap module so coverage is auditable as new skills land.
