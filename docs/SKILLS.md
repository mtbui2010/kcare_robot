# kcare_robot — Skill Reference (hierarchy + pre-conditions)

> Source of truth: [`kcare_robot/configs/skills_config.py`](../kcare_robot/configs/skills_config.py)
> (`SKILL_CONFIGS`: `skill_name → (module, func)`). This doc is generated from a
> read-only audit of the skill modules under `kcare_robot/skills/`. Line numbers
> are indicative and may drift as code changes.

**Skill contract** (every registered skill):

```python
def my_skill(node, **params) -> dict:
    # node = robot_agent.connect.ros.node.CustomNode (already spinning)
    # params = kwargs parsed from the caller (CLI "skill::inputs key=val" → {'inputs': ..., 'key': val})
    return {'isdone': bool, ...}   # MUST include 'isdone'; may add payload (ins, pose, dd, ...)
```

Execution: `SkillRegistry.execute(name, params, node)` →
[`robot_agent/core/skill_registry.py`](../../robot_agent/robot_agent/core/skill_registry.py).
Failures are surfaced as `{'isdone': False, 'msg': ...}` (most skills wrapped by
`@exception_handler`); the `UnifiedAgent` execution loop **stops on the first
`isdone == False`**.

---

## 1. Registered skills (44 names)

`detect`≡`find` and `approach`≡`approach_close` are **aliases** (same function under
two names). One entry (`enable`) is commented out.

### Perception — `skills/recognition.py`
| Skill | func | Purpose | Key params | Returns |
|---|---|---|---|---|
| `detect` *(alias of find)* | `find` | Detect objects via TCP detector, attach 3D features/grasps | `inputs='obj,obj2'`, `camera`, `detector`, `estimate_grasp` | `{isdone, ins:{obj:{loc_3d,pose_3d,box,score,...}}}` |
| `find` | `find` | Navigate (if `@loc`) + detect across views/locations, with retry | `inputs='obj@loc'`, `views`, `once` | `{isdone, ins}` |
| `find_arm` | `find_arm` | Wrist-cam detect + grasp pose (groundedsam) | `inputs='obj'`, `dmin/dmax`, thresholds | `{isdone, ins:{obj:{grasppose:[x,y,z,rz,w]}}}` |
| `grasp_succeed` | `grasp_succeed` | **Verify object in hand** via depth ROI near gripper | `crop_roi=[x0,y0,x1,y1]` | `{isdone, obj_depth}` (`isdone` = depth in window) |
| `get_side_pose_3d` | `get_side_pose_3d` | Free side-pose from head cam (fastsam) | `side`, `view`, `pose_3d` | `{isdone, pose_3d}` |

### Pick / drawer — `skills/pick.py`
| Skill | func | Purpose | Key params | Returns |
|---|---|---|---|---|
| `fine_move` | `fine_move` | Wrist-guided grasp with re-detect retry loop | `inputs='obj'`, `num_trials=2`, `dpull` | `{isdone, dd:[dx,dy,dz]}` |
| `pick` | `pick` | Full pick (open→approach→grasp→retract→verify) + TTS | `inputs='obj[@loc|num_trials]'`, `type` | `{isdone}` |
| `pick_card` | `pick_card` | Domain pick: blue surface → white handle | `inputs='box@loc'` | `{isdone}` |
| `open_drawer` | `open_drawer` | Approach + pull handle + retract | `inputs='drawer_loc'`, `stay_here` | `{isdone, object_from_drawer, pose_after_open, ...}` |
| `close_drawer` | `close_drawer` | Push handle to close (mirror of open) | `inputs='drawer_loc'`, `pose_after_open` | `{isdone}` |
| `stack` | `stack` | Hardcoded stacking demo | `inputs` (unused) | `{isdone}` |

### Approach / place-at — `skills/approach.py`
| Skill | func | Purpose | Key params | Returns |
|---|---|---|---|---|
| `approach` *(alias)* | `approach_close` | Move arm to approach pose of target | `inputs='[x,y,z]'\|'loc'\|'obj@loc'` | `{isdone, wrist_angle, mforward, ...}` |
| `approach_close` | `approach_close` | (same as above) | … | … |
| `approach_closef` | `approach_closef` | `approach_close(init_pose_fixed=True)` | … | … |
| `placeat` | `placeat` | Approach target → release/wipe → retract + TTS | `inputs='target'`, `to_wipe`, `dlift_up` | `{isdone}` |
| `placep` | `placep` | `placeat(stay_here=True)` (no base motion) | … | `{isdone}` |

### Place (high-level) — `skills/place.py`
| Skill | func | Purpose | Key params | Returns |
|---|---|---|---|---|
| `place` | `place` | Pick-if-needed, then place at reverse loc | `inputs='[target>>]rev_loc'`, `force` | `{isdone}` |
| `collect_card` | `collect_card` | Pick card (if needed) → dest → twist → fold | `inputs='box@dest'`, `turn_angle` | `{isdone}` |
| `return_card` | `return_card` | Return held card to source | `inputs='box@source'` | `{isdone}` |
| `wipe` | `wipe` | `place(to_wipe=True, wrist_angle=30)` | `inputs='towel>>spill'` | `{isdone}` |

### Arm primitives — `skills/arm.py`
| Skill | func | Purpose | Returns |
|---|---|---|---|
| `arm_joints` | `arm_joints` | Read 7-DOF joint angles (deg) | `{isdone, joints}` |
| `arm_pose` | `arm_pose` | Read EE pose (m + deg) | `{isdone, pose}` |
| `movel` | `movel` | Linear EE motion (abs/delta) | `{isdone}` |
| `movej` | `movej` | Joint motion (named pose / angles / deltas) | `{isdone}` |
| `movet` | `movet` | Tool-frame translate + rotate | `{isdone}` |
| `movelf` | `movelf` | `movel` + `movet` in sequence | `{isdone}` |

### Lift — `skills/lift.py`
| Skill | func | Purpose | Returns |
|---|---|---|---|
| `lift` | `lift` | Absolute height / named (`lowest/highest/home`) | `{isdone}` |
| `lift_state` | `lift_state` | Read current height | `{isdone, current_position}` |
| `dlift` | `dlift` | Relative lift (`lift_state`+`lift`) | `{isdone}` |

### Mobile base — `skills/mobile.py`
| Skill | func | Purpose | Returns |
|---|---|---|---|
| `mobile_pose` | `mobile_pose` | Read base pose `[x,y,rz]` | `{isdone, pose}` |
| `move` | `move` | **Navigate to named ENV location** (lift home→moveb→forward→fold) | `{isdone}` |
| `moveb` | `moveb` | Base nav to `(x,y,rz)` | `{isdone}` |
| `forward` | `forward` | Drive forward + heading correct | `{isdone}` |
| `turn` | `turn` | Rotate in place by Δ° (no-op if `|Δ|<5`) | `{isdone}` |
| `rotate` | `rotate` | Rotate to absolute heading | `{isdone}` |

### Head / gripper — `skills/head.py`, `skills/grip.py`
| Skill | func | Purpose | Returns |
|---|---|---|---|
| `head_state` | `head_state` | Read head pitch/yaw | `{isdone, current_ry, current_rz}` |
| `moveh` | `moveh` | Head to named/abs pose (`down/straight/up/left/front/right`) | `{isdone}` |
| `grip` | `grip` | Open/close gripper (`open`/`close`/mm) | `{isdone}` |

### Init & utilities
| Skill | module:func | Purpose | Returns |
|---|---|---|---|
| `init_arm` | `goto_ready:init_arm` | Enable arm → head+grip parallel → fold → home | `{isdone}` |
| `inform` | `inform:inform` | Query state (e.g. `light` brightness) + TTS | `{isdone, msg}` |
| `select_response` | `select:select_response` | TTS confirm + **hardcoded** `place(item@shelf>>sopha_table@living_room)` | `{isdone}` |
| `get3d` | `pointcloud:get3d` | Pixel `(x,y)` → 3D base-frame pose | `{isdone, pose}` |
| `llm` | `vlm:llm` | Query LLM via TCP client | `{isdone, msg}` |
| `rest_detect` | `rest_detect:rest_detect` | Example: head frame → inferix TCP detector | `{isdone, msg, result}` |

---

## 2. Hierarchy (composite → leaf)

**Composite skills** (orchestrate other registered skills + helpers):

```
pick ─► pick_no_sound* ─► grip(open) ─► approach_close ─► {fine_move | direct} ─► post_pick_retract* ─► grasp_succeed
                                                              └► fine_move ─► find_arm ─► movet ─► grip(close) ─► grasp_succeed
open_drawer ─► move ─► lift ─► movej ─► movel/movet ─► find_arm ─► fine_move ─► grip ─► dlift ─► retract_from_drawer*
close_drawer ─► movej ─► movel/movet ─► lift ─► [find_arm] ─► movet ─► retract_from_drawer*
pick_card ─► approach_close ─► movel ─► fine_move ─► movej
approach_close ─► resolve_pose_3d*(may call find/move) ─► move_forward_and_fold* ─► execute_approach_motion*(movel/movej/lift)
placeat ─► placeat_no_sound* ─► approach ─► grip(open)|wipe ─► retract_after_place*(movej/dlift/forward)
place ─► grasp_succeed ─► [pick] ─► placeat
collect_card ─► grasp_succeed ─► [pick_card] ─► approach ─► movet ─► movej ─► turn_base*
return_card ─► grasp_succeed ─► [collect_card] ─► approach ─► movel ─► grip ─► movet ─► movej
wipe ─► place(to_wipe=True)
move ─► moveh ─► moveb ─► forward ─► lift ─► movej(fold) ─► turn
find ─► find_once*(loop) ─► [move] ─► moveh ─► detect ─► moveh
find_arm ─► detect(detector=groundedsam, camera=arm, estimate_grasp=True)
init_arm ─► arm_enable ─► moveh ─► grip ─► movej ─► lift
select_response ─► place   (hardcoded destination)
```
`*` = internal helper (not a registered skill); `[x]` = conditional.

**Primitive skills** (call only ROS/TCP agents): `detect`, `grasp_succeed`,
`arm_joints`, `arm_pose`, `movel`, `movej`, `movet`, `movelf`, `grip`, `lift`,
`lift_state`, `dlift`, `mobile_pose`, `moveb`, `forward`, `turn`, `rotate`,
`head_state`, `moveh`, `get3d`, `llm`, `rest_detect`.

---

## 3. Pre-condition matrix

Legend: ✓ required · ~ conditional/inferred · ✗ none. "Inferred" = derived from
code logic, not an explicit guard.

| Skill | detected obj | holding obj | at location | arm reachable | open container | Notes (guard) |
|---|:--:|:--:|:--:|:--:|:--:|---|
| `detect` | ✗ | ✗ | ✗ | ✗ | ✗ | camera + TCP detector registered |
| `find` | ✗ | ✗ | ~ | ✗ | ✗ | calls `move` if `@loc` given |
| `find_arm` | ✗ | ✗ | ✗ | ✓ | ✗ | wrist cam live |
| `grasp_succeed` | ✗ | ~ | ✗ | ✓ | ✗ | checks depth ROI (post-grasp verify) |
| `fine_move` | ~ | ✗ | ✗ | ✓ | ✗ | re-detects via `find_arm`; retry loop |
| `pick` / `pick_no_sound` | ~ | ✗ | ~ | ✓ | ✗ | `approach_close` must pass; `grasp_succeed` verify |
| `open_drawer` | ✗ | ✗ | ✓ | ✓ | ✗ | ENV needs `handle`+`handle_height` |
| `close_drawer` | ~ | ✗ | ✓ | ✓ | ✓(drawer) | re-detects handle if no `pose_after_open` |
| `approach`/`approach_close` | ~ | ✗ | ~ | ✓ | ✗ | `resolve_pose_3d` may call `find`/`move` |
| `placeat`/`placep` | ✗ | ✓ | ✓ | ✓ | ✗ | needs object in gripper |
| `place` | ✗ | ~ | ~ | ✓ | ✗ | `grasp_succeed`; `pick` if empty or `force` |
| `wipe` | ✗ | ✓ | ✓ | ✓ | ✗ | wrapper of `place` |
| `collect_card`/`return_card` | ✗ | ~ | ✓ | ✓ | ✗ | domain card flow |
| `move` | ✗ | ✗ | ✓(ENV) | ✓ | ✗ | ENV dict lookup; folds arm |
| `movel/movej/movet/movelf` | ✗ | ✗ | ✗ | ✓ | ✗ | arm pose readable |
| `lift/dlift/lift_state` | ✗ | ✗ | ✗ | ✗ | ✗ | lift agent live; range-clipped |
| `moveb/forward/turn/rotate/mobile_pose` | ✗ | ✗ | ✗ | ✗ | ✗ | base agents live |
| `moveh/head_state/grip` | ✗ | ✗ | ✗ | ✗ | ✗ | agent live |
| `arm_joints/arm_pose` | ✗ | ✗ | ✗ | ✗ | ✗ | joint/pose agent live |
| `init_arm` | ✗ | ✗ | ✗ | ✓ | ✗ | `arm_enable` agent |
| `get3d/llm/rest_detect/inform` | ✗ | ✗ | ✗ | ✗ | ✗ | respective TCP/agent live |

---

## 4. Canonical macro-sequences

**Pick** (`pick_no_sound`): `grip(open)` → `approach_close` (resolve via find/loc/pose)
→ `fine_move | direct grasp` → `post_pick_retract` (grip close, lift, fold, base back)
→ `grasp_succeed`.

**Place** (`placeat_no_sound`): `approach` (resolve dest) → `grip(open) | wipe`
→ `retract_after_place` (fold, dlift, base back).

**Drawer open**: `move` → `lift`+`movej` → `movel(ry=-90)` → `movet(dz)` →
`movel(z=handle_height)` → `forward(mforward)` → `fine_move(handle)` → `grip(open)`
→ `dlift(+0.2)` → `retract_from_drawer`.

**Navigation** (`move`): if `home`/`base` special-cased; else
`moveh(front)`+`turn` → `moveb(env_loc)` → `lift(home)` → `forward(dforward)` →
`turn(target_mode)` → `moveh(target_mode)`+`movej(fold)`.

---

## 5. Checking / verification mechanisms

Verification is **per-skill, perception-based**. (There is now also a persistent
symbolic **`WorldState`** — `arrived/found/holding/opened/on/holding_since/found_pose`
— fed by `grace_namemap.apply_skill_effect` on each `isdone` success and surfaced
in the dashboard "Robot State" panel; but it is a *belief* layer, not a physical
post-condition. Only `arrived` is sensor-reconciled, so the per-skill checks below
remain the source of truth for manipulation.)

- **Grasp check** — `grasp_succeed` re-images the wrist camera and checks object
  depth inside a gripper ROI (`recognition.py`, depth window). The single most
  important post-condition signal for manipulation.
- **Retry loops** — `fine_move` retries `num_trials` times (re-detect → grasp →
  verify); `find` falls back across all ENV locations before failing.
- **`isdone` propagation** — every skill returns `isdone`; `UnifiedAgent` halts the
  plan on the first `False` (no automatic replan — that is the gap the GRACE
  integration fills, see [`ACTION_MAPPER_SPEC.md`](ACTION_MAPPER_SPEC.md) and
  [`CLOSED_LOOP_ARCHITECTURE.md`](CLOSED_LOOP_ARCHITECTURE.md)).

---

## 6. GRACE ↔ kcare action map

> Source of truth: [`kcare_robot/configs/grace_namemap.py`](../kcare_robot/configs/grace_namemap.py)
> — the robot-side half of the ActionMapper (see
> [`ACTION_MAPPER_SPEC.md`](ACTION_MAPPER_SPEC.md)). GRACE / pyplanner emit
> *abstract* CamelCase symbolic actions over CamelCase locations/objects; this
> module maps them onto the concrete skills cataloged above. Names go through an
> explicit table first, then a CamelCase fallback, so unknown (open-vocabulary)
> objects still attempt execution.

### 6.1 Action → skill (`SKILL_MAP`)

Every target name below is a registered skill from §1.

| GRACE action | kcare skill | `build_params` → `inputs` | Notes |
|---|---|---|---|
| `MoveTo` | `move` | `to_loc(Loc)` | ENV location key |
| `Find` | `find` | `to_obj(Obj)` | open-vocab detector string |
| `Pick` | `pick` | `to_obj(world.found or Obj)`, `type='fine_move'` | GRACE `Pick` carries no object; taken from world state |
| `Place` | `placeat` | `_to_obj_or_loc(Recept)` | receptacle resolved as location-then-object |
| `PutIn` | `placeat` | `_to_obj_or_loc(Cont)` | realized as place-into-open-container |
| `Open` | `open_drawer` | `to_loc(Cont)` | **drawers only** — non-drawer `Open` is unsupported |
| `Close` | `close_drawer` | `to_loc(Cont)` | drawers only; ActionMapper threads the matching `open_drawer`'s `pose_after_open` |

The **inverse** mapping (skill → symbolic effect) is
`grace_namemap.apply_skill_effect(world, skill, params, result, node)`, used by
the open-loop / direct path to update `WorldState` from a raw skill that just
ran (keyed on the kcare skill name, applied only on `isdone`): `find*` → `found`
+ `found_pose`; `pick`/`grasp` → `holding` (clears `found`/`found_pose`,
stamps `holding_since`); `placeat`/`place`/`put`/`putin`/`give` → clears
`holding`; `open_drawer`/`open` · `close_drawer`/`close` → `opened`. `move`'s
`arrived` comes from sensor reconcile, not this hook.

### 6.2 No-op actions (`NOOP_ACTIONS`)

Benign symbolic padding — the ActionMapper succeeds these without touching the
robot (`build_params` returns `{}`, caller decides `isdone`):

`Sit`, `LieOn`, `Serve`, `Wait`

### 6.3 Unsupported actions

Actions with **no kcare skill** — intentionally absent from `SKILL_MAP` and from
`NOOP_ACTIONS`, so they are *not* in `SUPPORTED_ACTIONS` and the ActionMapper
pre-screen warns the operator (see ACTION_MAPPER_SPEC §6):

`TurnOn`, `TurnOff`, `Wash` (and any non-drawer use of `Open`/`Close`).

`SUPPORTED_ACTIONS` = `SKILL_MAP.keys() ∪ NOOP_ACTIONS` (the 7 mapped actions +
the 4 no-ops = 11 advanceable actions).

### 6.4 VLM-verified actions (`VLM_ACTIONS`)

World-changing actions get a layer-3 (perception) post-check via `vlm_hook`:

`Pick`, `Place`, `PutIn`, `Open`, `Close`

- **`Pick`** → reuses the `grasp_succeed` skill (depth ROI near the gripper) —
  kcare's authoritative in-hand check.
- **`Place` / `PutIn` / `Open` / `Close`** → ask the `llm` skill a yes/no
  question if an LLM client is configured; otherwise the hook is skipped
  permissively. The current `llm` skill is text-only, so the non-`Pick` path is
  a heuristic stand-in (see the `vlm_hook` TODO for a true vision-language check).

The verifier is guarded end-to-end and **never raises** — on any failure it
returns permissive (`True`, …) so a flaky verifier does not stall a good plan.

### 6.5 Name tables

`grace_namemap.py` seeds two explicit lookup tables (with CamelCase fallbacks
`to_loc`/`to_obj`):

- **`LOCATIONS`** (`to_loc`, GRACE → ENV key): `LivingRoom`, `Bedroom`,
  `Kitchen`, `Restroom`, `Bathroom`(→restroom), `Table`, `DiningTable`,
  `SophaTable`/`SofaTable`(→sopha table), `BedTable`, `Shelf`, `Sink`, `Drawer`,
  `Home`. Unknown names fall back to `CamelCase → snake_case`. The result must be
  validated against the live site ENV (`robot_agent.skill_configs.ENV`); an
  unknown location surfaces as a replan-able failure.
- **`OBJECTS`** (`to_obj`, GRACE → detector string): `Cup`, `Mug`, `Can`,
  `Bottle`, `WaterBottle`, `Apple`, `Coke`, `Coffee`, `CoffeeMachine`, `Towel`,
  `Phone`, `Controller`, `Remote`(→controller), `Toothpaste`, `Pen`, `Light`,
  `Drawer`, `Handle`. Unknown names fall back to `CamelCase → "two words"`;
  the detector is open-vocabulary so the heuristic is usually acceptable.
- **`_to_obj_or_loc`** (used by `Place`/`PutIn`): prefers an explicit `LOCATIONS`
  entry, then an explicit `OBJECTS` entry, then the object-words fallback —
  because a receptacle may be a known ENV location *or* an object-like string.

> The `observe()` "Grounder" pass is a deliberate non-fatal stub: it returns
> `('', [])` rather than implicitly moving the head/base to probe the scene
> (see the in-code TODO and ACTION_MAPPER_SPEC §8).

---

## 7. File / line index

| Module | Skills (with approx line) |
|---|---|
| `skills/recognition.py` | `detect`/`find`(805), `find_arm`(841), `grasp_succeed`(877), `get_side_pose_3d`(892) |
| `skills/pick.py` | `fine_move`(50), `open_drawer`(126), `close_drawer`(237), `pick_no_sound`(337), `pick_card`(441), `stack`(472), `pick`(530) |
| `skills/approach.py` | `approach_close`(73), `approach_closef`(153), `approach`(160), `placeat_no_sound`(176), `placeat`(209), `placep`(230) |
| `skills/arm.py` | `arm_joints`(9), `arm_pose`(18), `movel`(28), `movej`(53), `movet`(101), `movelf`(120) |
| `skills/place.py` | `place`(34), `collect_card`(82), `return_card`(129), `wipe`(165) |
| `skills/lift.py` | `lift`(7), `lift_state`(31), `dlift`(38) |
| `skills/mobile.py` | `mobile_pose`(24), `moveb`(33), `turn`(43), `rotate`(52), `forward`(62), `move`(81) |
| `skills/head.py` | `head_state`(8), `moveh`(35) |
| `skills/grip.py` | `grip`(7) |
| `skills/goto_ready.py` | `init_arm`(12) |
| `skills/inform.py`,`select.py`,`pointcloud.py`,`vlm.py`,`rest_detect.py` | `inform`, `select_response`, `get3d`, `llm`, `rest_detect` |
| `configs/grace_namemap.py` | `SKILL_MAP`, `NOOP_ACTIONS`, `SUPPORTED_ACTIONS`, `VLM_ACTIONS`, `to_loc`/`to_obj`, `build_params`, `observe`, `vlm_hook` |
| helpers | `_pick_helpers.py`, `_approach_helpers.py`, `_place_helpers.py` |
