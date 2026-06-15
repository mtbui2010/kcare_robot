# Task tracking · per-step verify · voice output — design

Shipped design for: (1) a **TaskRecord** object that tracks / verifies / logs the plan,
(2) **layered verification after every step**, (3) **voice output** (robot + dashboard
announce milestones & failures). Extends the closed-loop design
([`CLOSED_LOOP_ARCHITECTURE.md`](CLOSED_LOOP_ARCHITECTURE.md)); planner-agnostic, lives
under `robot_agent/core/planning/`.

This document reflects the **as-built** code. The three planning modules
(`records.py`, `verify.py`, `announcer.py`), the driver wiring (`loop.py`), the TTS
mute flag (`utils.py`) and the frontend hook (`useVoiceOutput.ts`) all exist and ship;
remaining gaps are marked **TODO**.

Decisions (locked): layered verify **incl. VLM (toggleable)** · TaskRecord
**in-memory + JSONL per run** · voice **backend (robot speaker) + frontend
(dashboard)** · announce **milestones + failures, language follows `lang`
(vi/ko/en)**.

> Constraint preserved: `pyplanner/`, `paper_grace/` untouched; kcare **skills not
> edited** (skill TTS is *muted via a flag*, not removed); plan-level concerns live
> in the robot_agent planning layer.

---

## 1. The tracked object — `TaskRecord` / `StepRecord`

Single source of truth for tracking (live status), verification (holds results),
and logging (serialized to JSONL). Plain dataclasses, JSON-serializable. **No
pyplanner import**; timestamps are passed in by the caller (libs forbid
`Date.now`-style nondeterminism).

```python
# robot_agent/core/planning/records.py
@dataclass
class VerifyResult:
    layer: str                              # 'isdone' | 'symbolic' | 'vlm'
    ok: bool
    detail: str = ''                        # reason / message
    confidence: Optional[float] = None      # vlm only
    latency_ms: float = 0.0

@dataclass
class StepRecord:
    index: int
    action: str                             # GRACE action (MoveTo, Pick, …)
    object: str                             # CamelCase target
    skill: Optional[str]                    # mapped kcare skill (None for no-op)
    params: dict = field(default_factory=dict)
    status: str = 'pending'                 # pending|running|success|failed|skipped
    attempt: int = 1                        # increments across replans
    started_at: float = 0.0
    ended_at: Optional[float] = None
    result: dict = field(default_factory=dict)            # sanitized skill return
    verifies: list[VerifyResult] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)         # per-step log lines

    @property
    def verified(self) -> bool:
        """True iff at least one layer ran and every run layer passed."""
        return bool(self.verifies) and all(v.ok for v in self.verifies)

@dataclass
class TaskRecord:
    run_id: str                             # time-based id (passed in; no Date.now)
    task: str                               # NL instruction
    lang: str = 'en'
    obs: str = ''
    visible: list[str] = field(default_factory=list)
    planner: str = 'grace'                  # which backend produced the plan
    plan_meta: dict = field(default_factory=dict)         # PlanMetrics
    warnings: list[str] = field(default_factory=list)     # mapper.screen()
    steps: list[StepRecord] = field(default_factory=list)
    replans: list[dict] = field(default_factory=list)     # {at_index, failed, reason, suffix_len}
    status: str = 'planning'                # planning|running|success|failed|aborted|planned
    started_at: float = 0.0
    ended_at: Optional[float] = None
```

A `new_task_record(...)` helper constructs a fresh record in the `planning`
state:

```python
def new_task_record(
    run_id: str,
    task: str,
    lang: str = 'en',
    obs: str = '',
    visible: Optional[list[str]] = None,
    planner: str = 'grace',
    started_at: float = 0.0,
) -> TaskRecord: ...
```

It copies `visible` defensively (`list(visible) if visible else []`) and forces
`status='planning'`. All other fields use their dataclass defaults.

**Lifecycle** (driven by the closed-loop driver, `loop.py`):
`planning → running → (each StepRecord pending→running→success|failed) →
success|failed|aborted`. Plan-only runs short-circuit `running → planned` right
after the plan is emitted (no steps execute).

---

## 2. Logging — in-memory + JSONL per run

```python
# records.py
class RunLogger:
    def __init__(self, run: TaskRecord, path: str | os.PathLike): ...
        # parent dirs are created eagerly in __init__ so the first event() never
        # fails on a missing directory; path = <common_dir>/task_runs/<run_id>.jsonl
    def event(self, kind: str, t: float | None = None, **fields): ...
        # append one JSON line {"t": t?, "kind": kind, **fields};
        # "t" (caller-supplied) is included only when provided — never time.time()
    def snapshot(self): ...
        # append {"kind": "snapshot", "record": asdict(self.run)} as the final line
```

- **One JSONL file per run** under a robot-side data dir
  (`<common_dir>/task_runs/<run_id>.jsonl`, set by `ClosedLoop._resolve_cfg`),
  **never** inside `pyplanner/`.
- The driver routes **every** streamed WebSocket event through `RunLogger.event`
  → the JSONL is a replay of exactly what the UI saw + the final `TaskRecord`
  snapshot (written in the driver's `finally`).
- In-memory `TaskRecord` is the live object the driver mutates and streams from.
- Lines are written with `ensure_ascii=False` (so vi/ko text stays readable) and
  appended under a fresh `open('a')` per line — durable, no buffering surprises.
- (History/query across runs is out of scope per the chosen option; the per-run
  files already make later indexing trivial if wanted.)

### Serialization — numpy / NaN-safe `_to_jsonable`

Skill returns routinely carry numpy scalars/arrays and non-finite floats that
`json.dumps` would choke on (or emit invalid `NaN`/`Infinity` for). `RunLogger`
funnels every payload through `_to_jsonable(obj)` (then `_dumps`), which
recursively coerces:

- dataclasses → dict (via `asdict`)
- `set` / `frozenset` → **sorted** list (deterministic; falls back to `repr`-keyed
  sort when members are unorderable)
- `tuple` → list
- numpy scalars (`np.generic`) → python scalars; numpy arrays → list
  (numpy import is **guarded** — module stays stdlib-only at runtime if numpy is absent)
- `NaN` / `±inf` → `None`
- `bytes` / `bytearray` → utf-8 (`'replace'`) decoded string
- `Path` → `str`
- anything still unknown → `str(obj)` (logging never hard-fails)

---

## 3. Per-step verification — layered, VLM toggleable

```python
# robot_agent/core/planning/verify.py
class StepVerifier:
    def __init__(self, vlm_enabled: bool, vlm_actions: set[str], vlm_hook=None): ...

    def verify(self, step, skill_result, world, node) -> list[VerifyResult]:
        out = []
        # ── Layer 1: skill isdone (always, cheapest) ───────────────────────
        ok1 = bool(skill_result.get('isdone'))
        out.append(VerifyResult('isdone', ok1, skill_result.get('msg', '')))
        if not ok1:
            return out                       # short-circuit: physical failure

        # ── Layer 2: symbolic consistency (0 token) ────────────────────────
        ok2, why = symbolic_check(step, world)
        out.append(VerifyResult('symbolic', ok2, why))

        # ── Layer 3: VLM look-and-confirm (toggleable, world-changing only) ─
        if self.vlm_enabled and step.get('action') in self.vlm_actions and self.vlm_hook:
            try:
                ok3, conf, why3, ms = self.vlm_hook(step, node)   # robot-provided
                out.append(VerifyResult('vlm', bool(ok3), why3 or '',
                                        confidence=conf, latency_ms=float(ms or 0.0)))
            except Exception as exc:
                # never-fail: a broken/unavailable hook records a PASSING result
                out.append(VerifyResult('vlm', True, f'vlm hook error (skipped): {exc}'))
        return out
```

- **Layer 1 — `isdone`**: the existing skill signal (incl. in-skill checks like
  `grasp_succeed`, retry loops). Failure short-circuits → caller replans. `detail`
  carries `skill_result['msg']`.
- **Layer 2 — symbolic** (`symbolic_check(step, world)`): apply the step's effect
  to a **copy** of the mapper's `WorldState` and confirm consistency (mirror of
  GRACE's effect model, robot-side — no pyplanner import). `world` is **never
  mutated** (`world.copy()`). Returns `(ok, reason)`. Modeled actions and their
  precondition/effect:

  | action | precondition checked | effect on copy |
  |---|---|---|
  | `MoveTo` | — | `arrived = obj` |
  | `Find` | `obj` non-empty | `found = obj` |
  | `Pick` | must have `found`, gripper empty | `holding = obj or found`, `found = None` |
  | `Place` | must be `holding` | `holding = None` |
  | `PutIn` | `holding` set; container (`target or obj`) in `opened` | `holding = None` |
  | `Open` | `obj` not already open | `opened.add(obj)` |
  | `Close` | `obj` currently open | `opened.discard(obj)` |
  | `TurnOn` | `obj` not already on | `on.add(obj)` |
  | `TurnOff` | `obj` currently on | `on.discard(obj)` |
  | `Wash` | `obj` held or found | — |
  | `Serve` | must be `holding` | `holding = None` |
  | `Sit` / `LieOn` / `Wait` | — | — (pose/idle) |
  | *unknown* | — | passes through (don't block unmodeled steps) |

- **Layer 3 — VLM**: only runs when `vlm_enabled` **and**
  `step['action'] in vlm_actions` **and** a `vlm_hook` is supplied. The hook is
  **robot-specific** (camera + model), returns `(ok, confidence, reason,
  latency_ms)`, and asks a vision-language model *"did the robot succeed at
  `<action> <object>`?"*. It is **guarded / never-fail**: any exception is caught
  and recorded as a *passing* `vlm` result noting the skip, so a broken or
  unavailable hook is visible in logs but never triggers a replan.
- **Verdict policy** — `verdict(results) -> (ok, reason)`:
  - no results → `(False, 'no verification ran')` (mirrors `StepRecord.verified`
    requiring ≥1 layer to have run),
  - else ok iff **every** layer passed; `reason` = the first failing layer's
    `detail` (or `'<layer> failed'`).
- The `WorldState` contract is imported from `.base` (`# noqa: F401`, do not
  redefine).

**Wiring (shipped).** `ClosedLoop.run_blocking` constructs
`StepVerifier(cfg.vlm_enabled, set(nm.VLM_ACTIONS), nm.vlm_hook)`, where
`vlm_enabled` comes from `ROBOT_AGENT_VERIFY_VLM` (default off) and
`VLM_ACTIONS = {Pick, Place, PutIn, Open, Close}` + the `vlm_hook` come from
`kcare_robot/configs/grace_namemap.py`. **TODO:** the `vlm_hook`'s non-`Pick` path
is a text-LLM heuristic stand-in until a true vision-language check is wired (the
`Pick` path already uses the real `grasp_succeed` depth check).

---

## 4. Voice output — backend speaker + dashboard, single phrasebook

**Key idea (shipped): the backend computes ONE localized phrase per milestone via
the `Announcer`, returns it as the streamed event's `say`, and speaks it on the
robot. The frontend speaks the SAME string via `speechSynthesis`.** One
phrasebook, shared contract — no duplicated phrasing in TS + Python.

```python
# robot_agent/core/planning/announcer.py
class Announcer:
    def __init__(self, lang: str = 'en', speak_backend: bool = True): ...
    def say_for(self, kind: str, **ctx) -> str:    # localized phrase, no speech
    def announce(self, kind: str, **ctx) -> str:   # compute; if speak_backend, speak; return text
```

- **Phrasebook** `PHRASES[lang][kind]` ships `{vi, ko, en}` templates. Phrase
  **kinds actually present**: `task_start`, `step_start`, `step_success`,
  `step_fail`, `replan`, `done_success`, `done_fail`, `plan_ready`,
  `unsupported`. Templates use `{verb}`, `{object}`, and (for `step_fail`)
  `{reason}`. Examples:
  - vi: `step_start → "Đang {verb} {object}"`, `step_fail → "Lỗi khi {verb} {object}. {reason}"`, `plan_ready → "Đã tạo kế hoạch"`
  - ko: `step_start → "{object} {verb}를 진행합니다"`, `done_success → "작업을 완료했습니다"`, `plan_ready → "계획을 생성했습니다"`
  - en: `step_start → "I will {verb} {object}"`, `done_fail → "Task failed"`, `plan_ready → "Plan ready"`
- **Per-action `VERBS` table** (`VERBS[lang][action]`) injects `{verb}` from the
  step's `action` (only when `ctx` doesn't already carry `verb` and does carry
  `action`). Covered actions (with synonyms folded to one verb):

  | action(s) | vi | ko | en |
  |---|---|---|---|
  | `MoveTo`/`GoTo`/`Navigate` | đi tới | 이동 | move to |
  | `Find`/`Search`/`Detect` | tìm | 찾기 | find |
  | `Pick`/`PickUp`/`Grasp` | lấy | 집기 | pick |
  | `Place`/`PlaceAt`/`PutDown`/`PutIn` | đặt | 놓기 | place |
  | `Open` | mở | 열기 | open |
  | `Close` | đóng | 닫기 | close |

  Unknown action → `action.lower()` as the verb.
- **Resolution robustness**: unknown `lang` falls back to `en`; a missing `kind`
  falls back to the `en` template, then to a per-language `_GENERIC` phrase
  (`vi: "Đang thực hiện"`, `ko: "진행 중입니다"`, `en: "Working"`). Missing format
  keys render as `''` via a `_BlankDict` mapping (so a `step_fail` with no
  `reason` still produces a clean phrase); any format error falls back to the raw
  template.
- **Backend (robot speaks)**: `announce(...)` calls `_speak(text)`, which spawns a
  **daemon thread** that calls `robot_agent.utils.text2voice(text, lang=code,
  force=True)` (gTTS). `force=True` bypasses the skill mute flag so the plan-level
  Announcer owns speech. Speech is fully isolated — any exception in the thread is
  swallowed so it never breaks the control loop. `_GTTS_CODE` maps `{vi, ko, en}`
  → gTTS codes.
- **Frontend (dashboard speaks)**: `useVoiceOutput({ enabled, lang })`
  ([`robotapp/frontend/lib/useVoiceOutput.ts`](../../robotapp/frontend/lib/useVoiceOutput.ts))
  returns `{ speak, supported }`. `speak(event.say)` builds a
  `SpeechSynthesisUtterance`, sets `utterance.lang` via a local `langToBcp47`
  (`en→en-US`, `ko→ko-KR`, `vi→vi-VN`), and calls `window.speechSynthesis.speak`.
  It no-ops when disabled/unsupported or on empty text, and cancels queued/in-flight
  utterances when toggled off or on unmount. `speak` has stable identity (safe in
  effect deps); latest `enabled`/`lang` are read through refs. **Shipped wiring:**
  `page.tsx` calls `const { speak } = useVoiceOutput({enabled: voiceOut, lang: voiceLang})`,
  speaks `if (ev.say) speak(ev.say)` in the WebSocket `onmessage`, and exposes a
  header **🔊 voice** toggle (`voiceOut`); `voiceLang` follows the run's `lang`.

### Skill-level TTS mute flag

To avoid double-speak (kcare `pick`/`placeat`/`move` skills speak internally),
`robot_agent/utils.py` exposes a **process-global** mute flag:

```python
_SKILL_TTS_MUTED = False
def set_skill_tts_muted(v: bool): ...   # driver mutes skill TTS for the run
def skill_tts_muted() -> bool: ...
def text2voice(text, lang=None, run_thread=True, slow=False, force=False):
    if _SKILL_TTS_MUTED and not force:   # skill calls suppressed
        return
    ...
```

- The closed-loop driver calls `set_skill_tts_muted(True)` at the start of
  `run_blocking` (when `cfg.mute_skill_tts`, default `True`) and restores it in
  its `finally`; the Announcer passes `force=True` to bypass the flag.
- **No skill edits** — `text2voice` simply honors the flag.

**TODO (vi backend voice):** `text2voice` currently normalizes language with
`lang = 'en' if lang != 'ko' else lang`, so a `vi` request **falls back to `en`**
on the robot speaker even though gTTS supports `vi`. The frontend already speaks
`vi-VN`. To make backend `vi` real, relax this clamp (and/or add the offline
**viettts** path). Until then, backend voice is effectively ko/en only.

---

## 5. Event schema (driver → UI, and → both voices)

The closed-loop driver (`loop.py`) emits these structured events; voice-relevant
ones carry `say` (the `Announcer.announce(...)` return value). **Shipped** — field
names below match `ClosedLoop.run_blocking`:

| event | fields | spoken? |
|---|---|---|
| `task_start` | `task`, `run_id`, `world`, `say` | ✓ |
| `status` | `msg` | — |
| `plan` | `steps[]`, `plan_meta`, `warnings[]`, `world` | — |
| `warning` | `msg` (one per unsupported step) | — |
| `step_start` | `index`, `action`, `object`, `skill`, `say` | ✓ |
| `step_verify` | `index`, `verifies[]` (layer/ok/detail/confidence/latency_ms) | — |
| `step_done` | `index`, `status`, `result?`, `reason?`, `skill?`, `world`, `say` | ✓ (success/fail) |
| `replan` | `at_index`, `reason`, `world`, `say` | ✓ |
| `done` | `status` (`success`/`failed`/`planned`), `run_id`, `world`, `say` | ✓ |
| `error` | `msg`, `trace` | — |

Existing `step_start/step_done/done` are **kept** (frontend contract stable); the
driver only **adds** `action/object/verifies/say` and the `world` snapshot
(`world.to_dict()`). `RunLogger.event` persists every event, and the driver's
`finally` appends the final `snapshot`. The `plan_ready` phrase rides the `done`
event in plan-only mode (`status="planned"`).

### 5.1 `world` snapshot + the open-loop / direct path

`world` carries the live `WorldState.to_dict()`
(`arrived/found/holding/opened/on/holding_since/found_pose`) so the dashboard
"Robot State" panel stays current each step. The **open-loop / direct** executor
(`UnifiedAgent.run` / `run_direct`) has no GRACE step, so it emits a dedicated
**`world`** event (`{event:'world', world:{…}, found_pose_stale?}`) — once as an
initial snapshot and once after each step — via `_emit_world`, which applies the
`grace_namemap.apply_skill_effect` hook, reconciles `arrived`, computes
`found_pose_stale`, and persists. Operators can correct a stale belief live with
`PUT /agent/world` (the panel is editable mid-run).

---

## 6. How it plugs into the closed-loop (shipped)

Inside `ClosedLoop.run_blocking` (the planner-agnostic driver), per step
(condensed from `loop.py`):

```
rec = StepRecord(index=i, action, object, skill, params, status='running', started_at=ts)
emit('step_start', say=announcer.announce('step_start', action=…, object=arg_name))
result = self.sr.execute(skill, params, node)               # POST-CHECK happens in-skill
rec.result   = result
rec.verifies = verifier.verify(step, rec.result, world, node)   # layered verify
ok, reason   = verdict(rec.verifies)
emit('step_verify', verifies=_vr_list(rec.verifies))
if ok:
    rec.status = 'success'
    mapper.apply_effect(step, world)
    emit('step_done', status='success', result=rec.result,
         say=announcer.announce('step_success', action=…, object=arg_name))
else:
    rec.status = 'failed'
    emit('step_done', status='failed', result=rec.result, reason=reason,
         say=announcer.announce('step_fail', action=…, object=arg_name, reason=reason))
    obs, visible = nm.observe(node)                         # fresh observation
    ok_to_replan, steps, i, replans = self._maybe_replan(...)   # bounded by max_replans
    emit('replan', at_index=i, reason=reason, say=announcer.announce('replan'))
run.steps.append(rec)
```

`world` = the mapper's robot-side `WorldState` (no pyplanner import). Both VLM and
symbolic verify read it; symbolic verify operates on `world.copy()`, and only
successful steps mutate the real `world` via `mapper.apply_effect`. The driver
calls `set_skill_tts_muted(True)` for the run so only the Announcer speaks, and on
full success best-effort calls `planner.record_episode(task, completed)`.

---

## 7. Files

### ★ NEW (shipped)
| File | Purpose | Status |
|---|---|---|
| `robot_agent/core/planning/records.py` | `VerifyResult`, `StepRecord` (`.verified`), `TaskRecord`, `RunLogger` (JSONL, `_to_jsonable`), `new_task_record(...)` | ✅ built |
| `robot_agent/core/planning/verify.py` | `symbolic_check`, `StepVerifier` (isdone → symbolic → guarded VLM), `verdict` | ✅ built |
| `robot_agent/core/planning/announcer.py` | `Announcer` + `PHRASES[lang][kind]` + `VERBS`; daemon-thread backend TTS | ✅ built |
| `robot_agent/core/planning/loop.py` | `ClosedLoop` — builds/streams `TaskRecord`; threads `verifier`+`announcer`+`verdict`; writes JSONL; emits `action/object/verifies/say` | ✅ built |
| `robotapp/frontend/lib/useVoiceOutput.ts` | hook: speak `event.say` via `speechSynthesis` (+ enabled/lang, BCP-47) | ✅ built |

### ✎ MODIFIED
| File | Change | Status |
|---|---|---|
| `robot_agent/robot_agent/utils.py` | `set_skill_tts_muted`/`skill_tts_muted` flag; `text2voice(..., force=False)` honors it | ✅ done (vi clamp = **TODO**, see §4) |
| `robotapp/frontend/app/page.tsx` | wires events into `useVoiceOutput` (`speak(ev.say)`); header **🔊 voice** toggle; `voiceLang` follows run `lang` | ✅ done |
| `kcare_robot/kcare_robot/configs/grace_namemap.py` | `VLM_ACTIONS` + robot `vlm_hook(step, node)` (Pick = `grasp_succeed`) | ✅ built (true *vision* check = **TODO**, §3) |

### ✗ UNTOUCHED
`pyplanner/**`, `paper_grace/**`, `kcare_robot/skills/*.py` (TTS muted via flag, not
edited), `skill_registry.py`, `device_manager.py`.

---

## 8. Notes / decisions

- **VLM never-fail**: a missing/raising `vlm_hook` records a *passing* `vlm`
  result (with the error in `detail`) rather than failing the step — broken
  perception can't stall a good run. (Decided & built.)
- **VLM model for layer-3** (still open): reuse kcare's `llm`/vision TCP client or
  a dedicated VLM endpoint? The `vlm_hook` abstracts it; the `Pick` path falls back
  to the `grasp_succeed` depth check, while the other actions are a text-LLM
  heuristic until a true VLM frame check lands (**TODO**).
- **viettts vs gTTS for `vi`**: gTTS needs internet; viettts (local env) is offline.
  Backend `vi` is currently clamped to `en` (§4 TODO); frontend already speaks
  `vi-VN`.
- **Mute flag scope**: process-global (one run at a time per robot; concurrent
  UI+CLI on one robot is already documented as unsafe).
- **run_id/timestamps**: generated in the driver (real runtime, `time.strftime` +
  ms suffix) and passed into `records.py`/`RunLogger.event(t=…)`; `records.py`
  never calls `time.time()` itself (libs forbid `Date.now`-style nondeterminism).
- **Deterministic logs**: sets serialize as sorted lists and non-finite floats as
  `null`, so JSONL diffs/replays are stable and valid JSON.
