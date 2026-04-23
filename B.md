# B.md — Red Team Performance Improvement Task
## Your job: Make RED smarter, stealthier, and able to detect BLUE's traps

> **Run in parallel** with your teammate doing A.md (training the 4 models).
> All changes are code-only — no model training required from you.
> When teammate finishes A.md, update `base_agent.py` to load the 4 new specialists.

---

## Context: Current Red Team Problems

Based on `rewards_log.csv` analysis, RED is losing because:

1. **RED walks into BLUE honeypots repeatedly** — planner has honeypot avoidance but it only works when `honeypot_nodes_nearby` is populated in observation. Blue `PLACE_HONEYPOT` and `PLANT_BREADCRUMB` traps are not being counter-detected.
2. **RED doesn't recognize BLUE's `TAMPER_DEAD_DROP` trap** — when BLUE tampers a dead drop, RED reads it and gets poisoned intelligence (wrong node targets, fake suspicion scores). RED has no validation layer.
3. **RED aborts too late** — the abort gate (`suspicion > 0.80 AND blue_conf > 0.60`) means RED often gets caught before it triggers. By the time both conditions are met, `confirmed_detection_step` is already logged.
4. **RED Analyst and Operative act independently** — there's no coordination signal between the 4 red agents. Planner moves, Analyst reads wrong node, Operative plants trap at wrong location.
5. **RED trap actions are wasted** — `plant_false_trail`, `plant_temporal_decoy` are placed without checking if BLUE is actively investigating that zone. Trap budget is wasted.
6. **RED path selection ignores BLUE's `PLANT_BREADCRUMB` trap** — when BLUE plants breadcrumbs on a path, RED following that path gets penalized. RED has no awareness of breadcrumb poisoning.
7. **RED Exfiltrator never successfully exfiltrates** — it always arrives at HVT after the Planner has already exfiltrated one file, but then tries duplicate files.

---

## Change 1 — Improve Hybrid Specialist Routing in `base_agent.py`

**File:** `cipher/agents/base_agent.py`

**Problem:** `_is_hybrid_specialist()` only returns True for RED Planner. After A.md delivers 4 models, we need 4 specialists.

**What to do:**
Update `_is_hybrid_specialist()` and `_act_lora()` to support multiple trained agents:

```python
# In base_agent.py — replace _is_hybrid_specialist
def _is_hybrid_specialist(self) -> bool:
    """Returns True if this agent has a trained LoRA available in hybrid mode."""
    # RED specialists
    if self.team == "red" and self.role == "planner":
        return bool(os.path.exists(os.getenv(
            "RED_PLANNER_LORA_PATH",
            os.path.join("red trained", "cipher-red-planner-v2")
        )))
    if self.team == "red" and self.role == "analyst":
        return bool(os.path.exists(os.getenv(
            "RED_ANALYST_LORA_PATH",
            os.path.join("red trained", "cipher-red-analyst-v1")
        )))
    # BLUE specialists
    if self.team == "blue" and self.role == "surveillance":
        return bool(os.path.exists(os.getenv(
            "BLUE_SURVEILLANCE_LORA_PATH",
            os.path.join("blue trained", "cipher-blue-surveillance-v1")
        )))
    if self.team == "blue" and self.role == "threat_hunter":
        return bool(os.path.exists(os.getenv(
            "BLUE_THREAT_HUNTER_LORA_PATH",
            os.path.join("blue trained", "cipher-blue-threat-hunter-v1")
        )))
    return False
```

Also update `_act_lora()` to resolve the right adapter path per agent:
```python
def _act_lora(self) -> Action:
    # Map role to env var and default path
    _LORA_PATHS = {
        ("red", "planner"):      ("RED_PLANNER_LORA_PATH",      os.path.join("red trained", "cipher-red-planner-v2")),
        ("red", "analyst"):      ("RED_ANALYST_LORA_PATH",       os.path.join("red trained", "cipher-red-analyst-v1")),
        ("blue", "surveillance"):("BLUE_SURVEILLANCE_LORA_PATH", os.path.join("blue trained", "cipher-blue-surveillance-v1")),
        ("blue", "threat_hunter"):("BLUE_THREAT_HUNTER_LORA_PATH", os.path.join("blue trained", "cipher-blue-threat-hunter-v1")),
    }
    env_key, default_path = _LORA_PATHS.get((self.team, self.role), ("", ""))
    adapter_path = os.getenv(env_key, default_path) if env_key else default_path
    # ... rest of existing _act_lora body using adapter_path ...
```

---

## Change 2 — Dead Drop Integrity Check (Trap Detection)

**File:** `cipher/memory/dead_drop.py`  
**Problem:** BLUE's `TAMPER_DEAD_DROP` action corrupts dead drops but RED reads them blindly.

**What to do:** Add a `is_tampered()` check to `DeadDrop` or `DeadDropVault`.

Key signals a tampered drop will show:
- `suspicion` field in drop content is unusually low (< 0.05) — BLUE sets fake low suspicion to lure RED into moving aggressively
- `directive` field contains words like "proceed", "safe", "undetected" — BLUE-injected false confidence
- `timestamp` is in the future or impossibly old
- `checksum` mismatch (if the drop uses checksums from `dead_drop.py`)

**Logic to add in `DeadDropVault.read()`:**
```python
def _is_drop_tampered(self, drop) -> bool:
    """Heuristic tamper detection for dead drops."""
    content = getattr(drop, 'content', {}) or {}
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except Exception:
            return False  # can't parse = treat as untampered

    # Suspicion too low is suspicious
    suspicion = float(content.get('suspicion', 0.5))
    if suspicion < 0.05:
        return True

    # BLUE-injected optimistic keywords in directive
    directive = str(content.get('directive', '')).lower()
    tamper_keywords = ['proceed immediately', 'all clear', 'no threat', 'safe to exfil', 'undetected']
    if any(kw in directive for kw in tamper_keywords):
        return True

    # Verify checksum if available
    stored_hash = getattr(drop, 'checksum', None)
    if stored_hash:
        import hashlib
        computed = hashlib.sha256(str(content).encode()).hexdigest()[:16]
        if computed != stored_hash:
            return True

    return False
```

Then in `DeadDropVault.read()`, tag tampered drops:
```python
# After reading drops, mark tampered ones
for drop in drops:
    drop.tampered = self._is_drop_tampered(drop)
```

And in `_process_red_action()` inside `_episode_runner.py`, when `READ_DEAD_DROP`:
```python
# Add in the READ_DEAD_DROP branch
tampered_count = sum(1 for d in drops if getattr(d, 'tampered', False))
if tampered_count > 0:
    result['tampered_drops_detected'] = tampered_count
    # Increase suspicion slightly — RED knows something's wrong
    state.update_suspicion(0.05 * tampered_count)
```

---

## Change 3 — Smarter Abort Gate in `RedPlanner._stub_act()`

**File:** `cipher/agents/red/planner.py`  
**Problem:** Abort triggers at `suspicion > 0.80 AND blue_conf > 0.60`, but by then RED is already detected.

**What to do:** Add an **early-warning abort** based on rate of change, not just absolute value.

```python
# In RedPlanner.__init__, add:
self._suspicion_history: list[float] = []
self._blue_conf_history: list[float] = []

# In _stub_act(), before the current PRIORITY 2 abort block:
# Track history
self._suspicion_history.append(obs.estimated_suspicion)
self._blue_conf_history.append(blue_confidence)
if len(self._suspicion_history) > 5:
    self._suspicion_history.pop(0)
    self._blue_conf_history.pop(0)

# Early abort: if suspicion is rising FAST (delta > 0.15 in 2 steps) AND blue is rising
if len(self._suspicion_history) >= 3:
    recent_delta = self._suspicion_history[-1] - self._suspicion_history[-3]
    blue_delta   = self._blue_conf_history[-1] - self._blue_conf_history[-3]
    if recent_delta > 0.18 and blue_delta > 0.15 and obs.estimated_suspicion > 0.60:
        return Action(
            agent_id=self.agent_id,
            action_type=ActionType.ABORT,
            reasoning=(
                f"Early abort: suspicion rising fast (+{recent_delta:.2f} in 3 steps), "
                f"blue_conf rising (+{blue_delta:.2f}). Aborting before confirmed detection."
            ),
        )
```

Also in `reset()`, clear these histories:
```python
self._suspicion_history = []
self._blue_conf_history = []
```

---

## Change 4 — Trap-Aware Path Scoring in `RedPlanner._stub_act()`

**File:** `cipher/agents/red/planner.py`  
**Problem:** Path selection uses `suspicion_delta` from edge data, but ignores BLUE trap presence on nodes.

**What to do:** When scoring `safe_paths`, add a trap penalty layer.

In the path scoring section (around line 112–125 of `planner.py`):
```python
# Replace the existing `best = min(pool, key=...)` line with:

def _path_score(p: dict) -> tuple:
    """Lower score = better path. Penalize trapped nodes."""
    node_id = int(p.get("node", 0))
    base_delta = float(p.get("suspicion_delta", 0.0))
    zone_val   = int(p.get("zone", obs.current_zone))

    # Trap penalty: if we've seen a trap event at this node, penalize heavily
    trap_penalty = 0.0
    if hasattr(state, 'trap_registry'):  # won't be available in stub, but check
        pass  # trap_registry is episode-level, not available in planner directly
    
    # Heuristic: nodes we've visited that caused suspicion spikes are likely trapped
    recent_bad = getattr(self, '_nodes_with_suspicion_spikes', set())
    if node_id in recent_bad:
        trap_penalty = 0.50

    # Zone progress bonus (prefer deeper zones when safe)
    zone_bonus = -0.10 if zone_val > obs.current_zone else 0.0

    return (base_delta + trap_penalty + zone_bonus, -zone_val)

best = min(pool, key=_path_score)
```

Add tracking of suspicious nodes in `_stub_act()`:
```python
# At the start of _stub_act(), track nodes where suspicion spiked
if not hasattr(self, '_nodes_with_suspicion_spikes'):
    self._nodes_with_suspicion_spikes = set()
if obs.estimated_suspicion > 0.55 and len(self._suspicion_history) >= 2:
    if self._suspicion_history[-1] - self._suspicion_history[-2] > 0.10:
        # The node we just moved FROM likely caused this spike
        self._nodes_with_suspicion_spikes.add(obs.current_node)
```

---

## Change 5 — RED Exfiltrator Deduplication Fix

**File:** `cipher/agents/red/exfiltrator.py`  
**Problem:** Exfiltrator attempts to exfiltrate files that the Planner already exfiltrated in the same episode.

**What to do:** In `_stub_act()` of `RedExfiltrator`, cross-reference with a shared state signal.

The state object has `state.red_exfiltrated_files` — but agents don't see state directly.
The observation `RedObservation` should include already-exfiltrated files.

Check `cipher/environment/observation.py` → `generate_red_observation()`.
Add `exfiltrated_files: list[str]` to `RedObservation` if not already present:
```python
# In generate_red_observation(), add:
exfiltrated_files=list(state.red_exfiltrated_files),
```

Then in `RedExfiltrator._stub_act()`:
```python
# Before choosing target_file, exclude already-exfiltrated ones:
already_done = set(getattr(obs, 'exfiltrated_files', []))
remaining_targets = [f for f in obs.files_at_current_node if f not in already_done]
if not remaining_targets:
    # Nothing left to exfiltrate at this node
    return Action(
        agent_id=self.agent_id,
        action_type=ActionType.WAIT,
        reasoning="All target files already exfiltrated — waiting for planner."
    )
target_file = remaining_targets[0]
```

---

## Change 6 — Breadcrumb Trap Awareness in Observation

**File:** `cipher/environment/observation.py`  
**Problem:** `RedObservation` has `honeypot_nodes_nearby` but no equivalent for BLUE breadcrumb traps.

**What to do:** Add `breadcrumb_nodes_nearby: list[int]` to `RedObservation`.

In `generate_red_observation()`:
```python
# After computing honeypot_nodes_nearby, add:
breadcrumb_nodes_nearby = []
if hasattr(state, 'trap_registry') and state.trap_registry:
    for trap in state.trap_registry.active_blue_traps:
        from cipher.environment.traps import BlueTrapType
        if trap.trap_type == BlueTrapType.BREADCRUMB:
            if trap.target_node in adj_nodes:
                breadcrumb_nodes_nearby.append(trap.target_node)
```

Then in `RedObservation` dataclass (or TypedDict, wherever it's defined), add:
```python
breadcrumb_nodes_nearby: list[int] = field(default_factory=list)
```

And in `RedPlanner._stub_act()`, extend the safe path filter:
```python
breadcrumb_nodes = set(getattr(obs, 'breadcrumb_nodes_nearby', []) or [])
safe_paths = [
    p for p in available_paths
    if p.get("node") in set(obs.adjacent_nodes or [])
    and p.get("node") not in honeypot_nodes
    and p.get("node") not in breadcrumb_nodes   # ← NEW
]
```

And update the prompt text in `_build_messages()`:
```python
if breadcrumb_nodes_nearby := getattr(obs, 'breadcrumb_nodes_nearby', []):
    path_lines.append(f"BREADCRUMB TRAPS NEARBY (AVOID): {breadcrumb_nodes_nearby}")
```

---

## Change 7 — RED Agent Coordination via Dead Drop Signal

**File:** `cipher/agents/red/analyst.py`, `cipher/agents/red/operative.py`  
**Problem:** The 4 RED agents act independently. Planner moves but Analyst doesn't know where Planner went.

**What to do:** Use the dead drop vault as a lightweight coordination bus.

In `RedPlanner._stub_act()`, when selecting a move target, write the target node to a "coordination hint" in the dead drop:
```python
# After computing `target` node for MOVE action:
# Write a coordination drop (lightweight — just node target)
# This is separate from the full dead drop written on context reset
# Store as a class attribute so Analyst/Operative can read it from obs
self._last_planned_target = target  # e.g., node 31
```

The observation system can then expose `red_planner_last_target` in `RedObservation` (read from a file or shared config if needed — simplest: write to a temp file on disk like `coordination.json` in the vault dir).

In `RedAnalyst._stub_act()`:
```python
# Read coordination hint
planned_node = self._read_coordination_hint()
if planned_node is not None:
    # Analyst investigates the planned target node
    if planned_node in obs.adjacent_nodes:
        return Action(
            agent_id=self.agent_id,
            action_type=ActionType.MOVE,
            target_node=planned_node,
            reasoning=f"Coordinating with Planner — moving to planned target node {planned_node}."
        )

def _read_coordination_hint(self) -> int | None:
    import json, os
    hint_file = os.path.join("vault", "coordination.json")
    try:
        with open(hint_file) as f:
            data = json.load(f)
            return int(data.get("planned_target", 0)) or None
    except Exception:
        return None
```

---

## Change 8 — Smarter Trap Placement Timing

**File:** `cipher/agents/red/operative.py`  
**Problem:** RED Operative places traps (`plant_false_trail`) whenever possible, wasting budget on early steps when BLUE isn't actively investigating.

**What to do:** Only plant traps when BLUE is actively detecting (blue_detection_confidence > 0.40):

```python
# In RedOperative._stub_act(), before choosing a trap action:
blue_conf = float(getattr(obs, 'blue_detection_confidence', 0.0))
if obs.estimated_suspicion > 0.45 and blue_conf > 0.40:
    # BLUE is actively hunting — plant a distraction
    n_moves = sum(1 for a in self.action_history if a.action_type == ActionType.MOVE)
    if n_moves >= 3 and self._trap_budget_remaining > 0:
        trap_choice = random.choice([
            ActionType.PLANT_FALSE_TRAIL,
            ActionType.PLANT_TEMPORAL_DECOY,
        ])
        return Action(
            agent_id=self.agent_id,
            action_type=trap_choice,
            target_node=obs.current_node,
            reasoning=(
                f"BLUE actively investigating (conf={blue_conf:.2f}). "
                f"Planting {trap_choice.value} at current node to mislead."
            ),
        )
```

---

## Change 9 — Reduce False Escalation Vulnerability

**File:** `cipher/agents/red/planner.py`  
**Problem:** BLUE's `TRIGGER_FALSE_ESCALATION` trap causes the episode state to flag RED as "confirmed detected" even when it isn't. RED doesn't know this happened.

**What to do:** In `generate_red_observation()`, expose a signal when the detection confidence jumps dramatically without RED doing anything:

```python
# In generate_red_observation(), add:
# Detection spike detector — did blue_conf jump > 0.20 without RED action this step?
prev_conf = getattr(state, '_prev_blue_detection_confidence', 0.0)
conf_spike = (state.blue_detection_confidence - prev_conf) > 0.20
state._prev_blue_detection_confidence = state.blue_detection_confidence

# Add to RedObservation:
detection_spike_this_step=conf_spike,
```

In `RedPlanner._build_messages()`, warn about the spike:
```python
spike = getattr(obs, 'detection_spike_this_step', False)
if spike:
    path_lines.insert(0, "⚠️ WARNING: BLUE detection confidence spiked suddenly — likely TRIGGER_FALSE_ESCALATION trap. Do NOT abort on this signal alone.")
```

---

## Change 10 — Improve LLM Prompt for RED Planner (Live/Hybrid Mode)

**File:** `cipher/agents/prompts/red_planner.txt`  
**Problem:** The system prompt doesn't mention trap awareness or BLUE counter-traps.

**What to add** (append to end of existing prompt):
```
## BLUE TRAP AWARENESS
BLUE plants traps to catch you:
- HONEYPOT nodes: Entering one immediately raises your suspicion by +0.30. NEVER enter honeypot nodes.
- BREADCRUMB trails: Moving along a breadcrumb-marked path increases detection confidence for BLUE.
- TAMPERED DEAD DROPS: If a dead drop's directive sounds suspiciously optimistic ("all clear", "safe to proceed"), IGNORE it — it's poisoned.
- FALSE ESCALATION: If detection confidence spikes suddenly without you doing anything risky, BLUE triggered a false alarm trap. Do NOT abort based on this spike alone.

## COUNTER-TRAP STRATEGY
When suspicion > 0.50 AND blue_conf > 0.35:
1. Plant a false trail at your PREVIOUS node (not current) to redirect BLUE investigation.
2. Wait 1-2 steps for suspicion to decay before moving.
3. Choose the path with lowest suspicion_delta, avoiding all flagged nodes.

## WINNING CONDITIONS
- Exfiltrate ALL 3 target files from the HVT node. Partial exfil is still scored.
- Avoid confirmed detection (blue_detection_confidence >= 0.80).
- Complete within max_steps. Urgency increases after step 8.
```

---

## Change 11 — Session Startup: Validate Model Paths Before Running Hybrid

**File:** `main.py` (or wherever `LLM_MODE=hybrid` is initialized)

**What to do:** At startup in hybrid mode, check all specialist paths exist and log warnings:

```python
# At top of main.py, after imports, add a function:
def _validate_hybrid_models():
    from cipher.utils.llm_mode import is_hybrid_mode
    if not is_hybrid_mode():
        return
    import os
    specialists = {
        "RED Planner v2":      os.getenv("RED_PLANNER_LORA_PATH", "red trained/cipher-red-planner-v2"),
        "RED Analyst v1":      os.getenv("RED_ANALYST_LORA_PATH", "red trained/cipher-red-analyst-v1"),
        "BLUE Surveillance v1": os.getenv("BLUE_SURVEILLANCE_LORA_PATH", "blue trained/cipher-blue-surveillance-v1"),
        "BLUE ThreatHunter v1": os.getenv("BLUE_THREAT_HUNTER_LORA_PATH", "blue trained/cipher-blue-threat-hunter-v1"),
    }
    for name, path in specialists.items():
        if os.path.exists(path):
            print(f"  ✅ {name}: {path}")
        else:
            print(f"  ⚠️  {name} NOT FOUND at {path} — will fall back to NVIDIA NIM")

# Call at startup:
_validate_hybrid_models()
```

---

## Files You Will Touch (Summary)

| File | Change |
|------|--------|
| `cipher/agents/base_agent.py` | Multi-specialist hybrid routing |
| `cipher/agents/red/planner.py` | Early abort, trap-aware path scoring, breadcrumb awareness |
| `cipher/agents/red/operative.py` | Smarter trap timing |
| `cipher/agents/red/analyst.py` | Coordination hint reader, dedup fix |
| `cipher/agents/red/exfiltrator.py` | Exfil deduplication |
| `cipher/environment/observation.py` | Add `breadcrumb_nodes_nearby`, `detection_spike_this_step`, `exfiltrated_files` to RedObservation |
| `cipher/memory/dead_drop.py` | Tamper detection heuristic |
| `cipher/training/_episode_runner.py` | Tag tampered drops in READ_DEAD_DROP result |
| `cipher/agents/prompts/red_planner.txt` | Add BLUE trap awareness section |
| `main.py` | Hybrid model path validation at startup |

---

## Checklist
- [x] Change 1: Multi-specialist `_is_hybrid_specialist()` and `_act_lora()` in `base_agent.py` — `_LORA_PATH_MAP` dict routes 4 specialist agents
- [ ] Change 2: Tamper detection in `dead_drop.py` + episode runner — pending
- [x] Change 3: Rate-of-change early abort in `planner.py` — thresholds raised (0.60→0.78 susp, 0.18→0.20 delta) to avoid premature abort
- [x] Change 4: Trap-aware path scoring in `planner.py` — spike penalty 0.50, zone bonus tripled (-0.10→-0.30), zone pool gate raised (0.70→0.82)
- [x] Change 5: Exfil deduplication in `exfiltrator.py` + observation — cross-references `obs.exfiltrated_files`
- [x] Change 6: Breadcrumb awareness in `observation.py` + `planner.py` — `breadcrumb_nodes_nearby` excluded from safe paths
- [x] Change 7: Dead drop coordination bus in `analyst.py` + `planner.py` — `coordination.json` written by Planner, read by Analyst
- [x] Change 8: Smarter trap timing in `operative.py` — traps only when blue_conf > 0.45, wait threshold raised (0.60→0.82)
- [x] Change 9: False escalation detection in `observation.py` + `planner.py` — `detection_spike_this_step` flag + planner warning
- [x] Change 10: Update `red_planner.txt` + `red_operative.txt` + `red_exfiltrator.txt` prompts — major threshold fixes:
  - Evasion threshold: 0.50 → 0.72 (was causing waits at step 8 with 0.44 suspicion)
  - Abort threshold: 0.80 → 0.82 
  - Exfil safe threshold: 0.50 → 0.78
  - Zone progression mandate added with urgency warnings
  - Dead drop discipline: only on context reset, max 2/episode
  - Wait discipline: never wait twice in a row
  - Suspicion "RISING" label: now fires at 0.55 not 0.30
- [ ] Change 11: Hybrid model validation in `main.py` — pending
- [ ] Wait for A.md teammate to deliver 4 model zips, then place in correct folders + update `.env`
- [ ] Run `python main.py --hybrid --episodes 5` and confirm RED scores improve
