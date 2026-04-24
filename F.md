# F.md — Agent Intelligence Upgrades
## Your job: Fix RED looping, improve BLUE coordination, add adaptive curriculum

> **Can start immediately** — does NOT require A.md to finish.
> **Run in parallel** with teammates doing E.md and G.md.
>
> From the 10-episode run output, Episode 2 showed RED bouncing between nodes 4↔7↔2
> for 29 steps (zero progress). Episode 10 showed RED stuck in CRITICAL zone for 15+ steps
> doing nothing but `wait` and `abort` (guardrail-rejected). These are behavioral bugs
> that waste compute and make the simulation look broken to judges.

---

## Context: Why This Matters

**Problem 1**: RED cycles (n4↔n7↔n2 loop, detected in episode 2 output) — the anti-loop 
filter exists but wasn't working because the pool contained dicts (now fixed by D.md). 
However, RED still needs smarter strategic behavior to avoid PERIMETER stalling.

**Problem 2**: Semantic guardrail is rejecting `abort` repeatedly in episode 10 (7 times!). 
The guardrail logic is too strict — RED gets stuck doing `wait` in perpetuity when it 
should be moving instead.

**Problem 3**: BLUE win rate is only 30% over 10 episodes — BLUE needs better coordination 
to actually mount a real defense.

---

## Change 1 — Fix "Abort" Guardrail Spam

**File:** `cipher/agents/base_agent.py`

**What to do:**

Find the semantic guardrail for `abort` and fix the condition. Currently it rejects `abort` 
even when RED is genuinely trapped. The correct logic should be:

```python
# Current (broken): only allow abort when suspicion > 0.90 AND detection > 0.60
# Fix: also allow abort when RED has done 5+ consecutive waits with no progress

def _is_abort_warranted(self, obs) -> bool:
    """Allow abort when suspicion is critical OR when genuinely stuck."""
    susp = getattr(obs, "estimated_suspicion", 0.0)
    det = getattr(obs, "current_detection_confidence", 0.0)
    
    # Count consecutive waits in recent history
    recent_waits = sum(
        1 for a in self.action_history[-6:]
        if a.action_type in (ActionType.WAIT, ActionType.STAND_DOWN)
    )
    
    return (susp > 0.85 and det > 0.50) or recent_waits >= 4
```

**Location:** Look for the `ABORT` guardrail check in `base_agent.py` around the 
`_is_action_semantically_valid()` method. Extend the condition using `_is_abort_warranted`.

---

## Change 2 — RED Zone Advancement Forcing

**File:** `cipher/agents/red/planner.py`

**What to do:**

Add a "zone stall" override that forces RED to advance after N steps in the same zone:

```python
# In _stub_act() and also ensure _act_live() sees this via the observation text

# If RED has been in same zone for > 8 steps with suspicion < 0.80:
#   Force a zone-crossing move by boosting zone-advance paths by 3x in score
#   Add a warning to the observation prompt: "⚠ ZONE STALL DETECTED — ADVANCE NOW"

# Add to _red_observation_to_text() in base_agent.py:
if self._zone_stall_steps > 8:
    lines.append(
        f"⚠⚠ CRITICAL ZONE STALL: {self._zone_stall_steps} steps in Zone {obs.current_zone}! "
        f"You MUST cross to the next zone NOW regardless of suspicion. "
        f"Staying in the same zone is mission failure."
    )
```

**Track zone stall in the planner** — the `_zone_stall_steps` attribute already exists 
(check `planner.py`). The issue is the threshold is too conservative. Change it from 
whatever it is now to 6 steps (was probably 10+).

---

## Change 3 — BLUE Coordinated Sweep Protocol

**File:** `cipher/agents/blue/threat_hunter.py` and `cipher/agents/blue/surveillance.py`

**What to do:**

Currently BLUE agents act independently. When suspicion is high (>60%), they should 
coordinate a "sweep" — all four BLUE agents focus investigation on the same zone.

Add a simple coordination signal via the observation:
- In `_blue_observation_to_text()` in `base_agent.py`, add a field: 
  `"coordinated_sweep_zone": N` when `detection_confidence > 0.6`
- BLUE agents should then all target nodes in that zone

```python
# Add to _blue_observation_to_text in base_agent.py:
if obs.current_detection_confidence > 0.60:
    # Identify the zone with highest anomaly activity
    zone_scores = {}
    for anomaly in obs.anomaly_feed:
        z = anomaly.zone
        zone_scores[z] = zone_scores.get(z, 0) + anomaly.severity
    if zone_scores:
        hot_zone = max(zone_scores, key=zone_scores.get)
        lines.append(
            f"🚨 COORDINATED SWEEP ACTIVE: All BLUE agents focus on Zone {hot_zone}. "
            f"Prioritize investigate_node on Zone {hot_zone} nodes."
        )
```

---

## Change 4 — Adaptive Difficulty Based on Win Rate

**File:** `cipher/environment/scenario.py`

**What to do:**

The current difficulty escalates linearly. Instead, make it respond to the last 5-episode 
win rate:

```python
# In ScenarioGenerator.generate():
# Read last 5 rows of rewards_log.csv (without importing pandas — use csv module)
# If RED win rate in last 5 episodes > 0.70: increase difficulty by 0.05
# If RED win rate < 0.30: decrease difficulty by 0.05 (but min 0.20)
# This creates a true adversarial curriculum

def _get_recent_win_rate(self, n: int = 5) -> float:
    """Read last N episodes from rewards_log.csv and compute RED win rate."""
    import csv
    from pathlib import Path
    csv_path = Path("rewards_log.csv")
    if not csv_path.exists():
        return 0.5
    try:
        rows = list(csv.DictReader(open(csv_path)))
        recent = rows[-n:]
        if not recent:
            return 0.5
        red_wins = sum(1 for r in recent if float(r.get("red_total", 0)) > 0.5)
        return red_wins / len(recent)
    except Exception:
        return 0.5
```

---

## Change 5 — Anti-Honeypot Intelligence for RED

**File:** `cipher/agents/base_agent.py`

**What to do:**

Track which nodes caused suspicion spikes and add them to a "danger map" that persists 
across the episode. Currently `_nodes_with_suspicion_spikes` exists in the planner but 
isn't shared with other RED agents.

Add a **shared RED coordination object** (a simple dict in the dead drop or a module-level 
shared state dict) that all 4 RED agents read/write:

```python
# In cipher/agents/red/__init__.py (or a new cipher/agents/red/coordination.py):
_RED_DANGER_MAP: dict[int, float] = {}  # node_id -> danger_score

def mark_danger(node_id: int, score: float) -> None:
    _RED_DANGER_MAP[node_id] = max(_RED_DANGER_MAP.get(node_id, 0.0), score)

def get_danger_score(node_id: int) -> float:
    return _RED_DANGER_MAP.get(node_id, 0.0)

def clear_danger_map() -> None:
    _RED_DANGER_MAP.clear()
```

When a RED agent gets a suspicion spike (>0.15 jump in one step), call `mark_danger(current_node, spike_size)`. When choosing the next move, avoid nodes with danger_score > 0.3.

---

## Checklist

- [ ] Fix `abort` guardrail — allow abort after 4+ consecutive waits
- [ ] Fix zone stall threshold in `planner.py` — reduce from current value to 6 steps
- [ ] Add zone stall warning to RED observation text in `base_agent.py`
- [ ] Add BLUE coordinated sweep signal to blue observation text
- [ ] Implement adaptive difficulty in `scenario.py`
- [ ] Add RED shared danger map (`cipher/agents/red/coordination.py`)
- [ ] Test: run 10 episodes and verify no 29-step perimeter loops
- [ ] Test: run 10 episodes and verify BLUE win rate > 40%

---

## Files You Will Touch

| File | Change |
|------|--------|
| `cipher/agents/base_agent.py` | Fix abort guardrail, add zone stall warning, add BLUE sweep signal |
| `cipher/agents/red/planner.py` | Reduce zone_stall threshold, boost zone-advance priority |
| `cipher/agents/red/coordination.py` | [NEW] Shared RED danger map |
| `cipher/environment/scenario.py` | Adaptive difficulty based on recent win rate |

---

## Expected Impact

After F.md:
- Episode average steps should **drop from 143s to ~60s** (no more perimeter loops)
- RED win rate should remain ~70% (already good), but with shorter, cleaner episodes
- BLUE win rate should increase from 30% to 40-50% (better coordination)
- The `abort` guardrail spam (7 rejects in episode 10) should disappear
