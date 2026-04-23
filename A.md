# CIPHER — Your Work (A): Simulation Engine & Reward System

> **Owner**: You (primary developer)
> **Parallel track**: This work is fully independent of B.md — no shared file conflicts.
> **Goal**: Make the CIPHER simulation smarter, more realistic, and produce richer reward signals.

---

## Track A1 — Fix Agent Intelligence (Prompt Engineering)

**Problem**: BLUE's `deception_architect` keeps outputting `stand_down` with reasoning like "placing honeypot at node X" — action/reason mismatch. RED agents abort too early due to rising suspicion even when BLUE hasn't detected them.

**Files**: `cipher/agents/blue/deception_architect.py`, `cipher/agents/red/planner.py`

### Tasks
- [ ] Open `cipher/agents/blue/deception_architect.py` — find the system prompt
- [ ] Add explicit constraint: **"If you intend to place a trap, your action MUST be `place_honeypot`, not `stand_down`"**
- [ ] Add JSON schema reminder with all valid BLUE actions listed
- [ ] Open `cipher/agents/red/planner.py` — find the abort threshold logic
- [ ] Make abort trigger only when suspicion > 0.80 AND blue_confidence > 0.60 (currently fires too early)
- [ ] Add a "suspicion decay" hint to RED's observation: if no BLUE alerts last 3 steps, suspicion should feel lower

**Verify**: Run `python main.py --live` — should see zero `Action-reason mismatch` warnings

---

## Track A2 — Smarter RED Path Strategy

**Problem**: RED always takes the same 0→13→37 path every run. The episode log shows RED's planner is ignoring the graph topology in its prompt.

**Files**: `cipher/environment/observation.py`, `cipher/agents/red/planner.py`

### Tasks
- [ ] Open `cipher/environment/observation.py` → `generate_red_observation()`
- [ ] Add `"available_paths"` field to the RED observation dict: list of `{node, zone, hostname, suspicion_delta}` for all neighbors of `red_current_node`
- [ ] Add `"honeypot_nodes_nearby"` field: list of known honeypot node IDs adjacent to current node
- [ ] In `cipher/agents/red/planner.py`, update the system prompt to say:
  - "Choose the neighbor with lowest `suspicion_delta` when stealth is priority"
  - "Avoid nodes listed in `honeypot_nodes_nearby`"
- [ ] Verify RED takes different paths across 3 stub runs: `python main.py` (3 times)

---

## Track A3 — Reward Function Tuning

**Problem**: RED gets rewarded even after aborting (tactical withdrawal still earns partial rewards). BLUE doesn't get enough credit for successful honeypot traps.

**Files**: `cipher/rewards/red_reward.py`, `cipher/rewards/blue_reward.py`

### Tasks
- [ ] Open `cipher/rewards/red_reward.py`
- [ ] Find `abort_penalty` — increase it from current value to `-0.4` (currently too lenient)
- [ ] Add a `zone_progress_bonus`: `+0.05` per unique zone crossed (incentivizes forward movement)
- [ ] Open `cipher/rewards/blue_reward.py`
- [ ] Find `honeypot_rate` calculation — increase weight from `×1.0` to `×1.5`
- [ ] Add `trap_accuracy_bonus`: `+0.1` for each trap that actually caught RED (honeypot triggered AND RED was at that node)
- [ ] Run `python -m pytest tests/test_rewards*.py -v` to verify reward tests still pass

---

## Track A4 — Episode Termination Logic

**Problem**: Episodes end at `max_steps` even when RED is stuck in Zone 1 with 100% suspicion and no path forward. This wastes API budget on meaningless steps.

**Files**: `cipher/training/_episode_runner.py`, `cipher/environment/state.py`

### Tasks
- [ ] Open `cipher/environment/state.py` → `is_done()` method
- [ ] Add early termination: if `red_suspicion_score >= 0.95` AND `blue_detection_confidence >= 0.70`, auto-terminate with `terminal_reason = "detected"`
- [ ] Add stall detection: if RED has been on the same node for 4+ consecutive steps, auto-terminate with `terminal_reason = "stalled"`
- [ ] Track `red_steps_on_current_node` counter in `EpisodeState`
- [ ] Increment it in `record_movement()` — reset to 0 on move, increment on stay
- [ ] Run `python -m pytest tests/ -x -q` — all 290 tests should still pass

---

## Track A5 — Hybrid Mode Completion

**Problem**: `python main.py --hybrid` currently uses OpenRouter for ALL agents (same as `--live`). The RED Planner should route to a local model.

**Files**: `cipher/utils/llm_client.py`, `.env`, `cipher/utils/llm_mode.py`

### Tasks
- [ ] Open `cipher/utils/llm_mode.py` — confirm `is_hybrid_mode()` returns True when `LLM_MODE=hybrid`
- [ ] Open `cipher/utils/llm_client.py` → `_LOCAL_KEYS_IN_HYBRID` set (line ~27)
- [ ] Confirm it contains `"nvidia_model_red_planner"` — if not, add it
- [ ] In `.env`, ensure `LOCAL_MODEL_URL` points to your local LM Studio / Ollama instance
- [ ] Set `LOCAL_MODEL_NAME=llama3.2` (or whatever model you have locally)
- [ ] Test: `python main.py --hybrid` — RED Planner should use local, others use OpenRouter
- [ ] Confirm in logs: "LLMClient initialized: backend=hybrid"

---

## Verification Checklist

```bash
# All tests pass
python -m pytest tests/ -x -q

# Stub mode works fast (< 2s)
python main.py

# Live mode shows step ticker
python main.py --live

# No action-reason mismatch warnings
python main.py --live 2>&1 | findstr "mismatch"
```