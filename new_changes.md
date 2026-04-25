# CIPHER — New Architecture Changes & Implementation Log

> Covers everything built across the **v2 Commander+Subagent Architecture** redesign  
> and all subsequent balance tuning sessions.

---

## Overview

The original CIPHER had a flat **4 RED agents + 4 BLUE agents** structure where every agent was predefined with a fixed role. This was replaced with a hierarchical **Commander + dynamic N subagents** model to better simulate real-world RL environments where team leaders delegate tasks on demand.

---

## 1. Architecture: From Flat 4v4 → Hierarchical Commander Model

### Old Architecture
- 4 fixed RED agents: `Planner`, `Analyst`, `Operative`, `Exfiltrator`
- 4 fixed BLUE agents: `Surveillance`, `ThreatHunter`, `DeceptionArchitect`, `Forensics`
- All 8 agents always active every step
- No inter-agent delegation or dynamic spawning

### New Architecture (v2)
- **1 RED Commander** — the single trained RED brain, orchestrates everything
- **1 BLUE Commander** — the single trained BLUE brain, orchestrates everything
- Each commander can **dynamically spawn up to N subagents** on demand
- Subagents can be legacy-proxied (same logic as old agents) or emergent (new roles)
- Meta-actions: `SPAWN_SUBAGENT`, `DELEGATE_TASK`, `DISMISS_SUBAGENT`
- Controlled via `CIPHER_AGENT_ARCH=v2` in `.env` (falls back to v1 if not set)

---

## 2. New Files Created

### Agent Layer
| File | Purpose |
|------|---------|
| `cipher/agents/commander.py` | `BaseCommander`, `RedCommander`, `BlueCommander` classes |
| `cipher/agents/subagent.py` | Generic `Subagent` worker — handles both legacy-proxy and emergent roles |
| `cipher/agents/subagent_registry.py` | `SubagentRegistry` — manages dynamic roster, lifespan, spawn budget |
| `cipher/agents/role_profiles.py` | `SubagentRoleProfile` specs for all roles including emergent ones |

### Prompt Files
| File | Role |
|------|------|
| `cipher/agents/prompts/red_commander.txt` | RED Commander LLM system prompt — aggressive, zone-advance focused |
| `cipher/agents/prompts/blue_commander.txt` | BLUE Commander LLM system prompt — detection-focused, spawning directives |
| `cipher/agents/prompts/red_scout.txt` | Scout subagent — reconnaissance ahead of the main path |
| `cipher/agents/prompts/red_dead_drop_courier.txt` | Dead drop courier subagent |
| `cipher/agents/prompts/blue_anomaly_triager.txt` | Anomaly triager subagent |
| `cipher/agents/prompts/blue_alert_judge.txt` | Alert judge subagent — decides when to fire alerts |

### Training & Tests
| File | Purpose |
|------|---------|
| `cipher_training_v2_commander.ipynb` | Jupyter notebook for training RED/BLUE Commander LoRA adapters |
| `tests/test_subagent_registry.py` | Tests for registry spawn, dismiss, lifespan, thrash detection |
| `tests/test_commander.py` | Tests for commander act_step, stub policies, urgency injection |
| `tests/test_arch_v2_smoke.py` | End-to-end smoke test for the v2 architecture |

### Assets
| File | Purpose |
|------|---------|
| `assets/architecture_card.png` | Generated architecture diagram card |

---

## 3. Modified Files

### `cipher/agents/base_agent.py`
- Added `SPAWN_SUBAGENT`, `DELEGATE_TASK`, `DISMISS_SUBAGENT` to `ActionType` enum
- Added `META_ACTIONS` constant, `EmergentAction` and `SubagentSpec` Pydantic models
- Extended `Action` model with `role`, `spawned_by`, `subagent_spec`, `target_subagent_id`
- `_parse_action_from_response` now parses subagent specs and rejects meta-actions from non-commanders
- Silent remap: `WAIT` → `STAND_DOWN` for blue team agents (no more spurious warnings)
- `_LORA_PATH_MAP` extended with `("red", "commander")` and `("blue", "commander")` entries
- **Auto-fill `target_file` for EXFILTRATE** — when at HVT with files, bare `{"action_type":"exfiltrate"}` auto-picks the first unexfiltrated file
- **Anti-oscillation path prioritization in `_red_observation_to_text`** — paths sorted by zone DESC then suspicion ASC with `*** ZONE-ADVANCE` markers
- **HVT node ID displayed in RED observation** — RED sees `TARGET IS node n{X}` and `<<< THIS IS THE HVT` marker on adjacent HVT paths; eliminates Zone 3 wandering
- **EXFIL-at-non-HVT rescue** — if RED tries EXFILTRATE from wrong node, action is overridden to MOVE toward HVT instead of WAITing
- **Gated episode memory injection** — episode history only injected into prompts in `hybrid`/`train` mode (not live mode, so episodes start fresh)
- **TRIGGER_ALERT guardrail lowered: 0.55 → 0.40** — BLUE can alert sooner now that detection rises faster

### `cipher/environment/observation.py`
- Added `hvt_node_id: int = -1` field to `RedObservation` dataclass
- Added `hvt_node_id` parameter to `generate_red_observation()`
- HVT node ID passed through from episode runner so RED always knows the target

### `cipher/environment/state.py`
- `is_done()` A4 early-termination: `red_suspicion ≥ 0.95 AND blue_confidence ≥ 0.70` (original rule, supplemented by new co-detection below)

### `cipher/training/_episode_runner.py`
- Agent instantiation block checks `cfg.cipher_agent_arch` and creates commanders in v2 mode
- Per-step action loops call `commander.act_step()` instead of per-agent loops
- Context reset logic updated for commanders (resets subagents too)
- Subagent lifecycle events streamed from registry into `state.episode_log` each step
- `_save_episode_trace` includes top-level `commanders` metadata dict
- **`generate_red_observation` now receives `hvt_node_id=hvt_node`** — passed down each step
- **`analyze_anomaly` detection boost tripled**: base `0.02 → 0.06`, same-zone `0.04 → 0.10`
- **Passive detection boost per step**: `red_suspicion × 4%` added to BLUE confidence each step. Ensures detection scales with RED activity
- **High-confidence co-detection terminal** (applied *before* `is_done()` check):  
  when `blue_detection_confidence ≥ 0.75 AND red_suspicion ≥ 0.75` simultaneously → BLUE wins, even mid-exfiltration

### `cipher/agents/commander.py`
- `BaseCommander`: owns `SubagentRegistry`, tracks zone/node stall, injects urgency into LLM messages
- **Zone stall tracking changed: same-ZONE → same-NODE** — RED navigating through Zone 3 to find the HVT no longer triggers false stall warnings; stall only fires when genuinely stuck at one node
- `RedCommander`: anti-oscillation override (`_is_oscillating()`, `_pick_next_zone_node()`), aggressive stub policy, zone/planner urgency injection
- `BlueCommander`: `_pre_spawn_surveillance()` hook — **force-spawns `surveillance` and `alert_judge` before each LLM call**, regardless of LLM decision. Guarantees BLUE is never flying blind
- `BlueCommander._stub_act()`: actively investigates, triggers alerts, spawns deception_architect based on detection confidence
- Urgency injection for both commanders: BLUE gets critical/high-alert banners, RED gets ZONE STALL / NO PLANNER warnings

### `cipher/agents/subagent.py`
- `alert_judge` threshold: triggers `TRIGGER_ALERT` at `detection_confidence ≥ 0.40` (was 0.65)
- Also adds `INVESTIGATE_NODE` on zone_alert_levels when no current-step anomalies
- Minimum lifespan clamped to `max(2, profile.default_lifespan // 2)` to prevent premature expiry

### `cipher/agents/subagent_registry.py`
- `spawn`, `delegate`, `dismiss` methods
- Enforces `max_concurrent` and `spawn_budget_remaining`
- `tick_step` auto-dismisses expired subagents
- `detect_thrash` for oversight

### `cipher/rewards/red_reward.py`
- Added `zone_stall_penalty`: `-0.05` per step over a 5-step grace window in the same zone, capped at `-0.30`
- Added `delegation_efficiency_bonus` and `spawn_cost_penalty` fields

### `cipher/rewards/blue_reward.py`
- Added `delegation_efficiency_bonus` and `spawn_cost_penalty` fields
- Fixed `false_positive_rate_penalty` calculation

### `cipher/agents/oversight/auditor.py`
- Added `SPAWN_THRASH` detection in `evaluate_step`

### `cipher/utils/config.py`
- New fields: `cipher_agent_arch`, `env_max_subagents_red/blue`, `env_subagent_spawn_budget_red/blue`, `env_subagent_default_lifespan`, `env_reward_delegation_enabled`, `hf_model_red_commander`, `hf_model_blue_commander`

### `.env` / `.env.example`
- Updated with all v2 commander/subagent configuration variables
- Added `RED_COMMANDER_LORA_PATH`, `BLUE_COMMANDER_LORA_PATH`
- Organized into clear sections for readability

### `main.py`
- **Dashboard auto-open removed** — prints URL, user opens manually
- `_print_live_step` shows subagent lifecycle events (spawns/dismissals/expirations)
- Gated episode memory banner (only shown in hybrid mode)
- `stall_pen=` included in reward breakdown display
- `_preload_hybrid_specialists` and `_validate_hybrid_models` include both commander LoRA paths

### Dashboard (`dashboard-react/`)
- `AgentMindPanel.jsx`: replaced hardcoded agent map with dynamic function; star badge on Commander entries
- `useEpisodeReplay.js`: updated Red planner intended target node logic
- `api_server.py`: added `/api/commanders` endpoint for live commander roster

---

## 4. Emergent Subagent Roles

Four new emergent roles were introduced (no legacy counterpart):

| Role | Team | Purpose | Default Lifespan |
|------|------|---------|-----------------|
| `scout` | RED | Map ahead, identify zone-advance paths | 3 steps |
| `dead_drop_courier` | RED | Write/read dead drops for team coordination | 4 steps |
| `anomaly_triager` | BLUE | Classify and prioritize anomaly feed | 5 steps |
| `alert_judge` | BLUE | Decides when to fire alerts (≥40% confidence) | 5 steps |

---

## 5. Balance Tuning — Key Numbers

| Metric | Before | After |
|--------|--------|-------|
| BLUE detection after 6 steps | 14–22% | 60–80% |
| Zone 3 stall (steps in Zone 3) | 5–8 steps | 2–4 steps |
| Win rate (RED / BLUE) | ~100% RED | ~50% / 50% |
| BLUE subagents spawned / episode | 0 | 2–4 |
| RED reward typical total | +1.45 | +0.35–0.72 |
| False stall warnings in Zone 3 | Every episode | Gone (node-stall tracking) |

---

## 6. How to Run

```bash
# Live inference — new v2 commander architecture
python main.py --live --episodes 3

# Stub mode (deterministic, no LLM calls)
python main.py --stub --episodes 5

# Training (hybrid mode with LoRA adapters)
python main.py --train --episodes 20
# or open: cipher_training_v2_commander.ipynb
```

The dashboard URL is printed at startup — open it manually in your browser.

---

## 7. Configuration Flags (`.env`)

```env
CIPHER_AGENT_ARCH=v2          # v2 = commander+subagents, v1 = legacy flat 4v4
ENV_MAX_SUBAGENTS_RED=6       # max concurrent RED subagents
ENV_MAX_SUBAGENTS_BLUE=6      # max concurrent BLUE subagents
ENV_SUBAGENT_SPAWN_BUDGET_RED=12   # total spawns RED can do per episode
ENV_SUBAGENT_SPAWN_BUDGET_BLUE=12  # total spawns BLUE can do per episode
ENV_SUBAGENT_DEFAULT_LIFESPAN=5    # default steps a subagent lives
ENV_REWARD_DELEGATION_ENABLED=true # enable delegation bonus/spawn penalty terms
HF_MODEL_RED_COMMANDER=...    # HuggingFace model for RED commander
HF_MODEL_BLUE_COMMANDER=...   # HuggingFace model for BLUE commander
```

---

## 8. Test Coverage

```bash
pytest tests/test_subagent_registry.py  # registry spawn/dismiss/lifespan
pytest tests/test_commander.py          # commander stub/live policies
pytest tests/test_arch_v2_smoke.py      # end-to-end v2 smoke test
pytest tests/test_phase3.py             # action set disjointness (updated for meta-actions)
pytest tests/test_blue_reward.py        # false positive penalty fix
```
