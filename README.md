# CIPHER

Adversarial multi-agent environment where RED infiltrates and exfiltrates while BLUE detects, misdirects, and traps.

## Current Build Status

| Phase | What | Status |
|-------|------|--------|
| 1 — Skeleton | Project structure, 8 stub agents, dead drops | ✅ Complete |
| 2 — Environment | 50-node zone-based network, suspicion mechanics | ✅ Complete |
| 3 — LLM Integration | NVIDIA NIM, real agent reasoning | ✅ Complete |
| 4 — Agent Prompting | 8 specialized prompt templates, action parsing | ✅ Complete |
| 5 — Trap Layer | FalseTrail, HoneypotPoison, DeadDropTamper | ✅ Complete |
| 6 — Reward Functions | Full continuous rewards, reward_logger.py | ✅ Complete |
| 7 — Oversight Agent | OversightAuditor (9th agent), fleet bonus | ✅ Complete |
| 8 — Training Loop | Self-play, reward curves | 🔵 Next |
| 12 — Dashboard | Episode replay visualization | 🟡 In Progress |
| 14 — HuggingFace | Awaiting compute credits | ⬜ Not Started |

Tests passing: **~182** | Reward logging: **rewards_log.csv** | Fleet verdict: per-episode | Modes: `LLM_MODE=stub` (free) / `LLM_MODE=live` (NVIDIA)

## Project Layout

- `cipher/environment/graph.py` — enterprise topology generation.
- `cipher/environment/observation.py` — RED/BLUE observation asymmetry engine.
- `cipher/environment/state.py` — ground-truth episode state.
- `cipher/environment/traps.py` — Phase 5 trap registry and effect engine.
- `cipher/memory/dead_drop.py` — MEMENTO dead drop schema and vault.
- `cipher/agents/` — 8 role-specific RED/BLUE agents and prompts.
- `cipher/training/_episode_runner.py` — single-episode execution loop.

## Setup

### Option A: Conda (reproducible)

```bash
conda env create -f environment.yml
conda activate cipher
```

### Option B: pip

```bash
python -m venv .venv
. .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Quick Start

```bash
# Stub mode (free, no API calls)
python main.py

# Live mode (NVIDIA API)
python main.py --live

# Full test suite
pytest tests/ -v --tb=short

# Training loop (10 episodes, stub mode)
LLM_MODE=stub python -m cipher.training.loop --episodes 10

# Disable trace saving (on by default in main.py)
python main.py --no-trace
```

## Run

### Stub mode (no API calls)

```bash
python main.py
```

### Live mode (NVIDIA API)

```bash
python main.py --live
```

## Verification Commands (Phase 5)

```bash
# 1) Full suite
pytest tests/ -v --tb=short

# 2) Trap registry init
python -c "import os; os.environ['LLM_MODE']='stub'; from cipher.environment.traps import TrapRegistry; from cipher.utils.config import config; r=TrapRegistry(config); assert r.red_trap_budget==config.env_trap_budget_red; assert r.blue_trap_budget==config.env_trap_budget_blue; assert r.active_red_traps==[]; assert r.active_blue_traps==[]; print('TrapRegistry init: PASSED')"

# 3) Budget enforcement
python -c "import os; os.environ['LLM_MODE']='stub'; from cipher.environment.traps import TrapRegistry, RedTrapType; from cipher.utils.config import config; r=TrapRegistry(config); [r.place_red_trap(RedTrapType.FALSE_TRAIL,'operative',i,i,{}) for i in range(config.env_trap_budget_red)]; ok,_=r.place_red_trap(RedTrapType.FALSE_TRAIL,'operative',99,99,{}); assert not ok; print('Budget enforcement: PASSED')"

# 4) Main still runs and emits trap events
python main.py
```

## Notes

- Keep `LLM_MODE=stub` for offline testing.
- Trap events are logged as `TRAP EVENT:` lines in episode output.
- Dead-drop tampering intentionally preserves old integrity hash so RED can detect corruption via `verify() == False`.
- `main.py` now saves episode traces by default to support Phase 12 dashboard replay testing.
- Use `--no-trace` only when you explicitly want to skip trace file generation.
