# CIPHER

Adversarial multi-agent environment where RED infiltrates and exfiltrates while BLUE detects, misdirects, and traps.

## Current Implementation Status (Problem Statement Audit Through Phase 5)

- **Phase 1 — Skeleton and wiring:** complete (`main.py` runs, core modules wired, episode loop active).
- **Phase 2 — Enterprise network + asymmetric observations:** complete (50-node graph, zones, role-asymmetric observations, suspicion/anomaly mechanics, tests).
- **Phase 3 — Dead drop / MEMENTO layer:** complete (dead drop schema, integrity hash verification, vault read/write and discovery path support).
- **Phase 4 — NVIDIA LLM integration + prompts:** complete (LLM client, model-key routing, 8 prompt templates, robust action parsing, retry/backoff logic).
- **Phase 5 — Trap layer + deception mechanics:** complete (trap registry, RED/BLUE trap types, budget enforcement, expiry/trigger handling, dead-drop tamper flow, episode-runner integration, runtime trap events).

### Test Status

- Full suite: `111 passed`.
- Includes Phase 5 trap suite (`tests/test_phase5.py`) with 24 tests.

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
