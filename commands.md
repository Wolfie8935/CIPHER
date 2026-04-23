# CIPHER — Command Reference

## Quick Start

```bash
# Single episode (stub mode — instant, no GPU required)
python main.py

# 5-episode competition with standings table
python main.py --episodes 5

# All agents use NVIDIA NIM LLM (requires NVIDIA_API_KEY env var)
python main.py --live

# RED Planner uses trained LoRA specialist (requires "red trained/" folder)
python main.py --hybrid

# Longer episodes (more steps per episode)
python main.py --steps 30

# Show all agent debug logs (verbose)
python main.py --debug

# Training loop (default 10 episodes, writes rewards_log.csv)
python main.py --train

# Training loop with custom episode count
python main.py --train --train-episodes 50

# Verify all imports resolve correctly
python main.py --check
```

## Dashboard

```bash
# Launch the unified Replay + Live Training dashboard (port 8050)
python -m cipher.dashboard.app

# Then open: http://localhost:8050
# - Toggle between "Episode Replay" and "Live Training" modes
# - Replay mode: load episode trace JSON files, scrub through step by step
# - Live mode: 6 tabs auto-polling every 2s (Rewards, Dead Drops, Network Map, Oversight, Difficulty, Learning Curve)
```

## Tests

```bash
# Run all 290 tests
python -m pytest tests/

# Run with verbose output
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_phase8.py -v

# Run only tests matching a keyword
python -m pytest tests/ -k "reward"

# Stop on first failure
python -m pytest tests/ -x
```

## OpenEnv Compliance Verification

```bash
# Verify CIPHEREnv passes OpenEnv API contract
python verify_openenv.py
```

## Training Notebooks

```bash
# CIPHER_Training_Colab.ipynb  — Full RED Planner LoRA fine-tuning via Unsloth + GRPO
# Run in Google Colab (free T4 GPU) or local GPU
# Output: "red trained/cipher-red-planner/" (adapter weights)

# CIPHER_Training_BLUE_Colab.ipynb — BLUE Surveillance fine-tuning (bonus)
# Mirrors RED notebook structure for BLUE team agent
```

## Environment Variables

```bash
# LLM backend mode (default: stub)
LLM_MODE=stub    # Fast random/heuristic policies (training)
LLM_MODE=live    # All 8 agents use NVIDIA NIM API
LLM_MODE=hybrid  # RED Planner uses local LoRA, others use NIM

# NVIDIA NIM API key (required for live/hybrid modes)
NVIDIA_API_KEY=nvapi-...

# Override LoRA adapter path (default: "red trained/cipher-red-planner")
LORA_ADAPTER_PATH=path/to/adapter
```

## Output Files

| File | Description |
|------|-------------|
| `rewards_log.csv` | Per-episode reward components, verdicts, flags |
| `training_state.json` | Current training progress (episode count, win rates) |
| `training_events.jsonl` | Structured log of all training events |
| `prompt_evolution_log.jsonl` | Prompt update history (when/why prompts evolved) |
| `episode_traces/episode_N.json` | Full step-by-step episode replay data |
| `cipher/agents/prompts/*.txt` | Agent system prompts (evolved by PromptEvolver) |
| `red trained/cipher-red-planner/` | Trained LoRA adapter weights |

## Architecture Overview

```
CIPHER Environment (50-node enterprise network, 4 zones)
│
├── RED Team (4 agents, goal: exfiltrate classified file)
│   ├── Planner    — strategic route planning (LoRA-trained in hybrid mode)
│   ├── Analyst    — network reconnaissance
│   ├── Operative  — lateral movement + trap deployment
│   └── Exfiltrator— data extraction
│
├── BLUE Team (4 agents, goal: detect and block intrusion)
│   ├── Surveillance      — anomaly detection
│   ├── ThreatHunter      — active threat hunting
│   ├── DeceptionArchitect— honeypot placement + false escalations
│   └── Forensics         — post-incident path reconstruction
│
└── Oversight Auditor (1 agent, independent judge)
    └── Issues verdicts: red_dominates / blue_dominates / contested / degenerate
```

## Modes Explained

| Mode | Speed | Requires | Use Case |
|------|-------|----------|----------|
| `stub` | ~0.01s/ep | Nothing | Training, demos, CI |
| `live` | ~8min/ep | NVIDIA_API_KEY | Live LLM demo |
| `hybrid` | ~4min/ep | NVIDIA_API_KEY + LoRA adapter | Showcase trained RED Planner |
