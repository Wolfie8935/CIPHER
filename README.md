# CIPHER — Adversarial Multi-Agent RL Environment

**OpenEnv Hackathon | Theme 1: Multi-Agent Interactions**

CIPHER is a fully adversarial, asymmetric, partially observable multi-agent RL environment. A **RED team** of 4 LLM-powered agents infiltrates a procedurally generated 50-node enterprise network to steal a classified file. A **BLUE team** of 4 LLM agents defends using honeypots, dead drop tampering, and false escalations. An independent **Oversight Auditor** judges every episode.

```
python main.py          # Watch the competition — no setup needed
```

---

## What CIPHER Demonstrates

| Capability | How |
|-----------|-----|
| Multi-agent coordination | 8 simultaneous agents with conflicting objectives |
| Theory-of-mind reasoning | Agents model adversary beliefs via dead drops and deception |
| Scalable oversight | Oversight Auditor monitors all 8 agents, detects reward hacking |
| Uncapped rewards | Complexity multiplier scales with zone traversal (no ceiling) |
| Self-improvement | Prompt evolution every 10 episodes based on reward heuristics |
| Trained specialist | RED Planner fine-tuned via Unsloth GRPO (LoRA adapter included) |

---

## Build Status

| Phase | What | Status |
|-------|------|--------|
| 1 | Foundation, config, logging | ✅ |
| 2 | 50-node network, zones, state, observations | ✅ |
| 3 | All 8 LLM agents, stub/live/hybrid modes | ✅ |
| 4 | Scenario generation, auto-difficulty escalation | ✅ |
| 5 | Full trap system (12 traps, budget enforcement) | ✅ |
| 6 | Reward functions, RewardLogger, variance verified | ✅ |
| 7 | Oversight Auditor, fleet verdicts, reward hacking flags | ✅ |
| 8 | OpenEnv API compliance wrapper (CIPHEREnv) | ✅ |
| 9 | Prompt evolution learning loop | ✅ |
| 10 | Improvement metrics + Dashboard Learning Curve tab | ✅ |
| 11 | Google Colab Unsloth GRPO training notebook | ✅ |
| 12 | Replay Dashboard (port 8050, episode trace scrubber) | ✅ |
| 13 | Live Training Dashboard (unified, 6 tabs, auto-poll) | ✅ |

**Tests: 290 passing, 0 failing**

---

## Setup

```bash
# Option A: pip
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate cipher
```

---

## Run

```bash
# Single episode, stub mode (no API key needed, instant)
python main.py

# 5-episode competition with standings
python main.py --episodes 5

# Use trained RED Planner LoRA (requires "red trained/" folder)
python main.py --hybrid

# All agents use NVIDIA NIM LLMs (requires NVIDIA_API_KEY)
python main.py --live

# Training loop (10 episodes, writes rewards_log.csv)
python main.py --train

# Verify all 23 module imports resolve
python main.py --check
```

See `commands.md` for the full command reference.

---

## Dashboard

```bash
python -m cipher.dashboard.app
# Open: http://localhost:8050
```

Toggle between **Episode Replay** (step through saved traces) and **Live Training** (6 real-time tabs: Rewards, Dead Drops, Network Map, Oversight, Difficulty, Learning Curve).

Recommended judge demo:

```bash
# Terminal 1 — run training
python main.py --train --train-episodes 50

# Terminal 2 — watch it live
python -m cipher.dashboard.app
```

---

## OpenEnv Compliance

```bash
python verify_openenv.py   # CIPHEREnv reset/step/render/metadata verified
```

`CIPHEREnv` inherits from `openenv.env.env.Env`. One `env.step()` = one full episode. The trained agent is the RED Planner — its first action drives strategic direction for the episode.

---

## Hybrid Mode (Trained LoRA Specialist)

The `red trained/cipher-red-planner/` folder contains a Llama-3.2-1B LoRA adapter fine-tuned via Unsloth GRPO on 50 episodes of self-play. In hybrid mode, the RED Planner uses this local model while all other agents use NVIDIA NIM.

```bash
python main.py --hybrid   # loads adapter automatically
```

Training notebook: `CIPHER_Training_Colab.ipynb` (runs free on Google Colab T4).

---

## Tests

```bash
python -m pytest tests/         # 290 tests, ~6s
python -m pytest tests/ -v      # verbose
python -m pytest tests/ -k "reward"  # filter by keyword
```

---

## Project Layout

```
cipher/
├── agents/           8 role-specific agents (RED: Planner/Analyst/Operative/Exfiltrator,
│   └── prompts/      BLUE: Surveillance/ThreatHunter/DeceptionArchitect/Forensics)
├── environment/      Graph, state, observations, scenario generation, traps
├── rewards/          RED/BLUE/Oversight reward functions + RewardLogger
├── memory/           Dead drop vault (RED inter-agent memory)
├── training/         Episode runner, training loop, prompt evolver, improvement analyzer
├── dashboard/        Unified Dash app (replay + live)
├── utils/            Config, logger, LLM client, LoRA client, mode toggle
env_wrapper.py        OpenEnv-compliant CIPHEREnv
main.py               Competition display CLI
commands.md           Full command reference
```
