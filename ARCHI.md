# CIPHER — Architecture & Build Status

> Last updated: Phase 7 complete (per-step oversight flags + penalties)

## Build Status

```mermaid
flowchart TD
    classDef done fill:#1a7a1a,stroke:#2ecc40,color:#fff
    classDef inprogress fill:#7a5a00,stroke:#ffcc00,color:#fff
    classDef next fill:#1a3a5c,stroke:#4488ff,color:#fff
    classDef future fill:#2a2a2a,stroke:#666,color:#aaa

    P1["✅ Phase 1\nProject Skeleton\n87+ tests passing"]:::done
    P2["✅ Phase 2\nEnterprise Network\n50 nodes, 4 zones"]:::done
    P3["✅ Phase 3\nLLM Integration\nNVIDIA NIM live"]:::done
    P4["✅ Phase 4\nAgent Prompting\n8 specialized agents"]:::done
    P5["✅ Phase 5\nTrap Layer\nDeception mechanics"]:::done
    P6["✅ Phase 6\nReward Functions\nFull implementation"]:::done
    P7["✅ Phase 7\nOversight Agent\nFleet AI bonus"]:::done
    P8["🔵 Phase 8\nTraining Loop\nSelf-play + curves"]:::next
    P9["🔵 Phase 9\nForensics Agent\nOp graph reconstruction"]:::next
    P10["🔵 Phase 10\nAuto-Escalation\nScenario generator"]:::next
    P11["🔵 Phase 11\nEvaluation Suite\nBaseline comparison"]:::next
    P12["🟡 Phase 12\nEpisode Replay\nDashboard — IN PROGRESS"]:::inprogress
    P13["⬜ Phase 13\nLive Dashboard\nReal-time training viz"]:::future
    P14["⬜ Phase 14\nHuggingFace\nNOT STARTED — awaiting credits"]:::future
    P15["⬜ Phase 15\nPolish & Demo\nPitch preparation"]:::future

    P1 --> P2 --> P3 --> P4 --> P5 --> P6
    P6 --> P7 --> P8 --> P9 --> P10 --> P11
    P8 --> P13
    P12 --> P13
    P11 --> P15
    P14 --> P15
```

## System Architecture

```mermaid
flowchart TB
    classDef env fill:#0d3349,stroke:#4488ff,color:#fff
    classDef red fill:#3d0000,stroke:#ff4444,color:#fff
    classDef blue fill:#00003d,stroke:#4444ff,color:#fff
    classDef memory fill:#2d2d00,stroke:#ffcc00,color:#fff
    classDef reward fill:#003d00,stroke:#44ff44,color:#fff
    classDef infra fill:#1a1a1a,stroke:#888,color:#ccc
    classDef oversight fill:#2d1a00,stroke:#ff8800,color:#fff

    subgraph ENV["🌐 Environment — Phase 2 ✅"]
        GRAPH["Enterprise Network\n50 nodes, 4 zones\nZone 0→3 topology"]
        STATE["Episode State\nSuspicion mechanics\nTerminal conditions"]
        OBS["Asymmetric Observations\nRED: honeypots masked\nBLUE: anomaly feed only"]
        TRAPS["Trap Registry — Phase 5 ✅\nFalseTrail, HoneypotPoison\nBreadcrumb, DeadDropTamper"]
        SCENARIO["Scenario Generator\nDifficulty scoring\nMission briefings"]
    end

    subgraph RED["🔴 RED TEAM — Phase 3/4 ✅"]
        PLANNER["Planner\nLong-horizon strategy\nAbort decisions"]
        ANALYST["Analyst\nBelief map\nRisk estimation"]
        OPERATIVE["Operative\nStealth movement\nTrap placement"]
        EXFIL["Exfiltrator\nTarget acquisition\nExit sequencing"]
    end

    subgraph BLUE["🔵 BLUE TEAM — Phase 3/4 ✅"]
        SURV["Surveillance\nAnomaly feed\nPattern detection"]
        HUNTER["Threat Hunter\nActive investigation\nHypothesis pursuit"]
        DECEIVER["Deception Architect\nHoneypot placement\nDrop tampering"]
        FORENSICS["Forensics\nOp graph reconstruction\nHandoff detection"]
    end

    subgraph MEMORY["📦 MEMENTO Layer — Phase 1/3 ✅"]
        VAULT["Dead Drop Vault\nSHA-256 integrity\nObfuscated paths"]
        DROPS["Operation Briefs\nJSON schema\nToken budget enforcement"]
    end

    subgraph REWARDS["📊 Rewards — Phase 6/7 ✅"]
        RRED["RED Reward\nexfil × stealth × memory\n× complexity + penalties"]
        RBLUE["BLUE Reward\ndetection × speed\n× honeypot − false_positives"]
        ROVER["Oversight Signal\nReward hacking detection\nCollusion monitoring"]
        RFLAG["OversightFlag Penalties\nper-step severity penalties\napplied to episode totals"]
        RLOG["Reward Logger\nrewards_log.csv\nfleet_verdict + fleet_judgment"]
    end

    subgraph OVERSIGHT["🟡 Oversight Agent — Phase 7 ✅"]
        AUDITOR["OversightAuditor\n9th LLM agent\nFleet bonus + verdict"]
        OFLAG["OversightFlag Engine\nper-step anomaly flags\npersisted in episode_log"]
    end

    subgraph TRAINING["⚙️ Training — Phase 8 🔵"]
        LOOP["Training Loop\nSelf-play episodes\nFew-shot injection"]
        ESCALATOR["Scenario Escalator\nAdaptive difficulty\nTargets last winner"]
    end

    subgraph DASHBOARD["📺 Dashboard"]
        D12["Episode Replay — Phase 12 🟡\n5-panel visualization\nStep slider + network map"]
        D13["Live Training — Phase 13 ⬜\nReal-time reward curves\nLive anomaly feed"]
    end

    subgraph LLM["🤖 LLM Backend — Phase 3 ✅"]
        NVIDIA["NVIDIA NIM\nllama-3.1-70b (strategic)\nllama-3.1-8b (tactical)"]
        HF["HuggingFace ⬜\nPhase 14 — not started\nawaiting compute credits"]
    end

    RED -->|actions| ENV
    BLUE -->|actions| ENV
    ENV -->|red_obs honeypot-masked| RED
    ENV -->|blue_obs anomaly-feed-only| BLUE
    RED -->|write drops| MEMORY
    MEMORY -->|read on reset| RED
    BLUE -->|tamper if found| MEMORY
    ENV --> REWARDS
    REWARDS -->|training signal| TRAINING
    TRAINING -->|policy update| LLM
    LLM -->|completions| RED
    LLM -->|completions| BLUE
    OVERSIGHT -->|audit both teams| REWARDS
    OVERSIGHT -->|emit per-step flags| ENV
    TRAINING -->|episode traces| DASHBOARD
    ENV -->|episode traces JSON| DASHBOARD
```

## Reward Formula Summary

```mermaid
flowchart LR
    classDef positive fill:#003d00,stroke:#44ff44,color:#fff
    classDef negative fill:#3d0000,stroke:#ff4444,color:#fff
    classDef neutral fill:#0d1a3d,stroke:#4488ff,color:#fff

    subgraph RED_REWARD["RED Reward"]
        E["exfil_completeness\n[0, 1]"]:::positive
        S["× (1 − detection)\n[0, 1]"]:::positive
        M["× memory_efficiency\n[0, 1]"]:::positive
        C["× complexity_mult\n[1, 3]"]:::positive
        AP["+ abort_penalty\n0 or −0.3"]:::negative
        HP["+ honeypot_penalty\n−0.2 per trigger"]:::negative
    end

    subgraph BLUE_REWARD["BLUE Reward"]
        DA["detection_accuracy\n[0, 1]"]:::positive
        RS["× response_speed\ne^(−step/50)"]:::positive
        HR["× (1 + honeypot_rate)\n[1, 2]"]:::positive
        GR["+ graph_reconstruction\n[0, 1] Jaccard"]:::positive
        FP["− false_positive_rate\n[0, 1]"]:::negative
    end
```

## .env Key Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_BACKEND` | `nvidia` | LLM provider — nvidia only until Phase 14 |
| `LLM_MODE` | `stub` | `stub`=random (free), `live`=real API calls |
| `ENV_GRAPH_SIZE` | `50` | Network node count |
| `ENV_CONTEXT_RESET_INTERVAL` | `40` | Steps between RED memory resets |
| `ENV_HONEYPOT_DENSITY` | `0.15` | Fraction of nodes that are honeypots |
| `ENV_DEAD_DROP_MAX_TOKENS` | `512` | Token budget per dead drop |
| `ENV_TRAP_BUDGET_RED` | `3` | RED trap placements per episode |
| `ENV_TRAP_BUDGET_BLUE` | `5` | BLUE trap placements per episode |

## Quick Commands

```bash
# Run demo episode (stub mode, free)
python main.py

# Run demo episode (live LLM — costs API credits)
LLM_MODE=live python main.py

# Run full test suite
pytest tests/ -v

# Run training loop (stub mode, 10 episodes)
LLM_MODE=stub python -m cipher.training.loop --episodes 10

# Open dashboard (after generating episode traces)
python -m cipher.dashboard.app
# → http://localhost:8050

# View reward curves
cat rewards_log.csv
```
