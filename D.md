# D.md — Neural Engine: Performance Optimization & Analytics
## Your job: Squeeze every drop of performance and provide "Winning" data

> **Run in parallel** with teammate doing C.md (Storytelling & HF).
> This task focuses on the technical "depth" and efficiency that wins hackathons.

---

## Context: Scalability is a key judging criterion
Currently, our simulation is limited by LLM latency and context window bloat. We need to optimize the "Inner Loop" and provide advanced analytics to prove our RL approach is working.

---

## Change 1 — "Token-Squeeze": Advanced Prompt Compression

**Objective:** Reduce costs and latency by 30-50% while improving agent focus.

**File:** `cipher/utils/llm_client.py` or `cipher/agents/base_agent.py`

**What to do:**
1. **Dynamic History Truncation:** Instead of sending the full `episode_log`, implement a `compress_history()` function that only keeps the last 5 steps in detail and summarizes the earlier steps into 1-2 sentences (e.g., "Steps 1-10: RED moved through Zone 1 successfully").
2. **Schema Stripping:** Remove the "reasoning" from the prompt examples when not needed, or use a more compact JSON format for input.

---

## Change 2 — "CIPHER Analytics": The Elo & Heatmap Suite

**Objective:** Prove that the LoRA models (from A.md) are actually better.

**File:** `cipher/dashboard/analytics.py` (Create new module)

**What to do:**
1. **Elo Rating System:** Implement a simple Elo calculation between "Local LoRA Team" and "NVIDIA NIM Team". Track this over multiple training runs.
2. **Detection Heatmaps:** Create a Plotly heatmap showing which nodes in the 50-node topology are "Death Traps" (high detection rate).
3. **Reward Curve Visualization:** A clean graph showing `Reward vs Step` for Red and Blue in the dashboard.

---

## Change 3 — "Self-Play" Data Pipeline (The Flywheel)

**Objective:** Automated data collection for the next generation of models.

**File:** `cipher/training/loop.py`

**What to do:**
1. **Failure Mining:** When RED loses an episode in `--live` mode, flag it. Save the *exact* state and observations into a `data/finetune/failure_cases.jsonl`.
2. **Success Mining:** When BLUE catches RED, save the BLUE actions.
3. This creates a dataset that we can use to "Fine-tune out the bugs" in the next hackathon sprint.

---

## Change 4 — Multi-Node Scaling (Concurrency)

**Objective:** Run 100 episodes in the time it takes to run 10.

**File:** `cipher/training/loop.py`

**What to do:**
1. Refactor the `TrainingLoop` to use `ProcessPoolExecutor` for the environment stepping and `ThreadPoolExecutor` for the LLM calls across *multiple episodes* simultaneously.
2. Ensure `rewards_log.csv` is thread-safe (use a `Lock` or a background writer thread).

---

## Change 5 — "Winning Metrics" Dashboard Overlay

**Objective:** A high-level summary that judges can understand in 5 seconds.

**File:** `cipher/dashboard/app.py`
**What to do:**
Create a top-bar banner with:
- **Total Exfiltrations:** XX
- **Mean Time to Detection:** XX steps
- **Training Efficiency:** +XX% (vs baseline)
- **Model Confidence:** High/Low

---

## Files You Will Touch (Summary)

| File | Change |
|------|--------|
| `cipher/utils/llm_client.py` | Add prompt compression logic |
| `cipher/dashboard/analytics.py` | [NEW] Elo and Heatmap logic |
| `cipher/training/loop.py` | Multi-episode concurrency & failure mining |
| `cipher/dashboard/app.py` | Add "Analytics" tab and top-bar summary |

---

## Checklist
- [x] Implement `compress_history` in LLM client
- [x] Create the `Analytics` tab in the dashboard
- [x] Build the Heatmap generator for the 50-node network
- [x] Verify `rewards_log.csv` thread-safety
- [x] Log "Failure Case" dataset for future training
- [x] Benchmark "Hybrid" vs "Live" speed and cost
