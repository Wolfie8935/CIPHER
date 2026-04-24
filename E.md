# E.md — Evaluation Suite & Model Comparison Report
## Your job: Prove the LoRA models are better with hard numbers

> **Start after A.md (LoRA training) is complete.** You will need the 4 adapter folders:
> `red trained/cipher-red-planner-v2/`, `red trained/cipher-red-analyst-v1/`,
> `blue trained/cipher-blue-surveillance-v1/`, `blue trained/cipher-blue-threat-hunter-v1/`
>
> **Run in parallel** with teammates doing F.md and G.md.
> This task generates the "proof" that our RL approach works — the most important hackathon artifact.

---

## Context: Why Evaluation Matters

Judges will ask: "How do you know your trained models are better?" Right now we cannot answer that quantitatively. After A.md trains the LoRA adapters, E.md creates the systematic comparison:

- **Baseline**: All 8 agents use NVIDIA NIM (zero-shot)
- **Hybrid**: RED Planner + RED Analyst use LoRA; rest use NIM
- **Full LoRA**: All 4 RED + 2 BLUE specialists use LoRA adapters

We measure: RED win rate, mean steps per episode, mean exfil reward, mean detection rate.

---

## Change 1 — Evaluation Runner Script

**File:** `cipher/training/eval_runner.py` (create new)

**What to do:**

Create a script that runs **N episodes in each mode** and collects results into a comparison table:

```python
# Usage:
# python -m cipher.training.eval_runner --episodes 20 --modes stub hybrid
# Outputs: eval_results/comparison_YYYYMMDD_HHMMSS.json + .csv
```

**Implementation:**

```python
from cipher.training._episode_runner import run_episode
from cipher.environment.graph import generate_enterprise_graph
from cipher.environment.scenario import ScenarioGenerator
from cipher.utils.config import config

MODES = ["stub", "live", "hybrid"]  # live = NIM only, hybrid = NIM + LoRA

def run_eval(n_episodes: int, modes: list[str]) -> dict:
    """Run n_episodes per mode, collect metrics, return comparison dict."""
    results = {}
    sg = ScenarioGenerator()
    for mode in modes:
        os.environ["LLM_MODE"] = mode
        mode_results = []
        for ep in range(1, n_episodes + 1):
            scenario = sg.generate(ep)
            graph = generate_enterprise_graph(
                n_nodes=config.env_graph_size,
                honeypot_density=config.env_honeypot_density,
                seed=scenario.episode_seed,
            )
            result = run_episode(scenario=scenario, graph=graph, cfg=config,
                                 episode_number=ep, max_steps=30, verbose=False)
            mode_results.append({
                "episode": ep,
                "red_total": result["red_reward"].total,
                "blue_total": result["blue_reward"].total,
                "terminal_reason": getattr(result["state"], "terminal_reason", "max_steps"),
                "steps": getattr(result["state"], "step", 0),
                "exfil_count": len(getattr(result["state"], "exfiltrated_files", [])),
            })
        results[mode] = mode_results
    return results
```

**Output format** — save to `eval_results/comparison_TIMESTAMP.json`:
```json
{
  "modes": {"stub": [...], "hybrid": [...]},
  "summary": {
    "stub":   {"red_win_rate": 0.45, "avg_red": 0.31, "avg_steps": 18},
    "hybrid": {"red_win_rate": 0.72, "avg_red": 0.89, "avg_steps": 11}
  }
}
```

---

## Change 2 — Comparison Dashboard Panel

**File:** `cipher/dashboard/analytics.py` (extend existing)

**What to do:**

Add `build_model_comparison_chart(eval_json_path: str) -> go.Figure` that:
1. Reads the eval JSON produced in Change 1
2. Shows side-by-side grouped bar chart: RED win rate per mode
3. Shows a second chart: Mean episode length (shorter = RED more efficient)
4. Shows a third chart: Mean RED reward distribution (violin or box plot)

The analytics tab already has a placeholder — wire these charts into the "Analytics ★" tab in `live.py`.

---

## Change 3 — Convergence Curve

**File:** `cipher/dashboard/analytics.py` (extend)

Add `build_convergence_curve(rewards_csv_path: str) -> go.Figure` that plots:
- X-axis: Training episode number
- Y-axis: RED reward (rolling 10-episode average)
- Overlay: vertical lines where LoRA checkpoint was created
- Goal: Show the "hockey stick" — rewards improving after LoRA loading

---

## Change 4 — "Proof of Learning" Report Generator

**File:** `cipher/utils/report_gen.py` (create new)

Generate a Markdown/HTML report automatically:

```python
def generate_proof_of_learning_report(eval_json: dict) -> str:
    """Return a formatted Markdown string suitable for pasting into the HF card."""
    ...
```

**Report should include:**
1. Executive summary: "Hybrid mode achieves XX% higher RED win rate vs baseline"
2. Table: Metric | Baseline | Hybrid | Improvement
3. Interpretation: why each metric matters for multi-agent RL
4. Methodology: episode count, difficulty, graph size

---

## Checklist
- [ ] Create `cipher/training/eval_runner.py` with mode comparison
- [ ] Create `eval_results/` directory and output format
- [ ] Add `build_model_comparison_chart()` to analytics.py
- [ ] Add `build_convergence_curve()` to analytics.py
- [ ] Wire new charts into the "Analytics ★" tab
- [ ] Create `cipher/utils/report_gen.py` proof-of-learning report
- [ ] Add `--eval` flag to `main.py` that runs the eval suite

---

## Files You Will Touch

| File | Change |
|------|--------|
| `cipher/training/eval_runner.py` | [NEW] Mode comparison runner |
| `cipher/dashboard/analytics.py` | Add comparison + convergence charts |
| `cipher/dashboard/live.py` | Wire new charts into Analytics tab |
| `cipher/utils/report_gen.py` | [NEW] Proof-of-learning Markdown report |
| `main.py` | Add `--eval N` flag |

---

## Output for Judges

After running `python main.py --eval 20`, the dashboard "Analytics ★" tab will show:
- A bar chart: "Hybrid LoRA achieves +58% RED win rate vs NIM baseline"
- A convergence curve showing reward improvement over training episodes
- A downloadable PDF/HTML report

This is the **single most important artifact** for winning the hackathon.
