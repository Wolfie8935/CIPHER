# X.md — Submission Checklist

---

## DONE

### ✅ X1. `openenv.yaml` manifest
`openenv.yaml` created at root with all required fields: `name`, `version`, `entry_point`, `observation_type`, `action_type`, `reward_range`, `description`, agent roster, curriculum config, reward rubric, tags, quickstart. Verified by `check_submission.py`.

---

### ✅ X2. Plot assets committed to `assets/`
7 PNG files generated from `rewards_log.csv` (1,082 episodes) via `generate_plots.py`:
- `assets/baseline_vs_trained.png` — key comparison chart (0% → 70.5% win rate)
- `assets/reward_curves.png` — RED vs BLUE rolling reward + component breakdown
- `assets/elo_chart.png` — RED vs BLUE Elo over all episodes
- `assets/terminal_outcomes.png` — outcome distribution by training phase
- `assets/fleet_verdicts.png` — Oversight Auditor verdict shifts by phase
- `assets/win_rate_progression.png` — 50-ep rolling win rate across all 1,082 episodes
- `assets/architecture_card.png` — visual system architecture

---

### ✅ X3. README story rewrite
Full story-driven README with: Problem → Environment → Results (all 7 plots embedded with captions) → Why It Matters → Architecture → Reward Design → Training → Dashboard → Setup → Submission Links. Readable in 3–5 min.

---

### ✅ X5. Baseline vs trained comparison plot
`assets/baseline_vs_trained.png` shows three panels: mean RED reward per phase (−0.235 → +0.613), win rate per phase (0% → 70.5%), and full rolling reward curve across all 1,082 episodes with phase markers.

---

### ✅ X6. Plots embedded in README with captions
All 7 plots embedded in the Results section with one-line captions explaining what each shows. Verified by `check_submission.py` (90/90 pass).

---

### ✅ X8. Reserved MCP tool name check
CIPHER uses plain `Env` base class, not `MCPEnvironment` — no MCP tools exist at all. `check_submission.py` confirms clean. Not applicable.

---

### ✅ X11. Wandb run link
No Wandb run available. N/A — not penalised, just optional.

---

## TODO — Still Required Before Submission

### ⬜ X4. Verify training notebook connects to CIPHEREnv live
**Risk: MEDIUM-HIGH**

The guidelines require the training loop to call `CIPHEREnv.reset()` / `.step()` live — not just fine-tune on a static JSONL file. Both notebooks reference CIPHER (confirmed by `check_submission.py`) but it's not verified whether they run live episodes or only load `data/finetune/failure_cases.jsonl`.

**Action:** Open `CIPHER_Training_Colab.ipynb` and `cipher-training-red-v2.ipynb`. Confirm there is a cell that calls `env.reset()` and `env.step()` to generate training data before fine-tuning. If missing, add it.

---

### ⬜ X7. Mini-blog on HuggingFace + demo video
**Risk: HIGH — minimum requirement, non-negotiable**

- Write a short HuggingFace blog post (or <2 min YouTube video) covering: what the env does, what the agent learned, one result chart
- Link both from the README Submission Links table (currently has placeholders)

---

### ⬜ X9. HF Space URL in README
**Risk: HIGH — judges pull the environment from this URL**

After uploading to HuggingFace Spaces, replace the placeholder in the README Submission Links table with the real URL. The Dockerfile and `hf_app.py` are already fully configured for deployment.

---

### ⬜ X10. Colab notebook clean and runnable end-to-end
**Risk: MEDIUM — judges will try to re-run it**

Verify before submission:
- All `pip install` cells at the top (unsloth, trl, openenv)
- Repo clone or file upload step present (judges have no local files)
- GPU runtime instructions visible (T4 or better)
- Notebook saved **with outputs visible** — not cleared
- Reward/loss plots output inline so judges can see results without re-running

---

## Summary

| # | Task | Risk | Status |
|---|------|------|--------|
| X1 | `openenv.yaml` manifest | HIGH | ✅ Done |
| X2 | Plot assets in `assets/` | HIGH | ✅ Done |
| X3 | README story rewrite | HIGH | ✅ Done |
| X4 | Training notebook calls CIPHEREnv live | MEDIUM-HIGH | ⬜ Verify |
| X5 | Baseline vs trained comparison plot | HIGH | ✅ Done |
| X6 | Plots embedded in README with captions | HIGH | ✅ Done |
| X7 | HF blog post + demo video | HIGH | ⬜ Create & link |
| X8 | MCP reserved tool names | N/A | ✅ Clean |
| X9 | HF Space URL in README | HIGH | ⬜ After upload |
| X10 | Colab notebook runnable end-to-end | MEDIUM | ⬜ Verify |
| X11 | Wandb run link | LOW | N/A |

**7 done. 4 remaining: X4 (verify), X7 (create), X9 (after HF upload), X10 (verify).**
