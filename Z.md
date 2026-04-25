# CIPHER — Submission Command Reference

## 1. Local Verification (run first, in this order)

```bash
# Install dependencies
pip install -r requirements.txt

# OpenEnv compliance check (must be 7/7)
python verify_openenv.py

# Submission checklist (must be 0 failures)
python check_submission.py

# Single episode — stub mode (instant, no API key)
python main.py

# Full LLM mode — single episode
python main.py --live

# Multi-episode — check equal split
python main.py --live --episodes 5

# Training loop — stub (generates traces + rewards_log.csv)
python main.py --train --train-episodes 20

# Regenerate all plots (saves to plots/ and assets/)
python generate_plots.py

# Run full test suite
python -m pytest tests/ -v
```

## 2. React War Room Dashboard

```bash
# Terminal 1: Start API server
cd dashboard-react
python api_server.py          # → http://localhost:5001

# Terminal 2: Start React dev server
cd dashboard-react
npm install
npm run dev                   # → http://localhost:5173

# Or serve the built version:
npm run build
# Then api_server.py serves dist/ at http://localhost:5001
```

## 3. Docker Build & Test (local)

```bash
# Build the Docker image
docker build -t cipher-hf .

# Run locally on port 7860
docker run -p 7860:7860 \
  -e HF_TOKEN=your_hf_token \
  -e LLM_MODE=stub \
  cipher-hf

# Verify at http://localhost:7860
```

## 4. HuggingFace Spaces Deployment

### Option A — HF CLI push (recommended)

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login          # paste your HF_TOKEN when prompted

# Create a new Space (Docker SDK)
# Go to: https://huggingface.co/new-space
# - Space name: cipher-openenv
# - SDK: Docker
# - Visibility: Public

# Add secrets in Space settings:
#   HF_TOKEN  = hf_...
#   API_BASE_URL = https://router.huggingface.co/v1

# Push repo to the Space
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/cipher-openenv
git push hf main
```

### Option B — Direct file upload via web UI

```
1. Go to https://huggingface.co/new-space
2. SDK: Docker | Visibility: Public
3. Upload all files (or link GitHub repo)
4. Set secrets: HF_TOKEN, API_BASE_URL
5. Space auto-builds and serves on port 7860
```

## 5. OpenEnv Submission

```bash
# Verify compliance locally first
python verify_openenv.py

# The env wrapper is importable as:
from cipher.env_wrapper import make_env

env = make_env(max_steps=20, llm_mode="stub")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step("Move toward auth_gateway")
print(env.render())
```

## 6. Training Notebook (Colab)

```
1. Open CIPHER_Training_Colab.ipynb in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Runtime → Run all
4. Outputs saved to: red trained/cipher-red-planner/ (LoRA adapter)
```

## 7. Generate & Verify Plots

```bash
python generate_plots.py
# Saves to: plots/*.png AND assets/*.png

# Verify all 7 required assets exist:
ls assets/baseline_vs_trained.png
ls assets/reward_curves.png
ls assets/elo_chart.png
ls assets/terminal_outcomes.png
ls assets/fleet_verdicts.png
ls assets/win_rate_progression.png
ls assets/architecture_card.png
```

## 8. Final Pre-Push Checklist

```bash
# 1. All checks pass
python verify_openenv.py
python check_submission.py

# 2. Plots regenerated
python generate_plots.py

# 3. Tests pass
python -m pytest tests/ -v

# 4. Git commit
git add -A
git commit -m "v2: Commander+Subagent architecture, forensics, React War Room"
git push origin main

# 5. Push to HF Spaces (see Section 4)
git push hf main
```

## 9. Submission Links to Fill In

After deployment, update README.md `## Submission Links` table:

| Field | Value |
|---|---|
| GitHub repo | https://github.com/YOUR_USERNAME/cipher-openenv |
| HuggingFace Space | https://huggingface.co/spaces/YOUR_USERNAME/cipher-openenv |
| Training Notebook | Already linked in README |
| Demo Video | Upload 2-3 min to YouTube/Loom, paste URL |
| Mini-blog | HF community post or GitHub wiki page |

## 10. Key Files for Judges

| File | Purpose |
|---|---|
| `openenv.yaml` | OpenEnv manifest |
| `verify_openenv.py` | OpenEnv compliance (run this) |
| `cipher/env_wrapper.py` | `CIPHEREnv` class — reset/step/render |
| `Dockerfile` | HF Spaces deployment |
| `CIPHER_Training_Colab.ipynb` | Training notebook |
| `rewards_log.csv` | 1,082 episode telemetry |
| `README.md` | Full documentation |
| `assets/*.png` | All 7 result plots |
