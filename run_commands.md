# CIPHER — Run Commands Reference

Quick reference for every important command in the project.

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run stub training (fast — no API cost)
python -m cipher.training.loop --episodes 50

# Run all tests
pytest tests/ -v

# Run specific phase tests
pytest tests/test_phase10.py -v

# Verify environment setup
python verify_openenv.py

# Open live dashboard (http://localhost:8051)
python cipher/dashboard/live.py
```

---

## .env Modes

```bash
# Default: stub mode (random actions, instant)
LLM_MODE=stub
LLM_BACKEND=nvidia

# Live mode with NVIDIA NIM (real API calls)
LLM_MODE=live
LLM_BACKEND=nvidia

# Live competition mode: RED Planner uses trained local model, rest use NVIDIA NIM
LLM_MODE=live
LLM_BACKEND=hybrid
LOCAL_MODEL_URL=http://localhost:1234/v1   # LM Studio
LOCAL_MODEL_NAME=cipher-red-planner
```

---

## RunPod Training Setup

### Recommended GPU
- **RTX 3090** (24 GB VRAM) — ~$0.44/hr on RunPod
- RTX 4090 (24 GB VRAM) — ~$0.74/hr (faster, optional)
- A100 40GB — ~$1.89/hr (overkill, not needed)

**Pod settings:**
- Template: `RunPod PyTorch 2.2` (`runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`)
- Disk: 50 GB container storage
- Volume: 20 GB (for model weights)
- Enable SSH access

### Step 1 — Clone repo on RunPod (after you push to GitHub)

```bash
cd /workspace
git clone https://github.com/wolfie8935/OPENENV-FINAL cipher
cd cipher
```

### Step 2 — Install base dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Install training dependencies

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl>=0.12.0 datasets>=2.19.0 accelerate>=0.30.0 peft>=0.11.0 bitsandbytes>=0.43.0
```

### Step 4 — Set secrets

```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE    # from huggingface.co/settings/tokens
```

### Step 5 — Quick smoke test (5 minutes, verify everything works)

```bash
python CIPHER_Training_RunPod.py --quick
```

### Step 6 — Full training (8 hours)

```bash
# Recommended: run in tmux so SSH disconnect doesn't kill it
tmux new -s cipher_train
python CIPHER_Training_RunPod.py --epochs 3 --hf-push
# Detach: Ctrl+B then D
# Reattach later: tmux attach -t cipher_train
```

### Step 7 — Download model artifacts to your desktop

```bash
# From your local machine (replace POD_ID with your RunPod pod ID)
scp -P <port> root@<pod-ip>:/workspace/cipher/cipher_improvement.png ./
scp -P <port> root@<pod-ip>:/workspace/cipher/training_summary.json ./
scp -r -P <port> root@<pod-ip>:/workspace/cipher/cipher-red-planner/ ./

# Or pull from HuggingFace (easier)
# The training script pushes to: wolfie8935/cipher-red-planner-grpo
```

---

## Post-Training: Desktop Competition Mode

### Option A — LM Studio (recommended, no Linux/CUDA needed on Windows)

1. Download **LM Studio** → load `wolfie8935/cipher-red-planner-grpo`
2. Start the local server (default port 1234)
3. Update `.env`:
   ```
   LLM_BACKEND=hybrid
   LLM_MODE=live
   LOCAL_MODEL_URL=http://localhost:1234/v1
   LOCAL_MODEL_NAME=cipher-red-planner
   ```
4. Run competition:
   ```bash
   python -m cipher.training.loop --episodes 20
   ```

### Option B — vllm (Linux/WSL, faster inference)

```bash
pip install vllm
# Serve the merged model
python -m vllm.entrypoints.openai.api_server \
    --model ./cipher-red-planner-merged \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 1024

# In .env set:
# LOCAL_MODEL_URL=http://localhost:8000/v1
```

---

## Dashboard

```bash
# Live training dashboard (auto-refreshes every 2 seconds)
python cipher/dashboard/live.py
# Opens at: http://localhost:8051

# Tab guide:
#   Tab 1 — Training State    (current episode, policy updates)
#   Tab 2 — Rewards           (RED/BLUE reward curves per episode)
#   Tab 3 — Events            (trap fires, dead drops, exfil attempts)
#   Tab 4 — Dead Drops        (memory efficiency, vault contents)
#   Tab 5 — Prompt Evolution  (heuristic rules learned from rewards)
#   Tab 6 — Learning Curve    (rolling win rate, early/late comparison)
```

---

## Git Workflow

```bash
# Standard commit
git add -A
git commit -m "description"
git push

# After RunPod training (done automatically by the training script)
git add training_summary.json cipher_improvement.png rewards_log.csv
git commit -m "[RunPod] GRPO training complete"
git push
```

---

## Tests

```bash
# Full suite
pytest tests/ -v

# By phase
pytest tests/test_phase1.py tests/test_phase2.py tests/test_phase3.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"

# Single test
pytest tests/test_phase10.py::TestDashboardTab6Layout -v
```

---

## HuggingFace

```bash
# Manually push model (if --hf-push wasn't used during training)
python -c "
from huggingface_hub import HfApi
import os
api = HfApi()
api.upload_folder(
    folder_path='./cipher-red-planner',
    repo_id='wolfie8935/cipher-red-planner-grpo',
    token=os.environ['HF_TOKEN'],
)
"

# Pull model on a new machine
from huggingface_hub import snapshot_download
snapshot_download('wolfie8935/cipher-red-planner-grpo', local_dir='./cipher-red-planner')
```

---

## Episode Runner Shortcuts

```bash
# 10 live episodes with NVIDIA NIM
LLM_MODE=live LLM_BACKEND=nvidia python -m cipher.training.loop --episodes 10

# 100 stub episodes (fast, for reward analysis)
LLM_MODE=stub python -m cipher.training.loop --episodes 100

# Competition: hybrid (trained RED Planner + NVIDIA NIM for BLUE)
LLM_MODE=live LLM_BACKEND=hybrid python -m cipher.training.loop --episodes 20
```

---

## Budget Estimate (RunPod)

| Task | GPU | Est. Time | Cost |
|------|-----|-----------|------|
| Smoke test | RTX 3090 | 5 min | $0.04 |
| Full training (3 epochs) | RTX 3090 | ~7-8 hrs | ~$3.50 |
| Pod idle (tmux) | — | — | ~$0.02/hr |
| **Total budget** | | | **< $5** of your $10 |
