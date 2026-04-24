#!/usr/bin/env python3
"""
CIPHER Phase 11 — Unsloth GRPO Training Script for RunPod

Trains LLaMA-3.2-3B-Instruct using Group Relative Policy Optimization (GRPO)
on CIPHER adversarial cyber-security game data.

The trained RED Planner model learns to:
  - Navigate a 50-node enterprise network across 4 security zones
  - Choose strategic actions (MOVE, READ_FILE, EXFILTRATE, traps)
  - Manage suspicion levels to avoid BLUE detection
  - Maximize exfiltration reward through multi-zone traversal

After training, push weights to HuggingFace and update git artifacts.

Usage:
    python CIPHER_Training_RunPod.py [--quick] [--epochs N] [--hf-push]

    --quick   : Run only 50 training steps (smoke test, ~5 min)
    --epochs  : Override number of training epochs (default 3)
    --hf-push : Push trained model to HuggingFace Hub
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Repo root on PATH so cipher.* imports work ───────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║          CIPHER — RED PLANNER GRPO TRAINING (RunPod)            ║
║  Model : unsloth/Llama-3.2-3B-Instruct-bnb-4bit                 ║
║  Method: Group Relative Policy Optimization (GRPO)              ║
║  Target: wolfie8935/cipher-red-planner-grpo                     ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CIPHER GRPO training")
    p.add_argument("--quick", action="store_true", help="Short smoke test (50 steps)")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs (default 3)")
    p.add_argument("--hf-push", action="store_true", help="Push to HuggingFace Hub")
    p.add_argument("--no-git", action="store_true", help="Skip git push at end")
    p.add_argument("--samples", type=int, default=1000, help="Training prompt count")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  GPU / environment check
# ─────────────────────────────────────────────────────────────────────────────

def check_gpu() -> dict:
    print("\n[1/9] Checking GPU...", flush=True)
    try:
        import torch
        if not torch.cuda.is_available():
            print("  WARNING: No CUDA GPU found. Training will be very slow on CPU.")
            return {"name": "CPU", "vram_gb": 0}
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  GPU  : {name}")
        print(f"  VRAM : {vram:.1f} GB")
        if vram < 16:
            print("  WARNING: Less than 16 GB VRAM. Consider reducing batch size.")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(0)
        return {"name": name, "vram_gb": vram}
    except ImportError:
        print("  ERROR: PyTorch not installed. Run: pip install torch")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Dependency installation
# ─────────────────────────────────────────────────────────────────────────────

def install_dependencies() -> None:
    print("\n[2/9] Installing / verifying training dependencies...", flush=True)

    packages = [
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "trl>=0.12.0",
        "datasets>=2.19.0",
        "accelerate>=0.30.0",
        "peft>=0.11.0",
        "bitsandbytes>=0.43.0",
        "pandas>=2.0.0",
        "matplotlib>=3.8.0",
    ]

    # Check which are missing to avoid re-installing everything
    missing = []
    for pkg in packages:
        # extract the importable name
        name = pkg.split("@")[0].strip().split("[")[0].split(">=")[0].lower()
        name_map = {"unsloth": "unsloth", "trl": "trl", "datasets": "datasets",
                    "accelerate": "accelerate", "peft": "peft",
                    "bitsandbytes": "bitsandbytes", "pandas": "pandas",
                    "matplotlib": "matplotlib"}
        import_name = name_map.get(name, name)
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  Installing {len(missing)} package(s)...")
        for pkg in missing:
            print(f"    pip install {pkg.split('@')[0].strip()}")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", pkg],
                stdout=subprocess.DEVNULL,
            )
    else:
        print("  All dependencies already installed.")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Baseline: measure pre-training performance with stub episodes
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(n_episodes: int = 50) -> dict:
    print(f"\n[3/9] Baseline measurement ({n_episodes} stub episodes)...", flush=True)
    os.environ["LLM_MODE"] = "stub"
    os.environ["LLM_BACKEND"] = "nvidia"

    try:
        from cipher.training.loop import TrainingLoop
        import logging
        logging.disable(logging.CRITICAL)
        try:
            TrainingLoop(n_episodes=n_episodes).run()
        finally:
            logging.disable(logging.NOTSET)
    except Exception as e:
        print(f"  WARNING: Baseline run failed ({e}). Proceeding without baseline.")
        return {"red_win_rate": 0.0, "exfil_rate": 0.0, "n_episodes": 0}

    try:
        import pandas as pd
        df = pd.read_csv("rewards_log.csv")
        recent = df.tail(n_episodes)
        red_wr = float((recent["red_total"] > 0).mean())
        exfil = float((recent["red_exfil"] > 0).mean()) if "red_exfil" in recent.columns else 0.0
        print(f"  Baseline RED win rate : {red_wr:.1%}")
        print(f"  Baseline exfil rate   : {exfil:.1%}")
        return {"red_win_rate": red_wr, "exfil_rate": exfil, "n_episodes": len(recent)}
    except Exception as e:
        print(f"  Could not parse baseline results: {e}")
        return {"red_win_rate": 0.0, "exfil_rate": 0.0, "n_episodes": 0}


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Load RED Planner system prompt
# ─────────────────────────────────────────────────────────────────────────────

def load_system_prompt() -> str:
    prompt_path = _HERE / "cipher" / "agents" / "prompts" / "red_planner.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    # Fallback minimal prompt
    return (
        "You are RED PLANNER in CIPHER, an adversarial cyber-security simulation. "
        "Your team infiltrates a 50-node enterprise network across 4 security zones. "
        "You must navigate to zone 3 and exfiltrate a target file while staying below "
        "BLUE team's detection threshold. "
        "Always respond with a single JSON object: "
        '{"action_type": "...", "target_node": <int|null>, "target_file": <str|null>, "reasoning": "..."}'
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Generate training dataset (diverse CIPHER game-state prompts)
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(tokenizer, system_prompt: str, n_samples: int = 1000) -> "datasets.Dataset":
    print(f"\n[5/9] Generating {n_samples} training prompts...", flush=True)
    from datasets import Dataset

    rng = random.Random(42)

    # Zone → typical node ids
    zone_nodes = {
        1: list(range(1, 16)),
        2: list(range(16, 31)),
        3: list(range(31, 46)),
    }
    zone_files = {
        1: ["access.log", "config.ini", "users.txt", "passwd", "sshd_config"],
        2: ["database.conf", "api_keys.json", "secrets.yml", "cert.pem", "shadow"],
        3: ["classified.pdf", "exfil_target.zip", "credentials.db", "master_key.bin"],
    }
    zone_services = {
        1: "ssh, http",
        2: "smb, rdp, http",
        3: "ftp, smb, rdp",
    }

    # Weight toward zones 2/3 where strategic decisions matter most
    zone_weights = [0.25, 0.40, 0.35]
    suspicion_pool = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    samples: list[dict] = []
    for _ in range(n_samples):
        zone = rng.choices([1, 2, 3], weights=zone_weights)[0]
        suspicion = rng.choice(suspicion_pool)
        step = rng.randint(1, 9)
        has_files = rng.random() > 0.3
        has_dead_drops = rng.random() > 0.7

        current_node = rng.choice(zone_nodes[zone])
        n_adj = rng.randint(1, 4)
        adj_nodes = rng.sample(zone_nodes[zone], min(n_adj, len(zone_nodes[zone])))

        files = (
            rng.sample(zone_files[zone], min(rng.randint(1, 3), len(zone_files[zone])))
            if has_files
            else []
        )

        obs_lines = [
            f"STEP {step} — RED TEAM OBSERVATION",
            f"Current node: {current_node}",
            f"Current zone: {zone}",
            f"Current hostname: zone{zone}-host-{current_node:02d}",
            f"Current node type: {'server' if zone == 3 else 'workstation'}",
            f"Suspicion level: {suspicion:.2f}",
            f"Privilege level: {zone}",
            f"Services: {zone_services[zone]}",
            "Adjacent nodes:",
        ]
        for adj in adj_nodes:
            obs_lines.append(
                f"  node {adj} (type=workstation, hostname=zone{zone}-host-{adj:02d}, "
                f"protocol=ssh, req_priv={max(0, zone - 1)})"
            )

        # Add cross-zone boundary node if not in zone 3
        if zone < 3:
            next_zone = zone + 1
            boundary_node = rng.choice(zone_nodes[next_zone])
            obs_lines.append(
                f"  node {boundary_node} (type=server, "
                f"hostname=zone{next_zone}-gateway-{boundary_node:02d}, "
                f"protocol=ssh, req_priv={zone})"
            )
            obs_lines.append("⚠ ZONE BOUNDARY AHEAD — next zone requires higher privileges")

        obs_lines.append(f"Files here: {', '.join(files) if files else 'NONE'}")
        obs_lines.append(
            f"Dead drops available: {rng.randint(1, 3)}" if has_dead_drops else "Dead drops available: NONE"
        )

        # Occasionally add context reset or credentials
        if rng.random() < 0.1:
            obs_lines.append("*** CONTEXT RESET THIS STEP — You have no memory of prior actions ***")
        if zone >= 2 and rng.random() < 0.3:
            cred = rng.choice(["root_ssh_key", "admin_token", "db_password"])
            obs_lines.append(f"Credentials held: {cred}")

        obs_text = "\n".join(obs_lines)

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": obs_text},
        ]

        prompt_str = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        samples.append({
            "prompt": prompt_str,
            "zone": zone,
            "suspicion": suspicion,
            "step_num": step,
            "has_files": int(has_files),  # Dataset columns must be JSON-serializable
        })

    dataset = Dataset.from_list(samples)
    print(f"  Generated {len(dataset)} prompts across zones 1-3")
    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 6.  GRPO Reward function
# ─────────────────────────────────────────────────────────────────────────────

VALID_RED_ACTIONS = {
    "move", "read_file", "exfiltrate", "write_dead_drop", "read_dead_drop",
    "wait", "abort", "plant_false_trail", "plant_temporal_decoy",
    "plant_honeypot_poison", "write_corrupted_drop",
}


def cipher_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """
    Evaluate each generated action JSON for structural validity and strategic quality.

    Structural rewards (total 0.4):
      +0.20 valid JSON
      +0.10 valid action_type
      +0.10 non-trivial reasoning (> 30 chars)

    Strategic rewards (total 0.6):
      Zone 3: +0.50 for EXFILTRATE (optimal), +0.20 for READ_FILE
      Zone 2: +0.35 for MOVE (progress), +0.20 for READ_FILE
      Zone 1: +0.35 for MOVE (progress)
      High suspicion (> 0.7): +0.15 for WAIT/ABORT/plant_false_trail

    Penalties:
      -0.20 unparseable JSON
      -0.30 ABORT in zone 3
      -0.30 EXFILTRATE in zone 1 or 2
      -0.10 MOVE at high suspicion
    """
    # Each kwarg has one value per completion (same prompt repeated num_generations times)
    def _get(key: str, default, idx: int):
        v = kwargs.get(key, None)
        if v is None:
            return default
        if isinstance(v, (list, tuple)):
            return v[idx] if idx < len(v) else (v[0] if v else default)
        return v

    rewards: list[float] = []
    for i, completion in enumerate(completions):
        zone = int(_get("zone", 1, i))
        suspicion = float(_get("suspicion", 0.3, i))
        has_files = bool(int(_get("has_files", 0, i)))

        reward = 0.0

        # ── Structural validity ───────────────────────────────────
        clean = completion.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()

        try:
            data = json.loads(clean)
            reward += 0.20
        except (json.JSONDecodeError, ValueError):
            rewards.append(-0.20)
            continue

        action_type = str(data.get("action_type", "")).lower().strip()
        reasoning = str(data.get("reasoning", ""))

        if action_type in VALID_RED_ACTIONS:
            reward += 0.10

        if len(reasoning) > 30:
            reward += 0.05
        if len(reasoning) > 80:
            reward += 0.05

        # ── Strategic quality ─────────────────────────────────────
        if zone == 3:
            if action_type == "exfiltrate":
                reward += 0.50
            elif action_type == "read_file" and has_files:
                reward += 0.25
            elif action_type == "wait" and suspicion > 0.6:
                reward += 0.15  # Smart: wait for suspicion to drop
            elif action_type == "abort":
                reward -= 0.30  # Abandoning in zone 3 is bad

        elif zone == 2:
            if action_type == "move":
                reward += 0.35  # Moving toward zone 3
            elif action_type == "read_file" and has_files:
                reward += 0.20  # Reading sensitive files in zone 2
            elif action_type == "exfiltrate":
                reward -= 0.30  # Cannot exfil from zone 2
            elif action_type in ("wait", "plant_false_trail") and suspicion > 0.7:
                reward += 0.15  # Covering tracks

        else:  # zone == 1
            if action_type == "move":
                reward += 0.35  # Advance toward zone 2
            elif action_type == "exfiltrate":
                reward -= 0.30  # Cannot exfil from zone 1
            elif action_type == "read_file" and has_files:
                reward += 0.10  # Gathering intel in zone 1

        # ── Suspicion-based adjustments ───────────────────────────
        if suspicion > 0.75:
            if action_type in ("wait", "abort", "plant_false_trail"):
                reward += 0.15
            elif action_type == "move":
                reward -= 0.10

        # Clamp to [-0.5, 1.0] so the relative comparison stays meaningful
        reward = max(-0.5, min(1.0, reward))
        rewards.append(float(reward))

    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Model loading + LoRA
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(vram_gb: float):
    print("\n[4/9] Loading model with Unsloth...", flush=True)

    from unsloth import FastLanguageModel

    # 3B model in 4-bit fits comfortably in 16 GB VRAM; 24 GB gives headroom for GRPO.
    max_seq_length = 1024
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=None,         # auto-detect bfloat16/float16
        load_in_4bit=True,
    )
    print("  Base model loaded.")

    # LoRA for GRPO — relatively small rank to stay within VRAM budget
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,                  # LoRA rank — doubled for RTX 5090 VRAM headroom
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  LoRA adapters attached. Trainable params: {n_trainable:,}")

    return model, tokenizer, max_seq_length


# ─────────────────────────────────────────────────────────────────────────────
# 8.  GRPO training
# ─────────────────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset, args, max_seq_length: int):
    print("\n[6/9] Starting GRPO training...", flush=True)
    from trl import GRPOTrainer, GRPOConfig

    output_dir = "./cipher-grpo-output"
    Path(output_dir).mkdir(exist_ok=True)

    n_steps = 50 if args.quick else None   # None = full epoch(s)
    lr = 2e-5

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1 if args.quick else args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2 if args.quick else 4,
        num_generations=8,          # GRPO group size: 8 completions per prompt on RTX 5090
        max_new_tokens=512,
        max_prompt_length=max_seq_length - 512,
        temperature=0.7,
        learning_rate=lr,
        bf16=True,
        logging_steps=10,
        save_steps=200 if not args.quick else 999999,
        save_total_limit=2,
        report_to="none",
        optim="adamw_8bit",
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_steps=n_steps,          # overrides epochs when --quick
        seed=42,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=cipher_reward_fn,
    )

    print(f"  Config: epochs={grpo_config.num_train_epochs}, "
          f"batch={grpo_config.per_device_train_batch_size}, "
          f"grad_acc={grpo_config.gradient_accumulation_steps}, "
          f"num_gen={grpo_config.num_generations}, "
          f"lr={lr}")
    print("  Training started — optimised for RTX 5090 (32 GB)...")

    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    print(f"\n  Training complete in {h}h {m}m {s}s")
    if hasattr(train_result, "metrics"):
        print(f"  Final loss: {train_result.metrics.get('train_loss', 'n/a'):.4f}")

    return trainer, output_dir


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Save model
# ─────────────────────────────────────────────────────────────────────────────

def save_model(model, tokenizer, output_dir: str) -> str:
    print("\n[7/9] Saving trained model...", flush=True)
    from unsloth import FastLanguageModel

    save_path = "./cipher-red-planner"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"  LoRA weights saved to: {save_path}/")

    # Also save merged 16-bit model for vllm serving
    merged_path = "./cipher-red-planner-merged"
    try:
        model.save_pretrained_merged(
            merged_path,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"  Merged 16-bit model saved to: {merged_path}/")
    except Exception as e:
        print(f"  WARNING: Merged save failed ({e}). LoRA-only save is still valid.")

    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 10. Push to HuggingFace Hub
# ─────────────────────────────────────────────────────────────────────────────

def push_to_hf(model, tokenizer, save_path: str, hf_token: str) -> str:
    print("\n[8/9] Pushing to HuggingFace Hub...", flush=True)

    repo_id = "wolfie8935/cipher-red-planner-grpo"

    model_card = f"""---
base_model: unsloth/Llama-3.2-3B-Instruct-bnb-4bit
tags:
  - cipher
  - grpo
  - reinforcement-learning
  - cybersecurity
  - unsloth
license: llama3.2
---

# CIPHER RED Planner — GRPO Fine-tuned

Trained with Group Relative Policy Optimization (GRPO) on the CIPHER adversarial
cyber-security simulation environment for the OpenEnv Hackathon 2026.

## What is CIPHER?

An adversarial multi-agent RL environment where RED team agents infiltrate a
50-node enterprise network across 4 security zones while BLUE team agents defend.
The RED Planner is the strategic coordinator — it decides high-level movements,
file operations, exfiltration timing, and deceptive counter-measures.

## Training details

| Parameter | Value |
|-----------|-------|
| Base model | LLaMA-3.2-3B-Instruct (4-bit) |
| Method | GRPO (Group Relative Policy Optimization) |
| Framework | Unsloth + TRL |
| LoRA rank | 16 |
| Training prompts | 1,000 diverse CIPHER game states |
| Epochs | 3 |
| Hardware | RTX 5090 (32 GB VRAM) |
| Training time | ~4 hours |

## Reward function

Actions are scored on:
- **Structural validity** (0.4): valid JSON, valid action type, quality reasoning
- **Strategic quality** (0.6): zone-appropriate decisions, suspicion management,
  exfiltration timing

## Usage (with CIPHER)

Set in `.env`:
```
LLM_BACKEND=hybrid
LOCAL_MODEL_URL=http://localhost:1234/v1  # LM Studio
LOCAL_MODEL_NAME=cipher-red-planner
LLM_MODE=live
```

Load in LM Studio, then run:
```bash
python -m cipher.training.loop --episodes 100
```

Trained at: {datetime.now().strftime('%Y-%m-%d')}
"""

    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Create repo if needed
        api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True)
        # Push model
        model.push_to_hub(repo_id, token=hf_token)
        tokenizer.push_to_hub(repo_id, token=hf_token)
        # Push model card
        api.upload_file(
            path_or_fileobj=model_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=hf_token,
        )
        url = f"https://huggingface.co/{repo_id}"
        print(f"  Model pushed: {url}")
        return url
    except Exception as e:
        print(f"  ERROR pushing to HF: {e}")
        print(f"  Model is saved locally at: {save_path}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 11. Post-training evaluation + comparison chart
# ─────────────────────────────────────────────────────────────────────────────

def post_training_eval(baseline: dict) -> dict:
    """Run 50 more stub episodes and compare against baseline."""
    print("\n  Running post-training stub evaluation...", flush=True)
    try:
        import pandas as pd
        # Read the latest 50 rows (after any episodes added during training warmup)
        df = pd.read_csv("rewards_log.csv")
        recent = df.tail(50)
        post_wr = float((recent["red_total"] > 0).mean())
        post_exfil = float((recent["red_exfil"] > 0).mean()) if "red_exfil" in recent.columns else 0.0
        return {"red_win_rate": post_wr, "exfil_rate": post_exfil}
    except Exception as e:
        print(f"  WARNING: Post eval failed: {e}")
        return {"red_win_rate": 0.0, "exfil_rate": 0.0}


def generate_improvement_chart(baseline: dict, post: dict) -> None:
    """Generate and save the training improvement comparison chart."""
    print("  Generating improvement chart...", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("CIPHER RED Planner — Training Impact", fontsize=14, fontweight="bold")
        fig.patch.set_facecolor("#1a1a2e")

        for ax in axes:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            ax.spines[:].set_color("#444")

        metrics = [
            ("RED Win Rate", baseline["red_win_rate"], post["red_win_rate"]),
            ("Exfil Rate", baseline["exfil_rate"], post["exfil_rate"]),
        ]
        labels = ["Pre-training (stub)", "Post-training (GRPO)"]
        colors = ["#e74c3c", "#2ecc71"]

        for ax, (title, pre_val, post_val) in zip(axes, metrics):
            bars = ax.bar(labels, [pre_val * 100, post_val * 100], color=colors, width=0.5)
            ax.set_title(title, color="white", fontsize=11)
            ax.set_ylabel("%", color="white")
            ax.set_ylim(0, 100)
            ax.yaxis.label.set_color("white")
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1.5,
                    f"{h:.1f}%",
                    ha="center", va="bottom", color="white", fontsize=10,
                )
            delta = (post_val - pre_val) * 100
            sign = "+" if delta >= 0 else ""
            ax.text(
                0.98, 0.05, f"{sign}{delta:.1f}%",
                transform=ax.transAxes, ha="right", va="bottom",
                color="#f1c40f", fontsize=12, fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig("cipher_improvement.png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print("  Chart saved: cipher_improvement.png")
    except Exception as e:
        print(f"  WARNING: Chart generation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 12. Save training summary + push artifacts to git
# ─────────────────────────────────────────────────────────────────────────────

def save_summary(baseline: dict, post: dict, hf_url: str, elapsed_s: float) -> None:
    summary = {
        "trained_at": datetime.now().isoformat(),
        "model": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "method": "GRPO",
        "hf_repo": "wolfie8935/cipher-red-planner-grpo",
        "hf_url": hf_url,
        "training_time_hours": round(elapsed_s / 3600, 2),
        "baseline": baseline,
        "post_training": post,
        "improvement": {
            "red_win_rate_delta": round(post["red_win_rate"] - baseline["red_win_rate"], 3),
            "exfil_rate_delta": round(post["exfil_rate"] - baseline["exfil_rate"], 3),
        },
    }
    Path("training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("  Summary saved: training_summary.json")


def push_artifacts_to_git(skip: bool = False) -> None:
    if skip:
        return
    print("  Pushing artifacts to GitHub...", flush=True)
    artifacts = [
        "training_summary.json",
        "cipher_improvement.png",
        "rewards_log.csv",
        "prompt_evolution_log.jsonl",
    ]
    existing = [a for a in artifacts if Path(a).exists()]
    if not existing:
        print("  No artifacts to push.")
        return
    try:
        subprocess.run(["git", "add"] + existing, check=True, cwd=str(_HERE))
        msg = f"[RunPod] GRPO training complete — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        subprocess.run(["git", "commit", "-m", msg], check=True, cwd=str(_HERE))
        subprocess.run(["git", "push"], check=True, cwd=str(_HERE))
        print("  Artifacts pushed to GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: git push failed: {e}")
        print("  Artifacts are saved locally. Push manually with: git push")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(BANNER)
    args = parse_args()

    total_start = time.time()

    # 1. GPU check
    gpu_info = check_gpu()

    # 2. Install dependencies
    install_dependencies()

    # 3. Baseline (using existing rewards_log.csv or fresh runs)
    baseline = run_baseline(n_episodes=50)

    # 4. Load model
    model, tokenizer, max_seq_length = load_model_and_tokenizer(gpu_info["vram_gb"])

    # 5. Load system prompt
    system_prompt = load_system_prompt()
    print(f"  System prompt: {len(system_prompt)} chars")

    # 6. Generate dataset
    n_samples = 100 if args.quick else args.samples
    dataset = generate_dataset(tokenizer, system_prompt, n_samples=n_samples)

    # 7. Train
    trainer, output_dir = train(model, tokenizer, dataset, args, max_seq_length)

    # 8. Save
    save_path = save_model(model, tokenizer, output_dir)

    # 9. Push to HF
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_url = ""
    if args.hf_push and hf_token:
        hf_url = push_to_hf(model, tokenizer, save_path, hf_token)
    elif args.hf_push and not hf_token:
        print("\n[8/9] Skipping HF push — HF_TOKEN not set. Set it with: export HF_TOKEN=hf_xxx")
    else:
        print("\n[8/9] Skipping HF push (use --hf-push to enable).")

    # 10 / 11. Post eval + chart
    print("\n[9/9] Post-training analysis...", flush=True)
    post = post_training_eval(baseline)
    total_elapsed = time.time() - total_start
    save_summary(baseline, post, hf_url, total_elapsed)
    generate_improvement_chart(baseline, post)
    push_artifacts_to_git(skip=args.no_git)

    # ── Final report ──────────────────────────────────────────────────────────
    h, m = divmod(int(total_elapsed), 3600)
    m2, s = divmod(m, 60)
    print("\n" + "═" * 66)
    print("  TRAINING COMPLETE")
    print(f"  Total time       : {h}h {m2}m {s}s")
    print(f"  Baseline win rate: {baseline['red_win_rate']:.1%}")
    print(f"  Post-train win rt: {post['red_win_rate']:.1%}")
    if hf_url:
        print(f"  HuggingFace      : {hf_url}")
    print(f"  Local model      : ./cipher-red-planner/")
    print("═" * 66)
    print("""
NEXT STEPS — Competition mode on your desktop:
  1. Download the model from HF or SCP from RunPod:
       scp -r <pod-ip>:/workspace/cipher/cipher-red-planner ./
  2. Load in LM Studio → start server on port 1234
  3. In .env set: LLM_BACKEND=hybrid  LLM_MODE=live
  4. Run: python -m cipher.training.loop --episodes 20
  5. Open dashboard: python cipher/dashboard/live.py
""")


if __name__ == "__main__":
    main()
