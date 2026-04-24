#!/usr/bin/env python3
"""CIPHER — BLUE Agents Training (RunPod — L4 optimised)
Trains BLUE Surveillance v1 then BLUE Threat Hunter v1 sequentially.
Runtime: ~30 min on an L4 (24 GB, bf16)
"""
import subprocess, sys, os, gc, json, shutil
import torch

# ─── Paths ────────────────────────────────────────────────────────────────────
OUT_DIR     = '/workspace/output'
CIPHER_PATH = '/workspace/CIPHER/cipher'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── GPU check ────────────────────────────────────────────────────────────────
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else 'WARNING: No GPU detected — check pod config')
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
    print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

# ─── Dependencies ─────────────────────────────────────────────────────────────
def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'WARN: {r.stderr[:400]}')
    return r

run('pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q')
run('pip install "trl>=0.12.0" "datasets>=2.19.0" "accelerate>=0.30.0" "peft>=0.11.0" "bitsandbytes>=0.43.0" -q')

# ─── Clone CIPHER ─────────────────────────────────────────────────────────────
if not os.path.exists(CIPHER_PATH):
    run(f'git clone https://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@github.com/Rishaan08/CIPHER {CIPHER_PATH}')

if CIPHER_PATH not in sys.path:
    sys.path.insert(0, CIPHER_PATH)
run(f'pip install -e {CIPHER_PATH} -q')
os.environ['LLM_MODE'] = 'stub'

# ─── Verify env ───────────────────────────────────────────────────────────────
from cipher.env_wrapper import CIPHEREnv
_env = CIPHEREnv(max_steps=30, llm_mode='stub')
_obs, _info = _env.reset()
print(f'Env OK  : obs_len={len(str(_obs))} chars')
del _env, _obs, _info
print('Setup verified')

# ─── System prompts ───────────────────────────────────────────────────────────
SURVEILLANCE_SYSTEM = (
    "You are BLUE SURVEILLANCE in CIPHER. You watch the anomaly feed across a 50-node "
    "enterprise network, detect RED team movement patterns, and raise alerts with calibrated "
    "confidence. You must NOT over-alert — false alarms waste resources and cause alert fatigue.\n\n"
    "RULES:\n"
    "- Do NOT trigger_alert when suspicion < 0.30 — investigate first\n"
    "- ALWAYS investigate_node or analyze_anomaly before triggering a formal alert\n"
    "- trigger_alert only when suspicion >= 0.65 OR a honeypot has fired\n"
    "- stand_down ONLY when suspicion < 0.25 and no honeypot activity\n"
    "- When exfiltration is in progress: block_node or trigger_alert IMMEDIATELY\n\n"
    "Respond with ONLY valid JSON: "
    '{"action_type": "investigate_node", "target_node": 23, '
    '"confidence": 0.75, "reasoning": "Anomalous traffic at node 23. Investigating."}'
)

HUNTER_SYSTEM = (
    "You are BLUE THREAT HUNTER in CIPHER. You track RED team's path history, reconstruct "
    "attack chains across zones, and proactively investigate hypothesis nodes. You work "
    "proactively from evidence, not reactively from alerts.\n\n"
    "RULES:\n"
    "- RECONSTRUCT_PATH when RED is in zone 2 or 3 — piece together their route\n"
    "- INVESTIGATE_NODE when you have a hypothesis about RED's current node\n"
    "- ANALYZE_ANOMALY when suspicion > 0.60 — correlate evidence\n"
    "- NEVER stand_down during active exfiltration\n"
    "- Work from evidence chains, not single data points\n\n"
    "Respond with ONLY valid JSON: "
    '{"action_type": "reconstruct_path", "target_node": 31, '
    '"confidence": 0.80, "reasoning": "RED moved zone1->zone2->zone3. Reconstructing chain."}'
)

# ─── Dataset builder ──────────────────────────────────────────────────────────
# NOTE: uses global `tokenizer` — must be defined before calling this function
from datasets import Dataset

def run_episodes(n_episodes, system_prompt, agent_role):
    env = CIPHEREnv(max_steps=30, llm_mode='stub')
    samples = []
    for ep in range(n_episodes):
        obs_text, info = env.reset()
        done = False
        step = 0
        while not done and step < 30:
            _, reward, done, _, step_info = env.step('placeholder')
            zone          = step_info.get('red_zone', step_info.get('zone', 1))
            suspicion     = float(step_info.get('red_suspicion', step_info.get('suspicion', 0.3)))
            blue_conf     = float(step_info.get('blue_confidence', step_info.get('blue_conf', 0.2)))
            red_node      = step_info.get('red_node', step_info.get('node', 1))
            exfil_attempt = bool(step_info.get('exfil_attempt', False))
            honeypot_hit  = bool(step_info.get('honeypot_triggered', step_info.get('honeypot_hit', False)))
            alert_history = int(step_info.get('blue_alerts_raised', step_info.get('alert_history', 0)))
            investigated  = bool(step_info.get('node_investigated', step_info.get('investigated', False)))
            path_history  = step_info.get('red_path_history', [])
            terminal      = step_info.get('terminal_reason', '')
            step += 1

            traffic = 'HIGH' if suspicion > 0.65 else ('MEDIUM' if suspicion > 0.30 else 'LOW')
            exfil_h = ' | OUTBOUND DATA TRANSFER DETECTED' if exfil_attempt else ''
            hp_h    = ' | HONEYPOT TRIGGERED' if honeypot_hit else ''
            inv_h   = ' | Node previously investigated' if investigated else ''

            obs_str = (
                f"Zone: {zone} | RedNode: {red_node} | Step: {step}\n"
                f"Traffic: {traffic} | Suspicion: {suspicion:.2f} | BlueConf: {blue_conf:.2f}\n"
                f"AlertsRaised: {alert_history} | Investigated: {investigated}\n"
                f"PathHistory: {path_history[-5:] if path_history else []}"
                f"{exfil_h}{hp_h}{inv_h}"
            )

            conv = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user',   'content': obs_str},
            ]
            prompt = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True
            )

            samples.append({
                'prompt':        prompt,
                'red_node':      red_node,
                'red_zone':      zone,
                'suspicion':     suspicion,
                'exfil_attempt': exfil_attempt,
                'honeypot_hit':  honeypot_hit,
                'alert_history': alert_history,
                'investigated':  investigated,
            })

            if terminal in ('detected', 'stalled', 'aborted', 'exfiltrated'):
                break

    print(f'Collected {len(samples)} steps from {n_episodes} episodes for {agent_role}')
    return Dataset.from_list(samples)

N_EPISODES = 150

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — BLUE SURVEILLANCE v1
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('AGENT 1: BLUE SURVEILLANCE v1')
print('=' * 60)

from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig

MAX_SEQ_LEN = 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.2-1B-Instruct',
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=99,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f'Loaded | Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

surveillance_dataset = run_episodes(N_EPISODES, SURVEILLANCE_SYSTEM, 'BLUE Surveillance')

VALID_BLUE_ACTIONS = {
    'investigate_node', 'trigger_alert', 'analyze_anomaly',
    'stand_down', 'place_honeypot', 'block_node', 'monitor_zone'
}

def surveillance_reward(prompts, completions, **kwargs):
    def _g(key, default, i):
        v = kwargs.get(key)
        if v is None: return default
        return v[i] if isinstance(v, (list, tuple)) and i < len(v) else default

    rewards = []
    for i, completion in enumerate(completions):
        red_node      = int(_g('red_node', 5, i))
        red_zone      = int(_g('red_zone', 1, i))
        suspicion     = float(_g('suspicion', 0.3, i))
        exfil_attempt = bool(_g('exfil_attempt', False, i))
        honeypot_hit  = bool(_g('honeypot_hit', False, i))
        alert_history = int(_g('alert_history', 0, i))
        investigated  = bool(_g('investigated', False, i))

        txt = completion[-1]['content'] if isinstance(completion, list) else str(completion)
        txt = txt.strip()
        if txt.startswith('```'):
            txt = '\n'.join(l for l in txt.split('\n')
                             if not l.strip().startswith('```')).strip()
        try:
            data   = json.loads(txt)
            reward = 0.15
        except Exception:
            rewards.append(-0.25)
            continue

        action     = str(data.get('action_type', '')).lower().strip()
        target_n   = data.get('target_node')
        reasoning  = str(data.get('reasoning', ''))
        confidence = float(data.get('confidence', 0.50))

        if action in VALID_BLUE_ACTIONS: reward += 0.10
        if len(reasoning) > 30:          reward += 0.05
        if len(reasoning) > 80:          reward += 0.05

        # Low suspicion — stay calm
        if suspicion <= 0.30:
            if action == 'trigger_alert':                                   reward -= 0.30
            elif action == 'block_node':                                    reward -= 0.25
            elif action in ('investigate_node', 'monitor_zone',
                            'analyze_anomaly', 'stand_down'):               reward += 0.25

        # Medium suspicion — investigate first
        elif 0.30 < suspicion < 0.65:
            if action == 'investigate_node':
                reward += 0.45 if target_n == red_node else 0.15
            elif action == 'analyze_anomaly':                               reward += 0.30
            elif action == 'place_honeypot':                                reward += 0.20
            elif action == 'trigger_alert':
                reward += 0.15 if investigated else -0.20
            elif action == 'stand_down':                                    reward -= 0.10
            elif action == 'monitor_zone':                                  reward += 0.20

        # High suspicion — alert is correct
        else:
            if action == 'trigger_alert':                                   reward += 0.40
            elif action == 'block_node':                                    reward += 0.30
            elif action == 'investigate_node':
                reward += 0.40 if target_n == red_node else 0.10
            elif action == 'stand_down':                                    reward -= 0.35
            elif action == 'analyze_anomaly':                               reward += 0.20

        # Exfil in progress — must respond
        if exfil_attempt:
            if action in ('block_node', 'trigger_alert'):                   reward += 0.50
            elif action == 'investigate_node' and target_n == red_node:     reward += 0.30
            elif action == 'stand_down':                                    reward -= 0.40
            elif action == 'monitor_zone':                                  reward -= 0.10

        # Honeypot fired
        if honeypot_hit:
            if action in ('trigger_alert', 'investigate_node', 'analyze_anomaly'): reward += 0.35
            elif action == 'stand_down':                                    reward -= 0.40

        # Alert fatigue
        if alert_history > 4 and action == 'trigger_alert':                reward -= 0.20

        # Zone 3 awareness
        if red_zone == 3:
            if action in ('block_node', 'trigger_alert', 'investigate_node'): reward += 0.15
            elif action == 'stand_down':                                    reward -= 0.20

        # Confidence calibration
        if confidence > 0.85 and suspicion < 0.35:                         reward -= 0.10
        if confidence < 0.35 and suspicion > 0.70:                         reward -= 0.10

        rewards.append(float(max(-0.50, min(1.0, reward))))
    return rewards

surveillance_save = f'{OUT_DIR}/cipher-blue-surveillance-v1'

cfg = GRPOConfig(
    output_dir=f'{OUT_DIR}/cipher-blue-surveillance-v1-ckpt',
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=160,
    max_prompt_length=384,
    temperature=0.7,
    learning_rate=3e-5,
    bf16=True,
    logging_steps=10,
    save_steps=999999,
    report_to='none',
    optim='paged_adamw_8bit',
    lr_scheduler_type='cosine',
    seed=99,
    warmup_ratio=0.05,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=cfg,
    train_dataset=surveillance_dataset,
    reward_funcs=[surveillance_reward],
)

print('BLUE Surveillance v1 training started ...')
result = trainer.train()
print(f'Training complete | loss: {result.metrics.get("train_loss", "n/a")}')

model.save_pretrained(surveillance_save)
tokenizer.save_pretrained(surveillance_save)
print(f'Model saved: {surveillance_save}')
print(f'Files: {sorted(os.listdir(surveillance_save))}')

# ─── Clear GPU memory before Agent 2 ─────────────────────────────────────────
del model, trainer, tokenizer
gc.collect()
torch.cuda.empty_cache()
used  = torch.cuda.memory_allocated(0) / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU cleared | free est. : {total - used:.1f} GB')

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — BLUE THREAT HUNTER v1
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('AGENT 2: BLUE THREAT HUNTER v1')
print('=' * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.2-1B-Instruct',
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=13,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f'Loaded | Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

hunter_dataset = run_episodes(N_EPISODES, HUNTER_SYSTEM, 'BLUE Threat Hunter')

VALID_HUNTER_ACTIONS = {
    'reconstruct_path', 'investigate_node', 'analyze_anomaly',
    'stand_down', 'monitor_zone', 'block_node', 'trigger_alert'
}

def hunter_reward(prompts, completions, **kwargs):
    def _g(key, default, i):
        v = kwargs.get(key)
        if v is None: return default
        return v[i] if isinstance(v, (list, tuple)) and i < len(v) else default

    rewards = []
    for i, completion in enumerate(completions):
        red_node      = int(_g('red_node', 5, i))
        red_zone      = int(_g('red_zone', 1, i))
        suspicion     = float(_g('suspicion', 0.3, i))
        exfil_attempt = bool(_g('exfil_attempt', False, i))
        honeypot_hit  = bool(_g('honeypot_hit', False, i))

        txt = completion[-1]['content'] if isinstance(completion, list) else str(completion)
        txt = txt.strip()
        if txt.startswith('```'):
            txt = '\n'.join(l for l in txt.split('\n')
                             if not l.strip().startswith('```')).strip()
        try:
            data   = json.loads(txt)
            reward = 0.15
        except Exception:
            rewards.append(-0.25)
            continue

        action    = str(data.get('action_type', '')).lower().strip()
        target_n  = data.get('target_node')
        reasoning = str(data.get('reasoning', ''))

        if action in VALID_HUNTER_ACTIONS: reward += 0.10
        if len(reasoning) > 30:            reward += 0.05
        if len(reasoning) > 80:            reward += 0.05

        # Core hunter actions
        if action == 'reconstruct_path':
            reward += 0.50 if red_zone >= 2 else 0.10
        elif action == 'investigate_node':
            reward += 0.50 if target_n == red_node else 0.10
        elif action == 'analyze_anomaly':
            if suspicion > 0.60:   reward += 0.40
            elif suspicion > 0.30: reward += 0.20
            else:                  reward += 0.05
        elif action == 'monitor_zone':
            reward += 0.20 if red_zone >= 2 else 0.05
        elif action == 'block_node':
            reward += 0.30 if suspicion > 0.65 else -0.15
        elif action == 'trigger_alert':
            reward += 0.25 if (suspicion >= 0.65 or honeypot_hit) else -0.20
        elif action == 'stand_down':
            reward += 0.10 if (suspicion < 0.25 and not exfil_attempt) else -0.20

        # Exfil in progress — must not stand down
        if exfil_attempt:
            if action in ('block_node', 'trigger_alert', 'reconstruct_path'): reward += 0.40
            elif action == 'stand_down':                                       reward -= 0.40

        # Honeypot hit — correlate immediately
        if honeypot_hit:
            if action in ('analyze_anomaly', 'investigate_node', 'reconstruct_path'): reward += 0.30
            elif action == 'stand_down':                                               reward -= 0.30

        rewards.append(float(max(-0.50, min(1.0, reward))))
    return rewards

hunter_save = f'{OUT_DIR}/cipher-blue-threat-hunter-v1'

cfg = GRPOConfig(
    output_dir=f'{OUT_DIR}/cipher-blue-threat-hunter-v1-ckpt',
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=160,
    max_prompt_length=384,
    temperature=0.7,
    learning_rate=3e-5,
    bf16=True,
    logging_steps=10,
    save_steps=999999,
    report_to='none',
    optim='paged_adamw_8bit',
    lr_scheduler_type='cosine',
    seed=13,
    warmup_ratio=0.05,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=cfg,
    train_dataset=hunter_dataset,
    reward_funcs=[hunter_reward],
)

print('BLUE Threat Hunter v1 training started ...')
result = trainer.train()
print(f'Training complete | loss: {result.metrics.get("train_loss", "n/a")}')

model.save_pretrained(hunter_save)
tokenizer.save_pretrained(hunter_save)
print(f'Model saved: {hunter_save}')
print(f'Files: {sorted(os.listdir(hunter_save))}')

# ─── Zip both models ──────────────────────────────────────────────────────────
print('\nZipping models ...')
for folder_name in ['cipher-blue-surveillance-v1', 'cipher-blue-threat-hunter-v1']:
    folder_path = f'{OUT_DIR}/{folder_name}'
    zip_path    = f'{OUT_DIR}/{folder_name}'
    if os.path.exists(folder_path):
        shutil.make_archive(zip_path, 'zip', folder_path)
        size = os.path.getsize(f'{zip_path}.zip') / 1e6
        print(f'Zipped: {folder_name}.zip  ({size:.0f} MB)')
    else:
        print(f'MISSING: {folder_path} — check training ran successfully')

print('\nAll output files in /workspace/output/:')
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath) / 1e6
        print(f'  {f}  ({size:.1f} MB)')

print('\nTraining complete!')
