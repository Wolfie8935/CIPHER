#!/usr/bin/env python3
"""CIPHER — RED Agents Training (RunPod — L4 optimised)
Trains RED Planner v2 then RED Analyst v1 sequentially.
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
PLANNER_SYSTEM = (
    "You are RED PLANNER in CIPHER. Your mission: infiltrate a 50-node enterprise network, "
    "reach zone 3, and exfiltrate classified files from the High Value Target node. "
    "Staying idle is failure. You control strategic movement, zone progression, and abort decisions.\n\n"
    "RULES:\n"
    "- Move from zone 1 to zone 2 to zone 3 as fast as stealth allows\n"
    "- At a zone 3 HVT node: read files then exfiltrate\n"
    "- Use plant_false_trail or plant_temporal_decoy when suspicion > 0.50\n"
    "- ABORT only if suspicion > 0.80 AND blue_confidence > 0.60 simultaneously\n"
    "- NEVER wait when suspicion < 0.40 — movement beats idle\n"
    "- NEVER exfiltrate outside zone 3 HVT\n\n"
    "Respond with ONLY valid JSON: "
    '{"action_type": "move", "target_node": 12, "target_file": null, '
    '"reasoning": "Moving toward zone 2. Suspicion is low."}'
)

ANALYST_SYSTEM = (
    "You are RED ANALYST in CIPHER. Your mission: gather intelligence by reading files at nodes, "
    "writing dead drops so your team retains memory across context resets, and reading dead drops "
    "left by other agents. You do NOT move — that is the Planner's job. You do NOT exfiltrate — "
    "that is the Exfiltrator's job.\n\n"
    "RULES:\n"
    "- READ_FILE whenever files are present at your current node\n"
    "- WRITE_DEAD_DROP on any context reset to preserve team memory\n"
    "- READ_DEAD_DROP when drops are available from prior agents\n"
    "- Never move — issue WAIT if nothing else applies\n"
    "- Never exfiltrate — penalised heavily\n\n"
    "Respond with ONLY valid JSON: "
    '{"action_type": "read_file", "target_node": null, "target_file": "config.db", '
    '"reasoning": "Files present at node. Reading for intelligence."}'
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
            node          = step_info.get('red_node', step_info.get('node', 1))
            prev_zone     = step_info.get('prev_zone', max(1, zone - 1))
            files_here    = step_info.get('files_at_node', [])
            drops_avail   = step_info.get('dead_drops_available', 0)
            ctx_reset     = step_info.get('context_reset', False)
            steps_idle    = step_info.get('steps_idle', 0)
            adj_nodes     = step_info.get('adjacent_nodes', [])
            hp_near       = step_info.get('honeypot_nodes_nearby', [])
            terminal      = step_info.get('terminal_reason', '')
            step += 1

            obs_str = (
                f"Zone: {zone} | PrevZone: {prev_zone} | Node: {node} | Step: {step}\n"
                f"Suspicion: {suspicion:.2f} | BlueConfidence: {blue_conf:.2f} | StepsIdle: {steps_idle}\n"
                f"FilesHere: {files_here} | AdjacentNodes: {adj_nodes}\n"
                f"HoneypotNearby: {hp_near} | ContextReset: {ctx_reset}\n"
                f"DeadDropsAvailable: {drops_avail}"
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
                'zone':          zone,
                'prev_zone':     prev_zone,
                'suspicion':     suspicion,
                'blue_conf':     blue_conf,
                'steps_idle':    steps_idle,
                'context_reset': ctx_reset,
                'files_here':    len(files_here),
                'drops_avail':   drops_avail,
                'node':          node,
            })

            if terminal in ('detected', 'stalled', 'aborted', 'exfiltrated'):
                break

    print(f'Collected {len(samples)} steps from {n_episodes} episodes for {agent_role}')
    return Dataset.from_list(samples)

N_EPISODES = 150

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 — RED PLANNER v2
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('AGENT 1: RED PLANNER v2')
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
    random_state=42,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f'Loaded | Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

planner_dataset = run_episodes(N_EPISODES, PLANNER_SYSTEM, 'RED Planner')

VALID_RED_ACTIONS = {
    'move', 'read_file', 'exfiltrate', 'wait', 'abort',
    'plant_false_trail', 'plant_temporal_decoy', 'plant_honeypot_poison',
    'write_dead_drop', 'write_corrupted_drop'
}

def planner_reward(prompts, completions, **kwargs):
    def _g(key, default, i):
        v = kwargs.get(key)
        if v is None: return default
        return v[i] if isinstance(v, (list, tuple)) and i < len(v) else default

    rewards = []
    for i, completion in enumerate(completions):
        zone          = int(_g('zone', 1, i))
        prev_zone     = int(_g('prev_zone', zone, i))
        suspicion     = float(_g('suspicion', 0.3, i))
        blue_conf     = float(_g('blue_conf', 0.2, i))
        steps_idle    = int(_g('steps_idle', 0, i))
        context_reset = bool(_g('context_reset', False, i))

        txt = completion[-1]['content'] if isinstance(completion, list) else str(completion)
        txt = txt.strip()
        if txt.startswith('```'):
            txt = '\n'.join(l for l in txt.split('\n')
                             if not l.strip().startswith('```')).strip()
        try:
            data   = json.loads(txt)
            reward = 0.20
        except Exception:
            rewards.append(-0.30)
            continue

        action    = str(data.get('action_type', '')).lower().strip()
        reasoning = str(data.get('reasoning', ''))
        target_f  = data.get('target_file')
        target_n  = data.get('target_node')

        if action in VALID_RED_ACTIONS: reward += 0.10
        if len(reasoning) > 30:         reward += 0.05
        if len(reasoning) > 80:         reward += 0.05

        # Zone-progression bonus
        if zone > prev_zone:
            if prev_zone == 1 and zone == 2:   reward += 0.40
            elif prev_zone == 2 and zone == 3: reward += 0.60

        # Zone-based action scoring
        if zone == 3:
            if action == 'exfiltrate' and target_f: reward += 0.60
            elif action == 'read_file':             reward += 0.20
            elif action == 'move' and target_n:     reward += 0.05
            elif action == 'abort':                 reward -= 0.35
            elif action == 'wait':                  reward -= 0.10
        elif zone == 2:
            if action == 'move' and target_n:           reward += 0.35
            elif action == 'exfiltrate':                reward -= 0.40
            elif action == 'wait' and suspicion < 0.60: reward -= 0.15
        elif zone == 1:
            if action == 'move' and target_n:           reward += 0.30
            elif action == 'exfiltrate':                reward -= 0.40
            elif action == 'wait' and suspicion < 0.60: reward -= 0.15

        if suspicion < 0.40 and action == 'move':     reward += 0.20
        if steps_idle >= 2 and action == 'wait':       reward -= 0.20

        if suspicion > 0.75:
            if action in ('wait', 'plant_false_trail',
                          'plant_temporal_decoy', 'plant_honeypot_poison'):
                reward += 0.20
            elif action == 'move':                      reward -= 0.15

        if action == 'wait' and suspicion <= 0.75:     reward -= 0.10

        if action == 'abort':
            if suspicion > 0.80 and blue_conf > 0.60:  reward += 0.10
            else:                                       reward -= 0.35

        if suspicion > 0.50 and action in (
            'plant_false_trail', 'plant_honeypot_poison', 'plant_temporal_decoy'):
            reward += 0.15

        if context_reset and action == 'write_dead_drop': reward += 0.20

        rewards.append(float(max(-0.50, min(1.0, reward))))
    return rewards

planner_save = f'{OUT_DIR}/cipher-red-planner-v1'

cfg = GRPOConfig(
    output_dir=f'{OUT_DIR}/cipher-red-planner-v1-ckpt',
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
    seed=42,
    warmup_ratio=0.05,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=cfg,
    train_dataset=planner_dataset,
    reward_funcs=[planner_reward],
)

print('RED Planner v2 training started ...')
result = trainer.train()
print(f'Training complete | loss: {result.metrics.get("train_loss", "n/a")}')

model.save_pretrained(planner_save)
tokenizer.save_pretrained(planner_save)
print(f'Model saved: {planner_save}')
print(f'Files: {sorted(os.listdir(planner_save))}')

# ─── Clear GPU memory before Agent 2 ─────────────────────────────────────────
del model, trainer, tokenizer
gc.collect()
torch.cuda.empty_cache()
used  = torch.cuda.memory_allocated(0) / 1e9
total = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU cleared | free est. : {total - used:.1f} GB')

# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 — RED ANALYST v1
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('AGENT 2: RED ANALYST v1')
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
    random_state=7,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f'Loaded | Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

analyst_dataset = run_episodes(N_EPISODES, ANALYST_SYSTEM, 'RED Analyst')

def analyst_reward(prompts, completions, **kwargs):
    def _g(key, default, i):
        v = kwargs.get(key)
        if v is None: return default
        return v[i] if isinstance(v, (list, tuple)) and i < len(v) else default

    rewards = []
    for i, completion in enumerate(completions):
        files_here    = int(_g('files_here', 0, i))
        drops_avail   = int(_g('drops_avail', 0, i))
        context_reset = bool(_g('context_reset', False, i))

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
        reasoning = str(data.get('reasoning', ''))
        target_f  = data.get('target_file')

        if len(reasoning) > 30: reward += 0.05
        if len(reasoning) > 80: reward += 0.05

        # Core analyst actions
        if action == 'read_file':
            if files_here > 0:
                reward += 0.50
                if target_f: reward += 0.10
            else:
                reward -= 0.20
        elif action == 'write_dead_drop':
            reward += 0.50 if context_reset else 0.10
        elif action == 'read_dead_drop':
            reward += 0.40 if drops_avail > 0 else -0.10
        elif action == 'wait':
            if files_here == 0 and drops_avail == 0 and not context_reset:
                reward += 0.10
            else:
                reward -= 0.15
        elif action == 'move':
            reward -= 0.35
        elif action == 'exfiltrate':
            reward -= 0.50
        elif action == 'abort':
            reward -= 0.30

        rewards.append(float(max(-0.50, min(1.0, reward))))
    return rewards

analyst_save = f'{OUT_DIR}/cipher-red-analyst-v1'

cfg = GRPOConfig(
    output_dir=f'{OUT_DIR}/cipher-red-analyst-v1-ckpt',
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
    seed=7,
    warmup_ratio=0.05,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=cfg,
    train_dataset=analyst_dataset,
    reward_funcs=[analyst_reward],
)

print('RED Analyst v1 training started ...')
result = trainer.train()
print(f'Training complete | loss: {result.metrics.get("train_loss", "n/a")}')

model.save_pretrained(analyst_save)
tokenizer.save_pretrained(analyst_save)
print(f'Model saved: {analyst_save}')
print(f'Files: {sorted(os.listdir(analyst_save))}')

# ─── Zip both models ──────────────────────────────────────────────────────────
print('\nZipping models ...')
for folder_name in ['cipher-red-planner-v1', 'cipher-red-analyst-v1']:
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
