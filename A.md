# A.md — Hybrid Mode Training Task
## Your job: Train 2 RED + 2 BLUE models on Google Colab (T4 GPU) using Unsloth + GRPO

> **Run in parallel** with your teammate doing B.md (red team code improvements).
> When done, share the 4 zipped model folders so hybrid mode can load them.

---

## Context: Why Hybrid Is Broken

Currently `LLM_MODE=hybrid` routes:
- **RED Planner** → local LoRA model at `red trained/cipher-red-planner/`
- All other 7 agents → NVIDIA NIM API

Hybrid fails because:
1. The single RED Planner model was trained with only 200 scenarios and 3 epochs — it underfits badly
2. There is NO trained BLUE model at all — blue side is entirely NVIDIA NIM, which is slow and expensive
3. The reward function in the existing notebooks is too simple (no trap awareness, no dead-drop scoring)
4. `RED_PLANNER_LORA_PATH` env var must point to a valid adapter — if missing, it falls back to NIM silently

The goal: train **2 RED specialists** and **2 BLUE specialists**, each better than the current baseline, so hybrid mode actually runs with local inference for more agents.

---

## What You Need

### Hardware
- **4 separate Google Colab sessions** (free tier is fine, but Pro gives longer runtime)
- Each session: **Runtime → Change runtime type → T4 GPU**
- Each session needs ~25–35 min of GPU time

### Accounts / Access
- Google account for Colab
- GitHub access to clone: `https://github.com/wolfie8935/CIPHER`
  - Token: `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`

### Python Packages (auto-installed in each notebook)
```
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
trl>=0.12.0
datasets>=2.19.0
accelerate>=0.30.0
peft>=0.11.0
bitsandbytes>=0.43.0
torch (comes with Colab)
```

---

## Session 1 — RED Planner v2 (Improved)

> File to open/create: `CIPHER_Training_Colab.ipynb` (already exists in repo root)
> **Key changes from v1**: larger dataset (400 samples), more epochs (5), richer reward fn with trap awareness

### Cell 1 — Verify GPU
```python
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else '❌ NO GPU')
```

### Cell 2 — Install + Clone
```python
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install "trl>=0.12.0" "datasets>=2.19.0" "accelerate>=0.30.0" "peft>=0.11.0" "bitsandbytes>=0.43.0"
!git clone https://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@github.com/wolfie8935/CIPHER
import sys; sys.path.insert(0, '/content/CIPHER')
!pip install -e /content/CIPHER --quiet
print('✅ Done')
```

### Cell 3 — Load Model
```python
from unsloth import FastLanguageModel

MAX_SEQ_LEN = 768  # increased from 512

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Llama-3.2-1B-Instruct',
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,               # increased from 16 — more capacity
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    lora_alpha=64,      # increased from 32
    lora_dropout=0.05,  # small dropout to prevent overfitting
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=42,
)
params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'✅ RED Planner v2 loaded | Trainable: {params:,}')
```

### Cell 4 — Build Richer Dataset (400 samples)
```python
from datasets import Dataset
import random, json

VALID_ACTIONS = {
    'move','read_file','exfiltrate','wait','abort',
    'plant_false_trail','plant_temporal_decoy','plant_honeypot_poison',
    'write_dead_drop','write_corrupted_drop'
}

SYSTEM_PROMPT = """You are RED PLANNER in CIPHER, an adversarial cyber-security simulation.
Mission: Navigate through zones 1→2→3 and exfiltrate all target files from the High-Value Target node.
Stay below BLUE detection. Use traps to confuse BLUE. Manage suspicion carefully.

Respond with ONLY a single JSON object:
{"action_type": "move", "target_node": 25, "target_file": null, "reasoning": "Moving toward zone 3 via low-suspicion path."}

Valid actions: move, read_file, exfiltrate, wait, abort, plant_false_trail, plant_temporal_decoy, plant_honeypot_poison, write_dead_drop, write_corrupted_drop
Rules:
- NEVER exfiltrate unless at zone 3 HVT node
- ABORT only if suspicion > 0.80 AND blue_confidence > 0.60
- Use plant_false_trail to mislead BLUE when suspicion rises
- Use write_dead_drop at context resets to preserve memory"""

def red_planner_v2_reward(completions, **kwargs):
    def _get(key, default, idx):
        v = kwargs.get(key)
        if v is None: return default
        return v[idx] if isinstance(v,(list,tuple)) and idx < len(v) else default
    rewards = []
    for i, completion in enumerate(completions):
        zone      = int(_get('zone', 1, i))
        suspicion = float(_get('suspicion', 0.3, i))
        step      = int(_get('step', 1, i))
        blue_conf = float(_get('blue_conf', 0.2, i))
        traps_placed = int(_get('traps_placed', 0, i))
        context_reset = bool(_get('context_reset', False, i))

        clean = completion.strip()
        if clean.startswith('```'):
            clean = '\n'.join(l for l in clean.split('\n') if not l.strip().startswith('```')).strip()
        try:
            data = json.loads(clean)
            reward = 0.20
        except:
            rewards.append(-0.30)
            continue

        action    = str(data.get('action_type','')).lower().strip()
        reasoning = str(data.get('reasoning',''))
        target_f  = data.get('target_file')

        # Valid action
        if action in VALID_ACTIONS: reward += 0.10

        # Reasoning quality
        if len(reasoning) > 30: reward += 0.05
        if len(reasoning) > 80: reward += 0.05

        # Zone-based logic
        if zone == 3:
            if action == 'exfiltrate' and target_f: reward += 0.60  # MAIN OBJECTIVE
            elif action == 'read_file':              reward += 0.20
            elif action == 'move':                   reward += 0.05
            elif action == 'abort':                  reward -= 0.30
        elif zone == 2:
            if action == 'move':                     reward += 0.35
            elif action == 'exfiltrate':             reward -= 0.40  # wrong zone
        elif zone == 1:
            if action == 'move':                     reward += 0.30
            elif action == 'exfiltrate':             reward -= 0.40

        # Suspicion management
        if suspicion > 0.75:
            if action in ('wait','abort','plant_false_trail','plant_temporal_decoy'):
                reward += 0.20
            elif action == 'move':
                reward -= 0.15
        elif suspicion < 0.30:
            if action == 'move':  reward += 0.10  # low risk, keep moving

        # BLUE awareness — don't abort prematurely
        if blue_conf < 0.60 and action == 'abort':
            reward -= 0.25  # no need to abort yet

        # Trap usage is good when suspicion rising
        if suspicion > 0.50 and action in ('plant_false_trail','plant_honeypot_poison','plant_temporal_decoy'):
            reward += 0.15

        # Context reset: write dead drop is correct
        if context_reset and action == 'write_dead_drop':
            reward += 0.20

        rewards.append(float(max(-0.5, min(1.0, reward))))
    return rewards

rng = random.Random(42)
zone_nodes = {1: list(range(1,16)), 2: list(range(16,31)), 3: list(range(31,46))}
zone_files = {
    1: ['access.log','config.ini','users.txt','sshd_config'],
    2: ['database.conf','api_keys.json','secrets.yml','vault.key'],
    3: ['classified.pdf','exfil_target.zip','credentials.db','master_keys.enc']
}
samples = []
for _ in range(400):  # 2x more than before
    zone      = rng.choices([1,2,3], weights=[0.20,0.35,0.45])[0]  # more zone 3
    suspicion = round(rng.uniform(0.05, 0.95), 2)
    step      = rng.randint(1, 15)
    blue_conf = round(rng.uniform(0.0, 0.9), 2)
    traps_placed = rng.randint(0, 3)
    context_reset = rng.random() < 0.15
    node  = rng.choice(zone_nodes[zone])
    files = rng.sample(zone_files[zone], rng.randint(1,2))

    obs = (f"Zone: {zone} | Node: {node} | Suspicion: {suspicion} | "
           f"Step: {step} | BlueConf: {blue_conf} | Files: {', '.join(files)} | "
           f"TrapsPlaced: {traps_placed} | ContextReset: {context_reset}")
    conv = [{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':obs}]
    samples.append({
        'prompt': tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True),
        'zone': zone, 'suspicion': suspicion, 'step': step,
        'blue_conf': blue_conf, 'traps_placed': traps_placed, 'context_reset': context_reset,
    })

dataset = Dataset.from_list(samples)
print(f'✅ Dataset: {len(dataset)} RED Planner v2 scenarios')
```

### Cell 5 — Train (5 epochs, ~35 min)
```python
from trl import GRPOTrainer, GRPOConfig

cfg = GRPOConfig(
    output_dir='./cipher-red-planner-v2-output',
    num_train_epochs=5,           # up from 3
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=6,            # up from 4
    max_completion_length=192,    # up from 128
    max_prompt_length=384,        # up from 256
    temperature=0.8,
    learning_rate=3e-5,           # lower LR for stability
    fp16=True,
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
    train_dataset=dataset,
    reward_funcs=red_planner_v2_reward,
)

print('🚀 RED Planner v2 training started...')
result = trainer.train()
print(f'✅ DONE. Loss: {result.metrics.get("train_loss","n/a")}')

model.save_pretrained('./cipher-red-planner-v2')
tokenizer.save_pretrained('./cipher-red-planner-v2')
print('✅ Saved: ./cipher-red-planner-v2/')
```

### Cell 6 — Zip + Download
```python
import shutil
from google.colab import files
shutil.make_archive('cipher-red-planner-v2', 'zip', './cipher-red-planner-v2')
files.download('cipher-red-planner-v2.zip')
print('✅ Download started!')
```

---

## Session 2 — RED Analyst Specialist (NEW model)

> This is a brand new model. The RED Analyst (`red_analyst_01`) currently only uses NVIDIA NIM.
> We train a local specialist for it so hybrid mode can run it locally too.

- **Same Colab setup** (Cells 1 & 2 identical to Session 1)
- Load model same way but with `random_state=123`

### Dataset & Reward for RED Analyst
```python
ANALYST_SYSTEM_PROMPT = """You are RED ANALYST in CIPHER — your job is intelligence gathering and reconnaissance.
You analyze network topology, identify high-value targets, read files at nodes, and write dead drops for the team.
You do NOT initiate exfiltration — that's RED EXFILTRATOR's job.

Respond with ONLY a single JSON:
{"action_type": "read_file", "target_file": "api_keys.json", "target_node": null, "reasoning": "Gathering credentials at current node."}

Valid actions: read_file, write_dead_drop, read_dead_drop, wait, plant_temporal_decoy"""

def analyst_reward(completions, **kwargs):
    def _get(key, default, idx):
        v = kwargs.get(key)
        if v is None: return default
        return v[idx] if isinstance(v,(list,tuple)) and idx < len(v) else default
    rewards = []
    for i, completion in enumerate(completions):
        zone      = int(_get('zone', 1, i))
        suspicion = float(_get('suspicion', 0.3, i))
        has_files = bool(_get('has_files', True, i))
        has_drops = bool(_get('has_drops', False, i))

        clean = completion.strip()
        if clean.startswith('```'):
            clean = '\n'.join(l for l in clean.split('\n') if not l.strip().startswith('```')).strip()
        try:
            data = json.loads(clean); reward = 0.15
        except:
            rewards.append(-0.25); continue

        action    = str(data.get('action_type','')).lower().strip()
        reasoning = str(data.get('reasoning',''))
        ANALYST_ACTIONS = {'read_file','write_dead_drop','read_dead_drop','wait','plant_temporal_decoy'}

        if action in ANALYST_ACTIONS: reward += 0.15
        if len(reasoning) > 30: reward += 0.05
        if len(reasoning) > 80: reward += 0.05

        # Core analyst duty: gather intel
        if has_files and action == 'read_file':    reward += 0.40
        if has_drops and action == 'read_dead_drop': reward += 0.30
        if not has_files and action == 'read_file':  reward -= 0.20

        # Don't waste budget on wrong actions
        if action == 'exfiltrate': reward -= 0.50  # not analyst's job
        if action == 'move':       reward -= 0.10  # planner handles movement

        # Suspicion management
        if suspicion > 0.70 and action == 'wait':  reward += 0.15
        if suspicion > 0.70 and action == 'plant_temporal_decoy': reward += 0.20

        rewards.append(float(max(-0.5, min(1.0, reward))))
    return rewards

# Build 300 analyst scenarios
rng2 = random.Random(123)
samples2 = []
for _ in range(300):
    zone      = rng2.choices([1,2,3], weights=[0.35,0.40,0.25])[0]
    suspicion = round(rng2.uniform(0.05, 0.85), 2)
    has_files = rng2.random() < 0.70
    has_drops = rng2.random() < 0.30
    node  = rng2.randint(1, 45)
    obs = (f"Zone: {zone} | Node: {node} | Suspicion: {suspicion} | "
           f"HasFiles: {has_files} | HasDeadDrops: {has_drops}")
    conv = [{'role':'system','content':ANALYST_SYSTEM_PROMPT},{'role':'user','content':obs}]
    samples2.append({
        'prompt': tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True),
        'zone': zone, 'suspicion': suspicion, 'has_files': has_files, 'has_drops': has_drops,
    })
dataset2 = Dataset.from_list(samples2)
print(f'✅ Analyst dataset: {len(dataset2)} scenarios')
```

- **Train with same GRPOConfig** (3 epochs is enough for analyst since it's simpler)
- Save as `./cipher-red-analyst-v1/`
- Zip + download as `cipher-red-analyst-v1.zip`

---

## Session 3 — BLUE Surveillance Specialist (Improved)

> The existing blue notebook trains a generic "monitor". We now train **BlueSurveillance** (`blue_surveillance_01`) specifically.

- Same Colab setup (Cells 1 & 2)
- `random_state=99`

### Dataset & Reward for Blue Surveillance
```python
BLUE_SURV_PROMPT = """You are BLUE SURVEILLANCE in CIPHER — you watch network traffic across all nodes.
Your job: detect RED team movement patterns, identify anomalies, and feed intelligence to your team.
Focus on zone transitions, suspicious protocols, and honeypot triggers.

Respond with ONLY JSON:
{"action_type": "investigate_node", "target_node": 23, "reasoning": "Anomalous traffic spike at node 23 zone 2.", "confidence": 0.75}

Valid actions: investigate_node, trigger_alert, analyze_anomaly, stand_down"""

def blue_surv_reward(completions, **kwargs):
    def _get(key, default, idx):
        v = kwargs.get(key)
        if v is None: return default
        return v[idx] if isinstance(v,(list,tuple)) and idx < len(v) else default
    rewards = []
    for i, completion in enumerate(completions):
        red_node      = int(_get('red_node', 5, i))
        red_zone      = int(_get('red_zone', 1, i))
        suspicion     = float(_get('suspicion', 0.3, i))
        exfil_attempt = bool(_get('exfil_attempt', False, i))
        honeypot_hit  = bool(_get('honeypot_hit', False, i))

        clean = completion.strip()
        if clean.startswith('```'):
            clean = '\n'.join(l for l in clean.split('\n') if not l.strip().startswith('```')).strip()
        try:
            data = json.loads(clean); reward = 0.15
        except:
            rewards.append(-0.25); continue

        action     = str(data.get('action_type','')).lower().strip()
        target     = data.get('target_node')
        reasoning  = str(data.get('reasoning',''))
        confidence = float(data.get('confidence', 0.5))
        SURV_ACTIONS = {'investigate_node','trigger_alert','analyze_anomaly','stand_down'}

        if action in SURV_ACTIONS: reward += 0.10
        if len(reasoning) > 30: reward += 0.05
        if len(reasoning) > 80: reward += 0.05

        # Correct node investigation
        if action == 'investigate_node':
            if target == red_node:   reward += 0.50  # hit!
            elif target is not None: reward += 0.10  # investigating wrong node, at least active

        # Alert when warranted
        if suspicion >= 0.65:
            if action == 'trigger_alert':  reward += 0.40
            elif action == 'stand_down':   reward -= 0.35
        elif suspicion <= 0.25:
            if action == 'trigger_alert':  reward -= 0.30  # false alarm

        # Exfil attempt — must not stand down
        if exfil_attempt:
            if action == 'trigger_alert':     reward += 0.40
            elif action == 'investigate_node': reward += 0.20
            elif action == 'stand_down':       reward -= 0.50

        # Honeypot triggered — major intel event
        if honeypot_hit:
            if action in ('trigger_alert','investigate_node'): reward += 0.35
            elif action == 'stand_down': reward -= 0.40

        # Confidence calibration
        if confidence > 0.8 and suspicion < 0.40: reward -= 0.10  # overconfident
        if confidence < 0.4 and suspicion > 0.70: reward -= 0.10  # underconfident

        rewards.append(float(max(-0.5, min(1.0, reward))))
    return rewards

# 350 surveillance scenarios
rng3 = random.Random(77)
samples3 = []
for _ in range(350):
    red_zone      = rng3.choices([1,2,3], weights=[0.25,0.40,0.35])[0]
    red_node      = rng3.randint(1, 45)
    suspicion     = round(rng3.uniform(0.05, 0.95), 2)
    exfil_attempt = rng3.random() < 0.25
    honeypot_hit  = rng3.random() < 0.15
    traffic = 'HIGH' if suspicion > 0.6 else ('MEDIUM' if suspicion > 0.3 else 'LOW')
    obs = (f"Zone: {red_zone} | ActiveNode: {red_node} | Traffic: {traffic} | "
           f"Suspicion: {suspicion} | ExfilAttempt: {exfil_attempt} | HoneypotHit: {honeypot_hit}")
    conv = [{'role':'system','content':BLUE_SURV_PROMPT},{'role':'user','content':obs}]
    samples3.append({
        'prompt': tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True),
        'red_node': red_node, 'red_zone': red_zone, 'suspicion': suspicion,
        'exfil_attempt': exfil_attempt, 'honeypot_hit': honeypot_hit,
    })
dataset3 = Dataset.from_list(samples3)
print(f'✅ Surveillance dataset: {len(dataset3)} scenarios')
```

- Train 4 epochs. Save as `./cipher-blue-surveillance-v1/`
- Zip + download `cipher-blue-surveillance-v1.zip`

---

## Session 4 — BLUE Threat Hunter Specialist (NEW)

> `BlueThreatHunter` (`blue_threat_hunter_01`) focuses on tracking RED's path and reconstructing attack chains.

- Same Colab setup
- `random_state=55`

### System Prompt
```
You are BLUE THREAT HUNTER in CIPHER. You track RED team movement, reconstruct attack paths, 
and identify the most likely next target node. You specialize in path analysis and proactive blocking.

Respond with ONLY JSON:
{"action_type": "reconstruct_path", "target_node": 31, "reasoning": "RED likely heading to zone 3 HVT via node 31.", "confidence": 0.80}

Valid actions: investigate_node, reconstruct_path, analyze_anomaly, stand_down
```

### Reward Logic (summary — implement similar to above)
- `+0.50` for `reconstruct_path` when RED is in zone 2 or 3 (active threat tracking)
- `+0.40` for `investigate_node` hitting the actual RED node
- `+0.30` for `analyze_anomaly` when suspicion > 0.6
- `-0.35` for `stand_down` when exfil is being attempted
- `+0.20` for correct zone prediction in reasoning
- 300 scenarios, 4 epochs
- Save as `./cipher-blue-threat-hunter-v1/`
- Zip + download `cipher-blue-threat-hunter-v1.zip`

---

## After All 4 Sessions Complete

### What to hand off
Send these 4 zip files to the person doing B.md (or put in shared drive):
1. `cipher-red-planner-v2.zip`
2. `cipher-red-analyst-v1.zip`
3. `cipher-blue-surveillance-v1.zip`
4. `cipher-blue-threat-hunter-v1.zip`

### Where to place them locally
Unzip each into the project:
```
OPENENV FINAL/
  red trained/
    cipher-red-planner-v2/      ← unzip here
    cipher-red-analyst-v1/      ← unzip here
  blue trained/                 ← create this folder
    cipher-blue-surveillance-v1/
    cipher-blue-threat-hunter-v1/
```

### .env updates needed (tell the B.md person)
```env
RED_PLANNER_LORA_PATH=red trained/cipher-red-planner-v2
RED_ANALYST_LORA_PATH=red trained/cipher-red-analyst-v1
BLUE_SURVEILLANCE_LORA_PATH=blue trained/cipher-blue-surveillance-v1
BLUE_THREAT_HUNTER_LORA_PATH=blue trained/cipher-blue-threat-hunter-v1
```

> The B.md person will update `base_agent.py` → `_is_hybrid_specialist()` and `_act_lora()` to support multiple specialists.

---

## Checklist
- [ ] Session 1: RED Planner v2 trained and downloaded
- [ ] Session 2: RED Analyst v1 trained and downloaded
- [ ] Session 3: BLUE Surveillance v1 trained and downloaded
- [ ] Session 4: BLUE Threat Hunter v1 trained and downloaded
- [ ] All 4 zips shared with teammate
- [ ] Confirm each model folder has: `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.json`
