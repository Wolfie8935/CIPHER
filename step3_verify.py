import os
os.environ['LLM_MODE']='stub'
from pathlib import Path
for f in ['rewards_log.csv','training_events.jsonl','prompt_evolution_log.jsonl']:
    Path(f).unlink(missing_ok=True)
from cipher.training.loop import run_training
run_training(n_episodes=30, verbose=False)
import json
evols = [json.loads(l) for l in Path('prompt_evolution_log.jsonl').read_text().splitlines() if l.strip()]
print(f'Evolutions: {len(evols)} (expected >= 2)')
print(f'Rules in evo 1: RED={evols[0]["red_rules_count"]}, BLUE={evols[0]["blue_rules_count"]}')
assert len(evols) >= 2
print('PASSED')
