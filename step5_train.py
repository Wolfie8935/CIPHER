import os
os.environ['LLM_MODE']='stub'
from pathlib import Path
for f in ['rewards_log.csv','training_events.jsonl','prompt_evolution_log.jsonl']:
    Path(f).unlink(missing_ok=True)
from cipher.training.loop import run_training
run_training(n_episodes=50, verbose=False)
print('Done.')
