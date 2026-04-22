import pandas as pd, json
from pathlib import Path

df = pd.read_csv('rewards_log.csv')
print(f'Total episodes logged: {len(df)}')
print(f'RED mean reward (first 10): {df.head(10)["red_total"].mean():+.3f}')
print(f'RED mean reward (last 10):  {df.tail(10)["red_total"].mean():+.3f}')
print(f'RED win rate overall:       {(df["red_total"] > 0).mean()*100:.1f}%')
print(f'Abort rate:                 {(df["terminal_reason"] == "aborted").mean()*100:.1f}%')
print(f'Exfil rate:                 {(df["red_exfil"] > 0).mean()*100:.1f}%')

evols = [json.loads(l) for l in Path('prompt_evolution_log.jsonl').read_text().splitlines() if l.strip()]
print(f'Prompt evolutions fired:    {len(evols)}')
total_rules = sum(e["red_rules_count"] + e["blue_rules_count"] for e in evols)
print(f'Total rules injected:       {total_rules}')
