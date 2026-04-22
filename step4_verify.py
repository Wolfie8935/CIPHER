from pathlib import Path
for fname in ['red_planner.txt', 'red_operative.txt', 'red_exfiltrator.txt']:
    content = Path(f'cipher/agents/prompts/{fname}').read_text()
    rules = [l for l in content.splitlines() if l.startswith('- LEARNED:')]
    print(f'{fname}: {len(rules)} rules injected')
    for r in rules:
        print(f'   {r[:90]}')
