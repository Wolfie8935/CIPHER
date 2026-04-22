"""Phase 9 end-to-end verification (Step 3 + Step 4 from the spec)."""
import json
import os
from pathlib import Path

os.environ["LLM_MODE"] = "stub"

# Clean slate
for fname in ["rewards_log.csv", "training_events.jsonl", "prompt_evolution_log.jsonl"]:
    Path(fname).unlink(missing_ok=True)

from cipher.training.loop import run_training  # noqa: E402

run_training(n_episodes=30, verbose=False)

# ── Step 3: check evolutions ────────────────────────────────────────────────
evo_path = Path("prompt_evolution_log.jsonl")
assert evo_path.exists(), "prompt_evolution_log.jsonl not created!"

evols = [
    json.loads(line)
    for line in evo_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

print(f"Evolutions: {len(evols)} (expected >= 2)")
print(
    f"Rules in evolution 1: "
    f"RED={evols[0]['red_rules_count']}, "
    f"BLUE={evols[0]['blue_rules_count']}"
)
assert len(evols) >= 2, f"Expected >= 2 evolutions, got {len(evols)}"
print("Prompt evolution: PASSED")
print()

# ── Step 4: check prompt content ────────────────────────────────────────────
content = Path("cipher/agents/prompts/red_planner.txt").read_text(encoding="utf-8")
assert "LEARNED HEURISTICS" in content, "Prompt not updated!"

rules = [line for line in content.splitlines() if line.startswith("- LEARNED:")]
print(f"Rules appended to red_planner.txt: {len(rules)}")
for r in rules:
    print(" ", r[:90])
print("Prompt content: VERIFIED")
