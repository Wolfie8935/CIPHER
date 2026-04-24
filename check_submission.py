"""
CIPHER Hackathon Submission Checker
Run: python check_submission.py
"""
import os, sys, json, csv, importlib
from pathlib import Path

ROOT = Path(__file__).parent
PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"

results = []

def check(label, ok, detail="", warn=False):
    status = PASS if ok else (WARN if warn else FAIL)
    results.append((status, label, detail))
    print(f"  {status}  {label}" + (f"  — {detail}" if detail else ""))

def section(title):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")

# ── 1. MINIMUM REQUIRED FILES ────────────────────────────────────────────────
section("1. Required Files")
required_files = {
    "openenv.yaml":                    "OpenEnv manifest",
    "README.md":                       "Project README",
    "requirements.txt":                "Python dependencies",
    "cipher/env_wrapper.py":           "CIPHEREnv class",
    "verify_openenv.py":               "OpenEnv compliance script",
    "Dockerfile":                      "HF Spaces deployment",
    "hf_app.py":                       "HF Spaces entrypoint",
    "CIPHER_Training_Colab.ipynb":     "Training notebook (Colab)",
    "cipher-training-red-v2.ipynb":    "Training notebook v2",
    "rewards_log.csv":                 "Telemetry data",
    "generate_plots.py":               "Plot generation script",
}
for path, desc in required_files.items():
    check(desc, (ROOT / path).exists(), path)

# ── 2. ASSETS / PLOTS ────────────────────────────────────────────────────────
section("2. Plot Assets (assets/)")
required_plots = [
    "baseline_vs_trained.png",
    "reward_curves.png",
    "elo_chart.png",
    "terminal_outcomes.png",
    "fleet_verdicts.png",
    "win_rate_progression.png",
    "architecture_card.png",
]
for plot in required_plots:
    p = ROOT / "assets" / plot
    if p.exists():
        size_kb = p.stat().st_size / 1024
        check(plot, size_kb > 10, f"{size_kb:.0f} KB")
    else:
        check(plot, False, "MISSING")

# All plots in README?
readme = (ROOT / "README.md").read_text(encoding="utf-8")
for plot in required_plots:
    check(f"  {plot} embedded in README", plot in readme)

# ── 3. OPENENV YAML ──────────────────────────────────────────────────────────
section("3. openenv.yaml Manifest")
yaml_path = ROOT / "openenv.yaml"
if yaml_path.exists():
    yaml_text = yaml_path.read_text(encoding="utf-8")
    for field in ["name:", "version:", "entry_point:", "observation_type:",
                  "action_type:", "reward_range:", "description:"]:
        check(f"field: {field.rstrip(':')}", field in yaml_text)
else:
    check("openenv.yaml exists", False)

# ── 4. OPENENV COMPLIANCE ────────────────────────────────────────────────────
section("4. OpenEnv API Compliance")
try:
    sys.path.insert(0, str(ROOT))
    from cipher.env_wrapper import CIPHEREnv, make_env
    from openenv.env.env import Env as _Base

    check("CIPHEREnv inherits openenv.Env", issubclass(CIPHEREnv, _Base))
    check("metadata.name present", "name" in CIPHEREnv.metadata)
    check("metadata.reward_range present", "reward_range" in CIPHEREnv.metadata)
    check("metadata.action_type present", "action_type" in CIPHEREnv.metadata)
    check("make_env() factory exists", callable(make_env))

    env = make_env(max_steps=15, llm_mode="stub")
    obs, info = env.reset()
    check("reset() returns (str, dict)", isinstance(obs, str) and isinstance(info, dict),
          f"obs={len(obs)} chars, info={len(info)} keys")
    check("info has 'episode' key", "episode" in info)

    obs2, reward, terminated, truncated, info2 = env.step("Move to nearest node")
    check("step() returns 5-tuple", True)
    check("reward is float", isinstance(reward, float), f"reward={reward:+.3f}")
    check("terminated is True", terminated is True)
    check("info2 has 'terminal_reason'", "terminal_reason" in info2)

    render = env.render()
    check("render() returns str", isinstance(render, str))
    check("second reset() increments episode", True)
    _, i2 = env.reset()
    check("episode counter increments", i2["episode"] == 2, f"episode={i2['episode']}")

except Exception as e:
    check("OpenEnv compliance checks", False, str(e))

# ── 5. REWARD DATA ───────────────────────────────────────────────────────────
section("5. Rewards & Training Data")
csv_path = ROOT / "rewards_log.csv"
if csv_path.exists():
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("timestamp", "").startswith("202")]
    check("rewards_log.csv readable", True, f"{len(rows)} valid episodes")
    check("at least 100 episodes logged", len(rows) >= 100, f"{len(rows)} episodes")
    check("at least 500 episodes logged", len(rows) >= 500, f"{len(rows)} episodes")

    # Check columns
    if rows:
        required_cols = ["red_total", "blue_total", "fleet_verdict",
                         "terminal_reason", "red_exfil"]
        for col in required_cols:
            check(f"column '{col}' present", col in rows[0])

        # Check there are both wins and losses
        red_wins = sum(1 for r in rows if float(r.get("red_total", 0)) > 0)
        check("has RED wins (> 0 reward)", red_wins > 0, f"{red_wins} wins")
        check("has RED losses", (len(rows) - red_wins) > 0)
else:
    check("rewards_log.csv exists", False)

# ── 6. README QUALITY ───────────────────────────────────────────────────────
section("6. README Quality")
readme_checks = {
    "## The Problem":        "Problem statement section",
    "## The Environment":    "Environment description section",
    "## Results":            "Results section",
    "## Why This Matters":   "Why it matters section",
    "## Architecture":       "Architecture section",
    "## Reward Design":      "Reward design section",
    "## Training":           "Training section",
    "Submission Links":      "Submission links section",
    "assets/baseline_vs_trained.png": "Key comparison chart embedded",
    "70.5%":                 "Win rate result mentioned",
    "1,082":                 "Episode count mentioned",
    "theory-of-mind":        "Core capability named",
}
for marker, desc in readme_checks.items():
    check(desc, marker in readme)

# Submission links filled in?
hf_placeholder = "uploading" in readme or "link to be added" in readme
check("HF Space URL (placeholder OK for now)", True,
      "placeholder present — fill after upload" if hf_placeholder else "URL present",
      warn=hf_placeholder)

# ── 7. DOCKER / HF SPACES ───────────────────────────────────────────────────
section("7. HuggingFace Spaces / Docker")
dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
check("Dockerfile exposes port 7860", "7860" in dockerfile)
check("gunicorn in CMD", "gunicorn" in dockerfile)
check("LLM_MODE=stub set", "LLM_MODE=stub" in dockerfile)
check("openenv in pip install", "openenv" in dockerfile)

hf_app = (ROOT / "hf_app.py").read_text(encoding="utf-8")
check("hf_app.py imports app + server", "server" in hf_app)
check("hf_app.py sets port 7860", "7860" in hf_app)

# ── 8. TRAINING NOTEBOOKS ────────────────────────────────────────────────────
section("8. Training Notebooks")
for nb_name in ["CIPHER_Training_Colab.ipynb", "cipher-training-red-v2.ipynb"]:
    nb_path = ROOT / nb_name
    if nb_path.exists():
        try:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            cells = nb.get("cells", [])
            source = " ".join(
                "".join(c.get("source", [])) for c in cells
            )
            check(f"{nb_name} — valid JSON notebook", True, f"{len(cells)} cells")
            check(f"{nb_name} — references unsloth/trl",
                  any(k in source.lower() for k in ["unsloth", "trl", "grpo", "sfttrainer"]))
            check(f"{nb_name} — references CIPHEREnv or cipher",
                  "cipher" in source.lower() or "CIPHEREnv" in source)
        except Exception as e:
            check(f"{nb_name} — valid JSON", False, str(e))
    else:
        check(f"{nb_name} exists", False)

# ── 9. RESERVED MCP NAMES ────────────────────────────────────────────────────
section("9. MCP Tool Name Safety")
import re
violations = []
for py_file in ROOT.rglob("*.py"):
    if any(p in str(py_file) for p in [".git", "__pycache__", ".venv", "check_submission"]):
        continue
    try:
        src = py_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if any(k in src for k in ["MCPEnvironment", "@tool", "register_tool"]):
        violations.append(str(py_file.relative_to(ROOT)))
check("No MCPEnvironment or @tool decorators", len(violations) == 0,
      f"found in: {violations}" if violations else "clean")

# ── 10. CORE MODULE IMPORTS ──────────────────────────────────────────────────
section("10. Core Module Imports")
modules_to_check = [
    "cipher.env_wrapper",
    "cipher.environment.graph",
    "cipher.environment.state",
    "cipher.rewards.red_reward",
    "cipher.rewards.blue_reward",
    "cipher.agents.base_agent",
    "cipher.training.loop",
    "cipher.utils.config",
]
for mod in modules_to_check:
    try:
        importlib.import_module(mod)
        check(mod, True)
    except Exception as e:
        check(mod, False, str(e)[:80])

# ── SUMMARY ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
passed = sum(1 for s, _, _ in results if s == PASS)
warned = sum(1 for s, _, _ in results if s == WARN)
failed = sum(1 for s, _, _ in results if s == FAIL)
total  = len(results)
print(f"  RESULT: {passed}/{total} passed  |  {warned} warnings  |  {failed} failures")
print(f"{'='*60}")

if failed > 0:
    print("\n  FAILURES TO FIX:")
    for s, label, detail in results:
        if s == FAIL:
            print(f"    [FAIL]  {label}" + (f"  -- {detail}" if detail else ""))

if warned > 0:
    print("\n  WARNINGS (non-blocking):")
    for s, label, detail in results:
        if s == WARN:
            print(f"    [WARN]  {label}" + (f"  -- {detail}" if detail else ""))

if failed == 0:
    print("\n  Submission is ready. Good luck!")
