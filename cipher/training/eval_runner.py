"""
cipher/training/eval_runner.py

E.md Change 1 — Evaluation Runner Script.

Runs N episodes in each requested mode (stub / live / hybrid) and writes
a comparison JSON + CSV to eval_results/.

Usage (from project root, venv activated):
    python -m cipher.training.eval_runner --episodes 20 --modes stub hybrid
    python -m cipher.training.eval_runner --episodes 5 --modes stub live hybrid

Output files:
    eval_results/comparison_YYYYMMDD_HHMMSS.json
    eval_results/comparison_YYYYMMDD_HHMMSS.csv

NOTE (RunPod):  After training completes, update the LoRA adapter paths in
.env or pass via env vars:
    RED_PLANNER_LORA_PATH   = "red trained/cipher-red-planner-v1"
    RED_ANALYST_LORA_PATH   = "red trained/cipher-red-analyst-v1"
    BLUE_SURVEILLANCE_LORA_PATH  = "blue trained/cipher-blue-surveillance-v1"
    BLUE_THREAT_HUNTER_LORA_PATH = "blue trained/cipher-blue-threat-hunter-v1"
No other code changes are required — the hybrid routing in lora_client.py
already reads these env vars at runtime.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Ensure project root is importable when run as __main__ ────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Suppress noisy INFO logs unless DEBUG ─────────────────────────────────
import logging
if "--debug" not in sys.argv:
    logging.disable(logging.INFO)

from cipher.training._episode_runner import run_episode
from cipher.environment.graph import generate_enterprise_graph
from cipher.environment.scenario import ScenarioGenerator
from cipher.utils.config import config

# ── Valid modes ────────────────────────────────────────────────────────────
VALID_MODES = ["stub", "live", "hybrid"]

# ── Output directory ──────────────────────────────────────────────────────
EVAL_RESULTS_DIR = _PROJECT_ROOT / "eval_results"
EVAL_RESULTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation logic
# ─────────────────────────────────────────────────────────────────────────────


def run_eval(n_episodes: int, modes: list[str], max_steps: int = 30,
             verbose: bool = False) -> dict[str, Any]:
    """
    Run n_episodes per mode, collect metrics, return a nested comparison dict.

    Returns:
        {
            "modes": {
                "stub":   [{"episode": 1, "red_total": ..., ...}, ...],
                "hybrid": [...],
            },
            "summary": {
                "stub":   {"red_win_rate": 0.45, "avg_red": 0.31, "avg_steps": 18,
                           "avg_exfil": 0.8, "avg_detection_rate": 0.2},
                "hybrid": {...},
            },
            "meta": {
                "n_episodes": 20, "max_steps": 30,
                "timestamp": "...", "modes": [...]
            }
        }
    """
    results: dict[str, list[dict]] = {}
    sg = ScenarioGenerator()
    original_mode = os.environ.get("LLM_MODE", "stub")

    print(f"\n[eval_runner] Starting evaluation: {n_episodes} episodes × {len(modes)} mode(s)")

    for mode in modes:
        if mode not in VALID_MODES:
            print(f"[eval_runner] WARNING: unknown mode '{mode}', skipping.")
            continue

        print(f"\n[eval_runner] ── Mode: {mode.upper()} ──────────────────────")
        os.environ["LLM_MODE"] = mode

        mode_results: list[dict] = []
        for ep in range(1, n_episodes + 1):
            t0 = time.perf_counter()
            scenario = sg.generate(ep)
            graph = generate_enterprise_graph(
                n_nodes=config.env_graph_size,
                honeypot_density=config.env_honeypot_density,
                seed=scenario.episode_seed,
            )

            try:
                result = run_episode(
                    scenario=scenario,
                    graph=graph,
                    cfg=config,
                    episode_number=ep,
                    max_steps=max_steps,
                    verbose=verbose,
                    save_trace=False,
                )
            except Exception as exc:
                print(f"  [eval_runner] ep {ep} ERROR: {exc}")
                # Record a zero-reward row so totals stay balanced
                mode_results.append({
                    "episode": ep,
                    "mode": mode,
                    "red_total": 0.0,
                    "blue_total": 0.0,
                    "terminal_reason": "error",
                    "steps": 0,
                    "exfil_count": 0,
                    "elapsed_s": round(time.perf_counter() - t0, 2),
                    "error": str(exc),
                })
                continue

            elapsed = round(time.perf_counter() - t0, 2)

            # result is a dict when scenario+graph are passed
            red_reward  = result.get("red_reward")
            blue_reward = result.get("blue_reward")
            state       = result.get("state")

            red_total  = round(getattr(red_reward,  "total", 0.0), 4)
            blue_total = round(getattr(blue_reward, "total", 0.0), 4)
            term_reason = getattr(state, "terminal_reason", "max_steps") or "max_steps"
            steps_run   = int(getattr(state, "step", max_steps))
            exfil_count = len(getattr(state, "red_exfiltrated_files", []))
            detection   = round(float(getattr(state, "blue_detection_confidence", 0.0)), 4)

            row = {
                "episode":         ep,
                "mode":            mode,
                "red_total":       red_total,
                "blue_total":      blue_total,
                "terminal_reason": term_reason,
                "steps":           steps_run,
                "exfil_count":     exfil_count,
                "detection_conf":  detection,
                "elapsed_s":       elapsed,
            }
            mode_results.append(row)

            win_icon = "✓" if "exfil" in term_reason else "✗"
            print(
                f"  ep {ep:>3}/{n_episodes}  [{mode}]  "
                f"RED={red_total:+.3f}  BLUE={blue_total:+.3f}  "
                f"steps={steps_run}  exfil={exfil_count}  "
                f"reason={term_reason}  {win_icon}  ({elapsed:.1f}s)"
            )

        results[mode] = mode_results

    # Restore original LLM_MODE
    os.environ["LLM_MODE"] = original_mode

    # ── Build summary statistics ──────────────────────────────────────────
    summary: dict[str, dict] = {}
    for mode, rows in results.items():
        if not rows:
            summary[mode] = {}
            continue
        n = len(rows)
        red_wins  = sum(1 for r in rows if "exfil" in str(r.get("terminal_reason", "")))
        avg_red   = round(sum(r.get("red_total",  0.0) for r in rows) / n, 4)
        avg_blue  = round(sum(r.get("blue_total", 0.0) for r in rows) / n, 4)
        avg_steps = round(sum(r.get("steps",       0)  for r in rows) / n, 1)
        avg_exfil = round(sum(r.get("exfil_count", 0)  for r in rows) / n, 2)
        avg_det   = round(sum(r.get("detection_conf", 0.0) for r in rows) / n, 4)
        summary[mode] = {
            "n_episodes":      n,
            "red_win_rate":    round(red_wins / max(n, 1), 4),
            "avg_red":         avg_red,
            "avg_blue":        avg_blue,
            "avg_steps":       avg_steps,
            "avg_exfil":       avg_exfil,
            "avg_detection_rate": avg_det,
        }

    payload = {
        "modes":   results,
        "summary": summary,
        "meta": {
            "n_episodes": n_episodes,
            "max_steps":  max_steps,
            "timestamp":  datetime.now().isoformat(),
            "modes":      modes,
        },
    }
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────


def save_eval_results(payload: dict, timestamp: str | None = None) -> tuple[Path, Path]:
    """
    Persist evaluation results to:
        eval_results/comparison_TIMESTAMP.json
        eval_results/comparison_TIMESTAMP.csv

    Returns (json_path, csv_path).
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = EVAL_RESULTS_DIR / f"comparison_{ts}.json"
    csv_path  = EVAL_RESULTS_DIR / f"comparison_{ts}.csv"

    # ── JSON ─────────────────────────────────────────────────────────────
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ── CSV (flat rows) ───────────────────────────────────────────────────
    all_rows: list[dict] = []
    for _mode, rows in payload.get("modes", {}).items():
        all_rows.extend(rows)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    return json_path, csv_path


def print_comparison_table(payload: dict) -> None:
    """Print a human-readable comparison table to stdout."""
    summary = payload.get("summary", {})
    if not summary:
        print("[eval_runner] No summary data to display.")
        return

    modes = list(summary.keys())
    cols = ["red_win_rate", "avg_red", "avg_blue", "avg_steps", "avg_exfil", "avg_detection_rate"]
    col_labels = {
        "red_win_rate":        "RED Win Rate",
        "avg_red":             "Avg RED Reward",
        "avg_blue":            "Avg BLUE Reward",
        "avg_steps":           "Avg Steps",
        "avg_exfil":           "Avg Exfils",
        "avg_detection_rate":  "Avg Detection",
    }

    header = f"{'Metric':<22}" + "".join(f"{m.upper():>14}" for m in modes)
    print("\n" + "═" * (22 + 14 * len(modes)))
    print("  EVALUATION COMPARISON SUMMARY")
    print("═" * (22 + 14 * len(modes)))
    print(header)
    print("─" * (22 + 14 * len(modes)))

    baseline_mode = modes[0] if modes else None
    for col in cols:
        label = col_labels.get(col, col)
        row = f"  {label:<20}"
        baseline_val = summary.get(baseline_mode, {}).get(col, None) if baseline_mode else None
        for i, m in enumerate(modes):
            val = summary[m].get(col, None)
            if val is None:
                cell = "    n/a"
            elif col in ("red_win_rate", "avg_detection_rate"):
                cell = f"  {val*100:5.1f}%"
                if i > 0 and baseline_val is not None:
                    delta = (val - baseline_val) * 100
                    cell += f"({delta:+.1f}%)"
            elif col in ("avg_red", "avg_blue"):
                cell = f"  {val:+7.3f}"
                if i > 0 and baseline_val is not None:
                    delta = val - baseline_val
                    cell += f"({delta:+.3f})"
            else:
                cell = f"  {val:7.2f}"
                if i > 0 and baseline_val is not None:
                    delta = val - baseline_val
                    cell += f"({delta:+.2f})"
            row += f"{cell:>14}"
        print(row)

    print("═" * (22 + 14 * len(modes)) + "\n")

    # ── Improvement headline ──────────────────────────────────────────────
    if len(modes) >= 2:
        base = summary[modes[0]]
        comp = summary[modes[-1]]
        base_wr = base.get("red_win_rate", 0.0)
        comp_wr = comp.get("red_win_rate", 0.0)
        if base_wr > 1e-6:
            pct = (comp_wr - base_wr) / base_wr * 100
            direction = "higher" if pct >= 0 else "lower"
            print(
                f"  ► {modes[-1].upper()} mode achieves "
                f"{abs(pct):.1f}% {direction} RED win rate vs {modes[0].upper()} baseline\n"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CIPHER Evaluation Runner — compare stub / live / hybrid modes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--episodes", type=int, default=20,
        help="Number of episodes to run per mode",
    )
    p.add_argument(
        "--modes", nargs="+", default=["stub", "hybrid"],
        choices=VALID_MODES,
        help="Modes to compare",
    )
    p.add_argument(
        "--steps", type=int, default=30,
        help="Max steps per episode",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-step agent actions (very noisy)",
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.debug:
        logging.disable(logging.NOTSET)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload  = run_eval(
        n_episodes=args.episodes,
        modes=args.modes,
        max_steps=args.steps,
        verbose=args.verbose,
    )
    json_path, csv_path = save_eval_results(payload, ts)

    print_comparison_table(payload)
    print(f"[eval_runner] Results saved:")
    print(f"  JSON → {json_path}")
    print(f"  CSV  → {csv_path}")

    # ── Also generate the proof-of-learning report ────────────────────────
    try:
        from cipher.utils.report_gen import generate_proof_of_learning_report, save_report
        report_md = generate_proof_of_learning_report(payload)
        report_path = save_report(report_md, ts)
        print(f"  Report → {report_path}")
    except Exception as exc:
        print(f"  [eval_runner] report_gen skipped: {exc}")


if __name__ == "__main__":
    main()
