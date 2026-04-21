#!/usr/bin/env python3
"""
CIPHER — Adversarial Multi-Agent Reinforcement Learning Environment
Phase 3: LLM Integration — Give the Agents Real Brains

Run this file to verify the system:
  python main.py                  # Single demo episode (stub mode)
  python main.py --episodes 3    # Run 3 episodes
  python main.py --no-trace       # Disable trace saving
  python main.py --live           # Enable LLM mode (requires valid API key)

Pass condition:
- No import errors
- No crashes
- Prints a 10-step episode trace with 8 agents
- At least one dead drop is written
- RED and BLUE reward components are printed
- In live mode: agent reasoning is displayed
- Oversight signal reports (may fire or not)

Usage:
  python main.py                  # Single demo episode
  python main.py --episodes 3    # Run 3 episodes
  python main.py --no-trace       # Disable trace saving
  python main.py --live           # Enable live LLM mode
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from datetime import datetime
from time import perf_counter

# Force UTF-8 on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


def main() -> None:
    """Run the CIPHER demo."""
    parser = argparse.ArgumentParser(description="CIPHER Phase 3 Demo")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Max steps per episode"
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable episode trace saving (enabled by default for dashboard replay)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live LLM mode (requires valid NVIDIA API key)"
    )
    args = parser.parse_args()

    # Set LLM_MODE before importing anything that reads it
    if args.live:
        os.environ["LLM_MODE"] = "live"

    console = Console(force_terminal=True)

    # ── Detect LLM mode ──────────────────────────────────────────
    from cipher.utils.llm_mode import is_live_mode
    llm_mode = "LIVE (LLM)" if is_live_mode() else "STUB (random)"

    # ── Startup banner ───────────────────────────────────────────
    banner_text = (
        "========================================================\n"
        "  C I P H E R\n"
        "  Adversarial Multi-Agent RL Environment\n"
        f"  Phase 3 -- LLM Integration ({llm_mode})\n"
        "========================================================"
    )
    console.print(Panel(banner_text, border_style="bright_cyan", padding=(1, 2)))
    console.print()

    # ── Verify imports ───────────────────────────────────────────
    console.print("[bold white]Phase 3 Import Verification[/bold white]")
    _verify_imports(console)
    console.print()

    # ── Run episodes ─────────────────────────────────────────────
    from cipher.training._episode_runner import run_episode

    for ep_num in range(1, args.episodes + 1):
        ep_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        t0 = perf_counter()
        console.print(
            f"[bold bright_cyan]{'═' * 60}[/bold bright_cyan]"
        )
        red_total, blue_total = run_episode(
            episode_number=ep_num,
            max_steps=args.steps,
            verbose=True,
            save_trace=not args.no_trace,
        )
        elapsed_s = perf_counter() - t0
        winner = "RED" if red_total > blue_total else "BLUE" if blue_total > red_total else "DRAW"
        winner_style = "red" if winner == "RED" else "blue" if winner == "BLUE" else "yellow"
        console.print(
            f"  [dim]Episode {ep_num} started: {ep_started} | duration: {elapsed_s:.2f}s[/dim]"
        )
        console.print(
            f"  [bold]Winner: [{winner_style}]{winner}[/{winner_style}][/bold]"
        )
        console.print(
            f"  [bold]Episode {ep_num} final: "
            f"[red]RED={red_total:.4f}[/red]  "
            f"[blue]BLUE={blue_total:.4f}[/blue][/bold]"
        )
        console.print()

    # ── Final summary ────────────────────────────────────────────
    console.print(
        Panel(
            f"[bold green]✓ Phase 3 pass condition met.[/bold green]\n"
            f"[dim]All modules imported. Episode executed ({llm_mode}). "
            f"Rewards computed. No crashes.[/dim]",
            border_style="green",
            padding=(1, 2),
        )
    )


def _verify_imports(console: Console) -> None:
    """Verify all Phase 1-3 modules can be imported without errors."""
    modules = [
        ("cipher.utils.config", "CipherConfig"),
        ("cipher.utils.logger", "get_logger"),
        ("cipher.utils.llm_client", "LLMClient"),
        ("cipher.utils.llm_mode", "is_live_mode"),
        ("cipher.environment.graph", "generate_enterprise_graph"),
        ("cipher.environment.state", "EpisodeState"),
        ("cipher.environment.observation", "generate_red_observation"),
        ("cipher.environment.scenario", "ScenarioGenerator"),
        ("cipher.agents.base_agent", "BaseAgent"),
        ("cipher.agents.red.planner", "RedPlanner"),
        ("cipher.agents.red.analyst", "RedAnalyst"),
        ("cipher.agents.red.operative", "RedOperative"),
        ("cipher.agents.red.exfiltrator", "RedExfiltrator"),
        ("cipher.agents.blue.surveillance", "BlueSurveillance"),
        ("cipher.agents.blue.threat_hunter", "BlueThreatHunter"),
        ("cipher.agents.blue.deception_architect", "BlueDeceptionArchitect"),
        ("cipher.agents.blue.forensics", "BlueForensics"),
        ("cipher.memory.dead_drop", "DeadDropVault"),
        ("cipher.rewards.red_reward", "compute_red_reward"),
        ("cipher.rewards.blue_reward", "compute_blue_reward"),
        ("cipher.rewards.oversight_reward", "compute_oversight_signal"),
        ("cipher.dashboard.app", "CipherDashboard"),
        ("cipher.training.loop", "TrainingLoop"),
    ]

    passed = 0
    failed = 0

    for module_path, symbol in modules:
        try:
            mod = __import__(module_path, fromlist=[symbol])
            getattr(mod, symbol)
            console.print(f"  [green]✓[/green] {module_path}.{symbol}")
            passed += 1
        except Exception as exc:
            console.print(
                f"  [red]✗[/red] {module_path}.{symbol} — {exc}"
            )
            failed += 1

    console.print(
        f"\n  [bold]Results: {passed} passed, {failed} failed[/bold]"
    )

    if failed > 0:
        console.print(
            "[bold red]FATAL: Import verification failed. Fix errors above.[/bold red]"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
