#!/usr/bin/env python3
"""
CIPHER — Adversarial Multi-Agent Reinforcement Learning Environment
Phase 1: Project Skeleton — Full Episode Demo

Run this file to verify the Phase 1 skeleton is intact:
  python main.py

Pass condition:
- No import errors
- No crashes
- Prints a 10-step episode trace with 8 agents
- At least one dead drop is written
- RED and BLUE reward components are printed
- Oversight signal reports (may fire or not)

Usage:
  python main.py                  # Single demo episode
  python main.py --episodes 3    # Run 3 episodes
  python main.py --trace          # Save episode trace JSON
"""
from __future__ import annotations

import argparse
import io
import os
import sys

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
    """Run the CIPHER Phase 1 demo."""
    parser = argparse.ArgumentParser(description="CIPHER Phase 1 Demo")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument(
        "--steps", type=int, default=10, help="Max steps per episode"
    )
    parser.add_argument(
        "--trace", action="store_true", help="Save episode trace to disk"
    )
    args = parser.parse_args()

    console = Console(force_terminal=True)

    # ── Startup banner ───────────────────────────────────────────
    banner_text = (
        "========================================================\n"
        "  C I P H E R\n"
        "  Adversarial Multi-Agent RL Environment\n"
        "  Phase 1 -- Skeleton Verification\n"
        "========================================================"
    )
    console.print(Panel(banner_text, border_style="bright_cyan", padding=(1, 2)))
    console.print()

    # ── Verify imports ───────────────────────────────────────────
    console.print("[bold white]Phase 1 Import Verification[/bold white]")
    _verify_imports(console)
    console.print()

    # ── Run episodes ─────────────────────────────────────────────
    from cipher.training._episode_runner import run_episode

    for ep_num in range(1, args.episodes + 1):
        console.print(
            f"[bold bright_cyan]{'═' * 60}[/bold bright_cyan]"
        )
        red_total, blue_total = run_episode(
            episode_number=ep_num,
            max_steps=args.steps,
            verbose=True,
            save_trace=args.trace,
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
            "[bold green]✓ Phase 1 pass condition met.[/bold green]\n"
            "[dim]All modules imported. Episode executed. "
            "Rewards computed. No crashes.[/dim]",
            border_style="green",
            padding=(1, 2),
        )
    )


def _verify_imports(console: Console) -> None:
    """Verify all Phase 1 modules can be imported without errors."""
    modules = [
        ("cipher.utils.config", "CipherConfig"),
        ("cipher.utils.logger", "get_logger"),
        ("cipher.utils.llm_client", "LLMClient"),
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
