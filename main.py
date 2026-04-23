#!/usr/bin/env python3
"""
CIPHER — Adversarial Multi-Agent RL Environment
================================================
RED team (4 AI agents, including trained LoRA specialist) infiltrates a
50-node enterprise network to steal a classified file.
BLUE team (4 AI agents) defends using honeypots, traps, and forensics.
An Oversight Auditor judges both teams after every episode.

Usage:
  python main.py                          # 1 episode, stub mode
  python main.py --episodes 5            # 5-episode competition
  python main.py --steps 30              # longer episodes
  python main.py --live                  # all agents use NVIDIA NIM
  python main.py --hybrid                # RED Planner uses trained LoRA
  python main.py --train                 # training loop (10 episodes)
  python main.py --debug                 # show all agent debug logs
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

# ── Force UTF-8 on Windows ──────────────────────────────────────
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Suppress debug/info logs unless --debug passed ──────────────
# This must happen before any cipher imports to take effect
_DEBUG_MODE = "--debug" in sys.argv
if not _DEBUG_MODE:
    logging.disable(logging.INFO)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console(force_terminal=True)

# ── Zone labels ──────────────────────────────────────────────────
ZONE_NAMES = {0: "Perimeter", 1: "General", 2: "Sensitive", 3: "Critical (HVT)"}
ZONE_COLORS = {0: "dim white", 1: "cyan", 2: "yellow", 3: "bold red"}


def _suspicion_bar(level: float, width: int = 30) -> str:
    """Render a suspicion level as a colored ASCII bar."""
    filled = int(level * width)
    bar = "█" * filled + "░" * (width - filled)
    if level < 0.4:
        color = "green"
    elif level < 0.7:
        color = "yellow"
    else:
        color = "red"
    return f"[{color}]{bar}[/{color}] {level:.0%}"


def _determine_outcome(terminal_reason: str) -> tuple[str, str, str]:
    """
    Returns (winner, style, message) based on actual terminal reason.

    RED wins ONLY on successful exfiltration.
    BLUE wins on detection or max_steps (RED failed to exfil).
    DRAW on RED abort (voluntary withdrawal).
    """
    if terminal_reason in ("exfil_success", "exfiltration_complete", "exfil_complete"):
        return "RED", "bold red", "RED TEAM WINS — Classified file successfully exfiltrated!"
    elif terminal_reason == "detected":
        return "BLUE", "bold blue", "BLUE TEAM WINS — RED agent detected and neutralized!"
    elif terminal_reason == "aborted":
        return "DRAW", "bold yellow", "TACTICAL WITHDRAWAL — RED aborted due to rising suspicion."
    else:  # max_steps or unknown
        return "BLUE", "bold blue", "BLUE TEAM WINS — RED failed to complete mission in time!"


def _print_competition_header(episode_num: int, total_episodes: int, mode: str,
                               difficulty: float, max_steps: int) -> None:
    """Print the episode battle header."""
    if mode == "hybrid":
        red_label = "RED TEAM (Trained LoRA + 3 agents)"
        mode_badge = "[bold red]HYBRID[/bold red] — RED Planner uses fine-tuned Llama-3.2-1B"
    elif mode == "live":
        red_label = "RED TEAM (4 × NVIDIA NIM agents)"
        mode_badge = "[bold green]LIVE[/bold green] — All 8 agents use real LLM inference"
    else:
        red_label = "RED TEAM (4 agents, stub policy)"
        mode_badge = "[bold dim]STUB[/bold dim] — Fast random/heuristic policies"

    ep_label = f"Episode {episode_num}"
    if total_episodes > 1:
        ep_label += f" / {total_episodes}"

    header = (
        f"[bold white]{'═' * 58}[/bold white]\n"
        f"  [bold cyan]C I P H E R[/bold cyan]  —  [bold white]{ep_label}[/bold white]\n"
        f"  [bold red]{'🔴 ' + red_label:<35}[/bold red]  vs  [bold blue]🔵 BLUE TEAM (4 agents)[/bold blue]\n"
        f"  Mode: {mode_badge}\n"
        f"  Network: [cyan]50 nodes[/cyan] | Zones: [cyan]4[/cyan] | "
        f"Difficulty: [yellow]{difficulty:.2f}[/yellow] | "
        f"Max steps: [cyan]{max_steps}[/cyan]\n"
        f"[bold white]{'═' * 58}[/bold white]"
    )
    console.print(header)
    console.print()
    console.print(
        "  [bold red]OBJECTIVE[/bold red]: RED must traverse Zones 0→1→2→3 and "
        "exfiltrate the classified file\n"
        "  [bold blue]OBJECTIVE[/bold blue]: BLUE must detect, trap, and neutralize the intrusion\n"
    )


def _zone_badge(zone: int) -> str:
    icons = {0: "[dim white]◌ PERIMETER[/dim white]", 1: "[cyan]◉ GENERAL[/cyan]",
             2: "[yellow]◈ SENSITIVE[/yellow]", 3: "[bold red]◆ CRITICAL/HVT[/bold red]"}
    return icons.get(zone, f"Zone {zone}")


def _mini_bar(val: float, width: int = 20) -> str:
    filled = int(val * width)
    if val < 0.4:
        col = "green"
    elif val < 0.7:
        col = "yellow"
    else:
        col = "red"
    return f"[{col}]{'█' * filled}{'░' * (width - filled)}[/{col}] {val:.0%}"


def _print_live_step(step: int, max_steps: int, red_actions: list,
                     blue_actions: list, state: Any, elapsed_s: float) -> None:
    """Print a compact, informative per-step line during live/hybrid episodes."""
    from cipher.agents.base_agent import ActionType  # local import for safety

    # Red planner's primary action
    red_info = "[dim]waiting…[/dim]"
    for a in red_actions:
        if a and a.agent_id.startswith("red_planner"):
            atype = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
            node = f" → n{a.target_node}" if a.target_node is not None else ""
            file_ = f" [{a.target_file[:18]}]" if a.target_file else ""
            red_info = f"[red]{atype}{node}{file_}[/red]"
            break

    # Blue dominant action
    blue_counts: dict[str, int] = {}
    for a in blue_actions:
        if a:
            k = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
            blue_counts[k] = blue_counts.get(k, 0) + 1
    blue_info = " ".join(f"[blue]{k}×{v}[/blue]" for k, v in blue_counts.items()) or "[dim]—[/dim]"

    # Current zone
    zone = getattr(state, "red_current_zone", None)
    if zone is None:
        try:
            zone_raw = state.graph.nodes[state.red_current_node].get("zone")
            zone = zone_raw.value if hasattr(zone_raw, "value") else int(zone_raw)
        except Exception:
            zone = 0

    suspicion = float(getattr(state, "red_suspicion_score", 0.0))
    detection = float(getattr(state, "blue_detection_confidence", 0.0))
    exfil_count = len(getattr(state, "red_exfiltrated_files", []))

    console.print(
        f"  [bold white]Step {step:02d}/{max_steps}[/bold white]  "
        f"[dim]{elapsed_s:4.1f}s[/dim]  "
        f"Zone {_zone_badge(zone)}  "
        f"RED: {red_info}  │  "
        f"BLUE: {blue_info}  │  "
        f"Susp [red]{suspicion:.0%}[/red]  "
        f"Det [blue]{detection:.0%}[/blue]"
        + (f"  [bold green]✓ {exfil_count} file(s) exfil'd[/bold green]" if exfil_count else "")
    )


def _print_episode_battle(result: dict, episode_num: int, mode: str = "stub") -> None:
    """
    Print a structured narrative of the episode from both team perspectives.
    In live/hybrid mode also prints a full per-agent breakdown table.
    """
    state = result.get("state")
    if state is None:
        return

    terminal_reason = str(getattr(state, "terminal_reason", None) or "max_steps")
    steps_run = int(getattr(state, "step", 0))
    path_history = list(getattr(state, "red_path_history", []))
    exfiltrated = list(getattr(state, "red_exfiltrated_files", []))
    detection_conf = float(getattr(state, "blue_detection_confidence", 0.0))
    suspicion = float(getattr(state, "red_suspicion_score", 0.0))
    graph = getattr(state, "graph", None)
    episode_log = list(getattr(state, "episode_log", []))
    red_reward = result.get("red_reward")
    blue_reward = result.get("blue_reward")
    judgment = result.get("judgment")
    oversight = result.get("oversight")
    mismatch_count = result.get("action_reason_mismatch_count", 0)
    oversight_flags = result.get("oversight_flags", [])

    console.print(f"[bold white]  ── BATTLE LOG ──[/bold white]")
    console.print()

    # Parse key events from episode log
    key_events: list[str] = []
    prev_zone = 0
    _exfil_logged: set[str] = set()

    for entry in episode_log:
        step = entry.get("step", 0)
        agent_id = str(entry.get("agent_id", ""))
        action_type = str(entry.get("action_type", ""))
        payload = entry.get("payload", {}) or {}
        res = entry.get("result", {}) or {}

        if action_type == "move" and agent_id.startswith("red_planner"):
            target = payload.get("target_node")
            if target is not None and graph is not None:
                try:
                    zone_val = graph.nodes[target].get("zone")
                    if zone_val is not None:
                        new_zone = zone_val.value if hasattr(zone_val, "value") else int(zone_val)
                        hostname = graph.nodes[target].get("hostname", f"node_{target}")
                        zone_name = ZONE_NAMES.get(new_zone, f"Zone {new_zone}")
                        if new_zone > prev_zone:
                            key_events.append(
                                f"  Step {step:02d} | [bold red]RED ADVANCES[/bold red] "
                                f"Zone {prev_zone}→[bold]{new_zone}[/bold] "
                                f"([yellow]{zone_name}[/yellow]) via [cyan]{hostname}[/cyan]"
                            )
                            prev_zone = new_zone
                        elif new_zone == 3 and prev_zone == 3:
                            key_events.append(
                                f"  Step {step:02d} | [red]RED[/red] moves within "
                                f"Critical zone via [cyan]{hostname}[/cyan]"
                            )
                except Exception:
                    pass

        elif action_type == "exfiltrate":
            file_name = payload.get("target_file") or res.get("exfiltrated")
            if file_name and file_name not in _exfil_logged:
                _exfil_logged.add(str(file_name))
                success = res.get("success", False)
                reason = res.get("reason", "")
                if success or reason in ("exfil_success", "exfil_complete"):
                    key_events.append(
                        f"  Step {step:02d} | [bold red]EXFILTRATION[/bold red] "
                        f"[green]SUCCESS[/green] — [cyan]{file_name}[/cyan] stolen!"
                    )
                else:
                    key_events.append(
                        f"  Step {step:02d} | [red]EXFIL attempt[/red] "
                        f"[yellow]FAILED[/yellow] ({reason})"
                    )

        elif action_type == "trigger_alert" and agent_id.startswith("blue_"):
            correct = res.get("correct_alert", False)
            if correct:
                key_events.append(
                    f"  Step {step:02d} | [bold blue]BLUE ALERT[/bold blue] "
                    f"[green]CORRECT[/green] — RED agent located and flagged!"
                )
            else:
                key_events.append(
                    f"  Step {step:02d} | [blue]BLUE alert[/blue] "
                    f"[yellow]false positive[/yellow]"
                )

        elif action_type == "abort" and agent_id.startswith("red_"):
            if res.get("reason") == "abort_applied":
                key_events.append(
                    f"  Step {step:02d} | [bold yellow]RED ABORT[/bold yellow] "
                    f"— Mission abandoned"
                )

        # Trap events
        elif action_type in ("place_honeypot", "plant_breadcrumb", "trigger_false_escalation"):
            if agent_id.startswith("blue_"):
                node = payload.get("target_node", "?")
                key_events.append(
                    f"  Step {step:02d} | [blue]BLUE TRAP[/blue] "
                    f"[cyan]{action_type}[/cyan] planted at node {node}"
                )

    if key_events:
        for event in key_events[:25]:
            console.print(event)
    else:
        console.print(f"  [dim]Episode completed in {steps_run} steps — no major events[/dim]")

    console.print()

    # ── Final summary ─────────────────────────────────────────────
    winner, winner_style, outcome_msg = _determine_outcome(terminal_reason)

    console.print(f"  Red suspicion:    {_suspicion_bar(suspicion)}")
    console.print(f"  Blue confidence:  {_suspicion_bar(detection_conf)}")
    console.print()

    if path_history:
        path_str = " → ".join(str(n) for n in path_history[:12])
        if len(path_history) > 12:
            path_str += f" … ({len(path_history)} nodes total)"
        console.print(f"  RED path: [dim]{path_str}[/dim]")

    if exfiltrated:
        unique_exfil = list(dict.fromkeys(exfiltrated))  # deduplicate
        console.print(
            f"  Exfiltrated: [bold red]{len(unique_exfil)}[/bold red] unique file(s): "
            f"[cyan]{', '.join(unique_exfil[:3])}[/cyan]"
        )

    # ── Reward breakdown (live/hybrid only) ──────────────────────
    if mode in ("live", "hybrid") and red_reward is not None and blue_reward is not None:
        console.print()
        console.print("  [bold white]── REWARD BREAKDOWN ──[/bold white]")
        # RED
        rr = red_reward
        console.print(
            f"  [red]RED[/red]   total=[bold]{rr.total:+.3f}[/bold]  "
            f"exfil={getattr(rr,'exfiltration_completeness',0.0):+.3f}  "
            f"stealth={1.0 - getattr(rr,'detection_probability',0.0):+.3f}  "
            f"abort_pen={getattr(rr,'abort_penalty',0.0):+.3f}"
        )
        # BLUE
        br = blue_reward
        console.print(
            f"  [blue]BLUE[/blue]  total=[bold]{br.total:+.3f}[/bold]  "
            f"detection={getattr(br,'detection_accuracy_score',0.0):+.3f}  "
            f"honeypot={getattr(br,'honeypot_trigger_rate',0.0):+.3f}  "
            f"fp_pen={getattr(br,'false_positive_rate_penalty',0.0):+.3f}"
        )
        if judgment:
            verdict = getattr(judgment, "episode_verdict", "?")
            vcol = {"red_dominates": "red", "blue_dominates": "blue",
                    "contested": "yellow", "degenerate": "dim"}.get(verdict, "white")
            console.print(f"  Oversight verdict: [{vcol}]{verdict}[/{vcol}]")

    # ── Warnings ─────────────────────────────────────────────────
    if mismatch_count > 0:
        console.print(
            f"\n  [dim yellow]⚠ {mismatch_count} action-reason mismatch(es) detected "
            f"(agents said one thing, did another)[/dim yellow]"
        )
    if oversight_flags:
        console.print(
            f"  [dim yellow]⚑ {len(oversight_flags)} oversight flag(s) raised[/dim yellow]"
        )

    console.print()
    console.print(f"  [{winner_style}]{outcome_msg}[/{winner_style}]")
    console.print()



_STATE_FILE = Path("training_state.json")
_LIVE_STEPS_FILE = Path("live_steps.jsonl")
_AGENT_STATUS_FILE = Path("logs") / "agent_status.json"
_COST_LOG_FILE = Path("logs") / "api_costs.json"


def _write_run_state(state: dict) -> None:
    try:
        _STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def _append_live_step(data: dict) -> None:
    try:
        with open(_LIVE_STEPS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(data) + "\n")
    except Exception:
        pass


def _write_agent_status(data: dict) -> None:
    try:
        _AGENT_STATUS_FILE.parent.mkdir(exist_ok=True)
        _AGENT_STATUS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def _rename_trace(ep_num: int, mode: str) -> None:
    """Rename episode_NNNN.json → episode_NNN_TIMESTAMP_MODE.json after saving."""
    traces_dir = Path("episode_traces")
    old_path = traces_dir / f"episode_{ep_num:04d}.json"
    if old_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_path = traces_dir / f"episode_{ep_num:03d}_{ts}_{mode}.json"
        try:
            old_path.rename(new_path)
        except Exception:
            pass


def _get_step_callback_factory(run_id: str):
    def factory(ep_num: int):
        _ep_start_ref = [perf_counter()]
        def _cb(step, max_steps, red_actions, blue_actions, state):
            if step == 1:
                _ep_start_ref[0] = perf_counter()
            elapsed = perf_counter() - _ep_start_ref[0]
            _print_live_step(step, max_steps, red_actions, blue_actions, state, elapsed)
            try:
                red_info = ""
                for a in red_actions:
                    if a and a.agent_id.startswith("red_planner"):
                        atype = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
                        node = f" → n{a.target_node}" if a.target_node is not None else ""
                        red_info = f"{atype}{node}"
                        break
                blue_counts: dict = {}
                for a in blue_actions:
                    if a:
                        k = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
                        blue_counts[k] = blue_counts.get(k, 0) + 1
                blue_info = " ".join(f"{k}×{v}" for k, v in blue_counts.items())
                zone = getattr(state, "red_current_zone", None)
                if zone is None:
                    try:
                        zone_raw = state.graph.nodes[state.red_current_node].get("zone")
                        zone = zone_raw.value if hasattr(zone_raw, "value") else int(zone_raw)
                    except Exception:
                        zone = 0
                zone_names = {0: "Perimeter", 1: "General", 2: "Sensitive", 3: "Critical/HVT"}
                susp = round(float(getattr(state, "red_suspicion_score", 0.0)), 3)
                det = round(float(getattr(state, "blue_detection_confidence", 0.0)), 3)
                exfil = list(getattr(state, "red_exfiltrated_files", []))
                zone_label = zone_names.get(int(zone) if zone is not None else 0, "Unknown")
                step_data = {
                    "run_id": run_id,
                    "episode": ep_num,
                    "step": step,
                    "max_steps": max_steps,
                    "red_action": red_info or "waiting",
                    "blue_actions": blue_info or "—",
                    "suspicion": susp,
                    "detection": det,
                    "zone": zone_label,
                    "elapsed": round(elapsed, 1),
                    "exfil_count": len(exfil),
                    "exfil_files": exfil[-3:],
                    "timestamp": datetime.now().isoformat(),
                }
                _append_live_step(step_data)

                agents_detail: dict = {}
                for a in (red_actions or []) + (blue_actions or []):
                    if a:
                        atype = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
                        agents_detail[a.agent_id] = {
                            "action": atype,
                            "node": a.target_node,
                            "team": "red" if str(a.agent_id).startswith("red") else "blue",
                        }
                _write_agent_status({
                    "run_id": run_id,
                    "episode": ep_num,
                    "step": step,
                    "suspicion": susp,
                    "detection": det,
                    "zone": zone_label,
                    "agents": agents_detail,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception:
                pass
        return _cb
    return factory


def _validate_hybrid_models() -> None:
    """Check that all specialist LoRA adapters exist; warn if missing (Change 11)."""
    specialists = {
        "RED Planner v2":        os.getenv("RED_PLANNER_LORA_PATH",       "red trained/cipher-red-planner-v2"),
        "RED Analyst v1":        os.getenv("RED_ANALYST_LORA_PATH",        "red trained/cipher-red-analyst-v1"),
        "BLUE Surveillance v1":  os.getenv("BLUE_SURVEILLANCE_LORA_PATH",  "blue trained/cipher-blue-surveillance-v1"),
        "BLUE ThreatHunter v1":  os.getenv("BLUE_THREAT_HUNTER_LORA_PATH", "blue trained/cipher-blue-threat-hunter-v1"),
    }
    console.print("\n  [bold cyan]Hybrid specialist check:[/bold cyan]")
    for name, path in specialists.items():
        if os.path.exists(path):
            console.print(f"    [green]✓[/green] {name}: {path}")
        else:
            console.print(f"    [yellow]⚠[/yellow]  {name} NOT FOUND at '{path}' — falling back to NVIDIA NIM")
    console.print()


def main() -> None:
    """Entry point for CIPHER."""
    parser = argparse.ArgumentParser(
        description="CIPHER — Adversarial Multi-Agent RL Environment"
    )
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run (default: 1)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Max steps per episode (default: 30)")
    parser.add_argument("--live", action="store_true",
                        help="Use NVIDIA NIM for all agents")
    parser.add_argument("--hybrid", action="store_true",
                        help="RED Planner uses trained LoRA, others use NIM")
    parser.add_argument("--train", action="store_true",
                        help="Run training loop (updates prompts every 5 episodes)")
    parser.add_argument("--train-episodes", type=int, default=10,
                        help="Number of episodes for --train mode (default: 10)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--no-trace", action="store_true",
                        help="Skip saving episode traces")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-step agent actions (very verbose)")
    args = parser.parse_args()

    # Set LLM mode
    if args.live:
        os.environ["LLM_MODE"] = "live"
        mode = "live"
    elif args.hybrid:
        os.environ["LLM_MODE"] = "hybrid"
        mode = "hybrid"
    else:
        os.environ.setdefault("LLM_MODE", "stub")
        mode = "stub"

    # Change 11: validate specialist model paths when running hybrid
    if mode == "hybrid":
        _validate_hybrid_models()

    # Generate a unique run_id so the dashboard can isolate this run
    run_id = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.environ["CIPHER_RUN_ID"] = run_id
    run_started_at = datetime.now().isoformat()

    # Clear live step feed from any prior run
    try:
        _LIVE_STEPS_FILE.write_text("", encoding="utf-8")
    except Exception:
        pass

    if args.train:
        from cipher.training.loop import TrainingLoop
        n = args.train_episodes
        console.print(
            f"[bold cyan]CIPHER Training Loop[/bold cyan] — {n} episodes"
        )
        factory = _get_step_callback_factory(run_id) if mode in ("live", "hybrid") else None
        if factory:
            console.print("  [dim]Calling LLM agents in parallel — step ticker will print below:[/dim]\n")
        TrainingLoop(n_episodes=n).run(step_callback_factory=factory)
        return

    # Import episode runner after setting LLM_MODE
    from cipher.training._episode_runner import run_episode
    from cipher.environment.scenario import ScenarioGenerator
    from cipher.utils.config import config

    n_episodes = args.episodes
    max_steps = args.steps          # always set (default 30)
    runner_verbose = args.verbose   # False by default — main.py owns display
    save_trace = not args.no_trace

    _write_run_state({
        "status": "running",
        "current_episode": 0,
        "total_episodes": n_episodes,
        "llm_mode": mode,
        "run_id": run_id,
        "started_at": run_started_at,
        "last_updated": datetime.now().isoformat(),
    })

    scenario_gen = ScenarioGenerator()
    start_time = perf_counter()

    for ep_num in range(1, n_episodes + 1):
        scenario = scenario_gen.generate(ep_num)

        _print_competition_header(
            episode_num=ep_num,
            total_episodes=n_episodes,
            mode=mode,
            difficulty=scenario.difficulty,
            max_steps=max_steps,
        )

        ep_start = perf_counter()
        _step_times: list[float] = []

        # For live/hybrid: inject a per-step callback so we print progress as it happens
        step_callback = None
        if mode in ("live", "hybrid"):
            console.print(
                f"  [dim]Calling LLM agents in parallel — step ticker will print below:[/dim]\n"
            )
            factory = _get_step_callback_factory(run_id)
            step_callback = factory(ep_num)

        result = run_episode(
            scenario=scenario,
            graph=scenario.generated_graph,
            cfg=config,
            max_steps=max_steps,
            verbose=runner_verbose,
            save_trace=save_trace,
            episode_number=ep_num,
            step_callback=step_callback,
        )

        ep_elapsed = perf_counter() - ep_start
        if mode in ("live", "hybrid"):
            console.print(
                f"\n  [dim]Episode finished in {ep_elapsed:.1f}s "
                f"({ep_elapsed/max_steps:.1f}s/step avg)[/dim]\n"
            )

        if isinstance(result, dict):
            _print_episode_battle(result, ep_num, mode=mode)

        # Rename trace to include timestamp + mode for dashboard trace selector
        if save_trace:
            _rename_trace(ep_num, mode)

        # Update dashboard state after each episode
        _write_run_state({
            "status": "running",
            "current_episode": ep_num,
            "total_episodes": n_episodes,
            "llm_mode": mode,
            "run_id": run_id,
            "started_at": run_started_at,
            "last_updated": datetime.now().isoformat(),
        })

    # Estimate API cost (OpenRouter; stub = $0)
    total_steps_run = n_episodes * max_steps
    if mode == "live":
        # ~8 agents × ~800 tokens/call × $0.80/1M tokens (approx)
        estimated_cost_usd = round(total_steps_run * 8 * 800 * 0.80 / 1_000_000, 4)
    elif mode == "hybrid":
        estimated_cost_usd = round(total_steps_run * 7 * 800 * 0.80 / 1_000_000, 4)
    else:
        estimated_cost_usd = 0.0

    # Mark run complete
    _write_run_state({
        "status": "complete",
        "current_episode": n_episodes,
        "total_episodes": n_episodes,
        "llm_mode": mode,
        "run_id": run_id,
        "started_at": run_started_at,
        "last_updated": datetime.now().isoformat(),
        "estimated_cost_usd": estimated_cost_usd,
        "total_steps": total_steps_run,
    })

    elapsed = perf_counter() - start_time
    if n_episodes > 1:
        console.print(
            f"\n[dim]Completed {n_episodes} episodes in {elapsed:.1f}s "
            f"({elapsed/n_episodes:.1f}s avg)[/dim]"
        )


if __name__ == "__main__":
    main()
