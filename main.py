#!/usr/bin/env python3
"""
CIPHER — Adversarial Multi-Agent RL Environment
================================================
RED COMMANDER (1 trained brain) orchestrates N dynamic subagents that
infiltrate a 50-node enterprise network to steal a classified file.
BLUE COMMANDER (1 trained brain) orchestrates N dynamic subagents that
defend using honeypots, traps, and forensics.
An Oversight Auditor (9th LLM) judges both teams after every episode.
Architecture: CIPHER_AGENT_ARCH=v2 (commander + dynamic subagents).

Usage:
  python main.py                          # 1 episode, stub mode
  python main.py --episodes 5            # 5-episode competition
  python main.py --steps 30              # longer episodes
  python main.py --live                  # all agents use HuggingFace API
  python main.py --hybrid                # RED Planner uses trained LoRA
  python main.py --train                 # training loop (10 episodes)
  python main.py --debug                 # show all agent debug logs
  python main.py --demo                  # 3 judge episodes (stub / from .env)
  python main.py --demo --live           # same + green banner + HF agents
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
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

# ── G.md — Judge demo + session + eval banners (56-char ║ rows) ───
_BOX_INNER = 54  # characters between ║ and ║
_RST = "\033[0m"


def _ansi_framed_row(border: str, inner: str) -> str:
    inner = inner.replace("\n", " ")[:_BOX_INNER].ljust(_BOX_INNER)
    return f"{border}║{_RST}{inner}{border}║{_RST}"


def _ansi_framed_cap(border: str, cap: str) -> str:
    return f"{border}{cap}{_RST}"


def judge_demo_banner_ansi(llm_mode: str) -> str:
    """
    G.md (51–59) judge box. Border is always red (--demo). Footer line follows
    LLM mode (live / hybrid / stub).
    """
    border = "\033[91m"  # demo = red frame
    top = "╔" + "═" * _BOX_INNER + "╗"
    sep = "╠" + "═" * _BOX_INNER + "╣"
    bot = "╚" + "═" * _BOX_INNER + "╝"
    footer = {
        "live": "  2 Commanders + N subagents · 50-node network · Live  ",
        "hybrid": "  2 Commanders + N subagents · Hybrid LoRA + API       ",
        "stub": "  2 Commanders + N subagents · 50-node network · Stub  ",
    }.get(llm_mode, "  2 Commanders + N subagents · 50-node network · Stub  ")
    footer = footer[:_BOX_INNER].ljust(_BOX_INNER)
    lines = [
        _ansi_framed_cap(border, top),
        _ansi_framed_row(border, "  C I P H E R  — Judge Demo Mode                      "),
        _ansi_framed_row(border, "  OpenEnv Hackathon | Multi-Agent Adversarial RL      "),
        _ansi_framed_cap(border, sep),
        _ansi_framed_row(border, "  3 episodes: Exfiltration · Detection · Contested    "),
        _ansi_framed_row(border, footer),
        _ansi_framed_cap(border, bot),
    ]
    return "\n".join(lines)


def session_mode_banner_ansi(mode: str) -> str:
    """Non-demo session: stub=dark grey, live=green, hybrid=yellow borders."""
    top = "╔" + "═" * _BOX_INNER + "╗"
    sep = "╠" + "═" * _BOX_INNER + "╣"
    bot = "╚" + "═" * _BOX_INNER + "╝"
    if mode == "live":
        border = "\033[92m"
        title = "L I V E   I N F E R E N C E"
        sub = "  All 8 agents · HuggingFace API · traces → :8050   "
    elif mode == "hybrid":
        border = "\033[93m"
        title = "H Y B R I D   M O D E"
        sub = "  RED LoRA + HF API · BLUE API · Dash replay :8050 "
    else:
        border = "\033[90m"  # dark grey
        title = "S T U B   ( S A F E   /   F A S T )"
        sub = "  Heuristics · $0 API · use --demo or --live judges "
    row_title = f"  C I P H E R  — {title}".ljust(_BOX_INNER)[:_BOX_INNER]
    row_sub = sub[:_BOX_INNER].ljust(_BOX_INNER)
    lines = [
        _ansi_framed_cap(border, top),
        _ansi_framed_row(border, row_title),
        _ansi_framed_cap(border, sep),
        _ansi_framed_row(border, row_sub),
        _ansi_framed_cap(border, bot),
    ]
    return "\n".join(lines)


def eval_suite_banner_ansi(n_episodes: int) -> str:
    """--eval: blue border (E.md evaluation suite)."""
    border = "\033[94m"
    top = "╔" + "═" * _BOX_INNER + "╗"
    sep = "╠" + "═" * _BOX_INNER + "╣"
    bot = "╚" + "═" * _BOX_INNER + "╝"
    row3 = f"  --eval {n_episodes} · stub + hybrid · outputs below  "
    row3 = row3[:_BOX_INNER].ljust(_BOX_INNER)
    lines = [
        _ansi_framed_cap(border, top),
        _ansi_framed_row(border, "  C I P H E R  — E V A L   S U I T E   ( E . m d )    "),
        _ansi_framed_row(border, "  Stub + hybrid · eval_results/ + proof MD (E.md)     "),
        _ansi_framed_cap(border, sep),
        _ansi_framed_row(border, row3),
        _ansi_framed_cap(border, bot),
    ]
    return "\n".join(lines)

# ── ANSI color helpers (for demo + live ticker) ───────────────────
RED_ANSI   = "\033[91m"
BLUE_ANSI  = "\033[94m"
GREEN_ANSI = "\033[92m"
GOLD_ANSI  = "\033[93m"
DIM_ANSI   = "\033[90m"
RESET_ANSI = "\033[0m"


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
        red_label = "RED COMMANDER + subagents (Hybrid LoRA)"
        mode_badge = "[bold red]HYBRID[/bold red] — RED Commander uses trained LoRA; subagents via HF API"
    elif mode == "live":
        red_label = "RED COMMANDER + dynamic subagents (LLM)"
        mode_badge = "[bold green]LIVE[/bold green] — Commanders + all subagents use real LLM inference"
    else:
        red_label = "RED COMMANDER + subagents (stub heuristics)"
        mode_badge = "[bold dim]STUB[/bold dim] — Commander spawns default roster; heuristic policies"

    ep_label = f"Episode {episode_num}"
    if total_episodes > 1:
        ep_label += f" / {total_episodes}"

    header = (
        f"[bold white]{'═' * 58}[/bold white]\n"
        f"  [bold cyan]C I P H E R[/bold cyan]  —  [bold white]{ep_label}[/bold white]\n"
        f"  [bold red]{'🔴 ' + red_label:<40}[/bold red]  vs  [bold blue]🔵 BLUE COMMANDER + subagents[/bold blue]\n"
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


def _compute_zone_stall(state: Any) -> int:
    """Count consecutive steps RED has been in the current zone."""
    current_zone = int(getattr(state, "red_current_zone", 0))
    path = list(getattr(state, "red_path_history", []))
    graph = getattr(state, "graph", None)
    if not path or graph is None:
        return 0
    stall = 0
    for node in reversed(path):
        try:
            z_raw = graph.nodes[node].get("zone")
            z = z_raw.value if hasattr(z_raw, "value") else int(z_raw)
        except Exception:
            break
        if z == current_zone:
            stall += 1
        else:
            break
    return stall


def _print_live_step(step: int, max_steps: int, red_actions: list,
                     blue_actions: list, state: Any, elapsed_s: float) -> None:
    """ANSI step ticker for live/hybrid (G.md Change 3 — colours + timer)."""
    # Red commander / planner's primary action — prefer planner subagent, fall back to commander
    red_emergent_intent: str = ""
    red_info = "[dim]waiting…[/dim]"
    _red_primary = next(
        (a for a in red_actions if a and a.agent_id.startswith("red_planner")), None
    ) or next(
        (a for a in red_actions if a and a.agent_id.startswith("red_commander")), None
    ) or next(
        (a for a in red_actions if a and str(getattr(a, "agent_id", "")).startswith("red_")), None
    )
    if _red_primary:
        a = _red_primary
        atype = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
        node = f" → n{a.target_node}" if a.target_node is not None else ""
        file_ = f" [{a.target_file[:18]}]" if a.target_file else ""
        role_tag = f"[{getattr(a, 'role', '') or a.agent_id.split('_')[1]}] " if getattr(a, "role", None) else ""
        if atype == "emergent" and a.emergent_data:
            red_emergent_intent = a.emergent_data.intent
            red_info = f"[bold red]{role_tag}emergent:{red_emergent_intent}{node}[/bold red]"
        else:
            red_info = f"[red]{role_tag}{atype}{node}{file_}[/red]"

    # Blue dominant action — flag if any BLUE used emergent
    blue_counts: dict[str, int] = {}
    blue_emergent_intents: list[str] = []
    for a in blue_actions:
        if a:
            k = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
            blue_counts[k] = blue_counts.get(k, 0) + 1
            if k == "emergent" and a.emergent_data:
                blue_emergent_intents.append(a.emergent_data.intent)

    blue_parts = []
    for k, v in blue_counts.items():
        style = "bold blue" if k == "emergent" else "blue"
        blue_parts.append(f"[{style}]{k}×{v}[/{style}]")
    blue_info = " ".join(blue_parts) or "[dim]—[/dim]"

    # Current zone and stall
    zone = getattr(state, "red_current_zone", None)
    if zone is None:
        try:
            zone_raw = state.graph.nodes[state.red_current_node].get("zone")
            zone = zone_raw.value if hasattr(zone_raw, "value") else int(zone_raw)
        except Exception:
            zone = 0
    zone_plain = {0: "PERIMETER", 1: "GENERAL", 2: "SENSITIVE", 3: "CRITICAL/HVT"}.get(int(zone), f"Z{zone}")

    suspicion = float(getattr(state, "red_suspicion_score", 0.0))
    detection = float(getattr(state, "blue_detection_confidence", 0.0))
    exfil_count = len(getattr(state, "red_exfiltrated_files", []))

    # ── Compute active feature flags ──────────────────────────────
    zone_stall = _compute_zone_stall(state)
    stuck_hint_active = suspicion > 0.60 and zone_stall >= 3
    low_det_hint_active = detection < 0.40 and step > 10

    danger_nodes: dict = {}
    try:
        from cipher.agents.red.coordination import get_danger_nodes
        danger_nodes = get_danger_nodes(threshold=0.4)
    except Exception:
        pass

    consec_losses = 0
    try:
        from cipher.training.episode_memory import count_consecutive_losses
        consec_losses = count_consecutive_losses("red")
    except Exception:
        pass

    # Build FLAGS line — only print when at least one flag is active
    flags: list[str] = []
    if zone_stall >= 6:
        flags.append(f"[bold yellow]⚠ ZONE STALL ×{zone_stall}[/bold yellow]")
    elif zone_stall >= 3:
        flags.append(f"[yellow]stall×{zone_stall}[/yellow]")
    if stuck_hint_active:
        flags.append("[bold magenta]STUCK hint → emergent suggested[/bold magenta]")
    if low_det_hint_active:
        flags.append("[magenta]LOW-DET hint → emergent suggested[/magenta]")
    if danger_nodes:
        flags.append(f"[yellow]danger_map:{len(danger_nodes)} node(s)[/yellow]")
    if consec_losses >= 3:
        flags.append(f"[bold orange3]EXPLORE↑ temp=0.9 ({consec_losses} losses)[/bold orange3]")
    if red_emergent_intent:
        flags.append(f"[bold green]RED emergent→{red_emergent_intent}[/bold green]")
    if blue_emergent_intents:
        flags.append(f"[bold cyan]BLUE emergent→{', '.join(blue_emergent_intents)}[/bold cyan]")

    # Estimated total time
    est_total = f"~{int(elapsed_s / step * max_steps)}s" if step > 0 else "?"

    step_line = (
        f"  [bold white]Step {step:02d}/{max_steps}[/bold white]  "
        f"[dim]{elapsed_s:4.1f}s[/dim] [dim]est:{est_total}[/dim]  "
        f"Zone {_zone_badge(zone)}  "
        f"RED: {red_info}  │  "
        f"BLUE: {blue_info}  │  "
        f"Susp [red]{suspicion:.0%}[/red]  "
        f"Det [blue]{detection:.0%}[/blue]"
        + (f"  [bold green]✓ {exfil_count} exfil'd[/bold green]" if exfil_count else "")
    )
    console.print(step_line)

    # Feature flags line — shows exactly which new intelligence features fired this step
    if flags:
        console.print(f"  [dim]  └─ FLAGS:[/dim] {' │ '.join(flags)}")

    # ── Subagent lifecycle events for this step ───────────────────────
    # Pulled from episode_log entries tagged with action_type starting 'subagent_'
    try:
        ep_log = list(getattr(state, "episode_log", []) or [])
        spawn_lines: list[str] = []
        for entry in ep_log:
            if entry.get("step") != step:
                continue
            atype = str(entry.get("action_type", ""))
            if not atype.startswith("subagent_"):
                continue
            ev_type = atype.replace("subagent_", "")   # spawn / dismiss / expire / delegate / reject
            payload = entry.get("payload") or entry.get("action_payload") or {}
            role    = str(payload.get("role", "?"))
            team    = str(payload.get("team", "?"))
            sid     = str(entry.get("agent_id", "?"))
            reason  = str(payload.get("reason", ""))[:60]

            if ev_type == "spawn":
                color = "bold red" if team == "red" else "bold blue"
                brief = f" · {reason}" if reason else ""
                spawn_lines.append(
                    f"  [dim]  └─[/dim] [{color}]▶ SPAWN[/{color}] "
                    f"[{color}]{sid}[/{color}] [dim]({role})[/dim]{brief}"
                )
            elif ev_type in ("dismiss", "expire"):
                color = "red" if team == "red" else "blue"
                label = "✕ DISMISS" if ev_type == "dismiss" else "⌛ EXPIRED"
                spawn_lines.append(
                    f"  [dim]  └─[/dim] [{color}]{label}[/{color}] "
                    f"[dim]{sid} ({role})[/dim]"
                )
            elif ev_type == "delegate":
                color = "red" if team == "red" else "blue"
                spawn_lines.append(
                    f"  [dim]  └─[/dim] [{color}]↺ DELEGATE[/{color}] "
                    f"[dim]{sid} ({role}) · {reason}[/dim]"
                )
        for ln in spawn_lines:
            console.print(ln)
    except Exception:
        pass

    # Exfiltration success banner
    if exfil_count > 0:
        console.print(
            f"  [bold red]{'█' * 18} EXFILTRATION IN PROGRESS — {exfil_count} FILE(S) STOLEN {'█' * 18}[/bold red]"
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
            near_miss = res.get("near_miss", False)
            if correct:
                key_events.append(
                    f"  Step {step:02d} | [bold blue]BLUE ALERT[/bold blue] "
                    f"[green]CORRECT[/green] — RED agent located and flagged!"
                )
            elif near_miss:
                key_events.append(
                    f"  Step {step:02d} | [blue]BLUE alert[/blue] "
                    f"[cyan]NEAR MISS[/cyan] — 1 hop away from RED!"
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
        red_em = getattr(rr, "emergent_action_bonus", 0.0)
        red_em_str = (
            f"  [bold green]emergent_bonus={red_em:+.3f}[/bold green]"
            if red_em > 0 else ""
        )
        red_stall_str = ""
        if getattr(rr, "zone_stall_penalty", 0.0) < 0:
            red_stall_str = f"  stall_pen={rr.zone_stall_penalty:+.3f}"
        console.print(
            f"  [red]RED[/red]   total=[bold]{rr.total:+.3f}[/bold]  "
            f"exfil={getattr(rr,'exfiltration_completeness',0.0):+.3f}  "
            f"stealth={1.0 - getattr(rr,'detection_probability',0.0):+.3f}  "
            f"complexity={getattr(rr,'operation_complexity_multiplier',1.0):.2f}x  "
            f"abort_pen={getattr(rr,'abort_penalty',0.0):+.3f}"
            + red_stall_str
            + red_em_str
        )
        # BLUE
        br = blue_reward
        blue_em = getattr(br, "emergent_action_bonus", 0.0)
        blue_em_str = (
            f"  [bold green]emergent_bonus={blue_em:+.3f}[/bold green]"
            if blue_em > 0 else ""
        )
        blue_inv_eff = getattr(br, "investigation_effectiveness", 0.0)
        blue_inv_str = (
            f"  [bold cyan]inv_eff={blue_inv_eff:+.3f}[/bold cyan]"
            if blue_inv_eff > 0 else ""
        )
        console.print(
            f"  [blue]BLUE[/blue]  total=[bold]{br.total:+.3f}[/bold]  "
            f"detection={getattr(br,'detection_accuracy_score',0.0):+.3f}  "
            f"honeypot={getattr(br,'honeypot_trigger_rate',0.0):+.3f}  "
            f"fp_pen={getattr(br,'false_positive_rate_penalty',0.0):+.3f}"
            + blue_em_str + blue_inv_str
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
                # v2: prefer planner subagent, fall back to commander
                _cb_red = next(
                    (a for a in red_actions if a and a.agent_id.startswith("red_planner")), None
                ) or next(
                    (a for a in red_actions if a and a.agent_id.startswith("red_commander")), None
                ) or next(
                    (a for a in red_actions if a), None
                )
                if _cb_red:
                    a = _cb_red
                    atype = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
                    node = f" → n{a.target_node}" if a.target_node is not None else ""
                    red_info = f"{atype}{node}"
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


def _maybe_start_dashboard_process() -> None:
    """Start the replay dashboard in the background if port 8050 is not already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("localhost", 8050)) != 0:
            subprocess.Popen(
                [sys.executable, "-m", "cipher.dashboard.app"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


def _auto_launch_dashboard(delay: float = 4.0) -> None:
    """Print dashboard URL after a short delay — browser auto-open is disabled."""
    time.sleep(delay)
    console.print(
        f"  [dim cyan]Dashboard ready → http://localhost:8050  "
        f"(open manually)[/dim cyan]"
    )


def _ensure_judge_dashboard(delay: float = 4.0) -> None:
    """Spawn Dash on :8050 if needed; print URL (no auto browser open)."""
    _maybe_start_dashboard_process()
    threading.Thread(target=_auto_launch_dashboard, args=(delay,), daemon=True).start()


def _preload_hybrid_specialists() -> None:
    """
    Eagerly load all 4 LoRA specialists SEQUENTIALLY before the episode loop.

    Without this, models load lazily inside parallel threads at step 1, which
    causes interleaved print output and makes it look like models didn't load.
    Sequential pre-load also avoids OOM from 4 concurrent model.from_pretrained calls.
    """
    from cipher.agents.base_agent import BaseAgent
    from cipher.utils.lora_client import LoRAClient

    specialists = [
        ("red",  "commander",     "RED_COMMANDER_LORA_PATH",      "red trained/cipher-red-commander-v1"),
        ("blue", "commander",     "BLUE_COMMANDER_LORA_PATH",     "blue trained/cipher-blue-commander-v1"),
        ("red",  "planner",       "RED_PLANNER_LORA_PATH",        "red trained/cipher-red-planner-v1"),
        ("red",  "analyst",       "RED_ANALYST_LORA_PATH",         "red trained/cipher-red-analyst-v1"),
        ("blue", "surveillance",  "BLUE_SURVEILLANCE_LORA_PATH",   "blue trained/cipher-blue-surveillance-v1"),
        ("blue", "threat_hunter", "BLUE_THREAT_HUNTER_LORA_PATH",  "blue trained/cipher-blue-threat-hunter-v1"),
    ]

    client = LoRAClient()
    any_loaded = False
    for team, role, env_key, default_path in specialists:
        adapter_path = os.getenv(env_key, default_path)
        if not os.path.exists(adapter_path):
            continue  # Not found — will fall back to HF API at runtime
        try:
            client._load(adapter_path)  # No-op if already cached
            any_loaded = True
        except Exception as exc:
            console.print(f"  [yellow]⚠  LoRA pre-load failed for {adapter_path}: {exc}[/yellow]")

    if any_loaded:
        console.print("  [bold green]✓ All LoRA specialists pre-loaded — ready for inference[/bold green]\n")


def _validate_hybrid_models() -> None:
    """Check that all specialist LoRA adapters exist; warn if missing."""
    specialists = {
        "RED Commander":      os.getenv("RED_COMMANDER_LORA_PATH",       "red trained/cipher-red-commander-v1"),
        "BLUE Commander":     os.getenv("BLUE_COMMANDER_LORA_PATH",      "blue trained/cipher-blue-commander-v1"),
        "RED Planner":        os.getenv("RED_PLANNER_LORA_PATH",         "red trained/cipher-red-planner-v1"),
        "RED Analyst":        os.getenv("RED_ANALYST_LORA_PATH",         "red trained/cipher-red-analyst-v1"),
        "BLUE Surveillance":  os.getenv("BLUE_SURVEILLANCE_LORA_PATH",   "blue trained/cipher-blue-surveillance-v1"),
        "BLUE ThreatHunter":  os.getenv("BLUE_THREAT_HUNTER_LORA_PATH",  "blue trained/cipher-blue-threat-hunter-v1"),
    }
    console.print("\n  [bold cyan]Hybrid specialist check:[/bold cyan]")
    for name, path in specialists.items():
        if os.path.exists(path):
            console.print(f"    [green]✓[/green] {name}: [dim]{path}[/dim]")
        else:
            console.print(f"    [yellow]⚠[/yellow]  {name} not found at '[dim]{path}[/dim]' — will use HF API")
    console.print()


def _apply_scenario_seed(scenario: Any, new_seed: int) -> None:
    """G.md fixed seeds — regenerate graph + targets for a stable showcase topology."""
    from cipher.environment.graph import generate_enterprise_graph
    from cipher.utils.config import config as cfg

    scenario.episode_seed = new_seed
    scenario.target_files = [f"target_file_{new_seed}_{i:03d}" for i in range(3)]
    scenario.generated_graph = generate_enterprise_graph(
        n_nodes=cfg.env_graph_size,
        honeypot_density=cfg.env_honeypot_density,
        seed=new_seed,
    )


def _run_demo_mode(max_steps: int = 30, save_trace: bool = True) -> None:
    """
    Run 3 curated showcase episodes for judge demos.

    Episode 1 — Exfiltration showcase: low difficulty, RED has best chance to breach fast.
    Episode 2 — Blue Defence: elevated honeypot density, BLUE traps RED.
    Episode 3 — Contested: standard difficulty, long back-and-forth battle.
    """
    from cipher.training._episode_runner import run_episode
    from cipher.environment.scenario import ScenarioGenerator
    from cipher.utils.config import config

    mode = os.environ.get("LLM_MODE", "stub")
    run_id = os.environ.get("CIPHER_RUN_ID", f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # (ep, label, difficulty, honeypots, curated_seed) — G.md: fixed seeds for 3 arcs
    demo_configs = [
        (1, "RED EXFILTRATION SHOWCASE", 0.1, 2, 271828),
        (2, "BLUE HONEYPOT DEFENCE", 0.5, 12, 314159),
        (3, "CONTESTED BATTLE", 0.4, 7, 161803),
    ]

    scenario_gen = ScenarioGenerator()

    for ep_num, demo_label, difficulty_override, hp_override, demo_seed in demo_configs:
        console.print()
        console.print(
            f"[bold yellow]{'═' * 60}[/bold yellow]\n"
            f"  [bold white]DEMO EPISODE {ep_num}/3:[/bold white] "
            f"[bold cyan]{demo_label}[/bold cyan]\n"
            f"[bold yellow]{'═' * 60}[/bold yellow]"
        )
        console.print()

        scenario = scenario_gen.generate(ep_num)
        _apply_scenario_seed(scenario, demo_seed)
        scenario.difficulty = difficulty_override
        scenario.n_honeypots = hp_override

        _print_competition_header(
            episode_num=ep_num,
            total_episodes=3,
            mode=mode,
            difficulty=scenario.difficulty,
            max_steps=max_steps,
        )

        ep_start = perf_counter()
        step_callback = None
        if mode in ("live", "hybrid"):
            factory = _get_step_callback_factory(run_id)
            step_callback = factory(ep_num)

        result = run_episode(
            scenario=scenario,
            graph=scenario.generated_graph,
            cfg=config,
            max_steps=max_steps,
            verbose=False,
            save_trace=save_trace,
            episode_number=ep_num,
            step_callback=step_callback,
        )

        ep_elapsed = perf_counter() - ep_start
        if mode in ("live", "hybrid"):
            console.print(f"\n  [dim]Episode {ep_num} finished in {ep_elapsed:.1f}s[/dim]\n")

        if isinstance(result, dict):
            _print_episode_battle(result, ep_num, mode=mode)

            # Narrative report
            try:
                from cipher.utils.storyteller import generate_report
                _state = result.get("state")
                _ep_log = list(getattr(_state, "episode_log", []))
                _terminal = str(getattr(_state, "terminal_reason", "max_steps"))
                _rr = result.get("red_reward")
                _br = result.get("blue_reward")
                generate_report(
                    episode_num=ep_num,
                    episode_log=_ep_log,
                    outcome=_terminal,
                    red_reward=float(_rr.total) if _rr else 0.0,
                    blue_reward=float(_br.total) if _br else 0.0,
                    save=True,
                )
            except Exception:
                pass



        _write_run_state({
            "status": "running",
            "current_episode": ep_num,
            "total_episodes": 3,
            "llm_mode": mode,
            "run_id": run_id,
            "last_updated": datetime.now().isoformat(),
        })

        if ep_num < 3:
            console.print(
                f"\n[dim yellow]  → Dashboard updated. Switch to browser to see Episode {ep_num} replay.[/dim yellow]"
                f"\n[dim]  Starting Episode {ep_num + 1} in 2 seconds…[/dim]\n"
            )
            time.sleep(2)

    _write_run_state({
        "status": "complete",
        "current_episode": 3,
        "total_episodes": 3,
        "llm_mode": mode,
        "run_id": run_id,
        "last_updated": datetime.now().isoformat(),
    })

    console.print()
    console.print(
        "[bold green]╔══════════════════════════════════════════════════════╗[/bold green]\n"
        "[bold green]║  DEMO COMPLETE — Switch to browser for full replay   ║[/bold green]\n"
        "[bold green]║  http://localhost:8050                               ║[/bold green]\n"
        "[bold green]╚══════════════════════════════════════════════════════╝[/bold green]"
    )


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
                        help="Use HuggingFace API for all agents")
    parser.add_argument("--hybrid", action="store_true",
                        help="RED Planner uses trained LoRA, others use HF API")
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
    parser.add_argument("--video", action="store_true",
                        help="Generate CIPHER Cinema video highlights after each episode")
    parser.add_argument("--demo", action="store_true",
                        help="Run 3 curated judge episodes; red banner frame; footer reflects "
                             "LLM mode (--live / --hybrid / stub). Auto-launches dashboard.")
    # E.md Change 5 — evaluation flag
    parser.add_argument(
        "--eval", type=int, default=0, metavar="N",
        help="Run E.md evaluation suite: N episodes per mode (stub + hybrid). "
             "Saves results to eval_results/ and prints comparison table. "
             "Example: python main.py --eval 20",
    )
    args = parser.parse_args()

    # So .env / shell LLM_MODE matches banner + episodes (was: mode stuck "stub" if flags omitted)
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    except Exception:
        pass

    if args.live:
        mode = "live"
    elif args.hybrid:
        mode = "hybrid"
    else:
        os.environ.setdefault("LLM_MODE", "stub")
        _m = os.environ.get("LLM_MODE", "stub").strip().lower()
        mode = _m if _m in ("live", "hybrid", "stub") else "stub"
    os.environ["LLM_MODE"] = mode

    # Change 11: validate specialist model paths when running hybrid
    if mode == "hybrid":
        _validate_hybrid_models()

    # Generate a unique run_id so the dashboard can isolate this run
    run_id = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.environ["CIPHER_RUN_ID"] = run_id
    run_started_at = datetime.now().isoformat()

    # ── E.md Change 5 — --eval flag ──────────────────────────────
    if args.eval > 0:
        try:
            console.print(Text.from_ansi(eval_suite_banner_ansi(args.eval)))
        except Exception:
            print(eval_suite_banner_ansi(args.eval), flush=True)
        console.print()
        from cipher.training.eval_runner import run_eval, print_summary, save_results
        from cipher.utils.report_gen import generate_proof_of_learning_report, save_report
        eval_json = run_eval(
            n_episodes=args.eval,
            modes=["stub", "hybrid"],
            max_steps=args.steps,
            verbose=False,
        )
        print_summary(eval_json)
        saved = save_results(eval_json)
        report_md = generate_proof_of_learning_report(eval_json)
        rpath = save_report(report_md, saved["timestamp"])
        console.print(f"\n[green]✓[/green] Results → [dim]{saved['json']}[/dim]")
        console.print(f"[green]✓[/green] Report  → [dim]{rpath}[/dim]")
        return

    # ── Demo mode ────────────────────────────────────────────────
    if args.demo:
        # Rich renders ANSI reliably in Cursor / non-tty; raw print often shows no colour
        try:
            console.print(Text.from_ansi(judge_demo_banner_ansi(mode)))
        except Exception:
            print(judge_demo_banner_ansi(mode), flush=True)
        console.print("[bold cyan]Starting replay dashboard + browser…[/bold cyan]")
        _ensure_judge_dashboard(delay=4.0)
        _run_demo_mode(max_steps=args.steps, save_trace=not args.no_trace)
        return

    # Clear live step feed from any prior run
    try:
        _LIVE_STEPS_FILE.write_text("", encoding="utf-8")
    except Exception:
        pass

    # Clear previous thoughts file for a clean session
    try:
        thoughts_path = Path("logs") / "agent_thoughts.jsonl"
        thoughts_path.parent.mkdir(exist_ok=True)
        thoughts_path.write_text("", encoding="utf-8")
    except Exception:
        pass

    # ── Auto-launch React War Room for --live / --hybrid ─────────────────
    if mode in ("live", "hybrid"):
        # Serialize initial graph topology once scenario is generated later;
        # start the Flask API server now so the React app can connect.
        def _run_war_room_api() -> None:
            import importlib.util, sys as _sys
            api_path = Path(__file__).parent / "dashboard-react" / "api_server.py"
            if api_path.exists():
                spec = importlib.util.spec_from_file_location("api_server", api_path)
                mod  = importlib.util.module_from_spec(spec)
                _sys.modules["api_server"] = mod
                spec.loader.exec_module(mod)
                mod.app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
            if _s.connect_ex(("localhost", 5001)) != 0:
                threading.Thread(target=_run_war_room_api, daemon=True).start()
                console.print("  [bold cyan]✓ War Room API started on http://localhost:5001[/bold cyan]")
            else:
                console.print("  [dim]War Room API already running on port 5001[/dim]")

        console.print("  [dim]React War Room → http://localhost:5173  (run: cd dashboard-react && npm run dev)[/dim]\n")

    if args.train:
        from cipher.training.loop import TrainingLoop
        # Prioritize --episodes if explicitly provided in CLI
        n = args.train_episodes
        if "--episodes" in sys.argv:
            n = args.episodes
        
        console.print(
            f"[bold cyan]CIPHER Training Loop[/bold cyan] — {n} episodes"
        )
        factory = _get_step_callback_factory(run_id) if mode in ("live", "hybrid") else None
        if factory:
            console.print("  [dim]Calling LLM agents in parallel — step ticker will print below:[/dim]\n")
        TrainingLoop(n_episodes=n, max_steps=args.steps).run(
            step_callback_factory=factory,
            generate_video=args.video,
        )
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

    # ── Pre-load LoRA specialists (hybrid mode only) ─────────────
    # Must happen BEFORE the episode loop so models are cached and ready.
    # This avoids lazy loading inside parallel threads at step 1.
    if mode == "hybrid":
        _preload_hybrid_specialists()

    if mode in ("live", "hybrid"):
        print(session_mode_banner_ansi(mode), flush=True)
        console.print("[dim]Dash replay → http://localhost:8050 (browser opens shortly)[/dim]\n")
        _ensure_judge_dashboard(delay=4.0)
    elif mode == "stub":
        print(session_mode_banner_ansi("stub"), flush=True)
        console.print()

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
            # Show episode memory summary only in hybrid/train — live always starts fresh.
            try:
                if mode == "hybrid":
                    from cipher.training.episode_memory import get_recent_summary, count_consecutive_losses
                    ep_mem = get_recent_summary(n=3)
                    if ep_mem:
                        console.print("  [bold cyan]── EPISODE MEMORY (injected into agent prompts) ──[/bold cyan]")
                        for mem_line in ep_mem.splitlines():
                            console.print(f"  [cyan]{mem_line}[/cyan]")
                        console.print()
                    losses = count_consecutive_losses("red")
                    if losses >= 3:
                        console.print(
                            f"  [bold orange3]⚡ EXPLORATION PRESSURE: RED lost {losses} in a row "
                            f"→ temperature raised to 0.9 + exploration directive injected[/bold orange3]\n"
                        )
            except Exception:
                pass
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

            # ── Narrative post-mortem ──────────────────────────────────────
            try:
                from cipher.utils.storyteller import generate_report
                _state = result.get("state")
                _ep_log = list(getattr(_state, "episode_log", []))
                _terminal = str(getattr(_state, "terminal_reason", "max_steps"))
                _rr = result.get("red_reward")
                _br = result.get("blue_reward")
                _red_total = float(_rr.total) if _rr else 0.0
                _blue_total = float(_br.total) if _br else 0.0
                generate_report(
                    episode_num=ep_num,
                    episode_log=_ep_log,
                    outcome=_terminal,
                    red_reward=_red_total,
                    blue_reward=_blue_total,
                    save=True,
                )
                console.print(f"  [dim yellow]📰 Narrative report → episode_reports/episode_{ep_num:03d}_report.md[/dim yellow]")
            except Exception:
                pass

            # ── CIPHER Cinema video highlight ─────────────────────────────
            if args.video:
                try:
                    from cipher.utils.video_gen import generate_episode_video
                    from pathlib import Path as _Path
                    from datetime import datetime as _dt
                    _Path("episode_highlights").mkdir(exist_ok=True)
                    _state = result.get("state")
                    _rr = result.get("red_reward")
                    _br = result.get("blue_reward")

                    def _to_float(r):
                        if r is None:
                            return 0.0
                        if hasattr(r, "total"):
                            return float(r.total)
                        try:
                            return float(r)
                        except (TypeError, ValueError):
                            return 0.0

                    _ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                    _out = f"episode_highlights/episode_{ep_num:03d}_{_ts}.mp4"
                    _ep_log = [e for e in list(getattr(_state, "episode_log", []) or []) if isinstance(e, dict)]
                    _judgment = result.get("judgment")
                    _verdict = str(getattr(_judgment, "episode_verdict", "") or "").lower()
                    _vid_path = generate_episode_video(
                        {
                            "episode_num": ep_num,
                            "episode_log": _ep_log,
                            "outcome": str(getattr(_state, "terminal_reason", "max_steps")),
                            "verdict": _verdict,
                            "red_reward": _to_float(_rr),
                            "blue_reward": _to_float(_br),
                            "mode": mode,
                            "steps": int(getattr(_state, "step", 0)),
                        },
                        output_path=_out,
                    )
                    if _vid_path:
                        console.print(f"  [dim green]🎬 Video → {_vid_path}[/dim green]")
                except Exception as _ve:
                    console.print(f"  [dim]Video generation skipped: {_ve}[/dim]")



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

    # Estimate API cost (HuggingFace; stub = $0)
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
