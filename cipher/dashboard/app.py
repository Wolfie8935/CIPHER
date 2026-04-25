"""
cipher/dashboard/app.py

CIPHER Episode Replay Dashboard — Phase 12
Reads saved episode traces from episode_traces/ and renders an
interactive 5-panel visualization.

Usage:
    python -m cipher.dashboard.app
    Opens at http://localhost:8050

Does NOT run episodes. Only reads from disk.
"""

import json
import math
import re
from pathlib import Path
from collections import defaultdict

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import networkx as nx
from cipher.dashboard.live import create_live_layout, register_callbacks_on as register_live_callbacks
from cipher.dashboard.replay import (
    infer_runtime_mode,
    extract_trap_steps,
    extract_honeypot_trigger_steps,
    extract_forensics_path,
    operation_complexity_score,
    export_replay_html,
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTES
# ─────────────────────────────────────────────────────────────────────────────

DARK = {
    "bg":           "#0a0e1a",
    "surface":      "#111827",
    "surface2":     "#1c2333",
    "border":       "#2a3550",
    "text":         "#e2e8f0",
    "text_dim":     "#64748b",
    "text_muted":   "#374151",
    "red":          "#ff4444",
    "red_dim":      "#7f2222",
    "blue":         "#4488ff",
    "blue_dim":     "#224488",
    "yellow":       "#fbbf24",
    "purple":       "#a855f7",
    "green":        "#22c55e",
    "gray":         "#6b7280",
    "zone0":        "#4b5563",   # Perimeter – gray
    "zone1":        "#3b82f6",   # General   – blue
    "zone2":        "#f59e0b",   # Sensitive – amber
    "zone3":        "#ef4444",   # Critical  – red
    "plotly_tmpl":  "plotly_dark",
    "plot_bg":      "#111827",
    "paper_bg":     "#111827",
}

ZONE_LABELS = {0: "Perimeter", 1: "General", 2: "Sensitive", 3: "Critical"}

# ─────────────────────────────────────────────────────────────────────────────
# EPISODE TRACE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

TRACES_DIR = Path("episode_traces")


def get_available_traces():
    TRACES_DIR.mkdir(exist_ok=True)
    return sorted(
        TRACES_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_episode(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_timeline(data: dict) -> tuple[list, list, list]:
    """Return (steps, red_suspicion, blue_confidence) lists."""
    def _extract_numeric(entry: dict, keys: tuple[str, ...]) -> float | None:
        for key in keys:
            v = entry.get(key)
            if isinstance(v, (int, float)):
                return float(v)
        st = entry.get("state", {})
        if isinstance(st, dict):
            for key in keys:
                v = st.get(key)
                if isinstance(v, (int, float)):
                    return float(v)
        return None

    # Primary path: aggregate per-step metrics from any entry in that step.
    by_step: dict[int, dict[str, float | None]] = {}
    saw_signal = False
    for entry in data.get("episode_log", []):
        step_raw = entry.get("step", entry.get("t", None))
        if step_raw is None:
            continue
        try:
            step = int(step_raw)
        except Exception:
            continue
        slot = by_step.setdefault(step, {"red": None, "blue": None})
        red_v = _extract_numeric(entry, ("red_suspicion_score", "red_suspicion", "suspicion_score"))
        blue_v = _extract_numeric(entry, ("blue_detection_confidence", "blue_confidence", "detection_confidence"))
        if red_v is not None:
            slot["red"] = red_v
            saw_signal = True
        if blue_v is not None:
            slot["blue"] = blue_v
            saw_signal = True

    if by_step and saw_signal:
        steps = sorted(by_step.keys())
        red_vals: list[float] = []
        blue_vals: list[float] = []
        red_last = 0.0
        blue_last = 0.0
        for st in steps:
            red_here = by_step[st]["red"]
            blue_here = by_step[st]["blue"]
            if red_here is not None:
                red_last = float(red_here)
            if blue_here is not None:
                blue_last = float(blue_here)
            red_vals.append(max(0.0, min(1.0, red_last)))
            blue_vals.append(max(0.0, min(1.0, blue_last)))
        return steps, red_vals, blue_vals

    steps, red_vals, blue_vals = [], [], []

    # Fallback schema: per-step action logs + aggregate traces.
    max_step = int(data.get("step", 0) or 0)
    if max_step <= 0:
        max_step = max(
            [int(e.get("step", e.get("t", 0)) or 0) for e in data.get("episode_log", [])]
            or [0]
        )
    if max_step <= 0:
        return [], [], []

    steps = list(range(1, max_step + 1))
    move_cost_by_step = defaultdict(float)
    for move in data.get("red_movement_history", []):
        st = int(move.get("step", 0) or 0)
        if st <= 0:
            continue
        move_cost_by_step[st] += float(
            move.get("suspicion_cost", move.get("suspicion_delta", 0.0)) or 0.0
        )

    anomaly_by_step = defaultdict(float)
    for evt in data.get("blue_anomaly_history", []):
        st = int(evt.get("step", 0) or 0)
        if st <= 0:
            continue
        anomaly_by_step[st] += float(evt.get("severity", 0.0) or 0.0)

    red_s = 0.0
    blue_c = 0.0
    for st in steps:
        red_s = min(1.0, max(0.0, red_s + move_cost_by_step[st]))
        blue_c = min(1.0, max(0.0, blue_c * 0.85 + anomaly_by_step[st]))
        red_vals.append(red_s)
        blue_vals.append(blue_c)

    # Snap end points to final aggregate values when available.
    if red_vals and isinstance(data.get("red_suspicion_score"), (int, float)):
        red_vals[-1] = float(data["red_suspicion_score"])
    if blue_vals and isinstance(data.get("blue_detection_confidence"), (int, float)):
        blue_vals[-1] = float(data["blue_detection_confidence"])
    return steps, red_vals, blue_vals


def extract_red_path(data: dict) -> list[int]:
    """Extract ordered list of node IDs RED visited."""
    path = []
    seen_steps = set()
    for entry in data.get("episode_log", []):
        step = entry.get("step", entry.get("t"))
        if step in seen_steps:
            continue
        seen_steps.add(step)
        # Try various schema locations
        node = (
            entry.get("red_position")
            or entry.get("red_node")
            or entry.get("state", {}).get("red_position")
            or entry.get("state", {}).get("red_node")
        )
        if node is not None:
            path.append(int(node))
    if path:
        return path

    # Fallback schema.
    hist = data.get("red_path_history", [])
    if isinstance(hist, list) and hist:
        path = [int(n) for n in hist if n is not None]
    if not path:
        moves = sorted(
            data.get("red_movement_history", []),
            key=lambda m: int(m.get("step", 0) or 0),
        )
        for m in moves:
            if not path and m.get("from_node") is not None:
                path.append(int(m.get("from_node")))
            if m.get("to_node") is not None:
                path.append(int(m.get("to_node")))
    if not path and data.get("red_current_node") is not None:
        path = [int(data.get("red_current_node"))]
    return path


def extract_context_resets(data: dict) -> list[int]:
    resets = []
    for entry in data.get("episode_log", []):
        if entry.get("event") in ("context_reset", "ENV_CONTEXT_RESET") or entry.get(
            "context_reset"
        ):
            resets.append(entry.get("step", entry.get("t", 0)))
    if resets:
        return resets
    if isinstance(data.get("red_context_resets"), int):
        # Count only available in this schema, no exact reset steps.
        return [0] * int(data.get("red_context_resets", 0))
    return resets


def extract_dead_drops(data: dict) -> list[dict]:
    drops = []
    for entry in data.get("episode_log", []):
        if entry.get("event") == "dead_drop_written" or entry.get("dead_drop"):
            dd = entry.get("dead_drop", entry)
            if not isinstance(dd, dict):
                continue
            drops.append(
                {
                    "step": entry.get("step", entry.get("t", "?")),
                    "node": dd.get("node", dd.get("location", "?")),
                    "tokens": dd.get("token_count", dd.get("tokens", 0)),
                    "efficiency": dd.get("efficiency", dd.get("memory_efficiency", 0.0)),
                    "integrity": dd.get("integrity", "valid"),
                    "preview": (dd.get("continuation_directive") or dd.get("content") or "")[:80],
                    "written_by": dd.get("written_by", dd.get("agent", "RED")),
                }
            )
    if drops:
        return drops

    # Fallback schema.
    for dd in data.get("dead_drops_on_disk", []):
        if isinstance(dd, str):
            try:
                loaded = json.loads(Path(dd).read_text())
            except Exception:
                loaded = {}
            if isinstance(loaded, dict):
                objective = str(loaded.get("mission_status", {}).get("primary_objective", ""))
                node_guess = "?"
                m = re.search(r"(?:node|NODE)[_-]?(\d+)", objective)
                if m:
                    node_guess = int(m.group(1))
                drops.append(
                    {
                        "step": loaded.get("written_at_step", "?"),
                        "node": node_guess,
                        "tokens": loaded.get("token_count", 0),
                        "efficiency": loaded.get("memory_efficiency", 0.0),
                        "integrity": "valid" if loaded.get("integrity_hash") else "unknown",
                        "preview": (loaded.get("continuation_directive") or "")[:80],
                        "written_by": loaded.get("written_by", "RED"),
                    }
                )
            continue
        if not isinstance(dd, dict):
            continue
        drops.append(
            {
                "step": dd.get("step", dd.get("created_step", "?")),
                "node": dd.get("node", dd.get("node_id", dd.get("location", "?"))),
                "tokens": dd.get("token_count", dd.get("tokens", 0)),
                "efficiency": dd.get("efficiency", dd.get("memory_efficiency", 0.0)),
                "integrity": dd.get("integrity", "valid"),
                "preview": (dd.get("continuation_directive") or dd.get("content") or "")[:80],
                "written_by": dd.get("written_by", dd.get("agent", "RED")),
            }
        )
    return drops


def extract_actions(data: dict) -> list[dict]:
    actions = []
    for entry in data.get("episode_log", []):
        step = entry.get("step", entry.get("t", "?"))
        # Red actions
        for act in entry.get("red_actions", entry.get("actions_red", [])):
            actions.append(
                {
                    "step": step,
                    "team": "RED",
                    "agent": act.get("agent", act.get("agent_id", "RED")),
                    "action": act.get("action", act.get("action_type", "?")),
                    "target": act.get("target", act.get("node", "")),
                    "reasoning": act.get("reasoning", act.get("thought", "")),
                }
            )
        # Blue actions
        for act in entry.get("blue_actions", entry.get("actions_blue", [])):
            actions.append(
                {
                    "step": step,
                    "team": "BLUE",
                    "agent": act.get("agent", act.get("agent_id", "BLUE")),
                    "action": act.get("action", act.get("action_type", "?")),
                    "target": act.get("target", act.get("node", "")),
                    "reasoning": act.get("reasoning", act.get("thought", "")),
                }
            )
        # Events
        event = entry.get("event", "")
        if event:
            actions.append(
                {
                    "step": step,
                    "team": "EVENT",
                    "agent": "",
                    "action": event,
                    "target": "",
                    "reasoning": entry.get("detail", entry.get("message", "")),
                }
            )
    if actions:
        return actions

    # Fallback schema: one action per log entry.
    for entry in data.get("episode_log", []):
        step = entry.get("step", entry.get("t", "?"))
        agent_id = str(entry.get("agent_id", ""))
        action_type = str(entry.get("action_type", "?")).upper()
        payload = entry.get("payload", {}) or {}
        if not isinstance(payload, dict):
            payload = {}
        result = entry.get("result", {}) or {}
        if not isinstance(result, dict):
            result = {"success": True, "detail": str(result)}

        team = "EVENT"
        if agent_id.startswith("red_"):
            team = "RED"
        elif agent_id.startswith("blue_"):
            team = "BLUE"

        target = payload.get("target_node", payload.get("target_file", ""))
        reasoning = payload.get("reasoning", "")
        if result and (not result.get("success", True)):
            fail_reason = result.get("reason", "failed")
            reasoning = f"{reasoning} | result: {fail_reason}".strip(" |")

        actions.append(
            {
                "step": step,
                "team": team,
                "agent": agent_id.replace("_", " ").upper(),
                "action": action_type,
                "target": target,
                "reasoning": reasoning,
            }
        )

    # Include trap events when present.
    for evt in data.get("trap_events_log", data.get("traps_triggered_log", [])):
        if not isinstance(evt, dict):
            continue
        actions.append(
            {
                "step": evt.get("step", "?"),
                "team": "EVENT",
                "agent": "TRAP",
                "action": f"TRAP FIRED {str(evt.get('trap_type', '?')).upper()}",
                "target": evt.get("state_changes", {}),
                "reasoning": evt.get("effect_description", ""),
            }
        )
    return actions


def extract_rewards(data: dict) -> dict:
    r = data.get("rewards", data.get("final_rewards", {}))
    if not r:
        # Try flattened keys
        keys = [
            "red_reward", "blue_reward", "exfil", "stealth", "memory",
            "complexity", "abort_pen", "hp_pen", "red_total", "blue_total",
        ]
        r = {k: data.get(k, 0.0) for k in keys if k in data}
    if r:
        return r

    # Fallback schema: infer a reasonable reward breakdown from available episode metrics.
    terminal_reason = str(data.get("terminal_reason", "")).lower()
    red_exfil_files = data.get("red_exfiltrated_files", []) or []
    red_creds = data.get("red_credentials_acquired", []) or []
    trap_events = data.get("trap_events_log", data.get("traps_triggered_log", [])) or []
    red_susp = float(data.get("red_suspicion_score", 0.0) or 0.0)
    blue_conf = float(data.get("blue_detection_confidence", 0.0) or 0.0)
    resets = int(data.get("red_context_resets", 0) or 0)
    hp_triggered = data.get("blue_honeypots_triggered", []) or []
    blue_fp = float(data.get("blue_false_positives", 0) or 0.0)

    exfil = 0.35 * len(red_exfil_files)
    stealth = 0.6 - red_susp
    memory = -0.08 * resets
    complexity = 0.10 * len(red_creds) + 0.05 * len(trap_events)
    abort_pen = -0.5 if "abort" in terminal_reason else 0.0
    hp_pen = -0.15 * len(hp_triggered)
    red_total = exfil + stealth + memory + complexity + abort_pen + hp_pen

    detect_bonus = 0.9 if "detect" in terminal_reason else 0.0
    confidence_bonus = 0.2 * blue_conf
    honeypot_bonus = 0.12 * len(hp_triggered)
    fp_pen = -0.10 * blue_fp
    blue_total = detect_bonus + confidence_bonus + honeypot_bonus + fp_pen

    return {
        "exfil": exfil,
        "stealth": stealth,
        "memory": memory,
        "complexity": complexity,
        "abort_pen": abort_pen,
        "hp_pen": hp_pen,
        "red_total": red_total,
        "blue_total": blue_total,
    }


def build_graph_layout(data: dict) -> tuple[nx.Graph, dict, dict]:
    """Build NetworkX graph and node positions from episode data."""
    try:
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.utils.config import config
        seed = data.get("seed", data.get("episode_seed", 7961))
        G = generate_enterprise_graph(
            n_nodes=config.env_graph_size,
            honeypot_density=config.env_honeypot_density,
            seed=seed,
        )
        pos = nx.spring_layout(G, seed=42)
        node_meta = {}
        for nid in G.nodes():
            n = G.nodes[nid]
            zone_val = getattr(n.get("zone"), "value", n.get("zone", 0))
            node_meta[nid] = {
                "id": nid,
                "zone": zone_val,
                "type": str(n.get("node_type", "workstation")),
                "hostname": str(n.get("hostname", f"node-{nid}")),
                "files": n.get("files", []),
            }
        return G, pos, node_meta
    except Exception:
        pass

    # Fallback if import fails
    G = nx.Graph()
    node_meta = {}
    for i in range(50):
        G.add_node(i)
        zone = 0 if i < 10 else (1 if i < 25 else (2 if i < 40 else 3))
        node_meta[i] = {
            "id": i,
            "zone": zone,
            "type": "workstation",
            "hostname": f"node-{i:02d}",
        }
    seed = data.get("seed", data.get("episode_seed", 42))
    pos = nx.spring_layout(G, seed=seed, k=2.5)
    return G, pos, node_meta


def get_summary(data: dict) -> dict:
    outcome = (
        data.get("outcome")
        or data.get("terminal_condition")
        or data.get("result")
        or data.get("terminal_reason")
        or "UNKNOWN"
    )
    steps_from_log = len(set(e.get("step", e.get("t", 0)) for e in data.get("episode_log", [])))
    steps = int(data.get("step", 0) or steps_from_log)
    max_steps = data.get("max_steps", max(steps, 1))
    if isinstance(outcome, str):
        outcome = outcome.replace("_", " ").upper()
    rewards = extract_rewards(data)
    dead_drops = extract_dead_drops(data)
    resets = extract_context_resets(data)
    oversight = data.get("oversight_flags", data.get("oversight", []))
    return {
        "outcome": str(outcome).upper(),
        "steps": steps,
        "max_steps": max_steps,
        "rewards": rewards,
        "dead_drops": len(dead_drops),
        "resets": len(resets),
        "oversight": oversight if oversight else ["no flags fired"],
        "fleet_verdict": data.get("fleet_verdict", data.get("judgment", {})),
        "complexity": operation_complexity_score(data),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────


def build_network_figure(data: dict, step: int, C: dict) -> go.Figure:
    G, pos, node_meta = build_graph_layout(data)
    red_path = extract_red_path(data)
    drops = extract_dead_drops(data)
    drop_nodes = {int(d["node"]) for d in drops if str(d["node"]).isdigit()}

    # Nodes visited by RED up to current step
    moves = sorted(
        data.get("red_movement_history", []),
        key=lambda m: int(m.get("step", 0) or 0),
    )
    if moves:
        path_up_to = []
        for m in moves:
            st = int(m.get("step", 0) or 0)
            if st > step:
                continue
            if not path_up_to and m.get("from_node") is not None:
                path_up_to.append(int(m.get("from_node")))
            if m.get("to_node") is not None:
                path_up_to.append(int(m.get("to_node")))
        if not path_up_to and red_path:
            path_up_to = [red_path[0]]
    else:
        # Extract RED positions from episode_log successful move actions, step-filtered
        direct_path = []
        _seen_move_steps: set = set()
        for entry in sorted(data.get("episode_log", []), key=lambda e: int(e.get("step", 0))):
            ep_step = int(entry.get("step", 0))
            if ep_step > step:
                break
            if (entry.get("action_type") == "move"
                    and str(entry.get("agent_id", "")).startswith("red_")
                    and entry.get("result", {}).get("success")
                    and ep_step not in _seen_move_steps):
                tgt = (entry.get("payload") or {}).get("target_node")
                if tgt is not None:
                    direct_path.append(int(tgt))
                    _seen_move_steps.add(ep_step)
        if direct_path:
            path_up_to = direct_path
        elif red_path:
            path_up_to = red_path[: step + 1] if step < len(red_path) else red_path
        else:
            path_up_to = []
    red_current = path_up_to[-1] if path_up_to else None
    red_visited = set(path_up_to)
    forensics_path = extract_forensics_path(data)

    # Honeypot and BLUE investigated nodes
    honeypots = set()
    blue_investigated = set()
    for entry in data.get("episode_log", []):
        s = entry.get("step", entry.get("t", 0))
        if s > step:
            continue
        for hp in entry.get("honeypot_nodes", []):
            honeypots.add(int(hp))
        for bi in entry.get("blue_investigated", []):
            blue_investigated.add(int(bi))

    # ── Edge trace
    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    traces = [
        go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(
                width=0.9,
                color="rgba(148,163,184,0.20)" if C is DARK else "rgba(100,116,139,0.35)",
            ),
            hoverinfo="none", showlegend=False,
        )
    ]

    # ── RED path trail (gradient brightness)
    if len(path_up_to) > 1:
        n = len(path_up_to)
        for i in range(n - 1):
            a, b = path_up_to[i], path_up_to[i + 1]
            if a not in pos or b not in pos:
                continue
            intensity = 0.2 + 0.8 * (i / max(n - 2, 1))
            alpha = int(255 * intensity)
            color = f"rgba(255,68,68,{intensity:.2f})"
            traces.append(
                go.Scatter(
                    x=[pos[a][0], pos[b][0]],
                    y=[pos[a][1], pos[b][1]],
                    mode="lines",
                    line=dict(width=2.5 + 1.5 * intensity, color=color),
                    hoverinfo="none", showlegend=False,
                )
            )

    # ── BLUE Forensics reconstructed path overlay
    if len(forensics_path) > 1:
        f_nodes = []
        moves_by_step = {int(m.get("step", 0) or 0): m for m in moves if isinstance(m, dict)}
        # Limit overlay to current playback step using movement chronology.
        if moves_by_step:
            for st in sorted(moves_by_step.keys()):
                if st > step:
                    continue
                mv = moves_by_step[st]
                if mv.get("to_node") is not None:
                    f_nodes.append(int(mv.get("to_node")))
        if not f_nodes:
            f_nodes = forensics_path[: max(1, min(step + 1, len(forensics_path)))]
        for i in range(len(f_nodes) - 1):
            a, b = f_nodes[i], f_nodes[i + 1]
            if a not in pos or b not in pos:
                continue
            traces.append(
                go.Scatter(
                    x=[pos[a][0], pos[b][0]],
                    y=[pos[a][1], pos[b][1]],
                    mode="lines",
                    name="Forensics Path" if i == 0 else None,
                    line=dict(width=1.8, color=C["blue"], dash="dot"),
                    hoverinfo="none",
                    showlegend=(i == 0),
                )
            )

    # ── Nodes by zone
    zone_nodes = defaultdict(list)
    for nid in G.nodes():
        meta = node_meta.get(nid, {})
        zone = meta.get("zone", 0)
        zone_nodes[zone].append(nid)

    zone_colors = {
        0: C["zone0"], 1: C["zone1"], 2: C["zone2"], 3: C["zone3"]
    }

    for zone, znodes in sorted(zone_nodes.items()):
        xs = [pos[n][0] for n in znodes if n in pos]
        ys = [pos[n][1] for n in znodes if n in pos]
        labels = []
        hovers = []
        symbols = []
        sizes = []
        line_colors = []
        line_widths = []
        opacities = []

        for n in znodes:
            if n not in pos:
                continue
            meta = node_meta.get(n, {})
            hostname = meta.get("hostname", f"node-{n:02d}")
            ntype = meta.get("type", "workstation")
            label = hostname[:10]

            sym = "circle"
            size = 14
            lc = "rgba(0,0,0,0)"
            lw = 0
            op = 0.85

            if n in honeypots:
                sym = "diamond"
                lc = C["yellow"]
                lw = 2
            if n in blue_investigated:
                lc = C["blue"]
                lw = 2.5
            if n in red_visited and n != red_current:
                op = 1.0
                lc = C["red_dim"]
                lw = 1.5
            if n == red_current:
                sym = "circle"
                size = 20
                lc = C["red"]
                lw = 3
                op = 1.0

            hover = (
                f"<b>{hostname}</b><br>"
                f"Node: {n} | Zone {zone} ({ZONE_LABELS[zone]})<br>"
                f"Type: {ntype}<br>"
                + ("🍯 HONEYPOT<br>" if n in honeypots else "")
                + ("🔍 BLUE Investigated<br>" if n in blue_investigated else "")
                + ("📦 Dead Drop location<br>" if n in drop_nodes else "")
                + ("🔴 RED Current Position<br>" if n == red_current else
                   ("👣 RED Visited<br>" if n in red_visited else ""))
            )
            labels.append(label if n in red_visited or n in honeypots or n in drop_nodes else "")
            hovers.append(hover)
            symbols.append(sym)
            sizes.append(size)
            line_colors.append(lc)
            line_widths.append(lw)
            opacities.append(op)

        traces.append(
            go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                name=f"Z{zone} {ZONE_LABELS[zone]}",
                marker=dict(
                    size=sizes,
                    color=zone_colors[zone],
                    symbol=symbols,
                    opacity=opacities,
                    line=dict(color=line_colors, width=line_widths),
                ),
                text=labels,
                textfont=dict(size=7, color=C["text_dim"]),
                textposition="top center",
                hovertext=hovers,
                hoverinfo="text",
            )
        )

    # ── Dead drop markers
    if drop_nodes:
        ddx = [pos[n][0] for n in drop_nodes if n in pos]
        ddy = [pos[n][1] for n in drop_nodes if n in pos]
        traces.append(
            go.Scatter(
                x=ddx, y=ddy, mode="markers+text",
                name="Dead Drops",
                marker=dict(size=10, color=C["yellow"], symbol="square", opacity=0.9),
                text=["📦"] * len(ddx),
                textposition="middle center",
                hoverinfo="none", showlegend=True,
            )
        )

    # ── Trap event markers and false-trail chase arrows
    trap_nodes = []
    false_trail_arrows = []
    for evt in data.get("trap_events_log", data.get("traps_triggered_log", [])):
        if not isinstance(evt, dict):
            continue
        st = evt.get("step", 0)
        if isinstance(st, int) and st > step:
            continue
        state_changes = evt.get("state_changes", {}) or {}
        if isinstance(state_changes, dict):
            n = state_changes.get("fake_node", state_changes.get("node", state_changes.get("target_node")))
            if isinstance(n, (int, float, str)) and str(n).isdigit():
                trap_nodes.append(int(n))
            if "fake_node" in state_changes and red_current is not None and red_current in pos:
                fn = state_changes.get("fake_node")
                if isinstance(fn, (int, float, str)) and str(fn).isdigit() and int(fn) in pos:
                    false_trail_arrows.append((red_current, int(fn)))

    if trap_nodes:
        tx = [pos[n][0] for n in trap_nodes if n in pos]
        ty = [pos[n][1] for n in trap_nodes if n in pos]
        traces.append(
            go.Scatter(
                x=tx, y=ty, mode="markers+text",
                name="Trap Events",
                marker=dict(size=12, color=C["purple"], symbol="x", opacity=0.95),
                text=["⚡"] * len(tx),
                textposition="middle center",
                hoverinfo="none", showlegend=True,
            )
        )
    for idx, (src, dst) in enumerate(false_trail_arrows):
        traces.append(
            go.Scatter(
                x=[pos[src][0], pos[dst][0]],
                y=[pos[src][1], pos[dst][1]],
                mode="lines",
                name="False-Trail Chase" if idx == 0 else None,
                line=dict(width=1.5, color=C["yellow"], dash="dash"),
                hoverinfo="none",
                showlegend=(idx == 0),
            )
        )

    # ── RED position pulse
    if red_current is not None and red_current in pos:
        rx, ry = pos[red_current]
        traces.append(
            go.Scatter(
                x=[rx], y=[ry], mode="markers",
                name="RED Position",
                marker=dict(
                    size=28, color="rgba(255,68,68,0.2)",
                    line=dict(color=C["red"], width=2),
                ),
                hoverinfo="none", showlegend=True,
            )
        )

    # ── CLASH Marker (Blue investigating Red's current node)
    if red_current is not None and red_current in pos and red_current in blue_investigated:
        cx, cy = pos[red_current]
        traces.append(
            go.Scatter(
                x=[cx], y=[cy], mode="markers+text",
                name="⚔️ CLASH",
                marker=dict(
                    size=32, symbol="star", color=C["yellow"],
                    line=dict(color=C["red"], width=3),
                ),
                text=["⚔️ CLASH!"],
                textposition="top center",
                textfont=dict(size=14, color=C["yellow"]),
                hoverinfo="text",
                hovertext="⚔️ CLASH! Blue investigating Red's current node!",
                showlegend=True,
            )
        )

    fig = go.Figure(traces)
    fig.update_layout(
        template=C["plotly_tmpl"],
        paper_bgcolor=C["plot_bg"],
        plot_bgcolor=C["plot_bg"],
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            font=dict(size=10, color=C["text"]),
            x=1.0, xanchor="right", y=1.0,
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        hovermode="closest",
    )
    return fig


def build_timeline_figure(
    data: dict, step: int, C: dict, reward_history: list | None = None
) -> go.Figure:
    if reward_history is not None:
        # Training curve mode (Phase 13)
        ep_nums = list(range(len(reward_history)))
        red_r = [r.get("red_total", r.get("red", 0)) for r in reward_history]
        blue_r = [r.get("blue_total", r.get("blue", 0)) for r in reward_history]
        fig = go.Figure([
            go.Scatter(x=ep_nums, y=red_r, name="RED Reward", line=dict(color=C["red"], width=2)),
            go.Scatter(x=ep_nums, y=blue_r, name="BLUE Reward", line=dict(color=C["blue"], width=2)),
        ])
        fig.update_layout(
            template=C["plotly_tmpl"],
            paper_bgcolor=C["plot_bg"],
            plot_bgcolor=C["plot_bg"],
            margin=dict(l=40, r=10, t=10, b=30),
            xaxis_title="Episode",
            yaxis_title="Reward",
            legend=dict(font=dict(size=10)),
        )
        return fig

    steps, red_vals, blue_vals = extract_timeline(data)
    resets = extract_context_resets(data)
    honeypot_steps = extract_honeypot_trigger_steps(data)
    trap_steps = extract_trap_steps(data)

    shapes = []
    annotations = []

    # Detection threshold
    shapes.append(dict(
        type="line", x0=0, x1=max(steps) if steps else 1,
        y0=0.8, y1=0.8,
        line=dict(color=C["red_dim"], width=1.8, dash="dot"),
    ))
    annotations.append(dict(
        x=max(steps) if steps else 1, y=0.835,
        text="Detection threshold 0.80",
        font=dict(size=10, color=C["text_dim"], family="'Inter', sans-serif"),
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        align="right",
        bgcolor="rgba(15, 23, 42, 0.70)",
        bordercolor=C["border"],
        borderwidth=1,
        borderpad=3,
    ))

    # MEMENTO lines
    for rs in resets:
        if not isinstance(rs, int) or rs <= 0:
            continue
        shapes.append(dict(
            type="line", x0=rs, x1=rs, y0=0, y1=1,
            line=dict(color=C["purple"], width=1.5, dash="dash"),
        ))
        annotations.append(dict(
            x=rs, y=1.02, text="MEMENTO",
            font=dict(size=8, color=C["purple"], family="'JetBrains Mono', monospace"),
            showarrow=False,
            bgcolor="rgba(15, 23, 42, 0.50)",
        ))

    current_step = int(step or 0)
    visible_points = [(s, r, b) for s, r, b in zip(steps, red_vals, blue_vals) if s <= current_step]
    vis_steps = [p[0] for p in visible_points]
    vis_red_vals = [p[1] for p in visible_points]
    vis_blue_vals = [p[2] for p in visible_points]

    # Spike annotations on currently visible RED segment
    for i in range(1, len(vis_red_vals)):
        delta = vis_red_vals[i] - vis_red_vals[i - 1]
        if delta > 0.1:
            annotations.append(dict(
                x=vis_steps[i], y=vis_red_vals[i],
                text=f"▲{delta:.2f}",
                font=dict(size=8, color=C["red"], family="'JetBrains Mono', monospace"),
                showarrow=True, arrowhead=2,
                arrowcolor=C["red"], arrowsize=0.8, arrowwidth=1.1, ay=-20,
                bgcolor="rgba(15, 23, 42, 0.55)",
                borderpad=2,
            ))

    # Honeypot trigger annotations
    for hs in honeypot_steps:
        if steps and hs <= current_step:
            annotations.append(dict(
                x=hs, y=0.9, text="✅",
                font=dict(size=12, color=C["green"]),
                showarrow=False,
            ))

    # Trap event annotations
    for ts in trap_steps:
        if steps and ts <= current_step:
            annotations.append(dict(
                x=ts, y=0.72, text="⚡",
                font=dict(size=11, color=C["purple"]),
                showarrow=False,
            ))

    # Step cursor
    if steps:
        cur_x = min(current_step, max(steps))
        shapes.append(dict(
            type="line", x0=cur_x, x1=cur_x, y0=0, y1=1,
            line=dict(color=C["yellow"], width=1.5),
        ))

    traces = []
    if vis_steps:
        traces.append(go.Scatter(
            x=vis_steps, y=vis_red_vals,
            name="RED Suspicion", mode="lines+markers",
            line=dict(color=C["red"], width=2.4),
            marker=dict(size=5, symbol="circle", line=dict(width=0.8, color=C["plot_bg"])),
            hovertemplate="<b>RED Suspicion</b><br>Step %{x}<br>Score %{y:.3f}<extra></extra>",
        ))
        traces.append(go.Scatter(
            x=vis_steps, y=vis_blue_vals,
            name="BLUE Confidence", mode="lines+markers",
            line=dict(color=C["blue"], width=2.4),
            marker=dict(size=5, symbol="circle", line=dict(width=0.8, color=C["plot_bg"])),
            hovertemplate="<b>BLUE Confidence</b><br>Step %{x}<br>Score %{y:.3f}<extra></extra>",
        ))
    else:
        # No data: show helpful placeholder
        traces.append(go.Scatter(
            x=[0, 1], y=[0, 0],
            name="No data",
            mode="lines",
            line=dict(color=C["text_muted"], width=1, dash="dot"),
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        template=C["plotly_tmpl"],
        paper_bgcolor=C["plot_bg"],
        plot_bgcolor=C["plot_bg"],
        font=dict(
            family="'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            size=11,
            color=C["text"],
        ),
        margin=dict(l=52, r=22, t=30, b=44),
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            title=dict(text="Step", standoff=8),
            range=[0, max(steps) if steps else 10],
            tickfont=dict(size=10, color=C["text_dim"]),
            titlefont=dict(size=11, color=C["text"]),
            gridcolor="rgba(148, 163, 184, 0.16)",
            zeroline=False,
            automargin=True,
        ),
        yaxis=dict(
            title=dict(text="Score", standoff=8),
            range=[0, 1.1],
            tickfont=dict(size=10, color=C["text_dim"]),
            titlefont=dict(size=11, color=C["text"]),
            gridcolor="rgba(148, 163, 184, 0.20)",
            zeroline=False,
            automargin=True,
        ),
        legend=dict(
            orientation="h",
            x=0.0,
            xanchor="left",
            y=1.16,
            yanchor="top",
            font=dict(size=10, color=C["text"]),
            bgcolor="rgba(15, 23, 42, 0.72)",
            bordercolor=C["border"],
            borderwidth=1,
            traceorder="normal",
            itemsizing="constant",
            itemwidth=36,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.95)",
            bordercolor=C["border"],
            font=dict(size=10, color=C["text"]),
        ),
    )
    return fig


def build_reward_figure(rewards: dict, C: dict) -> go.Figure:
    component_keys = ["exfil", "stealth", "memory", "complexity", "abort_pen", "hp_pen"]
    labels, vals, colors = [], [], []
    for k in component_keys:
        v = float(rewards.get(k, 0.0))
        labels.append(k.replace("_", " ").title())
        vals.append(v)
        colors.append(C["red"] if v >= 0 else C["red_dim"])

    red_total = float(rewards.get("red_total", rewards.get("red", 0.0)))
    blue_total = float(rewards.get("blue_total", rewards.get("blue", 0.0)))

    fig = go.Figure()
    if any(v != 0 for v in vals):
        fig.add_trace(go.Bar(
            y=labels, x=vals,
            orientation="h",
            marker_color=colors,
            name="RED Components",
            text=[f"{v:+.2f}" for v in vals],
            textposition="auto",
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))
    else:
        # Fallback: just show red vs blue totals
        fig.add_trace(go.Bar(
            y=["RED Total", "BLUE Total"],
            x=[red_total, blue_total],
            orientation="h",
            marker_color=[C["red"], C["blue"]],
            text=[f"{red_total:+.2f}", f"{blue_total:+.2f}"],
            textposition="auto",
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ))

    y_vals = vals + [red_total, blue_total]
    max_abs = max([abs(v) for v in y_vals] + [0.5])
    y_lim = min(3.0, max(1.1, max_abs * 1.2))

    fig.update_layout(
        template=C["plotly_tmpl"],
        paper_bgcolor=C["plot_bg"],
        plot_bgcolor=C["plot_bg"],
        margin=dict(l=86, r=10, t=6, b=12),
        showlegend=False,
        xaxis=dict(
            title="Value",
            range=[-y_lim, y_lim],
            tickfont=dict(size=9),
            zeroline=True,
            zerolinecolor=C["border"],
            automargin=True,
        ),
        yaxis=dict(
            tickfont=dict(size=8),
            automargin=True,
        ),
        bargap=0.25,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def panel(title: str, children, C: dict, extra_style: dict | None = None) -> html.Div:
    style = {
        "background": C["surface"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "8px",
        "padding": "12px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "6px",
        **(extra_style or {}),
    }
    return html.Div([
        html.Div([
            html.Span("◉ ", style={"color": C["red"], "fontSize": "10px"}),
            html.Span(title, style={
                "fontSize": "10px",
                "fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
                "letterSpacing": "0.15em",
                "color": C["text_dim"],
                "fontWeight": "600",
            }),
        ], style={"marginBottom": "8px"}),
        *children,
    ], style=style)


def outcome_color(outcome: str, C: dict) -> str:
    o = outcome.upper()
    if "EXFIL" in o:
        return C["red"]
    if "DETECT" in o:
        return C["blue"]
    if "ABORT" in o:
        return C["gray"]
    if "MAX" in o:
        return C["yellow"]
    return C["text_dim"]


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap",
    ],
    assets_folder=str(Path(__file__).parent / "assets"),
    suppress_callback_exceptions=True,
    title="CIPHER · Episode Replay",
)
server = app.server

# ── Initial dark theme colours (client-side toggle switches between DARK/LIGHT)
C_INIT = DARK


def _make_winning_metrics_bar(C: dict) -> html.Div:
    """Read rewards_log.csv and build the Winning Metrics top bar."""
    try:
        from cipher.dashboard.analytics import compute_winning_metrics, load_rewards_df
        metrics = compute_winning_metrics(load_rewards_df())
    except Exception:
        metrics = {
            "total_exfils": 0, "mean_ttd": 0.0, "efficiency_pct": 0.0,
            "model_confidence": "N/A", "red_win_rate": 0.0, "blue_win_rate": 0.0,
            "n_episodes": 0,
        }

    eff = metrics["efficiency_pct"]
    eff_str = f"+{eff:.0f}%" if eff >= 0 else f"{eff:.0f}%"
    conf = metrics["model_confidence"]
    conf_color = C["green"] if conf == "HIGH" else C["yellow"] if conf == "MED" else C["gray"]

    tiles = [
        ("TOTAL EXFILS", str(metrics["total_exfils"]), C["red"]),
        ("MEAN TTD", f"{metrics['mean_ttd']:.1f} steps", C["blue"]),
        ("TRAIN EFFICIENCY", eff_str, C["green"] if eff >= 0 else C["red"]),
        ("MODEL CONFIDENCE", conf, conf_color),
        ("RED WIN RATE", f"{metrics['red_win_rate']*100:.0f}%", C["red"]),
        ("BLUE WIN RATE", f"{metrics['blue_win_rate']*100:.0f}%", C["blue"]),
        ("EPISODES", str(metrics["n_episodes"]), C["text_dim"]),
    ]

    children = []
    for label, value, color in tiles:
        children.append(html.Div([
            html.Div(value, style={
                "fontSize": "18px", "fontWeight": "800",
                "color": color, "fontFamily": "'JetBrains Mono', monospace",
            }),
            html.Div(label, style={
                "fontSize": "8px", "color": C["text_dim"],
                "letterSpacing": "0.12em", "fontFamily": "'JetBrains Mono', monospace",
                "marginTop": "2px",
            }),
        ], style={
            "padding": "8px 16px",
            "borderRight": f"1px solid {C['border']}",
            "textAlign": "center",
            "flex": "1",
        }))

    return html.Div(
        children,
        style={
            "display": "flex",
            "background": C["surface"],
            "borderBottom": f"2px solid {C['border']}",
            "alignItems": "stretch",
        },
        title="Aggregated metrics from all recorded episodes — run python main.py --live to generate",
    )


def make_layout():
    C = C_INIT
    traces = get_available_traces()
    trace_options = [{"label": t.name, "value": str(t)} for t in traces]
    no_traces = len(traces) == 0

    return html.Div([
        # ── Hidden stores
        dcc.Store(id="episode-data-store", data={}),
        dcc.Store(id="autoplay-store", data=False),
        dcc.Interval(id="autoplay-interval", interval=1200, disabled=True),
        dcc.Download(id="export-download"),

        # ══════════ TOP BAR ══════════
        html.Div([
            # Logo
            html.Div([
                *[html.Span(ch, style={
                    "color": C["red"] if i % 2 == 0 else C["text"],
                    "fontFamily": "'JetBrains Mono', monospace",
                    "fontSize": "22px",
                    "fontWeight": "700",
                    "letterSpacing": "0.3em",
                }) for i, ch in enumerate("CIPHER")],
            ], style={"display": "flex", "alignItems": "center", "gap": "2px"}),

            # Episode selector
            html.Div([
                html.Span("EPISODE TRACE", style={
                    "fontSize": "9px", "color": C["text_dim"],
                    "letterSpacing": "0.15em",
                    "fontFamily": "'JetBrains Mono', monospace",
                }),
                dcc.Dropdown(
                    id="episode-dropdown",
                    className="dashboard-dropdown",
                    options=trace_options,
                    value=str(traces[0]) if traces else None,
                    placeholder="No traces found — run python main.py",
                    clearable=False,
                    style={
                        "width": "clamp(180px, 32vw, 320px)",
                        "fontSize": "12px",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "backgroundColor": "#f8fafc",
                        "color": "#0f172a",
                        "border": "1px solid #94a3b8",
                        "borderRadius": "6px",
                    },
                ),
            ], style={"display": "flex", "flexDirection": "column", "gap": "2px", "flex": "1 1 260px"}),

            # Right controls
            html.Div([
                # LLM mode badge
                html.Div(
                    id="mode-badge",
                    style={
                        "fontSize": "10px",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "color": C["yellow"],
                        "border": f"1px solid {C['yellow']}",
                        "borderRadius": "4px",
                        "padding": "4px 10px",
                    },
                ),
                # Trace count
                html.Div(
                    f"[ {len(traces)} TRACE{'S' if len(traces) != 1 else ''} ]",
                    style={
                        "fontSize": "10px",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "color": C["text_dim"],
                        "border": f"1px solid {C['border']}",
                        "borderRadius": "4px",
                        "padding": "4px 10px",
                    },
                ),
            ], style={"display": "flex", "gap": "10px", "alignItems": "center", "flexWrap": "wrap", "justifyContent": "flex-end"}),
        ], id="top-bar", style={
            "background": C["surface"],
            "borderBottom": f"1px solid {C['border']}",
            "padding": "12px 20px",
            "display": "flex",
            "alignItems": "center",
            "gap": "30px",
            "justifyContent": "space-between",
            "flexWrap": "wrap",
            "rowGap": "8px",
        }),

        # ══════════ WINNING METRICS BANNER ══════════
        _make_winning_metrics_bar(C),

        # ── Visual Legend (judge-scannable in 10 seconds) ────────────
        html.Div([
            # One-line mission summary
            html.Div([
                html.Span("CIPHER: ", style={"color": "#ff4444", "fontWeight": "800",
                          "fontFamily": "'JetBrains Mono', monospace", "fontSize": "11px"}),
                html.Span(
                    "8 LLM agents play cyber-warfare. RED (4 agents) tries to steal a classified file. "
                    "BLUE (4 agents) defends with honeypots & traps. Oversight AI judges both teams.",
                    style={"fontSize": "11px", "color": C["text_dim"]},
                ),
            ], style={"marginBottom": "8px"}),
            # Icon legend grid
            html.Div([
                html.Div([
                    html.Span("🔴 ", style={"fontSize": "14px"}),
                    html.Span("RED PATH ", style={"color": "#ff4444", "fontWeight": "700"}),
                    html.Span("→ route RED took (Zone 0→1→2→3)", style={"color": C["text_dim"]}),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
                html.Div([
                    html.Span("🔵 ", style={"fontSize": "14px"}),
                    html.Span("BLUE CONFIDENCE ", style={"color": "#4488ff", "fontWeight": "700"}),
                    html.Span("detection level (0–100%)", style={"color": C["text_dim"]}),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
                html.Div([
                    html.Span("📦 ", style={"fontSize": "14px"}),
                    html.Span("DEAD DROP ", style={"color": "#fbbf24", "fontWeight": "700"}),
                    html.Span("encrypted memory node left by RED", style={"color": C["text_dim"]}),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
                html.Div([
                    html.Span("⚡ ", style={"fontSize": "14px"}),
                    html.Span("TRAP FIRED ", style={"color": "#a855f7", "fontWeight": "700"}),
                    html.Span("honeypot triggered / deception activated", style={"color": C["text_dim"]}),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
                html.Div([
                    html.Span("★ ", style={"fontSize": "14px", "color": "#ff4444"}),
                    html.Span("RED POSITION ", style={"color": "#ff4444", "fontWeight": "700"}),
                    html.Span("current node on network map", style={"color": C["text_dim"]}),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
                html.Div([
                    html.Span("◎ ", style={"fontSize": "14px", "color": "#fbbf24"}),
                    html.Span("HONEYPOT ", style={"color": "#fbbf24", "fontWeight": "700"}),
                    html.Span("disguised as FILE_SERVER — springs a trap", style={"color": C["text_dim"]}),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
                html.Div([
                    html.Span("📊 ", style={"fontSize": "14px"}),
                    html.Span("SUSPICION > 80% ", style={"color": "#ff4444", "fontWeight": "700"}),
                    html.Span("RED likely aborts", style={"color": C["text_dim"]}),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
                html.Div([
                    html.Span("⚖ ", style={"fontSize": "14px"}),
                    html.Span("OVERSIGHT VERDICT ", style={"color": "#22c55e", "fontWeight": "700"}),
                    html.Span("AI judge: red_dominates / blue_dominates / contested", style={"color": C["text_dim"]}),
                ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
                ], style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                "gap": "4px 16px",
                "fontSize": "10px",
                "fontFamily": "'JetBrains Mono', monospace",
                "lineHeight": "1.8",
            }),
            html.Div(
                "▶ QUICK START — click PLAY to scrub the replay",
                className="cipher-quick-start-badge",
                style={
                    "marginTop": "8px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "fontSize": "10px",
                    "fontWeight": "700",
                    "letterSpacing": "0.12em",
                    "color": C["green"],
                    "border": f"1px solid {C['green']}",
                    "borderRadius": "4px",
                    "padding": "6px 12px",
                    "width": "fit-content",
                },
            ),
        ], style={
            "padding": "8px 14px",
            "borderBottom": f"1px solid {C['border']}",
            "background": C["bg"],
        }),

        # ══════════ NO TRACES WARNING ══════════
        html.Div([
            html.Div([
                html.Div("⚠ NO EPISODE TRACES FOUND", style={
                    "color": C["yellow"], "fontSize": "16px",
                    "fontFamily": "'JetBrains Mono', monospace",
                    "marginBottom": "8px",
                }),
                html.Code(
                    "python main.py",
                    style={
                        "background": C["surface2"],
                        "color": C["green"],
                        "padding": "8px 16px",
                        "borderRadius": "4px",
                        "fontSize": "13px",
                        "fontFamily": "'JetBrains Mono', monospace",
                    },
                ),
                html.Div("Run the above command to generate episode traces.",
                         style={"color": C["text_dim"], "fontSize": "12px", "marginTop": "8px"}),
            ], style={
                "background": C["surface"],
                "border": f"1px solid {C['yellow']}",
                "borderRadius": "8px",
                "padding": "24px 32px",
                "textAlign": "center",
            }),
        ], style={
            "display": "flex" if no_traces else "none",
            "justifyContent": "center",
            "alignItems": "center",
            "height": "60vh",
        }, id="no-traces-warning"),

        # ══════════ MAIN GRID ══════════
        html.Div([

            # ── LEFT COLUMN (Episode Summary + Rewards + Dead Drops)
            html.Div([

                # Episode Summary
                panel("EPISODE SUMMARY", [
                    html.Div(id="summary-outcome", style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "15px",
                        "fontWeight": "700",
                        "textAlign": "center",
                        "padding": "9px 12px",
                        "border": f"1px solid {C['border']}",
                        "borderRadius": "10px",
                        "marginBottom": "10px",
                        "letterSpacing": "0.12em",
                        "textTransform": "uppercase",
                        "background": "linear-gradient(180deg, rgba(24,30,44,0.88), rgba(16,20,30,0.94))",
                        "boxShadow": "inset 0 1px 0 rgba(255,255,255,0.05), 0 6px 14px rgba(0,0,0,0.26)",
                    }),
                    html.Div(id="summary-stats", style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "8px",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "color": C["text"],
                    }),
                ], C, {"minHeight": "180px"}),

                # Reward Components
                panel("REWARD COMPONENTS", [
                    dcc.Graph(
                        id="reward-chart",
                        config={"displayModeBar": False},
                        style={"height": "190px"},
                    ),
                    html.Div(id="reward-totals", style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "10px",
                        "display": "flex",
                        "justifyContent": "space-around",
                        "marginTop": "0px",
                        "paddingTop": "2px",
                    }),
                ], C),

                # Difficulty Parameters (5.md)
                panel("DIFFICULTY PARAMETERS", [
                    html.Div(id="difficulty-panel", style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "10px",
                        "lineHeight": "1.7",
                        "color": C["text"],
                    }),
                ], C),

                # Dead Drop Inspector
                panel("DEAD DROP INSPECTOR 📦", [
                    html.Div(id="dead-drop-table", style={
                        "overflowY": "auto",
                        "maxHeight": "220px",
                        "fontSize": "10px",
                        "fontFamily": "'JetBrains Mono', monospace",
                    }),
                ], C),

            ], style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "10px",
                "width": "280px",
                "flexShrink": "0",
            }),

            # ── CENTER COLUMN (Timeline + Network + Slider)
            html.Div([

                # Suspicion Timeline
                panel("SUSPICION & DETECTION TIMELINE", [
                    dcc.Graph(
                        id="timeline-chart",
                        config={"displayModeBar": False},
                        style={"height": "170px"},
                    ),
                ], C),

                # Network Map
                panel("NETWORK MAP", [
                    dcc.Graph(
                        id="network-graph",
                        config={"displayModeBar": False, "scrollZoom": True},
                        style={"height": "420px"},
                    ),
                ], C, {"flex": "1"}),

                # Slider + Autoplay controls
                html.Div([
                    html.Button(
                        "⏮", id="btn-first",
                        title="Jump to start",
                        className="replay-ctrl-btn replay-ctrl-btn--icon",
                        style=_ctrl_btn_style(C),
                    ),
                    html.Button(
                        "⏪", id="btn-prev",
                        title="Step back",
                        className="replay-ctrl-btn replay-ctrl-btn--icon",
                        style=_ctrl_btn_style(C),
                    ),
                    html.Button(
                        "▶ PLAY", id="btn-autoplay",
                        className="replay-ctrl-btn replay-ctrl-btn--play",
                        style={**_ctrl_btn_style(C), "color": C["green"],
                               "border": f"1px solid {C['green']}", "minWidth": "96px"},
                    ),
                    html.Button(
                        "⏩", id="btn-next",
                        title="Step forward",
                        className="replay-ctrl-btn replay-ctrl-btn--icon",
                        style=_ctrl_btn_style(C),
                    ),
                    html.Button(
                        "⏭", id="btn-last",
                        title="Jump to end",
                        className="replay-ctrl-btn replay-ctrl-btn--icon",
                        style=_ctrl_btn_style(C),
                    ),
                    html.Button(
                        "↺ RESET", id="btn-jump-reset",
                        title="Jump to next context reset",
                        className="replay-ctrl-btn replay-ctrl-btn--reset",
                        style={**_ctrl_btn_style(C), "color": C["purple"], "border": f"1px solid {C['purple']}", "minWidth": "96px"},
                    ),
                    html.Button(
                        "⚡ TRAP", id="btn-jump-trap",
                        title="Jump to next trap event",
                        className="replay-ctrl-btn replay-ctrl-btn--trap",
                        style={**_ctrl_btn_style(C), "color": C["yellow"], "border": f"1px solid {C['yellow']}", "minWidth": "88px"},
                    ),
                    html.Div(
                        dcc.Slider(
                            id="step-slider",
                            min=0, max=1, step=1, value=0,
                            marks=None,
                            tooltip={"always_visible": False, "placement": "top"},
                            updatemode="drag",
                            className="slider-container",
                        ),
                        className="replay-slider-wrap",
                        style={"flex": "1", "minWidth": "260px"},
                    ),
                    html.Div(
                        "STEP 0 of 0",
                        id="step-label",
                        className="replay-step-label",
                        style={
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "12px",
                            "fontWeight": "700",
                            "color": "#f8fafc",
                            "whiteSpace": "nowrap",
                            "minWidth": "145px",
                            "textAlign": "right",
                            "background": "#0f172a",
                            "border": "1px solid #475569",
                            "borderRadius": "4px",
                            "padding": "7px 10px",
                        },
                    ),
                    # Speed selector
                    dcc.Dropdown(
                        id="speed-dropdown",
                        className="dashboard-dropdown",
                        options=[
                            {"label": "Fast (0.5s)", "value": 500},
                            {"label": "Normal (1.0s)", "value": 1000},
                            {"label": "Slow (1.5s)", "value": 1500},
                            {"label": "Very Slow (3.0s)", "value": 3000},
                        ],
                        value=1200,
                        clearable=False,
                        searchable=False,
                        style={
                            "width": "200px",
                            "minWidth": "200px",
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "fontFamily": "'JetBrains Mono', monospace",
                            "backgroundColor": "#0b1220",
                            "color": "#dbe6ff",
                            "border": "1px solid #32445f",
                            "borderRadius": "6px",
                            "flexShrink": "0",
                        },
                    ),
                    html.Button(
                        "📤 Export HTML Report", id="btn-export-html",
                        title="Export replay as shareable standalone HTML file",
                        className="replay-ctrl-btn replay-ctrl-btn--export",
                        style={**_ctrl_btn_style(C), "fontSize": "10px",
                               "color": C["green"], "border": f"1px solid {C['green']}",
                               "fontWeight": "700", "minWidth": "168px"},
                    ),
                    html.Div(
                        id="export-status",
                        className="replay-export-status",
                        style={
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "9px",
                            "color": C["text_dim"],
                            "minWidth": "120px",
                        },
                    ),
                ], style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "8px",
                    "padding": "8px 12px",
                    "background": C["surface"],
                    "border": f"1px solid {C['border']}",
                    "borderRadius": "8px",
                }, className="replay-controls-shell"),

            ], style={
                "display": "flex",
                "flexDirection": "column",
                "gap": "10px",
                "flex": "1",
                "minWidth": "0",
            }),

            # ── RIGHT COLUMN (Action Log)
            html.Div([
                panel("EPISODE ACTION LOG", [
                    # Filter bar
                    html.Div([
                        dcc.Dropdown(
                            id="log-filter-team",
                            className="dashboard-dropdown",
                            options=[
                                {"label": "ALL", "value": "ALL"},
                                {"label": "RED", "value": "RED"},
                                {"label": "BLUE", "value": "BLUE"},
                                {"label": "EVENTS", "value": "EVENT"},
                            ],
                            value="ALL",
                            clearable=False,
                            style={
                                "width": "90px",
                                "fontSize": "10px",
                                "fontFamily": "'JetBrains Mono', monospace",
                                "backgroundColor": "#f8fafc",
                                "color": "#0f172a",
                                "border": "1px solid #94a3b8",
                                "borderRadius": "4px",
                            },
                        ),
                    ], style={"marginBottom": "6px"}),
                    # Log content
                    html.Div(
                        id="action-log",
                        className="action-log-scroll",
                        style={
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "10px",
                            "lineHeight": "1.55",
                            "flex": "1",
                            "background": C["bg"],
                            "borderRadius": "6px",
                            "padding": "8px",
                            "minHeight": "500px",
                        },
                    ),
                ], C, {"flex": "1", "display": "flex", "flexDirection": "column"}),
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "width": "300px",
                "flexShrink": "0",
            }),

        ], id="main-grid", style={
            "display": "flex" if not no_traces else "none",
            "gap": "12px",
            "padding": "12px",
            "alignItems": "flex-start",
            "height": "auto",
            "minHeight": "calc(100vh - 120px)",
            "overflowX": "auto",
            "overflowY": "auto",
        }),

    ], id="root", style={
        "background": C["bg"],
        "minHeight": "100vh",
        "fontFamily": "'Inter', sans-serif",
        "color": C["text"],
    })




def _make_lore_layout():
    """Build the Lore tab - AI-powered narrative post-mortems."""
    try:
        from cipher.utils.storyteller import load_reports
        reports = load_reports()
    except Exception:
        reports = []

    if not reports:
        cards = [html.Div([
            html.Div('NO REPORTS YET', style={
                'color': DARK['yellow'], 'fontSize': '18px',
                'fontFamily': "'JetBrains Mono', monospace",
                'marginBottom': '12px', 'letterSpacing': '0.15em',
            }),
            html.Div('Reports auto-generate after each episode.', style={
                'color': DARK['text_dim'], 'fontSize': '12px', 'marginBottom': '8px',
            }),
        ], style={'textAlign': 'center', 'padding': '60px 0'})]
    else:
        cards = []
        for rep in reports:
            outcome_color = DARK['red'] if 'exfil' in rep['text'].lower() else (
                DARK['blue'] if 'detect' in rep['text'].lower() else DARK['yellow']
            )
            lines = rep['text'].split('\n')
            body_lines = [l for l in lines if not l.startswith('#') and l.strip() != '---']
            body = '\n'.join(body_lines).strip()
            ep_num = rep['episode']
            fname = rep['filename']
            cards.append(html.Div([
                html.Div(f'EPISODE {ep_num:03d} AFTER-ACTION REPORT', style={
                    'color': DARK['yellow'], 'fontSize': '11px',
                    'fontFamily': "'JetBrains Mono', monospace",
                    'fontWeight': '700', 'letterSpacing': '0.12em', 'marginBottom': '6px',
                }),
                html.Div(fname, style={
                    'color': DARK['text_dim'], 'fontSize': '9px',
                    'fontFamily': "'JetBrains Mono', monospace", 'marginBottom': '10px',
                }),
                html.Div(body, style={
                    'fontFamily': "'Inter', sans-serif", 'fontSize': '13px',
                    'lineHeight': '1.75', 'color': DARK['text'], 'whiteSpace': 'pre-wrap',
                }),
            ], style={
                'background': 'linear-gradient(135deg, #0d1117 0%, #111827 100%)',
                'border': '1px solid rgba(251,191,36,0.3)',
                'borderLeft': f'3px solid {outcome_color}',
                'borderRadius': '8px', 'padding': '16px 20px', 'marginBottom': '12px',
            }))

    n_reports = len(reports)
    return html.Div([
        html.Div([
            html.Div('[ THE DAILY BREACH ]', style={
                'fontFamily': "'JetBrains Mono', monospace", 'fontSize': '20px',
                'fontWeight': '900', 'color': DARK['yellow'], 'letterSpacing': '0.2em',
                'textShadow': '0 0 10px rgba(251,191,36,0.5)', 'marginBottom': '4px',
            }),
            html.Div(
                f'AI-powered cyber-warfare post-mortems  -  {n_reports} report(s)',
                style={'color': DARK['text_dim'], 'fontSize': '11px',
                       'fontFamily': "'JetBrains Mono', monospace"},
            ),
        ], style={'padding': '20px 24px', 'borderBottom': f"1px solid {DARK['border']}",
                  'background': DARK['surface']}),
        html.Div(cards, style={'padding': '20px 24px', 'maxWidth': '900px', 'margin': '0 auto'}),
        html.Div([
            html.Span('Reload the dashboard to see new reports after running episodes.',
                      style={'color': DARK['text_dim'], 'fontSize': '10px',
                             'fontFamily': "'JetBrains Mono', monospace"}),
        ], style={'padding': '12px 24px', 'borderTop': f"1px solid {DARK['border']}",
                  'textAlign': 'center'}),
    ], style={'background': DARK['bg'], 'minHeight': '80vh'})

def make_unified_layout():
    return html.Div(
        [
            # ── Top navigation bar with CIPHER description ───────────────
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("CIPHER", style={
                                "color": "#ff4444", "fontWeight": "800",
                                "fontSize": "16px", "letterSpacing": "3px",
                            }),
                            html.Span(" — Adversarial Multi-Agent RL", style={
                                "color": "#64748b", "fontSize": "11px", "marginLeft": "8px",
                            }),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(
                        "🔴 RED team infiltrates a 50-node enterprise network to steal a classified file  "
                        "|  🔵 BLUE team defends with traps & forensics  "
                        "|  ⚖ Oversight AI judges both teams",
                        style={
                            "color": "#475569", "fontSize": "10px",
                            "flex": "1", "textAlign": "center",
                        },
                    ),
                    html.Div(
                        [
                            html.Span("View: ", style={"fontSize": "11px", "color": "#94a3b8",
                                                        "marginRight": "4px"}),
                            dcc.RadioItems(
                                id="dashboard-mode",
                                options=[
                                    {"label": " Episode Replay", "value": "replay"},
                                    {"label": " Live Training", "value": "live"},
                                    {"label": " 📰 Lore", "value": "lore"},
                                ],
                                value="replay",
                                inline=True,
                                labelStyle={"marginRight": "16px", "fontSize": "12px"},
                                inputStyle={"marginRight": "4px"},
                                style={"color": "#e2e8f0",
                                       "fontFamily": "'JetBrains Mono', monospace"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                ],
                style={
                    "display": "flex", "alignItems": "center", "justifyContent": "space-between",
                    "gap": "12px", "padding": "8px 16px",
                    "background": "#020617", "borderBottom": "1px solid #1e293b",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(id="replay-container", children=make_layout()),
            html.Div(id="live-container", children=create_live_layout(),
                     style={"display": "none"}),
            # ── LORE TAB ───────────────────────────────────────────────
            html.Div(id="lore-container", children=_make_lore_layout(), style={"display": "none"}),
        ]
    )


def _ctrl_btn_style(C):
    return {
        "fontSize": "12px",
        "fontFamily": "'JetBrains Mono', monospace",
        "background": C["surface2"],
        "color": C["text"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "6px",
        "padding": "0 10px",
        "minHeight": "34px",
        "lineHeight": "1",
        "display": "inline-flex",
        "alignItems": "center",
        "justifyContent": "center",
        "fontWeight": "700",
        "letterSpacing": "0.04em",
        "cursor": "pointer",
        "whiteSpace": "nowrap",
        "transition": "all 0.16s ease",
    }


app.layout = make_unified_layout

# Register live-tab callbacks exactly once on this app instance.
register_live_callbacks(app)


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────


@app.callback(
    Output("replay-container", "style"),
    Output("live-container", "style"),
    Output("lore-container", "style"),
    Input("dashboard-mode", "value"),
)
def toggle_dashboard_mode(mode):
    hidden = {"display": "none"}
    visible = {"display": "block"}
    if mode == "live":
        return hidden, visible, hidden
    if mode == "lore":
        return hidden, hidden, visible
    return visible, hidden, hidden


@app.callback(
    Output("episode-data-store", "data"),
    Input("episode-dropdown", "value"),
    prevent_initial_call=False,
)
def load_episode_data(path):
    if not path:
        return {}
    try:
        return load_episode(path)
    except Exception as e:
        return {"_error": str(e)}


@app.callback(
    Output("mode-badge", "children"),
    Output("mode-badge", "style"),
    Input("episode-data-store", "data"),
    Input("episode-dropdown", "value"),
)
def update_mode_badge(data, path):
    mode = infer_runtime_mode(data if isinstance(data, dict) else {})
    if mode == "UNKNOWN" and path:
        path_str = str(path).lower()
        if "_live.json" in path_str:
            mode = "LIVE"
        elif "_hybrid.json" in path_str:
            mode = "HYBRID"
        elif "_stub.json" in path_str:
            mode = "STUB"

    is_live = mode in ("LIVE", "HYBRID")
    color = DARK["green"] if is_live else DARK["yellow"]
    style = {
        "fontSize": "10px",
        "fontFamily": "'JetBrains Mono', monospace",
        "color": color,
        "border": f"1px solid {color}",
        "borderRadius": "4px",
        "padding": "4px 10px",
    }
    return f"♦ {mode}", style


@app.callback(
    Output("step-slider", "max"),
    Output("step-slider", "marks"),
    Input("episode-data-store", "data"),
)
def update_slider_range(data):
    if not data or "_error" in data:
        return 1, {}
    log = data.get("episode_log", [])
    steps = [e.get("step", e.get("t", 0)) for e in log]
    max_step = int(data.get("step", 0) or (max(steps) if steps else 1))
    # Marks every 10 steps
    marks = {i: {"label": str(i), "style": {"fontSize": "8px"}}
             for i in range(0, max_step + 1, max(10, max_step // 10))}
    return max_step, marks


# ── Autoplay: toggle interval
@app.callback(
    Output("autoplay-interval", "disabled"),
    Output("autoplay-interval", "interval"),
    Output("btn-autoplay", "children"),
    Output("autoplay-store", "data"),
    Input("btn-autoplay", "n_clicks"),
    Input("speed-dropdown", "value"),
    State("autoplay-store", "data"),
    prevent_initial_call=True,
)
def toggle_autoplay(n_clicks, speed, is_playing):
    ctx = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    if trigger == "speed-dropdown":
        return not is_playing, speed or 1200, ("⏸ PAUSE" if is_playing else "▶ PLAY"), is_playing
    # Toggle
    new_state = not is_playing
    return not new_state, speed or 1200, ("⏸ PAUSE" if new_state else "▶ PLAY"), new_state


# ── Autoplay: advance step
@app.callback(
    Output("step-slider", "value"),
    Input("autoplay-interval", "n_intervals"),
    Input("btn-first", "n_clicks"),
    Input("btn-prev", "n_clicks"),
    Input("btn-next", "n_clicks"),
    Input("btn-last", "n_clicks"),
    Input("btn-jump-reset", "n_clicks"),
    Input("btn-jump-trap", "n_clicks"),
    State("step-slider", "value"),
    State("step-slider", "max"),
    State("episode-data-store", "data"),
    prevent_initial_call=True,
)
def advance_step(
    n_int,
    btn_first,
    btn_prev,
    btn_next,
    btn_last,
    btn_jump_reset,
    btn_jump_trap,
    current,
    max_val,
    data,
):
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    v = current or 0
    m = max_val or 1
    if trigger == "autoplay-interval":
        return (v + 1) if v < m else 0
    if trigger == "btn-first":
        return 0
    if trigger == "btn-prev":
        return max(0, v - 1)
    if trigger == "btn-next":
        return min(m, v + 1)
    if trigger == "btn-last":
        return m
    if trigger == "btn-jump-reset":
        reset_steps = sorted({int(s) for s in extract_context_resets(data or {}) if isinstance(s, int) and s > 0})
        for s in reset_steps:
            if s > v:
                return min(m, s)
        if reset_steps:
            return min(m, reset_steps[0])
        return no_update
    if trigger == "btn-jump-trap":
        trap_steps = extract_trap_steps(data or {})
        for s in trap_steps:
            if s > v:
                return min(m, s)
        if trap_steps:
            return min(m, trap_steps[0])
        return no_update
    return no_update


# ── Step label
@app.callback(
    Output("step-label", "children"),
    Input("step-slider", "value"),
    Input("step-slider", "max"),
)
def update_step_label(step, max_step):
    return f"STEP {step or 0:03d} of {max_step or 0:03d}"


@app.callback(
    Output("export-status", "children"),
    Output("export-download", "data"),
    Input("btn-export-html", "n_clicks"),
    State("episode-dropdown", "value"),
    State("episode-data-store", "data"),
    State("step-slider", "value"),
    prevent_initial_call=True,
)
def export_current_replay(n_clicks, trace_path, data, step):
    if not trace_path or not data or "_error" in data:
        return "No episode to export.", no_update
    try:
        net_fig = build_network_figure(data, step or 0, DARK)
        tl_fig = build_timeline_figure(data, step or 0, DARK)
        out_path = export_replay_html(
            trace_path=trace_path,
            network_fig=net_fig,
            timeline_fig=tl_fig,
        )
        # Also trigger browser download
        html_content = out_path.read_text(encoding="utf-8")
        filename = out_path.name
        return (
            f"✓ Saved: {out_path.name}",
            dcc.send_string(html_content, filename),
        )
    except Exception as e:
        return f"Export failed: {e}", no_update


# ── Main panels update
@app.callback(
    Output("network-graph", "figure"),
    Output("timeline-chart", "figure"),
    Output("reward-chart", "figure"),
    Output("reward-totals", "children"),
    Output("summary-outcome", "children"),
    Output("summary-outcome", "style"),
    Output("summary-stats", "children"),
    Output("difficulty-panel", "children"),
    Output("dead-drop-table", "children"),
    Output("action-log", "children"),
    Input("step-slider", "value"),
    Input("episode-data-store", "data"),
    Input("log-filter-team", "value"),
)
def update_all_panels(step, data, log_filter):
    C = DARK
    step = step or 0

    if not data or "_error" in data:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            template=C["plotly_tmpl"],
            paper_bgcolor=C["plot_bg"],
            plot_bgcolor=C["plot_bg"],
            annotations=[dict(
                text="No episode loaded" if not data else f"Error: {data.get('_error')}",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color=C["text_dim"]),
            )],
        )
        return (
            empty_fig, empty_fig, empty_fig,
            "", "NO EPISODE",
            {"color": C["text_dim"], "fontFamily": "'JetBrains Mono', monospace",
             "fontSize": "15px", "fontWeight": "700", "textAlign": "center",
             "padding": "9px 12px", "border": f"1px solid {C['border']}", "borderRadius": "10px",
             "letterSpacing": "0.12em", "textTransform": "uppercase",
             "background": "linear-gradient(180deg, rgba(24,30,44,0.88), rgba(16,20,30,0.94))"},
            "",
            html.Div("No episode loaded.", style={"color": C["text_dim"]}),
            html.Div("No episode loaded.", style={"color": C["text_dim"]}),
            html.Div("No episode loaded.", style={"color": C["text_dim"]}),
        )

    # Network figure
    net_fig = build_network_figure(data, step, C)

    # Timeline figure
    tl_fig = build_timeline_figure(data, step, C)

    # Rewards
    rewards = extract_rewards(data)
    rwd_fig = build_reward_figure(rewards, C)
    red_total = float(rewards.get("red_total", rewards.get("red", 0.0)))
    blue_total = float(rewards.get("blue_total", rewards.get("blue", 0.0)))
    reward_totals = html.Div([
        html.Span(f"RED: {red_total:+.3f}", style={"color": C["red"]}),
        html.Span(f"BLUE: {blue_total:+.3f}", style={"color": C["blue"]}),
    ], style={"display": "flex", "justifyContent": "space-around",
              "fontFamily": "'JetBrains Mono', monospace", "fontSize": "12px"})

    # Summary
    summary = get_summary(data)
    outcome = summary["outcome"]
    oc = outcome_color(outcome, C)
    outcome_style = {
        "color": oc,
        "fontFamily": "'JetBrains Mono', monospace",
        "fontSize": "15px",
        "fontWeight": "700",
        "textAlign": "center",
        "padding": "9px 12px",
        "border": f"1px solid {oc}",
        "borderRadius": "10px",
        "letterSpacing": "0.12em",
        "textTransform": "uppercase",
        "background": f"linear-gradient(180deg, {oc}20, rgba(16,20,30,0.94))",
        "boxShadow": f"inset 0 0 0 1px {oc}33, 0 8px 18px rgba(0,0,0,0.30)",
    }

    oversight_text = ", ".join(str(o) for o in summary["oversight"]) if summary["oversight"] else "none"
    trap_count = len(extract_trap_steps(data))
    metric_card_style = {
        "padding": "9px 10px",
        "border": f"1px solid {C['border']}",
        "borderRadius": "8px",
        "background": "linear-gradient(180deg, rgba(30,36,52,0.52), rgba(20,24,36,0.72))",
        "display": "flex",
        "alignItems": "baseline",
        "justifyContent": "space-between",
        "gap": "8px",
    }
    metric_label_style = {
        "color": C["text_dim"],
        "fontSize": "9px",
        "letterSpacing": "0.08em",
        "textTransform": "uppercase",
    }
    metric_value_style = {
        "fontSize": "14px",
        "fontWeight": "700",
        "letterSpacing": "0.04em",
    }
    stats = [
        html.Div([
            html.Span("Steps", style=metric_label_style),
            html.Span(f"{summary['steps']} / {summary['max_steps']}", style={**metric_value_style, "color": C["text"]}),
        ], style=metric_card_style),
        html.Div([
            html.Span("Complexity", style=metric_label_style),
            html.Span(f"{summary['complexity']:.2f}x", style={**metric_value_style, "color": C["green"]}),
        ], style=metric_card_style),
        html.Div([
            html.Span("RED Reward", style=metric_label_style),
            html.Span(f"{red_total:+.3f}", style={**metric_value_style, "color": C["red"]}),
        ], style=metric_card_style),
        html.Div([
            html.Span("BLUE Reward", style=metric_label_style),
            html.Span(f"{blue_total:+.3f}", style={**metric_value_style, "color": C["blue"]}),
        ], style=metric_card_style),
        html.Div([
            html.Span("Context Resets", style=metric_label_style),
            html.Span(str(summary["resets"]), style={**metric_value_style, "color": C["purple"]}),
        ], style=metric_card_style),
        html.Div([
            html.Span("Dead Drops", style=metric_label_style),
            html.Span(str(summary["dead_drops"]), style={**metric_value_style, "color": C["yellow"]}),
        ], style=metric_card_style),
        html.Div([
            html.Span("Trap Events", style=metric_label_style),
            html.Span(str(trap_count), style={**metric_value_style, "color": C["purple"]}),
        ], style=metric_card_style),
        html.Div([
            html.Span("Oversight", style=metric_label_style),
            html.Span(
                oversight_text,
                style={
                    "fontSize": "10px",
                    "fontWeight": "600",
                    "letterSpacing": "0.02em",
                    "color": C["yellow"] if oversight_text != "none" else C["green"],
                    "textAlign": "right",
                    "maxWidth": "160px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "whiteSpace": "nowrap",
                },
                title=oversight_text,
            ),
        ], style=metric_card_style),
    ]

    # Dead drop table
    drops = extract_dead_drops(data)
    if drops:
        header = html.Div([
            html.Div("Step", style={"width": "35px", "color": C["text_dim"]}),
            html.Div("Node", style={"width": "35px", "color": C["text_dim"]}),
            html.Div("Tok", style={"width": "35px", "color": C["text_dim"]}),
            html.Div("Eff", style={"width": "35px", "color": C["text_dim"]}),
            html.Div("Integrity", style={"width": "65px", "color": C["text_dim"]}),
            html.Div("Preview", style={"flex": "1", "color": C["text_dim"]}),
        ], style={"display": "flex", "gap": "4px", "borderBottom": f"1px solid {C['border']}",
                  "paddingBottom": "4px", "marginBottom": "4px"})
        rows = [header]
        for d in drops:
            integ = d.get("integrity", "valid")
            integ_str = "✅ Valid" if str(integ).lower() in ("valid", "ok", "intact", "1", "true") else "⚠ Tampered"
            integ_color = C["green"] if "Valid" in integ_str else C["red"]
            rows.append(html.Div([
                html.Div(str(d["step"]), style={"width": "35px"}),
                html.Div(str(d["node"]), style={"width": "35px"}),
                html.Div(str(d.get("tokens", 0)), style={"width": "35px"}),
                html.Div(f"{float(d.get('efficiency', 0)):.2f}", style={"width": "35px"}),
                html.Div(integ_str, style={"width": "65px", "color": integ_color}),
                html.Div(d.get("preview", "")[:40] + "…", style={"flex": "1", "color": C["text_dim"]}),
            ], style={
                "display": "flex", "gap": "4px",
                "padding": "4px 2px",
                "borderBottom": f"1px solid {C['border']}",
                "cursor": "pointer",
            }))
        dd_table = html.Div(rows)
    else:
        dd_table = html.Div(
            "No dead drops written this episode.",
            style={"color": C["text_dim"], "fontStyle": "italic", "padding": "8px 0"},
        )

    # Action log
    actions = extract_actions(data)
    if log_filter != "ALL":
        actions = [a for a in actions if a["team"] == log_filter]

    # Group by step
    by_step = defaultdict(list)
    for a in actions:
        by_step[a["step"]].append(a)

    log_els = []
    for s in sorted(by_step.keys()):
        if isinstance(s, int) and s > step:
            continue  # Only show up to current step
        step_actions = by_step[s]
        step_els = [html.Div([
            html.Span(f"STEP {s:03d}", className="action-log-step-label"),
            html.Span(f"{len(step_actions)} event{'s' if len(step_actions) != 1 else ''}", className="action-log-step-count"),
        ], className="action-log-step-header")]
        for act in step_actions:
            team = act["team"]
            action_str = act["action"]
            agent = act["agent"]
            target = act.get("target", "")
            reasoning = act.get("reasoning", "")

            if team == "RED":
                color = C["red"]
                team_label = "RED"
                action_type = "team"
                prefix = f"{agent} → {action_str}"
                if target:
                    prefix += f" → {target}"
            elif team == "BLUE":
                color = C["blue"]
                team_label = "BLUE"
                action_type = "team"
                prefix = f"{agent} → {action_str}"
                if target:
                    prefix += f" → {target}"
            elif "dead_drop" in action_str.lower() or "dead_drop" in str(target).lower():
                color = C["yellow"]
                team_label = "EVENT"
                action_type = "event dead-drop"
                prefix = f"📦 {action_str}"
            elif "trap" in action_str.lower() or "honeypot" in action_str.lower():
                color = C["purple"]
                team_label = "EVENT"
                action_type = "event trap"
                prefix = f"⚡ {action_str}"
            elif "context_reset" in action_str.lower() or "memento" in action_str.lower():
                color = C["text"]
                team_label = "SYSTEM"
                action_type = "event reset"
                prefix = f"══ MEMENTO RESET ══"
            else:
                color = C["text_dim"]
                team_label = "EVENT"
                action_type = "event"
                prefix = action_str

            row_class = f"action-log-row {action_type} {team.lower() if team in ('RED', 'BLUE') else ''}".strip()
            entry_els = [html.Div([
                html.Span(team_label, className=f"action-log-team-badge {team.lower() if team in ('RED', 'BLUE') else 'event'}"),
                html.Span(prefix, className="action-log-main-text", style={"color": color}),
            ], className=row_class)]
            if reasoning and len(str(reasoning)) > 2:
                entry_els.append(html.Div(
                    str(reasoning)[:120] + ("…" if len(str(reasoning)) > 120 else ""),
                    className="action-log-reasoning",
                ))
            step_els.append(html.Div(entry_els, className="action-log-entry"))
        log_els.append(html.Div(step_els, className="action-log-step-group"))

    if not log_els:
        log_els = [html.Div(
            "No actions at this step yet.",
            className="action-log-empty",
        )]

    # Difficulty panel (5.md)
    dp = data.get("difficulty_params") or {}
    if dp:
        _d_val = float(dp.get("difficulty", 0.3))
        _bar_w = f"{int(_d_val * 100)}%"
        _bar_color = (
            C["green"] if _d_val < 0.4
            else C["yellow"] if _d_val < 0.7
            else C["red"]
        )
        difficulty_content = html.Div([
            html.Div([
                html.Span("Difficulty:      ", style={"color": C["text_dim"]}),
                html.Span(f"{_d_val:.2f}", style={
                    "color": _bar_color, "fontWeight": "700"
                }),
            ]),
            html.Div([
                html.Div(style={
                    "height": "4px", "width": _bar_w,
                    "background": _bar_color, "borderRadius": "2px",
                    "marginBottom": "6px", "transition": "width 0.4s",
                }),
            ]),
            html.Div([
                html.Span("Honeypots:       ", style={"color": C["text_dim"]}),
                html.Span(f"{float(dp.get('honeypot_density', 0.1)):.0%}"),
            ]),
            html.Div([
                html.Span("Graph Size:      ", style={"color": C["text_dim"]}),
                html.Span(f"{dp.get('graph_size', 50)} nodes"),
            ]),
            html.Div([
                html.Span("Target Files:    ", style={"color": C["text_dim"]}),
                html.Span(str(dp.get("target_files", 3))),
            ]),
            html.Div([
                html.Span("BLUE Delay:      ", style={"color": C["text_dim"]}),
                html.Span(f"{dp.get('blue_response_delay', 0)} steps"),
            ]),
            html.Div([
                html.Span("Zone Lockdown:   ", style={"color": C["text_dim"]}),
                html.Span(f"≥{float(dp.get('zone_lockdown_threshold', 0.7)):.2f}"),
            ]),
            html.Div([
                html.Span("BLUE Trap Budget:", style={"color": C["text_dim"]}),
                html.Span(str(dp.get("trap_budget_blue", 5))),
            ]),
        ])
    else:
        difficulty_content = html.Div(
            "No difficulty data for this episode.",
            style={"color": C["text_dim"], "fontStyle": "italic"},
        )

    return (
        net_fig, tl_fig, rwd_fig,
        reward_totals,
        outcome, outcome_style,
        stats, difficulty_content, dd_table,
        html.Div(log_els, className="action-log-list"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CipherDashboard wrapper class (for verification scripts)
# ─────────────────────────────────────────────────────────────────────────────

class CipherDashboard:
    def __init__(self, config=None):
        self.config = config

    def get_available_traces(self):
        return [str(t) for t in get_available_traces()]

    def load_episode(self, path: str) -> dict:
        return load_episode(path)

    def build_network_figure(self, data: dict, step: int = 0) -> go.Figure:
        return build_network_figure(data, step, DARK)

    def build_timeline_figure(self, data: dict, reward_history=None) -> go.Figure:
        return build_timeline_figure(data, 0, DARK, reward_history)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Keep replay UI stable in normal runs: disable Dash devtools overlay
    # ("Errors/Callbacks" panel) and reloader side effects.
    app.run(debug=False, host="0.0.0.0", port=8050, use_reloader=False)