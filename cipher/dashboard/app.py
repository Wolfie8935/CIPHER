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
import os
import math
from pathlib import Path
from collections import defaultdict
from urllib.parse import quote

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

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

LIGHT = {
    "bg":           "#f0f4f8",
    "surface":      "#ffffff",
    "surface2":     "#e9eef6",
    "border":       "#cbd5e1",
    "text":         "#0f172a",
    "text_dim":     "#64748b",
    "text_muted":   "#94a3b8",
    "red":          "#dc2626",
    "red_dim":      "#fca5a5",
    "blue":         "#2563eb",
    "blue_dim":     "#bfdbfe",
    "yellow":       "#d97706",
    "purple":       "#7c3aed",
    "green":        "#16a34a",
    "gray":         "#9ca3af",
    "zone0":        "#9ca3af",
    "zone1":        "#3b82f6",
    "zone2":        "#f59e0b",
    "zone3":        "#ef4444",
    "plotly_tmpl":  "plotly_white",
    "plot_bg":      "#ffffff",
    "paper_bg":     "#ffffff",
}

ZONE_LABELS = {0: "Perimeter", 1: "General", 2: "Sensitive", 3: "Critical"}

# ─────────────────────────────────────────────────────────────────────────────
# EPISODE TRACE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

TRACES_DIR = Path("episode_traces")


def get_available_traces():
    TRACES_DIR.mkdir(exist_ok=True)
    return sorted(TRACES_DIR.glob("*.json"), reverse=True)


def load_episode(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_timeline(data: dict) -> tuple[list, list, list]:
    """Return (steps, red_suspicion, blue_confidence) lists."""
    steps, red_vals, blue_vals = [], [], []
    saw_signal = False
    for entry in data.get("episode_log", []):
        step = entry.get("step", entry.get("t", None))
        if step is None:
            continue
        has_red = any(
            k in entry
            for k in ("red_suspicion_score", "red_suspicion", "suspicion_score")
        ) or ("state" in entry and "red_suspicion_score" in entry.get("state", {}))
        has_blue = any(
            k in entry
            for k in ("blue_detection_confidence", "blue_confidence", "detection_confidence")
        ) or ("state" in entry and "blue_detection_confidence" in entry.get("state", {}))
        saw_signal = saw_signal or has_red or has_blue
        red_s = (
            entry.get("red_suspicion_score")
            or entry.get("red_suspicion")
            or entry.get("suspicion_score")
            or entry.get("state", {}).get("red_suspicion_score")
            or 0.0
        )
        blue_c = (
            entry.get("blue_detection_confidence")
            or entry.get("blue_confidence")
            or entry.get("detection_confidence")
            or entry.get("state", {}).get("blue_detection_confidence")
            or 0.0
        )
        steps.append(step)
        red_vals.append(float(red_s))
        blue_vals.append(float(blue_c))
    if steps and saw_signal:
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
    return r


def build_graph_layout(data: dict) -> tuple[nx.Graph, dict]:
    """Build NetworkX graph and node positions from episode data."""
    G = nx.Graph()
    node_meta = {}

    # Try to get graph info from trace
    graph_data = data.get("graph", data.get("network", {}))
    nodes_raw = graph_data.get("nodes", []) if graph_data else []
    edges_raw = graph_data.get("edges", []) if graph_data else []

    if nodes_raw:
        for n in nodes_raw:
            nid = int(n.get("id", n.get("node_id", 0)))
            G.add_node(nid)
            node_meta[nid] = n
        for e in edges_raw:
            try:
                if isinstance(e, (list, tuple)) and len(e) >= 2:
                    a, b = e[0], e[1]
                elif isinstance(e, dict):
                    a = e.get("source", e.get("src", e.get("from", e.get("u"))))
                    b = e.get("target", e.get("dst", e.get("to", e.get("v"))))
                else:
                    continue
                if a is None or b is None:
                    continue
                G.add_edge(int(a), int(b))
            except (TypeError, ValueError):
                continue
    else:
        # Reconstruct a plausible 50-node graph
        # Zone 0: nodes 0-9 (Perimeter), Zone 1: 10-24 (General),
        # Zone 2: 25-39 (Sensitive), Zone 3: 40-49 (Critical)
        for i in range(50):
            G.add_node(i)
            zone = 0 if i < 10 else (1 if i < 25 else (2 if i < 40 else 3))
            node_type = ["workstation", "file_server", "auth_gateway", "db_server"][i % 4]
            node_meta[i] = {
                "id": i,
                "zone": zone,
                "type": node_type,
                "hostname": f"corp-{['ws', 'fs', 'ag', 'db'][i % 4]}-{i:02d}",
                "files": [],
            }
        # Add edges with zone structure
        import random
        rng = random.Random(data.get("seed", data.get("episode_seed", 42)))
        # Intra-zone edges
        zone_groups = defaultdict(list)
        for nid, meta in node_meta.items():
            zone_groups[meta["zone"]].append(nid)
        for zone_nodes in zone_groups.values():
            for i, a in enumerate(zone_nodes):
                for b in zone_nodes[i + 1:]:
                    if rng.random() < 0.35:
                        G.add_edge(a, b)
        # Inter-zone edges (sparse)
        zones = list(zone_groups.values())
        for i in range(len(zones) - 1):
            for _ in range(3):
                a = rng.choice(zones[i])
                b = rng.choice(zones[i + 1])
                G.add_edge(a, b)

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
        path_up_to = red_path[: step + 1] if step < len(red_path) else red_path
    red_current = path_up_to[-1] if path_up_to else None
    red_visited = set(path_up_to)

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

    shapes = []
    annotations = []

    # Detection threshold
    shapes.append(dict(
        type="line", x0=0, x1=max(steps) if steps else 1,
        y0=0.8, y1=0.8,
        line=dict(color=C["red_dim"], width=1.5, dash="dot"),
    ))
    annotations.append(dict(
        x=max(steps) if steps else 1, y=0.82,
        text="⚠ Detection Threshold",
        font=dict(size=9, color=C["red_dim"]),
        showarrow=False, xanchor="right",
    ))

    # MEMENTO lines
    for rs in resets:
        shapes.append(dict(
            type="line", x0=rs, x1=rs, y0=0, y1=1,
            line=dict(color=C["purple"], width=1.5, dash="dash"),
        ))
        annotations.append(dict(
            x=rs, y=1.02, text="MEMENTO",
            font=dict(size=8, color=C["purple"]),
            showarrow=False,
        ))

    # Spike annotations on RED line
    for i in range(1, len(red_vals)):
        delta = red_vals[i] - red_vals[i - 1]
        if delta > 0.1:
            annotations.append(dict(
                x=steps[i], y=red_vals[i],
                text=f"▲{delta:.2f}",
                font=dict(size=8, color=C["red"]),
                showarrow=True, arrowhead=2,
                arrowcolor=C["red"], ay=-25,
            ))

    # Step cursor
    if steps and step < len(steps):
        cur_x = steps[step]
        shapes.append(dict(
            type="line", x0=cur_x, x1=cur_x, y0=0, y1=1,
            line=dict(color=C["yellow"], width=1.5),
        ))

    traces = []
    if steps:
        traces.append(go.Scatter(
            x=steps, y=red_vals,
            name="RED Suspicion", mode="lines+markers",
            line=dict(color=C["red"], width=2),
            marker=dict(size=4),
        ))
        traces.append(go.Scatter(
            x=steps, y=blue_vals,
            name="BLUE Confidence", mode="lines+markers",
            line=dict(color=C["blue"], width=2),
            marker=dict(size=4),
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
        margin=dict(l=40, r=10, t=10, b=30),
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(title="Step", range=[0, max(steps) if steps else 10]),
        yaxis=dict(title="Score", range=[0, 1.1]),
        legend=dict(
            orientation="h", x=0, y=1.1,
            font=dict(size=10),
        ),
        hovermode="x unified",
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
            x=labels, y=vals,
            marker_color=colors,
            name="RED Components",
            hovertemplate="%{x}: %{y:.3f}<extra></extra>",
        ))
    else:
        # Fallback: just show red vs blue totals
        fig.add_trace(go.Bar(
            x=["RED Total", "BLUE Total"],
            y=[red_total, blue_total],
            marker_color=[C["red"], C["blue"]],
        ))

    fig.update_layout(
        template=C["plotly_tmpl"],
        paper_bgcolor=C["plot_bg"],
        plot_bgcolor=C["plot_bg"],
        margin=dict(l=30, r=10, t=10, b=50),
        showlegend=False,
        xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
        yaxis=dict(title="Value", range=[-1.1, 1.1]),
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
    suppress_callback_exceptions=True,
    title="CIPHER · Episode Replay",
)
server = app.server

# ── Global CSS injected via inline style on root div
GLOBAL_CSS = """
  * { box-sizing: border-box; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2a3550; border-radius: 3px; }
  body { margin: 0; padding: 0; transition: background 0.3s; }
  .action-log-scroll { overflow-y: auto; max-height: 320px; }
  .action-log-scroll::-webkit-scrollbar-thumb { background: #2a3550; }
  .slider-container .rc-slider-track { background: #ff4444 !important; }
  .slider-container .rc-slider-handle { border-color: #ff4444 !important; }
"""

# ── Initial dark theme colours (client-side toggle switches between DARK/LIGHT)
C_INIT = DARK


def make_layout():
    C = C_INIT
    traces = get_available_traces()
    trace_options = [{"label": t.name, "value": str(t)} for t in traces]
    no_traces = len(traces) == 0

    return html.Div([
        # ── Hidden stores
        dcc.Store(id="theme-store", data="dark"),
        dcc.Store(id="episode-data-store", data={}),
        dcc.Store(id="autoplay-store", data=False),
        dcc.Interval(id="autoplay-interval", interval=1200, disabled=True),

        # ── Injected CSS
        html.Link(
            rel="stylesheet",
            href=f"data:text/css;charset=utf-8,{quote(GLOBAL_CSS)}",
        ),

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
                    options=trace_options,
                    value=str(traces[0]) if traces else None,
                    placeholder="No traces found — run python main.py",
                    clearable=False,
                    style={
                        "width": "clamp(180px, 32vw, 320px)",
                        "fontSize": "12px",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "backgroundColor": C["surface2"],
                        "color": C["text"],
                        "border": f"1px solid {C['border']}",
                        "borderRadius": "6px",
                    },
                ),
            ], style={"display": "flex", "flexDirection": "column", "gap": "2px", "flex": "1 1 260px"}),

            # Right controls
            html.Div([
                # LLM mode badge
                html.Div(
                    f"● {os.environ.get('LLM_MODE', 'stub').upper()} MODE",
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
                # Theme toggle
                html.Button(
                    "☀ LIGHT", id="theme-toggle",
                    style={
                        "fontSize": "10px",
                        "fontFamily": "'JetBrains Mono', monospace",
                        "background": "transparent",
                        "color": C["text_dim"],
                        "border": f"1px solid {C['border']}",
                        "borderRadius": "4px",
                        "padding": "4px 10px",
                        "cursor": "pointer",
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
                        "fontSize": "18px",
                        "fontWeight": "700",
                        "textAlign": "center",
                        "padding": "8px",
                        "border": f"1px solid {C['border']}",
                        "borderRadius": "6px",
                        "marginBottom": "6px",
                        "letterSpacing": "0.1em",
                    }),
                    html.Div(id="summary-stats", style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "11px",
                        "lineHeight": "1.8",
                        "color": C["text"],
                    }),
                ], C, {"minHeight": "180px"}),

                # Reward Components
                panel("REWARD COMPONENTS", [
                    dcc.Graph(
                        id="reward-chart",
                        config={"displayModeBar": False},
                        style={"height": "160px"},
                    ),
                    html.Div(id="reward-totals", style={
                        "fontFamily": "'JetBrains Mono', monospace",
                        "fontSize": "11px",
                        "display": "flex",
                        "justifyContent": "space-around",
                        "marginTop": "4px",
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
                        style=_ctrl_btn_style(C),
                    ),
                    html.Button(
                        "⏪", id="btn-prev",
                        title="Step back",
                        style=_ctrl_btn_style(C),
                    ),
                    html.Button(
                        "▶ PLAY", id="btn-autoplay",
                        style={**_ctrl_btn_style(C), "color": C["green"],
                               "border": f"1px solid {C['green']}"},
                    ),
                    html.Button(
                        "⏩", id="btn-next",
                        title="Step forward",
                        style=_ctrl_btn_style(C),
                    ),
                    html.Button(
                        "⏭", id="btn-last",
                        title="Jump to end",
                        style=_ctrl_btn_style(C),
                    ),
                    dcc.Slider(
                        id="step-slider",
                        min=0, max=1, step=1, value=0,
                        marks=None,
                        tooltip={"always_visible": True, "placement": "top"},
                        updatemode="drag",
                        className="slider-container",
                    ),
                    html.Div(
                        "STEP 0 of 0",
                        id="step-label",
                        style={
                            "fontFamily": "'JetBrains Mono', monospace",
                            "fontSize": "10px",
                            "color": C["text_dim"],
                            "whiteSpace": "nowrap",
                            "minWidth": "90px",
                            "textAlign": "right",
                        },
                    ),
                    # Speed selector
                    dcc.Dropdown(
                        id="speed-dropdown",
                        options=[
                            {"label": "Fast 0.5s", "value": 500},
                            {"label": "Normal 1s", "value": 1000},
                            {"label": "Slow 1.5s", "value": 1500},
                            {"label": "Very Slow 3s", "value": 3000},
                        ],
                        value=1200,
                        clearable=False,
                        style={
                            "width": "110px",
                            "fontSize": "10px",
                            "fontFamily": "'JetBrains Mono', monospace",
                            "backgroundColor": C["surface2"],
                            "color": C["text"],
                            "border": f"1px solid {C['border']}",
                            "borderRadius": "4px",
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
                }),

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
                                "backgroundColor": C["surface2"],
                                "color": C["text"],
                                "border": f"1px solid {C['border']}",
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
                            "lineHeight": "1.6",
                            "flex": "1",
                            "background": C["bg"],
                            "borderRadius": "4px",
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
            "gap": "10px",
            "padding": "10px",
            "alignItems": "flex-start",
            "height": "calc(100vh - 58px)",
            "overflowX": "auto",
            "overflowY": "hidden",
        }),

    ], id="root", style={
        "background": C["bg"],
        "minHeight": "100vh",
        "fontFamily": "'Inter', sans-serif",
        "color": C["text"],
    })


def _ctrl_btn_style(C):
    return {
        "fontSize": "12px",
        "fontFamily": "'JetBrains Mono', monospace",
        "background": C["surface2"],
        "color": C["text"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "4px",
        "padding": "5px 10px",
        "cursor": "pointer",
        "whiteSpace": "nowrap",
    }


app.layout = make_layout


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────


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
    Output("step-slider", "max"),
    Output("step-slider", "marks"),
    Input("episode-data-store", "data"),
)
def update_slider_range(data):
    if not data or "_error" in data:
        return 1, {}
    log = data.get("episode_log", [])
    steps = [e.get("step", e.get("t", 0)) for e in log]
    max_step = max(steps) if steps else 1
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
    State("step-slider", "value"),
    State("step-slider", "max"),
    prevent_initial_call=True,
)
def advance_step(n_int, btn_first, btn_prev, btn_next, btn_last, current, max_val):
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
    return no_update


# ── Step label
@app.callback(
    Output("step-label", "children"),
    Input("step-slider", "value"),
    Input("step-slider", "max"),
)
def update_step_label(step, max_step):
    return f"STEP {step or 0:03d} of {max_step or 0:03d}"


# ── Main panels update
@app.callback(
    Output("network-graph", "figure"),
    Output("timeline-chart", "figure"),
    Output("reward-chart", "figure"),
    Output("reward-totals", "children"),
    Output("summary-outcome", "children"),
    Output("summary-outcome", "style"),
    Output("summary-stats", "children"),
    Output("dead-drop-table", "children"),
    Output("action-log", "children"),
    Input("step-slider", "value"),
    Input("episode-data-store", "data"),
    Input("theme-store", "data"),
    Input("log-filter-team", "value"),
)
def update_all_panels(step, data, theme, log_filter):
    C = DARK if theme == "dark" else LIGHT
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
             "fontSize": "18px", "fontWeight": "700", "textAlign": "center",
             "padding": "8px", "border": f"1px solid {C['border']}", "borderRadius": "6px"},
            "", html.Div("No episode loaded.", style={"color": C["text_dim"]}),
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
        "fontSize": "18px",
        "fontWeight": "700",
        "textAlign": "center",
        "padding": "8px",
        "border": f"2px solid {oc}",
        "borderRadius": "6px",
        "letterSpacing": "0.1em",
    }

    oversight_text = ", ".join(str(o) for o in summary["oversight"]) if summary["oversight"] else "none"
    stats = [
        html.Div([
            html.Span("Steps:          ", style={"color": C["text_dim"]}),
            html.Span(f"{summary['steps']} / {summary['max_steps']}"),
        ]),
        html.Div([
            html.Span("RED Reward:     ", style={"color": C["text_dim"]}),
            html.Span(f"{red_total:+.3f}", style={"color": C["red"]}),
        ]),
        html.Div([
            html.Span("BLUE Reward:    ", style={"color": C["text_dim"]}),
            html.Span(f"{blue_total:+.3f}", style={"color": C["blue"]}),
        ]),
        html.Br(),
        html.Div([
            html.Span("Context Resets: ", style={"color": C["text_dim"]}),
            html.Span(str(summary["resets"]), style={"color": C["purple"]}),
        ]),
        html.Div([
            html.Span("Dead Drops:     ", style={"color": C["text_dim"]}),
            html.Span(str(summary["dead_drops"]), style={"color": C["yellow"]}),
        ]),
        html.Br(),
        html.Div([
            html.Span("Oversight:      ", style={"color": C["text_dim"]}),
            html.Span(oversight_text, style={"color": C["yellow"] if oversight_text != "none" else C["green"]}),
        ]),
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
        log_els.append(html.Div(
            f"── step {s:03d} ──",
            style={"color": C["text_muted"], "margin": "6px 0 2px 0", "fontSize": "9px"},
        ))
        for act in by_step[s]:
            team = act["team"]
            action_str = act["action"]
            agent = act["agent"]
            target = act.get("target", "")
            reasoning = act.get("reasoning", "")

            if team == "RED":
                color = C["red"]
                prefix = f"[R] {agent} → {action_str}"
                if target:
                    prefix += f" → {target}"
            elif team == "BLUE":
                color = C["blue"]
                prefix = f"[B] {agent} → {action_str}"
                if target:
                    prefix += f" → {target}"
            elif "dead_drop" in action_str.lower() or "dead_drop" in str(target).lower():
                color = C["yellow"]
                prefix = f"📦 {action_str}"
            elif "trap" in action_str.lower() or "honeypot" in action_str.lower():
                color = C["purple"]
                prefix = f"⚡ {action_str}"
            elif "context_reset" in action_str.lower() or "memento" in action_str.lower():
                color = C["text"]
                prefix = f"══ MEMENTO RESET ══"
            else:
                color = C["text_dim"]
                prefix = action_str

            entry_els = [html.Div(prefix, style={"color": color})]
            if reasoning and len(str(reasoning)) > 2:
                entry_els.append(html.Div(
                    str(reasoning)[:120] + ("…" if len(str(reasoning)) > 120 else ""),
                    style={"color": C["text_dim"], "fontStyle": "italic",
                           "fontSize": "9px", "paddingLeft": "12px"},
                ))
            log_els.append(html.Div(entry_els, style={"marginBottom": "2px"}))

    if not log_els:
        log_els = [html.Div(
            "No actions at this step yet.",
            style={"color": C["text_dim"], "fontStyle": "italic"},
        )]

    return (
        net_fig, tl_fig, rwd_fig,
        reward_totals,
        outcome, outcome_style,
        stats, dd_table,
        html.Div(log_els),
    )


# ── Theme toggle
@app.callback(
    Output("theme-store", "data"),
    Output("theme-toggle", "children"),
    Output("root", "style"),
    Output("top-bar", "style"),
    Output("main-grid", "style"),
    Input("theme-toggle", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=True,
)
def toggle_theme(n, current_theme):
    new_theme = "light" if current_theme == "dark" else "dark"
    C = LIGHT if new_theme == "light" else DARK
    btn_label = "🌙 DARK" if new_theme == "light" else "☀ LIGHT"
    root_style = {
        "background": C["bg"],
        "minHeight": "100vh",
        "fontFamily": "'Inter', sans-serif",
        "color": C["text"],
    }
    topbar_style = {
        "background": C["surface"],
        "borderBottom": f"1px solid {C['border']}",
        "padding": "12px 20px",
        "display": "flex",
        "alignItems": "center",
        "gap": "30px",
        "justifyContent": "space-between",
        "flexWrap": "wrap",
        "rowGap": "8px",
    }
    grid_style = {
        "display": "flex",
        "gap": "10px",
        "padding": "10px",
        "alignItems": "flex-start",
        "height": "calc(100vh - 58px)",
        "overflowX": "auto",
        "overflowY": "hidden",
    }
    return new_theme, btn_label, root_style, topbar_style, grid_style


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
    app.run(debug=True, host="0.0.0.0", port=8050)