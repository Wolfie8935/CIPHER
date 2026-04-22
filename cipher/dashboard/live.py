"""
Live training dashboard for CIPHER (Phase 13).

Runs a separate Dash app on port 8051 by default and reads:
- rewards_log.csv
- training_events.jsonl
- training_state.json
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dash_table, dcc, html

from cipher.utils.config import config

DARK_BG = "#0a0a0a"
PANEL_BG = "#111111"
BORDER_COLOR = "#2a2a2a"
RED_COLOR = "#ff4444"
BLUE_COLOR = "#4488ff"
GOLD_COLOR = "#ffaa00"
GREEN_COLOR = "#44cc88"
GRAY_COLOR = "#888888"
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#aaaaaa"
TEXT_MUTED = "#555555"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=PANEL_BG,
    plot_bgcolor=DARK_BG,
    font=dict(color=TEXT_SECONDARY, size=11),
    margin=dict(l=50, r=20, t=30, b=40),
    xaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#333"),
    yaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#333"),
)

app = dash.Dash(
    __name__,
    title="CIPHER - Live Training",
    update_title=None,
)
app.config.suppress_callback_exceptions = True
server = app.server
_REGISTERED_APPS: set[int] = set()


def _build_tab1_layout() -> html.Div:
    return html.Div(
        [
            dcc.Graph(id="t1-main-chart", style={"height": "280px"}),
            dcc.Graph(id="t1-red-breakdown", style={"height": "180px"}),
            dcc.Graph(id="t1-blue-breakdown", style={"height": "180px"}),
            html.Div(id="t1-stats"),
        ]
    )


def _build_tab2_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    dcc.Dropdown(
                        id="t2-filter",
                        options=[
                            {"label": "All drops", "value": "all"},
                            {"label": "RED only", "value": "red"},
                            {"label": "BLUE only", "value": "blue"},
                            {"label": "Tampered only", "value": "tampered"},
                        ],
                        value="all",
                        style={"width": "200px", "background": PANEL_BG},
                    ),
                ],
                style={"marginBottom": "8px"},
            ),
            html.Div(id="t2-stats", style={"marginBottom": "8px"}),
            dash_table.DataTable(
                id="t2-table",
                columns=[
                    {"name": "Ep", "id": "episode"},
                    {"name": "Step", "id": "step"},
                    {"name": "Node", "id": "node"},
                    {"name": "Agent", "id": "team"},
                    {"name": "Tokens", "id": "tokens"},
                    {"name": "Eff", "id": "efficiency"},
                    {"name": "Integrity", "id": "integrity"},
                    {"name": "Preview", "id": "detail"},
                ],
                style_cell={
                    "backgroundColor": PANEL_BG,
                    "color": TEXT_SECONDARY,
                    "border": f"1px solid {BORDER_COLOR}",
                    "fontSize": "11px",
                    "padding": "4px 8px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "maxWidth": "200px",
                },
                style_header={
                    "backgroundColor": "#1a1a1a",
                    "fontWeight": "bold",
                    "color": TEXT_PRIMARY,
                    "fontSize": "11px",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{integrity} = "tampered"'},
                        "color": RED_COLOR,
                    },
                ],
                page_size=20,
                sort_action="native",
            ),
        ]
    )


def _build_tab3_layout() -> html.Div:
    return html.Div(
        [
            dcc.Graph(id="t3-map", style={"height": "480px"}),
            html.Div(id="t3-events", style={"marginTop": "8px"}),
            html.Div(id="t3-stats", style={"marginTop": "8px"}),
        ]
    )


def _build_tab4_layout() -> html.Div:
    return html.Div(
        [
            html.Div(id="t4-stats", style={"marginBottom": "12px"}),
            html.Div(
                "Fleet Verdicts",
                style={"color": TEXT_MUTED, "fontSize": "11px", "marginBottom": "4px"},
            ),
            dash_table.DataTable(
                id="t4-verdicts",
                columns=[
                    {"name": "Ep", "id": "episode"},
                    {"name": "Verdict", "id": "fleet_verdict"},
                    {"name": "RED Bonus", "id": "oversight_red_adj"},
                    {"name": "BLUE Bonus", "id": "oversight_blue_adj"},
                    {"name": "Judgment", "id": "fleet_judgment"},
                ],
                style_cell={
                    "backgroundColor": PANEL_BG,
                    "color": TEXT_SECONDARY,
                    "border": f"1px solid {BORDER_COLOR}",
                    "fontSize": "11px",
                    "padding": "4px 8px",
                    "maxWidth": "300px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_header={
                    "backgroundColor": "#1a1a1a",
                    "color": TEXT_PRIMARY,
                    "fontWeight": "bold",
                    "fontSize": "11px",
                },
                style_data_conditional=[
                    {
                        "if": {"filter_query": '{fleet_verdict} = "red_dominates"'},
                        "color": RED_COLOR,
                    },
                    {
                        "if": {"filter_query": '{fleet_verdict} = "blue_dominates"'},
                        "color": BLUE_COLOR,
                    },
                    {
                        "if": {"filter_query": '{fleet_verdict} = "degenerate"'},
                        "color": GOLD_COLOR,
                    },
                ],
                page_size=15,
                sort_action="native",
            ),
            html.Div(
                "Oversight Flags",
                style={"color": TEXT_MUTED, "fontSize": "11px", "margin": "12px 0 4px"},
            ),
            html.Div(id="t4-flags"),
        ]
    )


def _build_tab5_layout() -> html.Div:
    return html.Div(
        [
            dcc.Graph(id="t5-main", style={"height": "280px"}),
            dcc.Graph(id="t5-scatter", style={"height": "240px"}),
            html.Div(id="t5-stats", style={"marginTop": "8px"}),
        ]
    )


def _build_tab6_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                "Reward Curves + Evolution Events",
                style={"color": TEXT_MUTED, "fontSize": "11px", "marginBottom": "6px"},
            ),
            dcc.Graph(id="t6-reward-chart", style={"height": "280px"}),
            html.Div(
                "Win Rate Curves (10-episode rolling)",
                style={"color": TEXT_MUTED, "fontSize": "11px", "margin": "12px 0 6px"},
            ),
            dcc.Graph(id="t6-winrate-chart", style={"height": "200px"}),
            html.Div(id="t6-stats", style={"marginTop": "12px"}),
        ]
    )


def _load_rewards_csv() -> Optional[pd.DataFrame]:
    try:
        csv_path = Path("rewards_log.csv")
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return None
        frame = pd.read_csv(csv_path)
        if "episode" in frame.columns:
            frame["episode"] = pd.to_numeric(frame["episode"], errors="coerce")
            frame = frame.dropna(subset=["episode"])
            if not frame.empty:
                frame["episode"] = frame["episode"].astype(int)
        for numeric_col in (
            "red_total",
            "blue_total",
            "red_exfil",
            "red_stealth",
            "red_memory",
            "red_complexity",
            "red_abort_penalty",
            "red_honeypot_penalty",
            "blue_detection",
            "blue_speed",
            "blue_fp_penalty",
            "blue_honeypot_rate",
            "blue_graph_reconstruction",
            "oversight_red_adj",
            "oversight_blue_adj",
        ):
            if numeric_col in frame.columns:
                frame[numeric_col] = pd.to_numeric(frame[numeric_col], errors="coerce")
        return frame if not frame.empty else None
    except Exception:
        return None


def _load_training_events() -> list[dict]:
    try:
        events_path = Path("training_events.jsonl")
        if not events_path.exists():
            return []
        text = events_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        rows: list[dict] = []
        for line in text.split("\n"):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows
    except Exception:
        return []


def _load_evolution_log() -> list[dict]:
    try:
        path = Path("prompt_evolution_log.jsonl")
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        rows: list[dict] = []
        for line in text.split("\n"):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows
    except Exception:
        return []


def _load_training_state() -> dict:
    try:
        state_path = Path("training_state.json")
        if not state_path.exists():
            return {
                "status": "idle",
                "current_episode": 0,
                "total_episodes": 0,
                "llm_mode": "unknown",
            }
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "error", "current_episode": 0, "total_episodes": 0}


def _moving_average(values: list, window: int = 10) -> list:
    if not values:
        return []
    arr = np.array(values, dtype=float)
    result = np.full_like(arr, np.nan)
    for idx in range(len(arr)):
        start = max(0, idx - window + 1)
        result[idx] = np.mean(arr[start : idx + 1])
    return result.tolist()


def _stat(label: str, value: str, color: str = TEXT_PRIMARY) -> html.Div:
    return html.Div(
        [
            html.Div(
                label,
                style={
                    "fontSize": "10px",
                    "color": TEXT_MUTED,
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                },
            ),
            html.Div(
                value,
                style={
                    "fontSize": "16px",
                    "fontWeight": "500",
                    "color": color,
                    "marginTop": "2px",
                },
            ),
        ],
        style={
            "background": PANEL_BG,
            "padding": "8px 12px",
            "borderRadius": "4px",
            "border": f"1px solid {BORDER_COLOR}",
            "minWidth": "100px",
        },
    )


_GRAPH_CACHE: dict[str, object] = {}


def _get_graph_positions():
    if "pos" not in _GRAPH_CACHE:
        try:
            import networkx as nx

            from cipher.environment.graph import generate_enterprise_graph

            graph = generate_enterprise_graph(
                n_nodes=config.env_graph_size,
                honeypot_density=config.env_honeypot_density,
                seed=7961,
            )
            pos = nx.spring_layout(graph, seed=42)
            _GRAPH_CACHE["graph"] = graph
            _GRAPH_CACHE["pos"] = pos
        except Exception:
            return None, None
    return _GRAPH_CACHE.get("graph"), _GRAPH_CACHE.get("pos")


def create_live_layout() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "CIPHER",
                        style={
                            "color": RED_COLOR,
                            "fontWeight": "700",
                            "fontSize": "18px",
                            "letterSpacing": "4px",
                        },
                    ),
                    html.Span(
                        " - LIVE",
                        id="live-indicator",
                        style={"color": GREEN_COLOR, "fontSize": "11px"},
                    ),
                    html.Span(
                        id="header-episode",
                        style={"color": TEXT_SECONDARY, "fontSize": "12px"},
                    ),
                    html.Span(id="header-status", style={"fontSize": "12px"}),
                    html.Span(id="header-red-avg", style={"color": RED_COLOR, "fontSize": "12px"}),
                    html.Span(
                        id="header-blue-avg",
                        style={"color": BLUE_COLOR, "fontSize": "12px"},
                    ),
                    html.Span(id="header-uptime", style={"color": TEXT_MUTED, "fontSize": "11px"}),
                ],
                style={
                    "display": "flex",
                    "gap": "20px",
                    "alignItems": "center",
                    "padding": "10px 20px",
                    "background": "#080808",
                    "borderBottom": f"1px solid {BORDER_COLOR}",
                },
            ),
            dcc.Tabs(
                id="main-tabs",
                value="tab-rewards",
                children=[
                    dcc.Tab(label="Reward Curves", value="tab-rewards"),
                    dcc.Tab(label="Dead Drops", value="tab-drops"),
                    dcc.Tab(label="Deception Map", value="tab-deception"),
                    dcc.Tab(label="Oversight Feed", value="tab-oversight"),
                    dcc.Tab(label="Difficulty Curve", value="tab-difficulty"),
                    dcc.Tab(label="Learning Curve", value="tab-evolution"),
                ],
                style={"fontFamily": "monospace", "fontSize": "12px"},
                colors={"border": BORDER_COLOR, "primary": RED_COLOR, "background": PANEL_BG},
            ),
            html.Div(
                id="tab-content",
                style={"padding": "16px", "background": DARK_BG, "minHeight": "600px"},
            ),
            dcc.Interval(
                id="interval-component",
                interval=config.dashboard_live_update_interval,
                n_intervals=0,
            ),
        ],
        style={
            "background": DARK_BG,
            "minHeight": "100vh",
            "fontFamily": "monospace",
            "color": TEXT_PRIMARY,
        },
    )


def render_tab(tab):
    if tab == "tab-rewards":
        return _build_tab1_layout()
    if tab == "tab-drops":
        return _build_tab2_layout()
    if tab == "tab-deception":
        return _build_tab3_layout()
    if tab == "tab-oversight":
        return _build_tab4_layout()
    if tab == "tab-difficulty":
        return _build_tab5_layout()
    if tab == "tab-evolution":
        return _build_tab6_layout()
    return html.Div("Unknown tab")


def update_header(_n):
    state = _load_training_state()
    frame = _load_rewards_csv()

    ep_text = f"Ep {state.get('current_episode', 0)} / {state.get('total_episodes', '?')}"
    status = str(state.get("status", "idle")).upper()
    status_color = {
        "RUNNING": GREEN_COLOR,
        "IDLE": GRAY_COLOR,
        "COMPLETE": BLUE_COLOR,
        "ERROR": RED_COLOR,
    }.get(status, GRAY_COLOR)
    status_el = html.Span(f"* {status}", style={"color": status_color, "fontSize": "12px"})

    red_avg = "RED avg: n/a"
    blue_avg = "BLUE avg: n/a"
    if frame is not None and len(frame) >= 1:
        last_ten = frame.tail(10)
        red_avg = f"RED(10): {last_ten['red_total'].mean():+.3f}"
        blue_avg = f"BLUE(10): {last_ten['blue_total'].mean():+.3f}"

    uptime = ""
    try:
        started_at = state.get("started_at")
        if started_at:
            delta = datetime.now() - datetime.fromisoformat(started_at)
            hours = int(delta.seconds // 3600)
            minutes = int((delta.seconds % 3600) // 60)
            seconds = int(delta.seconds % 60)
            uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except Exception:
        pass

    live_style = {"color": GREEN_COLOR if status == "RUNNING" else GRAY_COLOR, "fontSize": "11px"}
    return ep_text, status_el, red_avg, blue_avg, uptime, live_style


def update_tab1(_n):
    frame = _load_rewards_csv()
    empty = go.Figure().update_layout(**PLOTLY_LAYOUT)
    if frame is None or frame.empty:
        return empty, empty, empty, html.Div("No data yet.", style={"color": TEXT_MUTED})

    episodes = frame["episode"].tolist()
    red_vals = frame["red_total"].tolist()
    blue_vals = frame["blue_total"].tolist()
    red_ma = _moving_average(red_vals, window=10)
    blue_ma = _moving_average(blue_vals, window=10)

    fig_main = go.Figure()
    fig_main.add_trace(
        go.Scatter(x=episodes, y=red_vals, mode="lines", name="RED", line=dict(color=RED_COLOR, width=1.5))
    )
    fig_main.add_trace(
        go.Scatter(
            x=episodes,
            y=blue_vals,
            mode="lines",
            name="BLUE",
            line=dict(color=BLUE_COLOR, width=1.5),
        )
    )
    fig_main.add_trace(
        go.Scatter(
            x=episodes,
            y=red_ma,
            mode="lines",
            name="RED avg",
            line=dict(color=RED_COLOR, width=2, dash="dash"),
            opacity=0.5,
        )
    )
    fig_main.add_trace(
        go.Scatter(
            x=episodes,
            y=blue_ma,
            mode="lines",
            name="BLUE avg",
            line=dict(color=BLUE_COLOR, width=2, dash="dash"),
            opacity=0.5,
        )
    )
    fig_main.add_hline(y=0, line_color="#444", line_dash="dot")
    fig_main.update_layout(**PLOTLY_LAYOUT, title="Episode Rewards", legend=dict(orientation="h", y=1.1))

    fig_red = go.Figure()
    for col, name, color in [
        ("red_exfil", "Exfil", "#44cc88"),
        ("red_stealth", "Stealth", BLUE_COLOR),
        ("red_memory", "Memory", "#44cccc"),
        ("red_complexity", "Complexity", "#aa44ff"),
        ("red_abort_penalty", "Abort Pen", RED_COLOR),
        ("red_honeypot_penalty", "HP Pen", GOLD_COLOR),
    ]:
        if col in frame.columns:
            fig_red.add_trace(
                go.Scatter(
                    x=episodes,
                    y=frame[col].tolist(),
                    mode="lines",
                    name=name,
                    stackgroup="red",
                    line=dict(color=color, width=0.5),
                    opacity=0.7,
                )
            )
    fig_red.update_layout(**PLOTLY_LAYOUT, title="RED Components")

    fig_blue = go.Figure()
    for col, name, color in [
        ("blue_detection", "Detection", BLUE_COLOR),
        ("blue_speed", "Speed", "#44ccff"),
        ("blue_graph_reconstruction", "Recon", "#aa44ff"),
        ("blue_fp_penalty", "FP Pen", RED_COLOR),
        ("blue_honeypot_rate", "HP Rate", GREEN_COLOR),
    ]:
        if col in frame.columns:
            fig_blue.add_trace(
                go.Scatter(
                    x=episodes,
                    y=frame[col].tolist(),
                    mode="lines",
                    name=name,
                    stackgroup="blue",
                    line=dict(color=color, width=0.5),
                    opacity=0.7,
                )
            )
    fig_blue.update_layout(**PLOTLY_LAYOUT, title="BLUE Components")

    total = len(frame)
    red_wins = int((frame["red_total"] > 0).sum())
    blue_wins = int((frame["blue_total"] > 0).sum())
    best_red = frame.loc[frame["red_total"].idxmax()]
    best_blue = frame.loc[frame["blue_total"].idxmax()]
    most_term = frame["terminal_reason"].mode()[0] if "terminal_reason" in frame.columns else "n/a"
    stats = html.Div(
        [
            _stat("Total Episodes", str(total)),
            _stat("Best RED", f"Ep {int(best_red['episode'])} ({best_red['red_total']:+.3f})"),
            _stat("Best BLUE", f"Ep {int(best_blue['episode'])} ({best_blue['blue_total']:+.3f})"),
            _stat("RED wins", f"{red_wins} ({100 * red_wins // max(total, 1)}%)"),
            _stat("BLUE wins", f"{blue_wins} ({100 * blue_wins // max(total, 1)}%)"),
            _stat("Common terminal", str(most_term)),
        ],
        style={"display": "flex", "gap": "24px", "padding": "8px 0", "flexWrap": "wrap"},
    )

    return fig_main, fig_red, fig_blue, stats


def update_tab2(_n, filter_val):
    events = _load_training_events()
    drops = [event for event in events if event.get("event_type") == "dead_drop_written"]

    if filter_val == "red":
        drops = [drop for drop in drops if drop.get("team") == "red"]
    elif filter_val == "blue":
        drops = [drop for drop in drops if drop.get("team") == "blue"]
    elif filter_val == "tampered":
        drops = [drop for drop in drops if drop.get("integrity") == "tampered"]

    sorted_drops = sorted(drops, key=lambda row: row.get("episode", 0), reverse=True)
    table_data = [
        {
            "episode": row.get("episode", "?"),
            "step": row.get("step", "?"),
            "node": row.get("node", "?"),
            "team": row.get("team", "?"),
            "tokens": row.get("tokens", "?"),
            "efficiency": f"{float(row.get('efficiency', 0.0)):.2f}",
            "integrity": row.get("integrity", "valid"),
            "detail": str(row.get("detail", ""))[:60],
        }
        for row in sorted_drops
    ]

    total = len(drops)
    tampered = sum(1 for drop in drops if drop.get("integrity") == "tampered")
    avg_tokens = np.mean([drop.get("tokens", 0) for drop in drops]) if drops else 0
    stats = html.Div(
        [
            _stat("Total drops", str(total)),
            _stat("Avg tokens", f"{avg_tokens:.0f}"),
            _stat("Tampered", str(tampered), color=RED_COLOR if tampered > 0 else GREEN_COLOR),
        ],
        style={"display": "flex", "gap": "20px"},
    )
    return table_data, stats


def update_tab3(_n):
    graph, pos = _get_graph_positions()
    events = _load_training_events()
    trap_events = [event for event in events if event.get("event_type") == "trap_fired"]

    fig = go.Figure()
    if graph is not None and pos is not None:
        for src, dst in graph.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(color="#222", width=0.5),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        zone_colors = {0: GRAY_COLOR, 1: BLUE_COLOR, 2: GOLD_COLOR, 3: RED_COLOR}
        zone_names = {
            0: "Z0 Perimeter",
            1: "Z1 General",
            2: "Z2 Sensitive",
            3: "Z3 Critical",
        }

        trap_counts: dict[int, int] = {}
        for event in trap_events:
            node = event.get("node")
            if isinstance(node, int):
                trap_counts[node] = trap_counts.get(node, 0) + 1

        for zone in range(4):
            zone_nodes = []
            for node in graph.nodes():
                node_zone = graph.nodes[node].get("zone")
                zone_value = int(getattr(node_zone, "value", node_zone))
                if zone_value == zone:
                    zone_nodes.append(node)
            if not zone_nodes:
                continue

            xs = [pos[node][0] for node in zone_nodes]
            ys = [pos[node][1] for node in zone_nodes]
            sizes = [10 + 3 * trap_counts.get(node, 0) for node in zone_nodes]
            labels = [
                f"Node {node}<br>Zone {zone}<br>Traps: {trap_counts.get(node, 0)}"
                for node in zone_nodes
            ]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(size=sizes, color=zone_colors[zone], opacity=0.85),
                    name=zone_names[zone],
                    text=labels,
                    hoverinfo="text",
                )
            )

        triggered_nodes = list(
            {
                event.get("node")
                for event in trap_events
                if isinstance(event.get("node"), int) and event.get("node") in pos
            }
        )
        if triggered_nodes:
            tx = [pos[node][0] for node in triggered_nodes]
            ty = [pos[node][1] for node in triggered_nodes]
            fig.add_trace(
                go.Scatter(
                    x=tx,
                    y=ty,
                    mode="markers",
                    marker=dict(
                        size=20,
                        color="rgba(255,255,0,0.15)",
                        line=dict(color=GOLD_COLOR, width=2),
                    ),
                    name="Trap triggered",
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Network - Trap Activity",
        showlegend=True,
        legend=dict(x=1.01, y=1, bgcolor="rgba(0,0,0,0)"),
    )

    recent = sorted(trap_events, key=lambda row: row.get("episode", 0), reverse=True)[:15]
    event_rows = []
    for event in recent:
        color = GOLD_COLOR if event.get("team") == "blue" else RED_COLOR
        event_rows.append(
            html.Div(
                (
                    f"* Ep{event.get('episode', '?')} Step{event.get('step', '?')} "
                    f"Node{event.get('node', '?')} - {event.get('detail', '')}"
                ),
                style={
                    "color": color,
                    "fontSize": "11px",
                    "padding": "2px 0",
                    "borderBottom": f"1px solid {BORDER_COLOR}",
                },
            )
        )
    event_log = html.Div(event_rows or [html.Div("No trap events yet.", style={"color": TEXT_MUTED})])

    total_traps = len(trap_events)
    honeypots = sum(1 for event in trap_events if event.get("trap_type") == "honeypot")
    false_trails = sum(1 for event in trap_events if event.get("trap_type") == "false_trail")
    stats = html.Div(
        [
            _stat("Total trap events", str(total_traps)),
            _stat("Honeypots triggered", str(honeypots)),
            _stat("False trails", str(false_trails)),
        ],
        style={"display": "flex", "gap": "20px"},
    )
    return fig, event_log, stats


def update_tab4(_n):
    frame = _load_rewards_csv()
    if frame is None or frame.empty:
        return [], html.Div("No data.", style={"color": TEXT_MUTED}), html.Div()

    verdict_columns = ["episode", "fleet_verdict", "oversight_red_adj", "oversight_blue_adj", "fleet_judgment"]
    available = [column for column in verdict_columns if column in frame.columns]
    verdict_frame = frame[available].copy().sort_values("episode", ascending=False)
    if "fleet_judgment" in verdict_frame.columns:
        verdict_frame["fleet_judgment"] = verdict_frame["fleet_judgment"].astype(str).str[:80]
    verdict_data = verdict_frame.to_dict("records")

    flag_rows = []
    if "oversight_flags" in frame.columns:
        flagged = frame[frame["oversight_flags"] != "none"]
        for _, row in flagged.iterrows():
            flag_rows.append(
                html.Div(
                    f"! Ep{int(row['episode'])} - {row['oversight_flags']}",
                    style={
                        "color": RED_COLOR,
                        "fontSize": "11px",
                        "padding": "3px 0",
                        "borderBottom": f"1px solid {BORDER_COLOR}",
                    },
                )
            )
    flags_el = (
        html.Div(flag_rows)
        if flag_rows
        else html.Div(
            "No oversight flags fired this run.",
            style={"color": GREEN_COLOR, "fontSize": "11px"},
        )
    )

    total = len(frame)
    verdicts = frame["fleet_verdict"].value_counts() if "fleet_verdict" in frame.columns else pd.Series(dtype=int)
    stats = html.Div(
        [
            _stat("Episodes", str(total)),
            _stat("RED dominant", str(verdicts.get("red_dominates", 0))),
            _stat("BLUE dominant", str(verdicts.get("blue_dominates", 0))),
            _stat("Contested", str(verdicts.get("contested", 0))),
            _stat(
                "Degenerate",
                str(verdicts.get("degenerate", 0)),
                color=GOLD_COLOR if verdicts.get("degenerate", 0) > 0 else TEXT_SECONDARY,
            ),
        ],
        style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
    )
    return verdict_data, flags_el, stats


def update_tab5(_n):
    frame = _load_rewards_csv()
    events = _load_training_events()
    empty = go.Figure().update_layout(**PLOTLY_LAYOUT)
    if frame is None or frame.empty:
        return empty, empty, html.Div("No data yet.", style={"color": TEXT_MUTED})

    difficulty_map: dict[int, float] = {}
    for event in events:
        if event.get("event_type") == "episode_start":
            detail = str(event.get("detail", ""))
            try:
                diff_str = detail.split("difficulty=")[1].split(")")[0]
                difficulty_map[int(event.get("episode"))] = float(diff_str)
            except Exception:
                pass

    episodes = frame["episode"].tolist()
    difficulties = [difficulty_map.get(int(episode), None) for episode in episodes]

    fig_main = go.Figure()
    fig_main.add_trace(
        go.Scatter(
            x=episodes,
            y=frame["red_total"].tolist(),
            mode="lines",
            name="RED reward",
            line=dict(color=RED_COLOR, width=1.5),
        )
    )
    fig_main.add_trace(
        go.Scatter(
            x=episodes,
            y=frame["blue_total"].tolist(),
            mode="lines",
            name="BLUE reward",
            line=dict(color=BLUE_COLOR, width=1.5),
        )
    )
    valid_diffs = [(ep, diff) for ep, diff in zip(episodes, difficulties) if diff is not None]
    if valid_diffs:
        diff_eps, diff_vals = zip(*valid_diffs)
        fig_main.add_trace(
            go.Scatter(
                x=list(diff_eps),
                y=list(diff_vals),
                mode="lines",
                name="Difficulty",
                line=dict(color=GOLD_COLOR, width=2, dash="dot"),
                yaxis="y2",
            )
        )

    fig_main.update_layout(
        **PLOTLY_LAYOUT,
        title="Rewards vs Difficulty",
        yaxis2=dict(overlaying="y", side="right", showgrid=False, color=GOLD_COLOR, range=[0, 1]),
        legend=dict(orientation="h", y=1.1),
    )

    fig_scatter = go.Figure()
    if valid_diffs and len(valid_diffs) >= 2:
        diff_eps_list = list(diff_eps)
        red_rewards_for_diff: list[Optional[float]] = []
        for ep in diff_eps_list:
            row = frame[frame["episode"] == ep]
            red_rewards_for_diff.append(float(row.iloc[0]["red_total"]) if not row.empty else None)

        clean = [(d, r) for d, r in zip(diff_vals, red_rewards_for_diff) if r is not None]
        if len(clean) >= 2:
            cx, cy = zip(*clean)
            fig_scatter.add_trace(
                go.Scatter(
                    x=list(cx),
                    y=list(cy),
                    mode="markers",
                    marker=dict(color=RED_COLOR, size=6, opacity=0.7),
                    name="RED reward",
                )
            )
            coeffs = np.polyfit(cx, cy, 1)
            trend_x = np.linspace(min(cx), max(cx), 50)
            trend_y = np.polyval(coeffs, trend_x)
            fig_scatter.add_trace(
                go.Scatter(
                    x=trend_x.tolist(),
                    y=trend_y.tolist(),
                    mode="lines",
                    line=dict(color=GOLD_COLOR, width=1.5, dash="dash"),
                    name="Trend",
                )
            )

    fig_scatter.update_layout(
        **PLOTLY_LAYOUT,
        title="Difficulty vs RED Reward",
        xaxis_title="Difficulty",
        yaxis_title="RED Reward",
    )

    current_diff = difficulties[-1] if difficulties and difficulties[-1] is not None else "n/a"
    start_diff = next((diff for diff in difficulties if diff is not None), "n/a")
    max_diff = max((diff for diff in difficulties if diff is not None), default="n/a")
    corr_text = "n/a"
    if valid_diffs and len(valid_diffs) >= 3:
        try:
            diff_arr = np.array(diff_vals, dtype=float)
            red_arr = np.array(
                [
                    float(frame[frame["episode"] == ep].iloc[0]["red_total"])
                    for ep in diff_eps
                    if not frame[frame["episode"] == ep].empty
                ],
                dtype=float,
            )
            if len(diff_arr) == len(red_arr) and len(diff_arr) >= 2:
                corr = float(np.corrcoef(diff_arr, red_arr)[0, 1])
                corr_text = f"{corr:+.3f}"
        except Exception:
            pass

    stats = html.Div(
        [
            _stat("Start difficulty", f"{start_diff}" if isinstance(start_diff, str) else f"{start_diff:.2f}"),
            _stat(
                "Current difficulty",
                f"{current_diff}" if isinstance(current_diff, str) else f"{current_diff:.2f}",
            ),
            _stat("Max difficulty", f"{max_diff}" if isinstance(max_diff, str) else f"{max_diff:.2f}"),
            _stat("Corr (diff->RED)", corr_text),
        ],
        style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
    )
    return fig_main, fig_scatter, stats


def update_tab6(_n):
    from cipher.training.improvement_analyzer import ImprovementAnalyzer

    analyzer = ImprovementAnalyzer()
    frame = _load_rewards_csv()
    evols = _load_evolution_log()
    empty = go.Figure().update_layout(**PLOTLY_LAYOUT)

    # ── Top chart: rewards (raw gray) + rolling avg (colored) + evolution lines ──
    if frame is None or frame.empty:
        fig_reward = empty
    else:
        episodes = frame["episode"].tolist()
        red_vals = frame["red_total"].tolist()
        blue_vals = frame["blue_total"].tolist()
        red_ma = _moving_average(red_vals, window=10)
        blue_ma = _moving_average(blue_vals, window=10)
        evo_eps = [e.get("episode", 0) for e in evols]

        fig_reward = go.Figure()
        fig_reward.add_trace(
            go.Scatter(
                x=episodes, y=red_vals,
                mode="lines", name="RED (raw)",
                line=dict(color="#555555", width=1),
                opacity=0.5,
            )
        )
        fig_reward.add_trace(
            go.Scatter(
                x=episodes, y=blue_vals,
                mode="lines", name="BLUE (raw)",
                line=dict(color="#335588", width=1),
                opacity=0.5,
            )
        )
        fig_reward.add_trace(
            go.Scatter(
                x=episodes, y=red_ma,
                mode="lines", name="RED 10-ep avg",
                line=dict(color=RED_COLOR, width=2.5),
            )
        )
        fig_reward.add_trace(
            go.Scatter(
                x=episodes, y=blue_ma,
                mode="lines", name="BLUE 10-ep avg",
                line=dict(color=BLUE_COLOR, width=2.5),
            )
        )
        fig_reward.add_hline(y=0, line_color="#444", line_dash="dot")

        # Gold vertical lines at each evolution event
        for ep in evo_eps:
            fig_reward.add_vline(
                x=ep, line_color=GOLD_COLOR, line_width=1.5, line_dash="dash",
                annotation_text=f"Evo@{ep}",
                annotation_font_size=9,
                annotation_font_color=GOLD_COLOR,
            )

        fig_reward.update_layout(
            **PLOTLY_LAYOUT,
            title="Reward Curves + Evolution Annotations",
            xaxis_title="Episode",
            yaxis_title="Reward",
            legend=dict(orientation="h", y=1.12),
        )

    # ── Middle chart: win rate curves ──
    win_data = analyzer.compute_rolling_win_rates(window=10)
    if not win_data["episodes"]:
        fig_winrate = empty
    else:
        fig_winrate = go.Figure()
        fig_winrate.add_trace(
            go.Scatter(
                x=win_data["episodes"],
                y=[v * 100 for v in win_data["red_win_rate"]],
                mode="lines",
                name="RED Win %",
                line=dict(color=RED_COLOR, width=2.5),
            )
        )
        fig_winrate.add_trace(
            go.Scatter(
                x=win_data["episodes"],
                y=[v * 100 for v in win_data["blue_win_rate"]],
                mode="lines",
                name="BLUE Win %",
                line=dict(color=BLUE_COLOR, width=2.5),
            )
        )
        # Evolution lines on win rate chart too
        for ep in [e.get("episode", 0) for e in evols]:
            fig_winrate.add_vline(
                x=ep, line_color=GOLD_COLOR, line_width=1.2, line_dash="dash",
            )
        fig_winrate.add_hline(y=50, line_color="#444", line_dash="dot")
        fig_winrate.update_layout(
            **PLOTLY_LAYOUT,
            title="Rolling Win Rate (10-episode window)",
            xaxis_title="Episode",
            yaxis_title="Win Rate (%)",
            legend=dict(orientation="h", y=1.12),
        )
        fig_winrate.update_yaxes(range=[0, 100])

    # ── Bottom stats strip ──
    early_late = analyzer.compute_early_late_comparison()
    evo_summary = analyzer.get_evolution_summary()

    def _delta_color(v: float) -> str:
        return GREEN_COLOR if v > 0 else (RED_COLOR if v < 0 else TEXT_SECONDARY)

    total_rules = evo_summary["total_red_rules"] + evo_summary["total_blue_rules"]

    stats = html.Div(
        [
            html.Div(
                "Training Improvement Summary",
                style={
                    "color": TEXT_MUTED,
                    "fontSize": "10px",
                    "textTransform": "uppercase",
                    "letterSpacing": "1px",
                    "marginBottom": "8px",
                    "width": "100%",
                },
            ),
            _stat(
                "Early RED avg",
                f"{early_late['early_red_avg']:+.3f}",
                color=RED_COLOR,
            ),
            _stat(
                "Late RED avg",
                f"{early_late['late_red_avg']:+.3f}",
                color=RED_COLOR,
            ),
            _stat(
                "RED Improvement",
                f"{early_late['red_improvement']:+.3f}",
                color=_delta_color(early_late["red_improvement"]),
            ),
            _stat(
                "Early Exfil Rate",
                f"{early_late['early_exfil_rate'] * 100:.0f}%",
            ),
            _stat(
                "Late Exfil Rate",
                f"{early_late['late_exfil_rate'] * 100:.0f}%",
                color=_delta_color(early_late["exfil_delta"]),
            ),
            _stat(
                "Exfil Delta",
                f"{early_late['exfil_delta'] * 100:+.0f}%",
                color=_delta_color(early_late["exfil_delta"]),
            ),
            _stat(
                "Early Abort Rate",
                f"{early_late['early_abort_rate'] * 100:.0f}%",
            ),
            _stat(
                "Late Abort Rate",
                f"{early_late['late_abort_rate'] * 100:.0f}%",
                color=_delta_color(-early_late["abort_delta"]),
            ),
            _stat(
                "Abort Delta",
                f"{early_late['abort_delta'] * 100:+.0f}%",
                color=_delta_color(-early_late["abort_delta"]),
            ),
            _stat(
                "Evolutions Applied",
                str(evo_summary["total_evolutions"]),
                color=GOLD_COLOR,
            ),
            _stat(
                "Rules Added",
                str(total_rules),
                color=GOLD_COLOR,
            ),
        ],
        style={"display": "flex", "gap": "16px", "padding": "8px 0", "flexWrap": "wrap"},
    )

    return fig_reward, fig_winrate, stats


# ---------------------------------------------------------------------------
# App layout and callback wiring
# ---------------------------------------------------------------------------

def get_live_dashboard() -> dash.Dash:
    """Return the live training dashboard Dash app instance."""
    return app


def register_callbacks_on(target_app: dash.Dash) -> None:
    """
    Register all live-training callbacks on *target_app*.

    Called once at startup for the module-level ``app``, and can be called
    again to wire the same callbacks onto a fresh app instance (used in tests).
    """
    app_id = id(target_app)
    if app_id in _REGISTERED_APPS:
        return
    _REGISTERED_APPS.add(app_id)

    @target_app.callback(
        [
            Output("header-episode", "children"),
            Output("header-status", "children"),
            Output("header-red-avg", "children"),
            Output("header-blue-avg", "children"),
            Output("header-uptime", "children"),
            Output("live-indicator", "style"),
        ],
        Input("interval-component", "n_intervals"),
    )
    def _header(n):
        return update_header(n)

    @target_app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def _tab_content(tab):
        return render_tab(tab)

    @target_app.callback(
        [
            Output("t1-main-chart", "figure"),
            Output("t1-red-breakdown", "figure"),
            Output("t1-blue-breakdown", "figure"),
            Output("t1-stats", "children"),
        ],
        Input("interval-component", "n_intervals"),
    )
    def _tab1(n):
        return update_tab1(n)

    @target_app.callback(
        [
            Output("t2-table", "data"),
            Output("t2-stats", "children"),
        ],
        [
            Input("interval-component", "n_intervals"),
            Input("t2-filter", "value"),
        ],
    )
    def _tab2(n, filter_val):
        return update_tab2(n, filter_val)

    @target_app.callback(
        [
            Output("t3-map", "figure"),
            Output("t3-events", "children"),
            Output("t3-stats", "children"),
        ],
        Input("interval-component", "n_intervals"),
    )
    def _tab3(n):
        return update_tab3(n)

    @target_app.callback(
        [
            Output("t4-verdicts", "data"),
            Output("t4-flags", "children"),
            Output("t4-stats", "children"),
        ],
        Input("interval-component", "n_intervals"),
    )
    def _tab4(n):
        return update_tab4(n)

    @target_app.callback(
        [
            Output("t5-main", "figure"),
            Output("t5-scatter", "figure"),
            Output("t5-stats", "children"),
        ],
        Input("interval-component", "n_intervals"),
    )
    def _tab5(n):
        return update_tab5(n)

    @target_app.callback(
        [
            Output("t6-reward-chart", "figure"),
            Output("t6-winrate-chart", "figure"),
            Output("t6-stats", "children"),
        ],
        Input("interval-component", "n_intervals"),
    )
    def _tab6(n):
        return update_tab6(n)


# Wire callbacks onto the module-level app at import time.
app.layout = create_live_layout()
register_callbacks_on(app)


if __name__ == "__main__":
    app.run(debug=False, port=8051)
