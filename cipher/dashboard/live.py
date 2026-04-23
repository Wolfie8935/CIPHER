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
from dash import Input, Output, State, dash_table, dcc, html

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


def _tab_header(title: str, description: str) -> html.Div:
    """Consistent header for each tab explaining what you're looking at."""
    return html.Div(
        [
            html.Div(title, style={
                "fontSize": "14px", "fontWeight": "700",
                "color": TEXT_PRIMARY, "marginBottom": "4px",
            }),
            html.Div(description, style={
                "fontSize": "11px", "color": TEXT_MUTED,
                "borderLeft": f"2px solid {BORDER_COLOR}",
                "paddingLeft": "8px", "marginBottom": "12px",
            }),
        ]
    )


def _build_tab1_layout() -> html.Div:
    return html.Div(
        [
            _tab_header(
                "Reward Curves — RED vs BLUE (Current Run)",
                "Each bar = one episode. RED (attacker) scores high for exfiltrating data stealthily. "
                "BLUE (defender) scores high for detecting and blocking. "
                "Shows only the current run — start a new run to see a fresh trace.",
            ),
            dcc.Graph(id="t1-main-chart", style={"height": "260px"}),
            html.Div(
                [
                    html.Div(dcc.Graph(id="t1-red-breakdown", style={"height": "180px"}),
                             style={"flex": "1"}),
                    html.Div(dcc.Graph(id="t1-blue-breakdown", style={"height": "180px"}),
                             style={"flex": "1"}),
                ],
                style={"display": "flex", "gap": "8px"},
            ),
            html.Div(id="t1-stats"),
            html.Div(id="t1-live-feed"),
        ]
    )


def _build_tab2_layout() -> html.Div:
    return html.Div(
        [
            # B1: Store caches last filter value so we skip recompute on interval ticks
            dcc.Store(id="t2-filter-cache", data="all"),
            _tab_header(
                "Dead Drops — RED Team Memory System",
                "RED agents leave encrypted 'dead drops' (messages) at nodes so they can "
                "resume after context resets. BLUE can tamper with drops to feed RED false intel. "
                "High efficiency (close to 1.0) = compact, useful messages.",
            ),
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
            _tab_header(
                "Network Map — Trap Heatmap",
                "The 50-node enterprise network. Zone 0=Perimeter (gray) → Zone 1=General (blue) → "
                "Zone 2=Sensitive (amber) → Zone 3=Critical/HVT (red). "
                "Node size = trap activity (larger = more trap events there). "
                "Gold rings = active traps. RED path = red trail. BLUE investigated = blue marks.",
            ),
            dcc.Graph(id="t3-map", style={"height": "480px"}),
            html.Div(id="t3-events", style={"marginTop": "8px"}),
            html.Div(id="t3-stats", style={"marginTop": "8px"}),
        ]
    )


def _build_tab4_layout() -> html.Div:
    return html.Div(
        [
            _tab_header(
                "Oversight Feed — Independent AI Judge",
                "An Oversight Auditor LLM watches all 8 agents and issues verdicts after every episode. "
                "Verdicts: red_dominates (RED succeeded), blue_dominates (BLUE blocked), "
                "contested (close match), degenerate (neither team performed well). "
                "Flags fire when reward hacking or collusion is suspected. "
                "Fleet bonuses (±0.15) are applied to final rewards based on verdict.",
            ),
            html.Div(id="t4-stats", style={"marginBottom": "12px"}),
            html.Div(
                "Fleet Verdicts by Episode",
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
            _tab_header(
                "Difficulty Curve — Auto-Escalating Challenge",
                "Difficulty starts at 0.30 and auto-escalates based on recent win rate "
                "(increases if RED wins >60% of last 10 episodes, decreases if <30%). "
                "This forces agents to generalize rather than memorize. "
                "Expected pattern: rewards slightly decrease as difficulty rises, then recover as agents adapt.",
            ),
            dcc.Graph(id="t5-main", style={"height": "280px"}),
            dcc.Graph(id="t5-scatter", style={"height": "240px"}),
            html.Div(id="t5-stats", style={"marginTop": "8px"}),
        ]
    )


def _build_tab6_layout() -> html.Div:
    return html.Div(
        [
            _tab_header(
                "Learning Progress — Are Agents Getting Better?",
                "Top chart: raw rewards (dim) + 10-ep moving average (bright). "
                "Gold vertical lines = prompt evolution events — watch trend change after each. "
                "Gap chart: RED minus BLUE per episode (positive = RED winning the arms race). "
                "Win rate chart: 10-ep rolling %, 50% = perfectly balanced. "
                "Stats: early vs late comparison — positive RED improvement = learning occurred.",
            ),
            dcc.Graph(id="t6-reward-chart", style={"height": "220px"}),
            dcc.Graph(id="t6-gap-chart", style={"height": "160px"}),
            dcc.Graph(id="t6-winrate-chart", style={"height": "180px"}),
            html.Div(id="t6-stats", style={"marginTop": "12px"}),
        ]
    )


def _build_tab_logs_layout() -> html.Div:
    """B4-style: Live step-by-step narrative log."""
    return html.Div(
        [
            _tab_header(
                "Live Step Log — Real-Time Episode Narrative",
                "Each row = one simulation step. Shows RED action, BLUE response, zone, "
                "suspicion & detection levels. Updates live during --live / --hybrid runs. "
                "Stub mode: no live steps (deterministic, near-instant).",
            ),
            html.Div(id="t-logs-agent-bar", style={"marginBottom": "8px"}),
            html.Div(id="t-logs-feed",
                     style={"background": "#050505", "padding": "10px",
                            "borderRadius": "4px", "border": f"1px solid {BORDER_COLOR}",
                            "maxHeight": "520px", "overflowY": "auto",
                            "fontFamily": "monospace", "fontSize": "11px"}),
        ]
    )


def _build_tab_history_layout() -> html.Div:
    """B5: Cross-run episode history."""
    return html.Div(
        [
            _tab_header(
                "Episode History — All Runs",
                "Aggregated view across all runs stored in SQLite. "
                "Compare RED vs BLUE rewards, outcomes, and trends. "
                "Table shows the last 30 episodes across every run.",
            ),
            html.Div(
                [
                    html.Div(dcc.Graph(id="th-rewards-chart", style={"height": "220px"}),
                             style={"flex": "1"}),
                    html.Div(dcc.Graph(id="th-outcomes-chart", style={"height": "220px"}),
                             style={"flex": "1"}),
                ],
                style={"display": "flex", "gap": "8px"},
            ),
            html.Div(id="th-run-selector", style={"margin": "8px 0"}),
            dash_table.DataTable(
                id="th-table",
                columns=[
                    {"name": "Run", "id": "run_id"},
                    {"name": "Mode", "id": "llm_mode"},
                    {"name": "Ep", "id": "episode"},
                    {"name": "Steps", "id": "steps"},
                    {"name": "Outcome", "id": "terminal_reason"},
                    {"name": "RED", "id": "red_total"},
                    {"name": "BLUE", "id": "blue_total"},
                    {"name": "Verdict", "id": "fleet_verdict"},
                ],
                style_cell={
                    "backgroundColor": PANEL_BG, "color": TEXT_SECONDARY,
                    "border": f"1px solid {BORDER_COLOR}", "fontSize": "11px",
                    "padding": "4px 8px", "maxWidth": "160px",
                    "overflow": "hidden", "textOverflow": "ellipsis",
                },
                style_header={
                    "backgroundColor": "#1a1a1a", "color": TEXT_PRIMARY,
                    "fontWeight": "bold", "fontSize": "11px",
                },
                style_data_conditional=[
                    {"if": {"filter_query": '{terminal_reason} = "exfil_success"'},
                     "color": GREEN_COLOR},
                    {"if": {"filter_query": '{terminal_reason} = "detected"'},
                     "color": BLUE_COLOR},
                    {"if": {"filter_query": '{terminal_reason} = "aborted"'},
                     "color": GOLD_COLOR},
                ],
                page_size=15, sort_action="native",
            ),
        ]
    )


def _load_rewards_csv() -> Optional[pd.DataFrame]:
    import time
    csv_path = Path("rewards_log.csv")
    for attempt in range(3):
        try:
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
        except (OSError, pd.errors.ParserError):
            # File may be locked by the training loop — brief wait then retry
            if attempt < 2:
                time.sleep(0.05)
        except Exception:
            return None
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


def _get_run_frame() -> Optional[pd.DataFrame]:
    """Load current-run data. SQLite primary (B2), CSV timestamp-filtered fallback."""
    # Try SQLite first (B2)
    db_frame = _load_db_run_frame()
    if db_frame is not None and not db_frame.empty:
        for col in ("red_total", "blue_total", "red_exfil", "red_stealth", "red_memory",
                    "red_complexity", "red_abort_penalty", "red_honeypot_penalty",
                    "blue_detection", "blue_speed", "blue_fp_penalty", "blue_honeypot_rate",
                    "blue_graph_reconstruction", "oversight_red_adj", "oversight_blue_adj"):
            if col in db_frame.columns:
                db_frame[col] = pd.to_numeric(db_frame[col], errors="coerce")
        if "episode" in db_frame.columns:
            db_frame["episode"] = pd.to_numeric(db_frame["episode"], errors="coerce").astype("Int64")
        return db_frame

    # CSV fallback: filter by run started_at
    frame = _load_rewards_csv()
    if frame is None:
        return None
    state = _load_training_state()
    started_at = state.get("started_at")
    if started_at and "timestamp" in frame.columns:
        try:
            cutoff = pd.to_datetime(started_at)
            ts = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame = frame[ts >= cutoff].copy()
        except Exception:
            pass
    return frame if not frame.empty else None


def _load_live_steps() -> list[dict]:
    """Load per-step live feed written by main.py during live/hybrid runs."""
    try:
        steps_path = Path("live_steps.jsonl")
        if not steps_path.exists():
            return []
        text = steps_path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        rows = []
        for line in text.split("\n"):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        return rows
    except Exception:
        return []


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
_TAB2_CACHE: dict[str, object] = {}  # B1: filter-result cache


def _load_agent_status() -> dict:
    """B4: Read per-agent last-action status written by main.py step callback."""
    try:
        path = Path("logs") / "agent_status.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _estimate_api_cost() -> str:
    """Return a human-readable API cost estimate from training_state.json."""
    state = _load_training_state()
    cost = state.get("estimated_cost_usd")
    mode = state.get("llm_mode", "stub")
    if mode == "stub" or cost is None:
        return "$0.00 (stub)"
    return f"~${float(cost):.4f}"


def _load_db_run_frame() -> Optional[pd.DataFrame]:
    """B2: Load current-run data from SQLite (fast, no file-lock issues)."""
    try:
        from cipher.utils.telemetry_db import get_db
        state = _load_training_state()
        run_id = state.get("run_id")
        rows = get_db().get_all_episodes(run_id=run_id) if run_id else []
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception:
        return None


def _load_db_all_frame(n: int = 200) -> Optional[pd.DataFrame]:
    """Load last N episodes across all runs from SQLite (for History tab)."""
    try:
        from cipher.utils.telemetry_db import get_db
        rows = get_db().get_last_n_episodes(n)
        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception:
        return None


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
    _tab_style = {"fontFamily": "monospace", "fontSize": "11px", "padding": "6px 12px"}
    _tab_sel_style = {**_tab_style, "borderTop": f"2px solid {RED_COLOR}",
                      "background": PANEL_BG, "color": TEXT_PRIMARY}
    return html.Div(
        [
            # ── Top header bar ───────────────────────────────────────────
            html.Div(
                [
                    html.Div([
                        html.Span("CIPHER", style={
                            "color": RED_COLOR, "fontWeight": "700",
                            "fontSize": "16px", "letterSpacing": "4px",
                        }),
                        html.Span(" LIVE", id="live-indicator",
                                  style={"color": GREEN_COLOR, "fontSize": "10px",
                                         "marginLeft": "4px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Span(id="header-episode",
                              style={"color": TEXT_SECONDARY, "fontSize": "12px"}),
                    html.Span(id="header-status", style={"fontSize": "12px"}),
                    html.Span(id="header-red-avg",
                              style={"color": RED_COLOR, "fontSize": "11px"}),
                    html.Span(id="header-blue-avg",
                              style={"color": BLUE_COLOR, "fontSize": "11px"}),
                    html.Span(id="header-run-id",
                              style={"color": GOLD_COLOR, "fontSize": "10px",
                                     "fontFamily": "monospace"}),
                    html.Span(id="header-cost",
                              style={"color": GREEN_COLOR, "fontSize": "10px"}),
                    html.Span(id="header-uptime",
                              style={"color": TEXT_MUTED, "fontSize": "10px"}),
                ],
                style={
                    "display": "flex", "gap": "16px", "alignItems": "center",
                    "padding": "8px 16px", "background": "#070707",
                    "borderBottom": f"1px solid {BORDER_COLOR}", "flexWrap": "wrap",
                },
            ),
            # ── B4: Agent status bar (1s interval) ───────────────────────
            html.Div(id="agent-status-bar",
                     style={"padding": "4px 16px", "background": "#0a0a0a",
                            "borderBottom": f"1px solid {BORDER_COLOR}",
                            "minHeight": "24px"}),
            # ── Tabs ─────────────────────────────────────────────────────
            dcc.Tabs(
                id="main-tabs",
                value="tab-rewards",
                children=[
                    dcc.Tab(label="Rewards",       value="tab-rewards",
                            style=_tab_style, selected_style=_tab_sel_style),
                    dcc.Tab(label="Live Logs",     value="tab-logs",
                            style=_tab_style, selected_style=_tab_sel_style),
                    dcc.Tab(label="Dead Drops",    value="tab-drops",
                            style=_tab_style, selected_style=_tab_sel_style),
                    dcc.Tab(label="Network Map",   value="tab-deception",
                            style=_tab_style, selected_style=_tab_sel_style),
                    dcc.Tab(label="Oversight",     value="tab-oversight",
                            style=_tab_style, selected_style=_tab_sel_style),
                    dcc.Tab(label="Difficulty",    value="tab-difficulty",
                            style=_tab_style, selected_style=_tab_sel_style),
                    dcc.Tab(label="Learning",      value="tab-evolution",
                            style=_tab_style, selected_style=_tab_sel_style),
                    dcc.Tab(label="History",       value="tab-history",
                            style=_tab_style, selected_style=_tab_sel_style),
                ],
                colors={"border": BORDER_COLOR, "primary": RED_COLOR, "background": "#0a0a0a"},
            ),
            html.Div(
                id="tab-content",
                children=render_tab("tab-rewards"),
                style={"padding": "12px 16px", "background": DARK_BG, "minHeight": "600px"},
            ),
            dcc.Interval(id="interval-component",
                         interval=config.dashboard_live_update_interval, n_intervals=0),
            # B4: fast 1s interval just for agent status bar + live logs
            dcc.Interval(id="interval-fast", interval=1500, n_intervals=0),
        ],
        style={"background": DARK_BG, "minHeight": "100vh",
               "fontFamily": "monospace", "color": TEXT_PRIMARY},
    )


def render_tab(tab):
    """Render ALL tab layouts but only show the active one.

    This is the critical fix for 'Callback error updating t2-table' etc.:
    Dash fires interval callbacks for ALL registered outputs regardless of
    which tab is visible. If a tab's components don't exist in the DOM,
    Dash raises a callback error. By always rendering all tabs (hidden via
    display:none), all IDs are always present.
    """
    tabs_cfg = [
        ("tab-rewards",    _build_tab1_layout),
        ("tab-logs",       _build_tab_logs_layout),
        ("tab-drops",      _build_tab2_layout),
        ("tab-deception",  _build_tab3_layout),
        ("tab-oversight",  _build_tab4_layout),
        ("tab-difficulty", _build_tab5_layout),
        ("tab-evolution",  _build_tab6_layout),
        ("tab-history",    _build_tab_history_layout),
    ]
    children = []
    for tab_id, builder in tabs_cfg:
        style = {} if tab == tab_id else {"display": "none"}
        children.append(html.Div(builder(), style=style, id=f"tab-wrapper-{tab_id}"))
    return html.Div(children)


def update_header(_n):
    state = _load_training_state()
    frame = _get_run_frame()

    ep_text = f"Ep {state.get('current_episode', 0)} / {state.get('total_episodes', '?')}"
    status = str(state.get("status", "idle")).upper()
    status_color = {
        "RUNNING": GREEN_COLOR,
        "IDLE": GRAY_COLOR,
        "COMPLETE": BLUE_COLOR,
        "ERROR": RED_COLOR,
    }.get(status, GRAY_COLOR)
    status_el = html.Span(f"* {status}", style={"color": status_color, "fontSize": "12px"})

    red_avg = "RED: n/a"
    blue_avg = "BLUE: n/a"
    if frame is not None and len(frame) >= 1:
        last_n = frame.tail(min(10, len(frame)))
        red_avg = f"RED({len(last_n)}ep): {last_n['red_total'].mean():+.3f}"
        blue_avg = f"BLUE({len(last_n)}ep): {last_n['blue_total'].mean():+.3f}"

    run_id = state.get("run_id", "")
    llm_mode = state.get("llm_mode", "stub").upper()
    run_badge = f"[{llm_mode}] {run_id}" if run_id else f"[{llm_mode}]"

    uptime = ""
    try:
        started_at = state.get("started_at")
        if started_at:
            delta = datetime.now() - datetime.fromisoformat(started_at)
            total_secs = int(delta.total_seconds())
            hours = total_secs // 3600
            minutes = (total_secs % 3600) // 60
            seconds = total_secs % 60
            uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except Exception:
        pass

    cost_text = _estimate_api_cost()
    live_style = {"color": GREEN_COLOR if status == "RUNNING" else GRAY_COLOR, "fontSize": "10px"}
    return ep_text, status_el, red_avg, blue_avg, run_badge, cost_text, uptime, live_style


def update_tab1(_n):
    frame = _get_run_frame()
    empty = go.Figure().update_layout(**PLOTLY_LAYOUT)
    _no_data = html.Div("No episode data yet for this run.", style={"color": TEXT_MUTED})
    if frame is None or frame.empty:
        return empty, empty, empty, _no_data, html.Div()

    episodes = frame["episode"].tolist()
    red_vals = frame["red_total"].tolist()
    blue_vals = frame["blue_total"].tolist()
    red_ma = _moving_average(red_vals, window=5)
    blue_ma = _moving_average(blue_vals, window=5)

    # Bar chart per episode is much cleaner than spaghetti lines
    fig_main = go.Figure()
    fig_main.add_trace(go.Bar(
        x=episodes, y=red_vals, name="RED reward",
        marker_color=RED_COLOR, opacity=0.75, offsetgroup=0,
    ))
    fig_main.add_trace(go.Bar(
        x=episodes, y=blue_vals, name="BLUE reward",
        marker_color=BLUE_COLOR, opacity=0.75, offsetgroup=1,
    ))
    if len(episodes) >= 3:
        fig_main.add_trace(go.Scatter(
            x=episodes, y=red_ma, mode="lines", name="RED avg (5ep)",
            line=dict(color="#ff8888", width=2, dash="dash"),
        ))
        fig_main.add_trace(go.Scatter(
            x=episodes, y=blue_ma, mode="lines", name="BLUE avg (5ep)",
            line=dict(color="#88aaff", width=2, dash="dash"),
        ))
    fig_main.add_hline(y=0, line_color="#444", line_dash="dot")
    fig_main.update_layout(
        **PLOTLY_LAYOUT,
        title="Episode Rewards — This Run",
        barmode="group",
        legend=dict(orientation="h", y=1.1),
    )
    fig_main.update_xaxes(title="Episode", tickmode="linear", dtick=1)
    fig_main.update_yaxes(title="Reward")

    # Stacked bar breakdown (cleaner than area for few episodes)
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
            fig_red.add_trace(go.Bar(
                x=episodes, y=frame[col].tolist(),
                name=name, marker_color=color, opacity=0.85,
            ))
    fig_red.update_layout(
        **PLOTLY_LAYOUT, title="RED Reward Components",
        barmode="relative",
        legend=dict(orientation="h", y=1.1, font=dict(size=10)),
    )
    fig_red.update_xaxes(title="Episode", tickmode="linear", dtick=1)

    fig_blue = go.Figure()
    for col, name, color in [
        ("blue_detection", "Detection", BLUE_COLOR),
        ("blue_speed", "Speed", "#44ccff"),
        ("blue_graph_reconstruction", "Recon", "#aa44ff"),
        ("blue_fp_penalty", "FP Pen", RED_COLOR),
        ("blue_honeypot_rate", "HP Rate", GREEN_COLOR),
    ]:
        if col in frame.columns:
            fig_blue.add_trace(go.Bar(
                x=episodes, y=frame[col].tolist(),
                name=name, marker_color=color, opacity=0.85,
            ))
    fig_blue.update_layout(
        **PLOTLY_LAYOUT, title="BLUE Reward Components",
        barmode="relative",
        legend=dict(orientation="h", y=1.1, font=dict(size=10)),
    )
    fig_blue.update_xaxes(title="Episode", tickmode="linear", dtick=1)

    total = len(frame)
    red_wins = int((frame["red_total"] > 0).sum())
    blue_wins = int((frame["blue_total"] > 0).sum())
    best_red = frame.loc[frame["red_total"].idxmax()]
    best_blue = frame.loc[frame["blue_total"].idxmax()]
    most_term = frame["terminal_reason"].mode()[0] if "terminal_reason" in frame.columns else "n/a"

    # Live step feed (most recent 8 steps)
    live_steps = _load_live_steps()
    step_rows = []
    for s in reversed(live_steps[-8:]):
        susp_pct = f"{s.get('suspicion', 0):.0%}"
        det_pct = f"{s.get('detection', 0):.0%}"
        exfil = f" ✓{s['exfil_count']}x" if s.get("exfil_count") else ""
        color = RED_COLOR if s.get("suspicion", 0) > 0.7 else TEXT_SECONDARY
        step_rows.append(html.Div(
            f"Ep{s.get('episode','?')} S{s.get('step','?'):02d}/{s.get('max_steps','?')}  "
            f"{s.get('zone','?')[:8]}  "
            f"RED:{s.get('red_action','?')[:20]}  "
            f"BLUE:{s.get('blue_actions','?')[:22]}  "
            f"Susp:{susp_pct}  Det:{det_pct}{exfil}",
            style={"fontSize": "10px", "color": color, "fontFamily": "monospace",
                   "padding": "2px 0", "borderBottom": f"1px solid {BORDER_COLOR}"},
        ))

    live_feed = html.Div([
        html.Div("Live Step Feed", style={"fontSize": "11px", "color": TEXT_MUTED,
                                           "margin": "10px 0 4px", "letterSpacing": "1px"}),
        html.Div(
            step_rows if step_rows else [html.Div("— no live steps yet (stub mode or waiting) —",
                                                   style={"color": TEXT_MUTED, "fontSize": "10px"})],
            style={"background": "#0d0d0d", "padding": "8px", "borderRadius": "4px",
                   "border": f"1px solid {BORDER_COLOR}"},
        ),
    ])

    state_info = _load_training_state()
    run_id_txt = state_info.get("run_id", "")
    mode_txt = state_info.get("llm_mode", "stub").upper()
    stats = html.Div(
        [
            _stat("Run", f"{mode_txt}", color={"LIVE": GREEN_COLOR, "HYBRID": GOLD_COLOR}.get(mode_txt, GRAY_COLOR)),
            _stat("Episodes this run", str(total)),
            _stat("Best RED", f"Ep {int(best_red['episode'])} ({best_red['red_total']:+.3f})"),
            _stat("Best BLUE", f"Ep {int(best_blue['episode'])} ({best_blue['blue_total']:+.3f})"),
            _stat("RED wins", f"{red_wins} ({100 * red_wins // max(total, 1)}%)"),
            _stat("BLUE wins", f"{blue_wins} ({100 * blue_wins // max(total, 1)}%)"),
            _stat("Terminal", str(most_term)),
        ],
        style={"display": "flex", "gap": "16px", "padding": "8px 0", "flexWrap": "wrap"},
    )

    return fig_main, fig_red, fig_blue, stats, live_feed


def update_tab2(_n, filter_val, cached_filter):
    """B1: Only recompute when filter changes; return cached result on interval ticks."""
    # If this is an interval tick (not a filter change), return cached data
    from dash import callback_context
    triggered = [t["prop_id"] for t in callback_context.triggered]
    is_filter_change = any("t2-filter" in t for t in triggered)

    if not is_filter_change and "data" in _TAB2_CACHE and _TAB2_CACHE.get("filter") == filter_val:
        cached = _TAB2_CACHE["data"]
        return cached[0], cached[1], filter_val  # data, stats, new_cache

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
    avg_tokens = np.mean([drop.get("tokens") or 0 for drop in drops]) if drops else 0
    stats = html.Div(
        [
            _stat("Total drops", str(total)),
            _stat("Avg tokens", f"{avg_tokens:.0f}"),
            _stat("Tampered", str(tampered), color=RED_COLOR if tampered > 0 else GREEN_COLOR),
        ],
        style={"display": "flex", "gap": "20px"},
    )
    _TAB2_CACHE["filter"] = filter_val
    _TAB2_CACHE["data"] = (table_data, stats)
    return table_data, stats, filter_val


def update_tab3(_n):
    """B3: Edges cached after first build; only node traces rebuilt each tick."""
    graph, pos = _get_graph_positions()
    events = _load_training_events()
    trap_events = [event for event in events if event.get("event_type") == "trap_fired"]

    # B3: build edge traces once and cache them
    if "edge_traces" not in _GRAPH_CACHE and graph is not None and pos is not None:
        edge_traces = []
        for src, dst in graph.edges():
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None], mode="lines",
                line=dict(color="#1e1e1e", width=0.5),
                hoverinfo="none", showlegend=False,
            ))
        _GRAPH_CACHE["edge_traces"] = edge_traces

    edge_traces = _GRAPH_CACHE.get("edge_traces", [])
    node_traces = []

    if graph is not None and pos is not None:
        zone_colors = {0: GRAY_COLOR, 1: BLUE_COLOR, 2: GOLD_COLOR, 3: RED_COLOR}
        zone_names = {0: "Z0 Perimeter", 1: "Z1 General", 2: "Z2 Sensitive", 3: "Z3 Critical"}

        trap_counts: dict[int, int] = {}
        for event in trap_events:
            node = event.get("node")
            if isinstance(node, int):
                trap_counts[node] = trap_counts.get(node, 0) + 1

        # Latest RED position from live steps
        live_steps = _load_live_steps()
        red_node = None
        if live_steps:
            last = live_steps[-1]
            raw = last.get("red_action", "")
            import re
            m = re.search(r"→ n(\d+)", raw)
            if m:
                red_node = int(m.group(1))

        for zone in range(4):
            zone_nodes = [
                n for n in graph.nodes()
                if int(getattr(graph.nodes[n].get("zone"), "value", graph.nodes[n].get("zone", 0))) == zone
            ]
            if not zone_nodes:
                continue
            xs = [pos[n][0] for n in zone_nodes]
            ys = [pos[n][1] for n in zone_nodes]
            sizes = [10 + 4 * trap_counts.get(n, 0) for n in zone_nodes]
            opacities = [1.0 if n == red_node else 0.8 for n in zone_nodes]
            labels = [
                f"Node {n}<br>Zone {zone} {zone_names[zone]}<br>"
                f"Traps: {trap_counts.get(n, 0)}"
                + (" ← RED HERE" if n == red_node else "")
                for n in zone_nodes
            ]
            node_traces.append(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=sizes, color=zone_colors[zone], opacity=0.82,
                            line=dict(width=2 if any(n == red_node for n in zone_nodes) else 0,
                                      color=RED_COLOR)),
                name=zone_names[zone], text=labels, hoverinfo="text",
            ))

        triggered_nodes = list({
            event.get("node") for event in trap_events
            if isinstance(event.get("node"), int) and event.get("node") in pos
        })
        if triggered_nodes:
            node_traces.append(go.Scatter(
                x=[pos[n][0] for n in triggered_nodes],
                y=[pos[n][1] for n in triggered_nodes],
                mode="markers",
                marker=dict(size=22, color="rgba(255,200,0,0.12)",
                            line=dict(color=GOLD_COLOR, width=2)),
                name="Trap triggered", hoverinfo="skip",
            ))

        if red_node is not None and red_node in pos:
            node_traces.append(go.Scatter(
                x=[pos[red_node][0]], y=[pos[red_node][1]],
                mode="markers+text",
                marker=dict(size=16, color=RED_COLOR, symbol="star",
                            line=dict(color="white", width=1)),
                text=["RED"], textposition="top center",
                textfont=dict(color=RED_COLOR, size=9),
                name="RED position", hoverinfo="skip",
            ))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_layout(
        title="Network Map — Zone Topology & Trap Activity",
        showlegend=True,
        legend=dict(x=1.01, y=1, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

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
    frame = _get_run_frame()
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
    frame = _get_run_frame()
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
            if max(cx) - min(cx) > 1e-8:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
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
            if (len(diff_arr) == len(red_arr) and len(diff_arr) >= 2
                    and np.std(diff_arr) > 1e-8 and np.std(red_arr) > 1e-8):
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
    frame = _get_run_frame()
    if frame is not None:
        analyzer._df = frame.copy()
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

    # ── Middle chart: RED-minus-BLUE gap per episode ──
    fig_gap = go.Figure()
    if frame is not None and not frame.empty:
        red_vals_g = frame["red_total"].tolist()
        blue_vals_g = frame["blue_total"].tolist()
        episodes_g = frame["episode"].tolist()
        gap_vals = [r - b for r, b in zip(red_vals_g, blue_vals_g)]
        gap_ma = _moving_average(gap_vals, window=10)
        fig_gap.add_trace(go.Bar(
            x=episodes_g, y=gap_vals,
            name="RED-BLUE gap (raw)",
            marker_color=[RED_COLOR if v > 0 else BLUE_COLOR for v in gap_vals],
            opacity=0.55,
        ))
        fig_gap.add_trace(go.Scatter(
            x=episodes_g, y=gap_ma,
            mode="lines", name="Gap 10-ep avg",
            line=dict(color=GOLD_COLOR, width=2),
        ))
        fig_gap.add_hline(y=0, line_color="#555", line_dash="dash")
        for ep in evo_eps:
            fig_gap.add_vline(
                x=ep, line_color=GOLD_COLOR, line_width=1.0, line_dash="dot",
            )
    fig_gap.update_layout(
        **PLOTLY_LAYOUT,
        title="RED−BLUE Reward Gap per Episode (positive = RED winning)",
        xaxis_title="Episode",
        yaxis_title="Gap (RED − BLUE)",
        showlegend=True,
        legend=dict(orientation="h", y=1.15, font=dict(size=9)),
    )

    # ── Bottom chart: win rate curves ──
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

    return fig_reward, fig_gap, fig_winrate, stats


def update_agent_status_bar(_n):
    """B4: Render per-agent status bar from logs/agent_status.json."""
    status = _load_agent_status()
    if not status or "agents" not in status:
        return html.Span("— agent status unavailable (stub mode or waiting) —",
                         style={"color": TEXT_MUTED, "fontSize": "10px"})
    agents = status.get("agents", {})
    step = status.get("step", "?")
    ep = status.get("episode", "?")
    zone = status.get("zone", "?")
    susp = status.get("suspicion", 0)
    det = status.get("detection", 0)

    chips = []
    for agent_id, info in agents.items():
        team = info.get("team", "?")
        action = info.get("action", "?")
        node = info.get("node")
        label = f"{agent_id.split('_')[1] if '_' in agent_id else agent_id}: {action}"
        if node is not None:
            label += f"→n{node}"
        color = RED_COLOR if team == "red" else BLUE_COLOR
        chips.append(html.Span(
            label,
            style={"color": color, "fontSize": "10px", "marginRight": "14px",
                   "fontFamily": "monospace", "background": "#111",
                   "padding": "1px 6px", "borderRadius": "3px",
                   "border": f"1px solid {color}33"},
        ))

    return html.Div([
        html.Span(f"Ep{ep} S{step} | {zone} | Susp:{susp:.0%} Det:{det:.0%} | ",
                  style={"color": TEXT_MUTED, "fontSize": "10px", "marginRight": "8px"}),
        *chips,
    ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap"})


def update_tab_logs(_n):
    """Live step-by-step narrative log from live_steps.jsonl."""
    steps = _load_live_steps()
    agent_status = _load_agent_status()

    if not steps:
        feed = [html.Div(
            "No live steps yet. Run with --live or --hybrid to see real-time step narrative.",
            style={"color": TEXT_MUTED, "fontSize": "11px", "padding": "20px"},
        )]
    else:
        feed = []
        # Group by episode
        current_ep = None
        for s in steps:
            ep = s.get("episode", "?")
            step = s.get("step", "?")
            max_s = s.get("max_steps", "?")
            zone = s.get("zone", "?")
            red = s.get("red_action", "—")
            blue = s.get("blue_actions", "—")
            susp = s.get("suspicion", 0)
            det = s.get("detection", 0)
            elapsed = s.get("elapsed", 0)
            exfil = s.get("exfil_count", 0)
            files = s.get("exfil_files", [])
            ts = s.get("timestamp", "")[:19].replace("T", " ")

            if ep != current_ep:
                current_ep = ep
                feed.append(html.Div(
                    f"━━━ Episode {ep} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                    style={"color": GOLD_COLOR, "fontSize": "11px", "margin": "8px 0 4px",
                           "fontWeight": "bold"},
                ))

            susp_bar = "█" * int(susp * 20) + "░" * (20 - int(susp * 20))
            det_bar = "█" * int(det * 20) + "░" * (20 - int(det * 20))
            susp_color = RED_COLOR if susp > 0.7 else (GOLD_COLOR if susp > 0.4 else GREEN_COLOR)

            row_parts = [
                html.Span(f"S{step:02d}/{max_s} ", style={"color": TEXT_MUTED}),
                html.Span(f"[{zone[:10]:10s}] ", style={"color": GRAY_COLOR}),
                html.Span("RED: ", style={"color": RED_COLOR, "fontWeight": "bold"}),
                html.Span(f"{red[:22]:22s} ", style={"color": "#ff8888"}),
                html.Span("│ BLUE: ", style={"color": TEXT_MUTED}),
                html.Span(f"{blue[:28]:28s} ", style={"color": "#88aaff"}),
                html.Span("│ Susp:", style={"color": TEXT_MUTED}),
                html.Span(f"{susp:.0%}", style={"color": susp_color, "fontWeight": "bold"}),
                html.Span(f" Det:{det:.0%}", style={"color": BLUE_COLOR}),
                html.Span(f" {elapsed:.0f}s", style={"color": TEXT_MUTED}),
            ]
            if exfil:
                row_parts.append(html.Span(
                    f" ✓ EXFIL: {', '.join(files[:2])}",
                    style={"color": GREEN_COLOR, "fontWeight": "bold"},
                ))

            feed.append(html.Div(
                row_parts,
                style={"padding": "2px 0", "borderBottom": f"1px solid #111",
                       "lineHeight": "1.6"},
            ))

    # Agent status at top of log tab
    agent_bar = update_agent_status_bar(0)
    return agent_bar, html.Div(list(reversed(feed)),  # newest on top
                               style={"fontFamily": "monospace", "fontSize": "11px"})


def update_tab_history(_n):
    """B5: Cross-run episode history from SQLite."""
    frame = _load_db_all_frame(n=200)
    empty = go.Figure().update_layout(**PLOTLY_LAYOUT)

    if frame is None or frame.empty:
        return empty, empty, html.Div("No history in SQLite yet — run at least one episode.",
                                      style={"color": TEXT_MUTED}), []

    for col in ("red_total", "blue_total"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "episode" in frame.columns:
        frame["episode"] = pd.to_numeric(frame["episode"], errors="coerce")

    # Rewards chart: RED/BLUE grouped by run
    fig_rewards = go.Figure()
    if "run_id" in frame.columns:
        for run_id_val in frame["run_id"].unique():
            sub = frame[frame["run_id"] == run_id_val]
            mode_label = sub["llm_mode"].iloc[0].upper() if "llm_mode" in sub.columns else ""
            short_id = str(run_id_val)[-8:] if run_id_val else "?"
            label = f"{mode_label}/{short_id}"
            fig_rewards.add_trace(go.Scatter(
                x=list(range(1, len(sub) + 1)),
                y=sub["red_total"].tolist(),
                mode="lines+markers", name=f"RED {label}",
                line=dict(color=RED_COLOR, width=1.5), marker=dict(size=4),
                opacity=0.7,
            ))
            fig_rewards.add_trace(go.Scatter(
                x=list(range(1, len(sub) + 1)),
                y=sub["blue_total"].tolist(),
                mode="lines+markers", name=f"BLUE {label}",
                line=dict(color=BLUE_COLOR, width=1.5, dash="dot"), marker=dict(size=4),
                opacity=0.7,
            ))
    else:
        fig_rewards.add_trace(go.Bar(x=frame["episode"].tolist(), y=frame["red_total"].tolist(),
                                     name="RED", marker_color=RED_COLOR, opacity=0.75))
        fig_rewards.add_trace(go.Bar(x=frame["episode"].tolist(), y=frame["blue_total"].tolist(),
                                     name="BLUE", marker_color=BLUE_COLOR, opacity=0.75))
    fig_rewards.update_layout(**PLOTLY_LAYOUT, title="All Runs: RED vs BLUE Rewards",
                               barmode="group", legend=dict(orientation="h", y=1.1, font=dict(size=9)))

    # Outcomes chart: terminal reason bar
    fig_outcomes = go.Figure()
    if "terminal_reason" in frame.columns:
        counts = frame["terminal_reason"].value_counts()
        outcome_colors = {
            "exfil_success": GREEN_COLOR, "detected": BLUE_COLOR,
            "aborted": GOLD_COLOR, "max_steps": GRAY_COLOR,
        }
        fig_outcomes.add_trace(go.Bar(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            marker_color=[outcome_colors.get(k, GRAY_COLOR) for k in counts.index],
            opacity=0.85,
        ))
    fig_outcomes.update_layout(**PLOTLY_LAYOUT, title="Terminal Reason Distribution",
                                xaxis_title="Outcome", yaxis_title="Count",
                                showlegend=False)

    # Run selector info
    run_selector = html.Div([
        html.Span(f"Total runs: {frame['run_id'].nunique() if 'run_id' in frame.columns else '?'}  |  ",
                  style={"color": TEXT_MUTED, "fontSize": "11px"}),
        html.Span(f"Total episodes: {len(frame)}",
                  style={"color": TEXT_SECONDARY, "fontSize": "11px"}),
    ], style={"margin": "4px 0"})

    # Table: last 30 episodes
    show_cols = ["run_id", "llm_mode", "episode", "steps", "terminal_reason",
                 "red_total", "blue_total", "fleet_verdict"]
    disp = frame[[c for c in show_cols if c in frame.columns]].tail(30).copy()
    if "run_id" in disp.columns:
        disp["run_id"] = disp["run_id"].astype(str).str[-14:]
    for col in ("red_total", "blue_total"):
        if col in disp.columns:
            disp[col] = disp[col].apply(lambda x: f"{float(x):+.3f}" if pd.notna(x) else "?")
    table_data = disp.fillna("?").to_dict("records")

    return fig_rewards, fig_outcomes, run_selector, table_data


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
            Output("header-run-id", "children"),
            Output("header-cost", "children"),
            Output("header-uptime", "children"),
            Output("live-indicator", "style"),
        ],
        Input("interval-component", "n_intervals"),
    )
    def _header(n):
        return update_header(n)

    # B4: agent status bar on fast 1.5s interval
    @target_app.callback(
        Output("agent-status-bar", "children"),
        Input("interval-fast", "n_intervals"),
    )
    def _agent_bar(n):
        return update_agent_status_bar(n)

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
            Output("t1-live-feed", "children"),
        ],
        Input("interval-component", "n_intervals"),
        prevent_initial_call=True,
    )
    def _tab1(n):
        return update_tab1(n)

    # B4 + Live Logs: fast interval for live logs tab
    @target_app.callback(
        [
            Output("t-logs-agent-bar", "children"),
            Output("t-logs-feed", "children"),
        ],
        Input("interval-fast", "n_intervals"),
        prevent_initial_call=True,
    )
    def _tab_logs(n):
        return update_tab_logs(n)

    # B1: Tab 2 with Store caching — 3 outputs now (data, stats, cache update)
    @target_app.callback(
        [
            Output("t2-table", "data"),
            Output("t2-stats", "children"),
            Output("t2-filter-cache", "data"),
        ],
        [
            Input("interval-component", "n_intervals"),
            Input("t2-filter", "value"),
        ],
        State("t2-filter-cache", "data"),
        prevent_initial_call=True,
    )
    def _tab2(n, filter_val, cached_filter):
        return update_tab2(n, filter_val, cached_filter)

    @target_app.callback(
        [
            Output("t3-map", "figure"),
            Output("t3-events", "children"),
            Output("t3-stats", "children"),
        ],
        Input("interval-component", "n_intervals"),
        prevent_initial_call=True,
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
        prevent_initial_call=True,
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
        prevent_initial_call=True,
    )
    def _tab5(n):
        return update_tab5(n)

    @target_app.callback(
        [
            Output("t6-reward-chart", "figure"),
            Output("t6-gap-chart", "figure"),
            Output("t6-winrate-chart", "figure"),
            Output("t6-stats", "children"),
        ],
        Input("interval-component", "n_intervals"),
        prevent_initial_call=True,
    )
    def _tab6(n):
        return update_tab6(n)

    # B5: Episode History tab
    @target_app.callback(
        [
            Output("th-rewards-chart", "figure"),
            Output("th-outcomes-chart", "figure"),
            Output("th-run-selector", "children"),
            Output("th-table", "data"),
        ],
        Input("interval-component", "n_intervals"),
        prevent_initial_call=True,
    )
    def _tab_history(n):
        return update_tab_history(n)


# Wire callbacks onto the module-level app at import time.
app.layout = create_live_layout()
register_callbacks_on(app)


if __name__ == "__main__":
    app.run(debug=False, port=8051)
