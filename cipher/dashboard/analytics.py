"""
cipher/dashboard/analytics.py

CIPHER Analytics Engine — Elo ratings, detection heatmaps, reward curves.

Provides:
  - compute_elo()          : Elo tracking across episodes / team comparisons
  - build_detection_heatmap() : Which nodes are "Death Traps" (high detection)
  - build_reward_curves()  : Reward vs Step / Episode for RED and BLUE
  - build_elo_chart()      : Elo rating over time chart
  - load_rewards_df()      : Safe CSV loader returning a pandas DataFrame
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Colour palette (matches live.py DARK theme) ────────────────────────────
RED_COLOR = "#ff4444"
BLUE_COLOR = "#4488ff"
GOLD_COLOR = "#ffaa00"
GREEN_COLOR = "#44cc88"
PURPLE_COLOR = "#a855f7"
PANEL_BG = "#111111"
DARK_BG = "#0a0a0a"
BORDER_COLOR = "#2a2a2a"
TEXT_SECONDARY = "#aaaaaa"

_PLOTLY_BASE = dict(
    paper_bgcolor=PANEL_BG,
    plot_bgcolor=DARK_BG,
    font=dict(color=TEXT_SECONDARY, size=11),
    margin=dict(l=50, r=20, t=35, b=40),
    xaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#333"),
    yaxis=dict(gridcolor="#1a1a1a", zerolinecolor="#333"),
)

REWARDS_CSV = Path("rewards_log.csv")
EVENTS_JSONL = Path("training_events.jsonl")

# ── Elo constants ─────────────────────────────────────────────────────────
_ELO_K = 32
_ELO_BASE = 1500


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────


def load_rewards_df() -> pd.DataFrame:
    """Return rewards_log.csv as a DataFrame, or an empty one on failure."""
    if not REWARDS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(REWARDS_CSV)
        # Deduplicate: keep the first row per episode (same episode can appear
        # multiple times if multiple runs wrote to the same file).
        if "episode" in df.columns:
            df = df.drop_duplicates(subset=["episode"], keep="first")
        return df
    except Exception:
        return pd.DataFrame()


def load_events() -> list[dict]:
    """Return training_events.jsonl as a list of dicts."""
    if not EVENTS_JSONL.exists():
        return []
    events: list[dict] = []
    try:
        for line in EVENTS_JSONL.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return events


# ─────────────────────────────────────────────────────────────────────────────
# Elo rating system
# ─────────────────────────────────────────────────────────────────────────────


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo(
    df: pd.DataFrame | None = None,
    team_a_label: str = "RED (HF)",
    team_b_label: str = "BLUE (Defender)",
) -> dict[str, Any]:
    """
    Compute Elo ratings for RED vs BLUE across all episodes.

    Returns dict with:
      elo_red      : list of RED Elo values (one per episode)
      elo_blue     : list of BLUE Elo values (one per episode)
      episodes     : list of episode numbers
      final_red    : final RED Elo
      final_blue   : final BLUE Elo
    """
    if df is None:
        df = pd.DataFrame()

    if df.empty or "red_total" not in df.columns:
        return {"elo_red": [], "elo_blue": [], "episodes": [], "final_red": _ELO_BASE, "final_blue": _ELO_BASE}

    elo_red = _ELO_BASE
    elo_blue = _ELO_BASE
    elo_red_hist: list[float] = []
    elo_blue_hist: list[float] = []
    eps: list[int] = []

    for _, row in df.iterrows():
        red_r = float(row.get("red_total", 0))
        blue_r = float(row.get("blue_total", 0))

        # Determine outcome from rewards + terminal_reason
        reason = str(row.get("terminal_reason", "")).lower()
        if "exfil" in reason or red_r > blue_r + 0.1:
            score_red, score_blue = 1.0, 0.0
        elif "detect" in reason or blue_r > red_r + 0.1:
            score_red, score_blue = 0.0, 1.0
        else:
            score_red, score_blue = 0.5, 0.5

        exp_red = _expected_score(elo_red, elo_blue)
        exp_blue = 1.0 - exp_red

        elo_red += _ELO_K * (score_red - exp_red)
        elo_blue += _ELO_K * (score_blue - exp_blue)

        elo_red_hist.append(round(elo_red, 1))
        elo_blue_hist.append(round(elo_blue, 1))
        eps.append(int(row.get("episode", len(eps) + 1)))

    return {
        "elo_red": elo_red_hist,
        "elo_blue": elo_blue_hist,
        "episodes": eps,
        "final_red": round(elo_red, 1),
        "final_blue": round(elo_blue, 1),
        "team_a_label": team_a_label,
        "team_b_label": team_b_label,
    }


def build_elo_chart(df: pd.DataFrame | None = None) -> go.Figure:
    """Plotly figure showing Elo rating trajectory for RED and BLUE."""
    data = compute_elo(df)
    eps = data["episodes"]

    fig = go.Figure()

    if eps:
        fig.add_trace(go.Scatter(
            x=eps, y=data["elo_red"],
            name=f"RED Elo (final {data['final_red']})",
            line=dict(color=RED_COLOR, width=2.5),
            mode="lines+markers",
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=eps, y=data["elo_blue"],
            name=f"BLUE Elo (final {data['final_blue']})",
            line=dict(color=BLUE_COLOR, width=2.5),
            mode="lines+markers",
            marker=dict(size=5),
        ))
        # Baseline
        fig.add_hline(y=_ELO_BASE, line=dict(color="#333", dash="dot"), annotation_text="Baseline 1500")

    fig.update_layout(
        title=dict(text="Elo Rating — RED vs BLUE", font=dict(size=13, color=TEXT_SECONDARY)),
        xaxis_title="Episode",
        yaxis_title="Elo Rating",
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
        **_PLOTLY_BASE,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Detection heatmap
# ─────────────────────────────────────────────────────────────────────────────


def _extract_node_detection_scores(events: list[dict] | None = None) -> dict[int, float]:
    """
    Aggregate detection/trap events per node from training_events.jsonl.

    Returns dict: node_id → detection_weight (higher = more dangerous).
    """
    if events is None:
        events = load_events()

    node_counts: dict[int, float] = {}
    for ev in events:
        node = ev.get("node")
        if node is None:
            continue
        try:
            nid = int(node)
        except (TypeError, ValueError):
            continue

        etype = str(ev.get("event_type", ""))
        if "detection" in etype or "exfil" in etype:
            node_counts[nid] = node_counts.get(nid, 0.0) + 1.5
        elif "trap" in etype or "honeypot" in etype:
            node_counts[nid] = node_counts.get(nid, 0.0) + 1.0
        elif "dead_drop" in etype:
            node_counts[nid] = node_counts.get(nid, 0.0) + 0.5

    return node_counts


def build_detection_heatmap(n_nodes: int = 50, events: list[dict] | None = None) -> go.Figure:
    """
    Create a 50-node detection heatmap grid.

    Nodes with high detection / trap activity appear bright red ("Death Traps").
    """
    node_scores = _extract_node_detection_scores(events)

    # Arrange nodes in a 5×10 grid
    cols = 10
    rows = (n_nodes + cols - 1) // cols
    z_matrix: list[list[float]] = []
    text_matrix: list[list[str]] = []

    for r in range(rows):
        z_row: list[float] = []
        t_row: list[str] = []
        for c in range(cols):
            nid = r * cols + c
            if nid < n_nodes:
                score = node_scores.get(nid, 0.0)
                z_row.append(score)
                t_row.append(f"n{nid}: {score:.1f}")
            else:
                z_row.append(0.0)
                t_row.append("")
        z_matrix.append(z_row)
        text_matrix.append(t_row)

    # Zone labels for y-axis (rows map roughly to zones)
    y_labels = [f"Row {r}" for r in range(rows)]

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=8, color="white"),
        colorscale=[
            [0.0, "#0a0e1a"],
            [0.3, "#1e3a5f"],
            [0.6, "#c0392b"],
            [1.0, "#ff0000"],
        ],
        showscale=True,
        colorbar=dict(
            title="Risk Score",
            titleside="right",
            thickness=10,
            tickfont=dict(color=TEXT_SECONDARY, size=9),
            titlefont=dict(color=TEXT_SECONDARY, size=10),
        ),
        hovertemplate="Node %{text}<br>Risk: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text="Network Death Trap Heatmap — Node Risk by Detection Activity",
                   font=dict(size=12, color=TEXT_SECONDARY)),
        xaxis=dict(title="Node Column", showgrid=False, tickfont=dict(size=9, color=TEXT_SECONDARY)),
        yaxis=dict(title="Node Row", showgrid=False, ticktext=y_labels, tickvals=list(range(rows)),
                   tickfont=dict(size=9, color=TEXT_SECONDARY)),
        paper_bgcolor=PANEL_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT_SECONDARY, size=11),
        margin=dict(l=60, r=80, t=50, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Reward curve visualization
# ─────────────────────────────────────────────────────────────────────────────


def build_reward_curves(df: pd.DataFrame | None = None) -> go.Figure:
    """
    Build a Reward vs Episode line chart for RED and BLUE with moving averages.
    """
    if df is None:
        df = pd.DataFrame()

    fig = go.Figure()

    if df.empty or "red_total" not in df.columns:
        fig.update_layout(
            title="Reward Curves (no data yet)",
            **_PLOTLY_BASE,
        )
        return fig

    eps = list(range(1, len(df) + 1))
    red_vals = df["red_total"].tolist()
    blue_vals = df["blue_total"].tolist()

    # Raw traces (faded)
    fig.add_trace(go.Scatter(
        x=eps, y=red_vals, name="RED raw",
        line=dict(color=RED_COLOR, width=1, dash="dot"),
        opacity=0.4, mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=eps, y=blue_vals, name="BLUE raw",
        line=dict(color=BLUE_COLOR, width=1, dash="dot"),
        opacity=0.4, mode="lines",
    ))

    # 5-episode moving averages (bold)
    window = min(5, max(1, len(eps) // 4))
    if len(eps) >= window:
        red_ma = pd.Series(red_vals).rolling(window, min_periods=1).mean().tolist()
        blue_ma = pd.Series(blue_vals).rolling(window, min_periods=1).mean().tolist()
        fig.add_trace(go.Scatter(
            x=eps, y=red_ma, name=f"RED ({window}-ep avg)",
            line=dict(color=RED_COLOR, width=2.5), mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=eps, y=blue_ma, name=f"BLUE ({window}-ep avg)",
            line=dict(color=BLUE_COLOR, width=2.5), mode="lines",
        ))

    fig.add_hline(y=0, line=dict(color="#444", dash="dot"))

    fig.update_layout(
        title=dict(text="Reward vs Episode — RED Attack vs BLUE Defense", font=dict(size=13, color=TEXT_SECONDARY)),
        xaxis_title="Episode",
        yaxis_title="Total Reward",
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
        **_PLOTLY_BASE,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# "Winning Metrics" top-bar summary
# ─────────────────────────────────────────────────────────────────────────────


def compute_winning_metrics(df: pd.DataFrame | None = None) -> dict[str, Any]:
    """
    Return a dict of high-level KPIs for the "Winning Metrics" top banner.

    Keys:
      total_exfils        : int  — episodes where RED exfiltrated
      mean_ttd            : float — mean steps to detection (BLUE wins)
      efficiency_pct      : float — % change in RED reward from first→last 5 episodes
      model_confidence    : str  — "HIGH" / "MED" / "LOW"
      red_win_rate        : float — fraction of RED wins
      blue_win_rate       : float — fraction of BLUE wins
      n_episodes          : int
    """
    if df is None:
        df = pd.DataFrame()

    if df.empty or "red_total" not in df.columns:
        return {
            "total_exfils": 0,
            "mean_ttd": 0.0,
            "efficiency_pct": 0.0,
            "model_confidence": "N/A",
            "red_win_rate": 0.0,
            "blue_win_rate": 0.0,
            "n_episodes": 0,
        }

    n = len(df)
    reasons = df.get("terminal_reason", pd.Series([""] * n)).fillna("").str.lower()

    # Exfiltrations (episodes where RED exfiltrated)
    exfil_mask = reasons.str.contains("exfil") | (df["red_total"] > 0.8)
    total_exfils = int(exfil_mask.sum())

    # Detection episodes for mean time-to-detect
    detect_mask = reasons.str.contains("detect")
    if "steps" in df.columns:
        ttd_steps = df.loc[detect_mask, "steps"]
    else:
        ttd_steps = pd.Series(dtype=float)
    mean_ttd = float(ttd_steps.mean()) if not ttd_steps.empty else 0.0

    # Training efficiency: RED reward improvement first 5 vs last 5 episodes
    batch = max(1, min(5, n // 2))
    early_avg = df["red_total"].iloc[:batch].mean()
    late_avg = df["red_total"].iloc[-batch:].mean()
    if abs(early_avg) > 1e-9:
        efficiency_pct = (late_avg - early_avg) / abs(early_avg) * 100.0
    else:
        efficiency_pct = 0.0

    # Model confidence: mean BLUE detection across last 10 episodes
    if "blue_detection" in df.columns:
        recent_det = df["blue_detection"].iloc[-10:].mean()
    else:
        recent_det = df["blue_total"].iloc[-10:].mean()

    if recent_det > 0.65:
        model_confidence = "HIGH"
    elif recent_det > 0.35:
        model_confidence = "MED"
    else:
        model_confidence = "LOW"

    red_wins = int(exfil_mask.sum())
    blue_wins = int(detect_mask.sum())

    return {
        "total_exfils": total_exfils,
        "mean_ttd": round(mean_ttd, 1),
        "efficiency_pct": round(efficiency_pct, 1),
        "model_confidence": model_confidence,
        "red_win_rate": round(red_wins / max(n, 1), 2),
        "blue_win_rate": round(blue_wins / max(n, 1), 2),
        "n_episodes": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# E.md Change 2 — Model Comparison Chart
# ─────────────────────────────────────────────────────────────────────────────

_MODE_COLORS = {"stub": "#6366f1", "live": "#22d3ee", "hybrid": "#f59e0b", "full_lora": "#10b981"}


def _empty_comparison(msg: str = "No eval data yet") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16, color="#94a3b8"))
    fig.update_layout(**_PLOTLY_BASE, height=300)
    return fig


def find_latest_eval_json(eval_dir: str = "eval_results") -> str | None:
    """Return the path to the most recent comparison_*.json, or None."""
    import glob
    files = sorted(glob.glob(f"{eval_dir}/comparison_*.json"), reverse=True)
    return files[0] if files else None


def build_model_comparison_chart(eval_json_path: str | None = None) -> tuple[go.Figure, go.Figure, go.Figure]:
    """
    E.md Change 2 — Three comparison figures from eval_runner output.

    Returns:
        (win_rate_fig, steps_fig, reward_violin_fig)
    """
    import json, pathlib

    if eval_json_path is None:
        eval_json_path = find_latest_eval_json()
    if eval_json_path is None or not pathlib.Path(eval_json_path).exists():
        empty = _empty_comparison("Run `python main.py --eval 20` to generate data")
        return empty, empty, empty

    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    modes = list(summary.keys())
    if not modes:
        empty = _empty_comparison("Empty eval JSON")
        return empty, empty, empty

    # ── Win Rate bar chart ────────────────────────────────────────────────
    wr_fig = go.Figure()
    wr_fig.add_trace(go.Bar(
        x=[m.upper() for m in modes],
        y=[summary[m].get("red_win_rate", 0) * 100 for m in modes],
        marker_color=[_MODE_COLORS.get(m, "#94a3b8") for m in modes],
        text=[f"{summary[m].get('red_win_rate', 0)*100:.1f}%" for m in modes],
        textposition="outside",
        name="RED Win Rate",
    ))
    wr_fig.update_layout(
        **_PLOTLY_BASE,
        title="RED Win Rate by Mode",
        yaxis_title="Win Rate (%)",
        yaxis_range=[0, 110],
        height=320,
    )

    # ── Avg Steps bar chart ───────────────────────────────────────────────
    steps_fig = go.Figure()
    steps_fig.add_trace(go.Bar(
        x=[m.upper() for m in modes],
        y=[summary[m].get("avg_steps", 0) for m in modes],
        marker_color=[_MODE_COLORS.get(m, "#94a3b8") for m in modes],
        text=[f"{summary[m].get('avg_steps', 0):.1f}" for m in modes],
        textposition="outside",
        name="Avg Steps",
    ))
    steps_fig.update_layout(
        **_PLOTLY_BASE,
        title="Avg Episode Length (shorter = RED more efficient)",
        yaxis_title="Steps",
        height=320,
    )

    # ── RED Reward violin / box ───────────────────────────────────────────
    violin_fig = go.Figure()
    episodes = data.get("episodes", {})
    for mode in modes:
        eps = episodes.get(mode, [])
        rewards = [e.get("red_total", 0) for e in eps]
        if rewards:
            violin_fig.add_trace(go.Violin(
                y=rewards,
                name=mode.upper(),
                box_visible=True,
                meanline_visible=True,
                fillcolor=_MODE_COLORS.get(mode, "#94a3b8"),
                line_color=_MODE_COLORS.get(mode, "#94a3b8"),
                opacity=0.7,
            ))
    if not violin_fig.data:
        for mode in modes:
            violin_fig.add_trace(go.Bar(
                x=[mode.upper()],
                y=[summary[mode].get("avg_red", 0)],
                marker_color=_MODE_COLORS.get(mode, "#94a3b8"),
                name=mode.upper(),
            ))
    violin_fig.update_layout(
        **_PLOTLY_BASE,
        title="RED Reward Distribution by Mode",
        yaxis_title="RED Total Reward",
        height=340,
    )

    return wr_fig, steps_fig, violin_fig


# ─────────────────────────────────────────────────────────────────────────────
# E.md Change 3 — Convergence Curve
# ─────────────────────────────────────────────────────────────────────────────

def _infer_lora_checkpoints(rewards_csv: "pd.DataFrame") -> list[int]:
    """Heuristic: find episodes where reward jumped sharply (LoRA loaded)."""
    if "red_total" not in rewards_csv.columns or len(rewards_csv) < 5:
        return []
    r = rewards_csv["red_total"].rolling(3).mean().fillna(0)
    deltas = r.diff().fillna(0)
    threshold = deltas.std() * 2.0
    return list(rewards_csv.index[deltas > threshold])


def build_convergence_curve(rewards_csv_path: str = "rewards_log.csv") -> go.Figure:
    """
    E.md Change 3 — Rolling RED reward convergence curve with LoRA checkpoint lines.

    X: episode number | Y: rolling-10 avg RED reward
    Vertical lines: estimated LoRA checkpoint injection points
    """
    import pathlib

    fig = go.Figure()

    if not pathlib.Path(rewards_csv_path).exists():
        return _empty_comparison(f"No rewards log at {rewards_csv_path}")

    try:
        df = pd.read_csv(rewards_csv_path)
    except Exception as e:
        return _empty_comparison(f"Failed to read rewards CSV: {e}")

    if df.empty or "red_total" not in df.columns:
        return _empty_comparison("rewards_log.csv has no red_total column")

    df = df.reset_index(drop=True)
    df["episode"] = df.get("episode", pd.Series(range(1, len(df) + 1)))
    df["rolling_red"] = df["red_total"].rolling(10, min_periods=1).mean()

    # Raw scatter
    fig.add_trace(go.Scatter(
        x=df["episode"], y=df["red_total"],
        mode="markers",
        marker=dict(color="#6366f1", size=4, opacity=0.4),
        name="Raw RED reward",
    ))

    # Rolling average
    fig.add_trace(go.Scatter(
        x=df["episode"], y=df["rolling_red"],
        mode="lines",
        line=dict(color="#f59e0b", width=2.5),
        name="Rolling avg (10 ep)",
    ))

    # LoRA checkpoint vertical lines
    checkpoints = _infer_lora_checkpoints(df)
    for cp in checkpoints:
        ep = df["episode"].iloc[cp] if cp < len(df) else cp
        fig.add_vline(
            x=ep,
            line_dash="dash",
            line_color="#10b981",
            annotation_text="LoRA loaded",
            annotation_position="top right",
            annotation_font_color="#10b981",
        )

    fig.update_layout(
        **_PLOTLY_BASE,
        title="RED Reward Convergence Curve",
        xaxis_title="Training Episode",
        yaxis_title="RED Reward",
        height=380,
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig

