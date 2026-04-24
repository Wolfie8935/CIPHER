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
    team_a_label: str = "RED (NIM)",
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

# Colour palette for modes (stub → dim, live → blue, hybrid → gold/green)
_MODE_COLORS: dict[str, str] = {
    "stub":   "#888888",
    "live":   "#4488ff",
    "hybrid": "#ffaa00",
    "full":   "#44cc88",
}
_FALLBACK_COLORS = ["#cc88ff", "#ff8844", "#44ffcc"]


def _mode_color(mode: str, idx: int = 0) -> str:
    return _MODE_COLORS.get(mode.lower(), _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])


def build_model_comparison_chart(eval_json_path: str | Path) -> tuple[go.Figure, go.Figure, go.Figure]:
    """
    Read the eval_runner comparison JSON and return three Plotly figures:

    1. Grouped bar chart — RED win rate per mode
    2. Grouped bar chart — Mean episode length per mode (shorter = more efficient RED)
    3. Box/violin plot  — RED reward distribution per mode

    Args:
        eval_json_path: Path to ``eval_results/comparison_TIMESTAMP.json``.

    Returns:
        (fig_win_rate, fig_steps, fig_reward_dist)
    """
    path = Path(eval_json_path)
    if not path.exists():
        empty = _empty_comparison_figure("No eval data — run:  python main.py --eval 20")
        return empty, empty, empty

    try:
        import json as _json
        data = _json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        empty = _empty_comparison_figure(f"Error loading eval JSON: {exc}")
        return empty, empty, empty

    summary = data.get("summary", {})
    modes   = list(summary.keys())
    if not modes:
        empty = _empty_comparison_figure("Eval JSON has no summary — re-run eval_runner.")
        return empty, empty, empty

    colors  = [_mode_color(m, i) for i, m in enumerate(modes)]

    # ── Figure 1: RED Win Rate (grouped bar) ─────────────────────────────
    win_rates  = [summary.get(m, {}).get("red_win_rate", 0.0) * 100 for m in modes]
    fig_wr = go.Figure(go.Bar(
        x=[m.upper() for m in modes],
        y=win_rates,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in win_rates],
        textposition="outside",
        hovertemplate="Mode: %{x}<br>RED Win Rate: %{y:.1f}%<extra></extra>",
    ))
    # Baseline reference line (first mode)
    if len(win_rates) > 1:
        fig_wr.add_hline(
            y=win_rates[0], line=dict(color="#555", dash="dot"),
            annotation_text=f"Baseline ({modes[0].upper()})",
            annotation_font=dict(size=9, color=TEXT_SECONDARY),
        )
    fig_wr.update_layout(
        title=dict(
            text="RED Win Rate by Mode  —  Higher is Better for RED Team",
            font=dict(size=13, color=TEXT_SECONDARY),
        ),
        xaxis_title="LLM Mode",
        yaxis_title="RED Win Rate (%)",
        yaxis=dict(range=[0, min(100, max(win_rates or [0]) * 1.3 + 10)],
                   gridcolor="#1a1a1a", zerolinecolor="#333"),
        showlegend=False,
        **_PLOTLY_BASE,
    )

    # ── Figure 2: Mean Steps per Episode (grouped bar) ────────────────────
    avg_steps = [summary.get(m, {}).get("avg_steps", 0.0) for m in modes]
    fig_steps = go.Figure(go.Bar(
        x=[m.upper() for m in modes],
        y=avg_steps,
        marker_color=colors,
        text=[f"{v:.1f}" for v in avg_steps],
        textposition="outside",
        hovertemplate="Mode: %{x}<br>Avg Steps: %{y:.1f}<extra></extra>",
    ))
    if len(avg_steps) > 1:
        fig_steps.add_hline(
            y=avg_steps[0], line=dict(color="#555", dash="dot"),
            annotation_text=f"Baseline ({modes[0].upper()})",
            annotation_font=dict(size=9, color=TEXT_SECONDARY),
        )
    fig_steps.update_layout(
        title=dict(
            text="Mean Episode Length by Mode  —  Shorter = More Efficient RED",
            font=dict(size=13, color=TEXT_SECONDARY),
        ),
        xaxis_title="LLM Mode",
        yaxis_title="Avg Steps",
        showlegend=False,
        **_PLOTLY_BASE,
    )

    # ── Figure 3: RED Reward Distribution (violin / box) ─────────────────
    all_rows: dict[str, list[float]] = {}
    for mode_key, rows in data.get("modes", {}).items():
        all_rows[mode_key] = [float(r.get("red_total", 0.0)) for r in rows if isinstance(r, dict)]

    fig_dist = go.Figure()
    for i, m in enumerate(modes):
        vals = all_rows.get(m, [])
        if vals:
            fig_dist.add_trace(go.Violin(
                y=vals,
                name=m.upper(),
                box_visible=True,
                meanline_visible=True,
                line_color=_mode_color(m, i),
                fillcolor=_mode_color(m, i),
                opacity=0.6,
            ))
        else:
            # Fallback to a single-point scatter if no row data
            avg = summary.get(m, {}).get("avg_red", 0.0)
            fig_dist.add_trace(go.Bar(
                x=[m.upper()], y=[avg],
                marker_color=_mode_color(m, i),
                name=m.upper(),
            ))
    fig_dist.add_hline(y=0, line=dict(color="#444", dash="dot"))
    fig_dist.update_layout(
        title=dict(
            text="RED Reward Distribution by Mode",
            font=dict(size=13, color=TEXT_SECONDARY),
        ),
        xaxis_title="LLM Mode",
        yaxis_title="RED Total Reward",
        violinmode="overlay",
        **_PLOTLY_BASE,
    )

    return fig_wr, fig_steps, fig_dist


def _empty_comparison_figure(msg: str) -> go.Figure:
    """Return a minimal blank figure with a centred annotation."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=12, color=TEXT_SECONDARY),
    )
    fig.update_layout(**_PLOTLY_BASE)
    return fig


def find_latest_eval_json() -> Path | None:
    """Return the most recent comparison JSON in eval_results/, or None."""
    try:
        from pathlib import Path as _P
        results_dir = _P(__file__).resolve().parent.parent.parent / "eval_results"
        jsons = sorted(results_dir.glob("comparison_*.json"), reverse=True)
        return jsons[0] if jsons else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# E.md Change 3 — Convergence Curve
# ─────────────────────────────────────────────────────────────────────────────


def build_convergence_curve(
    rewards_csv_path: str | Path | None = None,
    window: int = 10,
    lora_checkpoint_episodes: list[int] | None = None,
) -> go.Figure:
    """
    Plot RED reward convergence curve with a rolling average and optional
    vertical lines marking where LoRA checkpoints were created.

    Args:
        rewards_csv_path: Path to ``rewards_log.csv``.
                          Defaults to the project-root rewards_log.csv.
        window:           Rolling window size (default 10).
        lora_checkpoint_episodes: List of episode numbers where a LoRA
                          checkpoint was saved.  If None, the function tries
                          to infer them from ``training_events.jsonl``.

    Returns:
        A Plotly Figure.
    """
    # ── Load data ─────────────────────────────────────────────────────────
    if rewards_csv_path is None:
        csv_path = REWARDS_CSV
    else:
        csv_path = Path(rewards_csv_path)

    df = load_rewards_df() if csv_path == REWARDS_CSV else _load_csv_safe(csv_path)

    fig = go.Figure()

    if df.empty or "red_total" not in df.columns:
        fig.add_annotation(
            text="No reward data yet — run training first.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=12, color=TEXT_SECONDARY),
        )
        fig.update_layout(
            title=dict(text="RED Reward Convergence Curve (no data)", font=dict(size=13, color=TEXT_SECONDARY)),
            **_PLOTLY_BASE,
        )
        return fig

    episodes  = list(range(1, len(df) + 1))
    red_vals  = df["red_total"].tolist()

    # Rolling average
    w = min(window, max(1, len(episodes) // 4))
    red_ma = pd.Series(red_vals).rolling(w, min_periods=1).mean().tolist()

    # ── Raw reward (faded) ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=episodes, y=red_vals,
        name="RED reward (raw)",
        line=dict(color=RED_COLOR, width=1, dash="dot"),
        opacity=0.35,
        mode="lines",
    ))

    # ── Rolling average (prominent) ───────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=episodes, y=red_ma,
        name=f"RED {w}-ep rolling avg",
        line=dict(color=RED_COLOR, width=2.5),
        mode="lines",
    ))

    # ── BLUE moving average (context) ────────────────────────────────────
    if "blue_total" in df.columns:
        blue_vals = df["blue_total"].tolist()
        blue_ma   = pd.Series(blue_vals).rolling(w, min_periods=1).mean().tolist()
        fig.add_trace(go.Scatter(
            x=episodes, y=blue_ma,
            name=f"BLUE {w}-ep rolling avg",
            line=dict(color=BLUE_COLOR, width=1.5, dash="dot"),
            mode="lines",
            opacity=0.6,
        ))

    fig.add_hline(y=0, line=dict(color="#444", dash="dot"))

    # ── LoRA checkpoint vertical lines ────────────────────────────────────
    checkpoints = lora_checkpoint_episodes or _infer_lora_checkpoints()
    for ep in checkpoints:
        fig.add_vline(
            x=ep,
            line=dict(color=GOLD_COLOR, width=1.5, dash="dash"),
            annotation_text=f"LoRA ckpt @{ep}",
            annotation_font=dict(size=9, color=GOLD_COLOR),
        )

    # ── Trend annotation: first vs last 10% ──────────────────────────────
    n = len(red_vals)
    if n >= 20:
        batch = max(1, n // 10)
        early = float(np.mean(red_vals[:batch]))
        late  = float(np.mean(red_vals[-batch:]))
        delta = late - early
        sign  = "+" if delta >= 0 else ""
        fig.add_annotation(
            text=f"Early avg: {early:+.3f}<br>Late avg: {late:+.3f}<br>Δ = {sign}{delta:.3f}",
            xref="paper", yref="paper",
            x=0.98, y=0.05,
            showarrow=False,
            align="right",
            font=dict(size=10, color=GOLD_COLOR),
            bgcolor=PANEL_BG,
            bordercolor=BORDER_COLOR,
            borderwidth=1,
        )

    fig.update_layout(
        title=dict(
            text="RED Reward Convergence Curve  —  'Hockey Stick' shows LoRA learning",
            font=dict(size=13, color=TEXT_SECONDARY),
        ),
        xaxis_title="Training Episode",
        yaxis_title=f"RED Reward ({w}-ep rolling avg)",
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
        **_PLOTLY_BASE,
    )
    return fig


def _load_csv_safe(path: Path) -> pd.DataFrame:
    """Load a CSV safely; return empty DataFrame on failure."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _infer_lora_checkpoints() -> list[int]:
    """
    Try to infer LoRA checkpoint episodes from training_events.jsonl.
    Looks for events with type containing 'checkpoint' or 'lora'.
    Returns a list of episode numbers (may be empty).
    """
    events = load_events()
    checkpoints: list[int] = []
    for ev in events:
        etype = str(ev.get("event_type", "")).lower()
        if "checkpoint" in etype or "lora" in etype:
            ep = ev.get("episode")
            try:
                checkpoints.append(int(ep))
            except (TypeError, ValueError):
                pass
    return sorted(set(checkpoints))
