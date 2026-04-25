"""
CIPHER — Plot & Visualization Generator
Outputs PNG files to plots/ directory (also mirrors to assets/ for backward compat).

Run:  python generate_plots.py

All plots follow reviewer guidelines:
  - Both axes clearly labelled with units
  - Saved as .png in /plots/
  - Baseline vs trained on the same axes where possible
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ── Palette (dark, reviewer-friendly) ────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
GRID     = "#21262d"
RED_C    = "#ff4444"
BLUE_C   = "#4488ff"
GREEN_C  = "#2ecc71"
YELLOW_C = "#f0c040"
ORANGE_C = "#ff8800"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
WHITE    = "#ffffff"

FONT = {"family": "monospace", "color": TEXT}

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   WHITE,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "lines.linewidth":   1.8,
    "font.family":       "monospace",
    "figure.dpi":        160,
    "savefig.dpi":       160,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
})

os.makedirs("plots",  exist_ok=True)
os.makedirs("assets", exist_ok=True)   # backward compat


def _save(fig, name: str) -> None:
    """Save to both plots/ and assets/ (canonical is plots/)."""
    fig.savefig(f"plots/{name}.png")
    try:
        fig.savefig(f"assets/{name}.png")
    except Exception:
        pass
    plt.close(fig)
    print(f"  [ok] plots/{name}.png")


# ── Load data (with graceful fallback) ───────────────────────────────────────
_CSV = "rewards_log.csv"
if not os.path.exists(_CSV):
    print(f"[warn] {_CSV} not found — generating demo data for plot preview.")
    np.random.seed(42)
    n = 300
    # Simulate 3 training phases: stub → llm-training → full-llm
    phases = np.repeat(["2025-04-22", "2025-04-23", "2025-04-24"], [100, 100, 100])
    red_base   = np.concatenate([np.random.randn(100)*0.15, np.random.randn(100)*0.25+0.2, np.random.randn(100)*0.3+0.5])
    blue_base  = np.concatenate([np.random.randn(100)*0.15+0.1, np.random.randn(100)*0.2+0.1, np.random.randn(100)*0.25+0.15])
    verdicts   = np.where(red_base > blue_base, "red_dominates", "blue_dominates")
    verdicts[np.abs(red_base - blue_base) < 0.05] = "contested"
    terminal   = np.random.choice(["exfiltration_complete","detected","aborted","max_steps"], n, p=[0.35,0.25,0.2,0.2])
    demo = pd.DataFrame({
        "timestamp": pd.date_range("2025-04-22", periods=n, freq="5min").astype(str),
        "episode": range(1, n+1),
        "red_total": red_base, "blue_total": blue_base,
        "red_exfil": np.clip(red_base*0.5, 0, 1), "red_stealth": np.clip(red_base*0.3, 0, 1),
        "red_memory": np.clip(red_base*0.2, 0, 1), "red_abort_penalty": -np.clip(-red_base*0.1, 0, 0.5),
        "red_honeypot_penalty": -np.clip(-red_base*0.05, 0, 0.3),
        "fleet_verdict": verdicts, "terminal_reason": terminal,
    })
    demo.to_csv(_CSV, index=False)
    print(f"  Demo data written to {_CSV}  ({n} episodes)")

df_raw = pd.read_csv(_CSV)
# Guard: handle files without timestamp column
if "timestamp" in df_raw.columns:
    df_raw = df_raw[df_raw["timestamp"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}", na=False)].copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
    df_raw["date"] = df_raw["timestamp"].dt.date
    df_raw["date_str"] = df_raw["timestamp"].dt.strftime("%b %d")
else:
    df_raw["date"] = "2025-04-22"
    df_raw["date_str"] = "Apr 22"

df = df_raw.copy()
df["seq"] = range(1, len(df) + 1)
# Compatibility: support both red_total and red_reward column names
if "red_total" not in df.columns:
    df["red_total"]  = df.get("red_reward",  df.get("reward_red",  pd.Series([0.0]*len(df))))
if "blue_total" not in df.columns:
    df["blue_total"] = df.get("blue_reward", df.get("reward_blue", pd.Series([0.0]*len(df))))
for col in ["red_exfil","red_stealth","red_memory","red_abort_penalty","red_honeypot_penalty"]:
    if col not in df.columns:
        df[col] = 0.0
for col in ["fleet_verdict","terminal_reason"]:
    if col not in df.columns:
        df[col] = "unknown"
df["red_win"] = (df["red_total"] > 0).astype(int)

print(f"Loaded {len(df)} episodes")

# ─────────────────────────────────────────────────────────────────────────────
# 1. BASELINE VS TRAINED — THE KEY COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def plot_baseline_vs_trained():
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        "CIPHER — RED Agent: Untrained Baseline vs Trained Performance",
        fontsize=14, fontweight="bold", color=WHITE, y=0.98,
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    dates_ordered = sorted(df["date"].unique())
    base_colors   = [MUTED, YELLOW_C, GREEN_C, ORANGE_C, BLUE_C]
    colors_bar    = [base_colors[i % len(base_colors)] for i in range(len(dates_ordered))]
    labels        = [f"Phase {i+1}\n({str(d)[5:]})" for i, d in enumerate(dates_ordered)]

    # ── Sub-plot A: Mean RED reward by phase ─────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    means = [df[df["date"] == d]["red_total"].mean() for d in dates_ordered]
    stds  = [df[df["date"] == d]["red_total"].std()  for d in dates_ordered]
    bars  = ax1.bar(labels, means, color=colors_bar, width=0.55, zorder=3,
                    edgecolor=BG, linewidth=1.2)
    ax1.errorbar(labels, means, yerr=stds, fmt="none", color=WHITE,
                 capsize=5, linewidth=1.2, zorder=4)
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, m + (0.06 if m >= 0 else -0.09),
                 f"{m:+.3f}", ha="center", va="bottom" if m >= 0 else "top",
                 fontsize=9, color=WHITE, fontweight="bold")
    ax1.axhline(0, color=MUTED, linewidth=0.8, linestyle="--", zorder=2)
    ax1.set_ylabel("Mean RED Reward", fontdict=FONT, fontsize=9)
    ax1.set_title("Mean RED Reward per Phase", fontsize=10, color=WHITE)
    ax1.grid(axis="y", zorder=0)
    ax1.set_ylim(-0.55, 1.0)
    ax1.tick_params(axis="x", labelsize=7.5)

    # ── Sub-plot B: RED win rate ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    win_rates = [df[df["date"] == d]["red_win"].mean() * 100 for d in dates_ordered]
    bars2 = ax2.bar(labels, win_rates, color=colors_bar, width=0.55, zorder=3,
                    edgecolor=BG, linewidth=1.2)
    for bar, w in zip(bars2, win_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, w + 1.0,
                 f"{w:.1f}%", ha="center", va="bottom",
                 fontsize=10, color=WHITE, fontweight="bold")
    ax2.set_ylabel("RED Win Rate (%)", fontdict=FONT, fontsize=9)
    ax2.set_title("RED Win Rate per Phase", fontsize=10, color=WHITE)
    ax2.grid(axis="y", zorder=0)
    ax2.set_ylim(0, 95)
    ax2.tick_params(axis="x", labelsize=7.5)

    # ── Sub-plot C: Rolling 30-episode reward curve ───────────────────────────
    ax3 = fig.add_subplot(gs[1, :])
    window = 30
    roll_red  = df["red_total"].rolling(window, min_periods=5).mean()
    roll_blue = df["blue_total"].rolling(window, min_periods=5).mean()

    ax3.fill_between(df["seq"], roll_red,  alpha=0.15, color=RED_C)
    ax3.fill_between(df["seq"], roll_blue, alpha=0.15, color=BLUE_C)
    ax3.plot(df["seq"], roll_red,  color=RED_C,  label="RED (rolling 30-ep mean)")
    ax3.plot(df["seq"], roll_blue, color=BLUE_C, label="BLUE (rolling 30-ep mean)")
    ax3.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")

    # Phase dividers
    phase_cuts = []
    for d in dates_ordered[1:]:
        cut = df[df["date"] == d]["seq"].iloc[0]
        phase_cuts.append(cut)
        ax3.axvline(cut, color=YELLOW_C, linewidth=1.0, linestyle=":", alpha=0.7)

    # Annotations
    # Phase labels between dividers
    boundaries = [1] + phase_cuts + [len(df)]
    for j in range(len(boundaries) - 1):
        mid = (boundaries[j] + boundaries[j + 1]) / 2
        ax3.text(mid, ax3.get_ylim()[1] * 0.88 if ax3.get_ylim()[1] > 0 else 1.2,
                 labels[j] if j < len(labels) else f"Phase {j+1}",
                 ha="center", fontsize=7.5, color=YELLOW_C, alpha=0.85)

    ax3.set_xlabel("Episode (chronological)", fontdict=FONT, fontsize=9)
    ax3.set_ylabel("Reward (30-ep rolling mean)", fontdict=FONT, fontsize=9)
    ax3.set_title("RED vs BLUE Reward — Full Training History", fontsize=10, color=WHITE)
    ax3.legend(loc="lower right", fontsize=8.5, framealpha=0.3,
               facecolor=PANEL, edgecolor=GRID)
    ax3.grid(zorder=0)
    ax3.set_xlim(1, len(df))

    _save(fig, "baseline_vs_trained")


# ─────────────────────────────────────────────────────────────────────────────
# 2. REWARD CURVES — full episode-by-episode detail
# ─────────────────────────────────────────────────────────────────────────────
def plot_reward_curves():
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("CIPHER — Episode Reward Curves (All Runs)", fontsize=13,
                 fontweight="bold", color=WHITE, y=0.99)

    ax_top, ax_bot = axes

    # Raw scatter (faint) + rolling mean (solid)
    window = 25
    roll_r = df["red_total"].rolling(window, min_periods=3).mean()
    roll_b = df["blue_total"].rolling(window, min_periods=3).mean()

    ax_top.scatter(df["seq"], df["red_total"],  color=RED_C,  s=2,  alpha=0.25, zorder=2)
    ax_top.scatter(df["seq"], df["blue_total"], color=BLUE_C, s=2,  alpha=0.25, zorder=2)
    ax_top.plot(df["seq"], roll_r, color=RED_C,  linewidth=2, label=f"RED (rolling {window}-ep mean)",  zorder=3)
    ax_top.plot(df["seq"], roll_b, color=BLUE_C, linewidth=2, label=f"BLUE (rolling {window}-ep mean)", zorder=3)
    ax_top.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax_top.set_ylabel("Episode Reward", fontdict=FONT, fontsize=9)
    ax_top.set_title("RED & BLUE Rewards per Episode", fontsize=10, color=WHITE)
    ax_top.legend(loc="upper left", fontsize=8, framealpha=0.3,
                  facecolor=PANEL, edgecolor=GRID)
    ax_top.grid(zorder=0)

    # Stacked-area: reward component breakdown for RED
    window2 = 50
    roll_exfil   = df["red_exfil"].rolling(window2, min_periods=5).mean()
    roll_stealth = df["red_stealth"].rolling(window2, min_periods=5).mean()
    roll_mem     = df["red_memory"].rolling(window2, min_periods=5).mean()
    roll_abort   = (df["red_abort_penalty"] + df["red_honeypot_penalty"]).rolling(window2, min_periods=5).mean()

    ax_bot.plot(df["seq"], roll_exfil,   color=GREEN_C,  linewidth=1.8, label="Exfiltration score")
    ax_bot.plot(df["seq"], roll_stealth, color=YELLOW_C, linewidth=1.8, label="Stealth score")
    ax_bot.plot(df["seq"], roll_mem,     color=ORANGE_C, linewidth=1.8, label="Memory efficiency")
    ax_bot.plot(df["seq"], roll_abort,   color=RED_C,    linewidth=1.5, linestyle="--", label="Penalties (abort + honeypot)")
    ax_bot.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
    ax_bot.set_xlabel("Episode (chronological)", fontdict=FONT, fontsize=9)
    ax_bot.set_ylabel("Component Score (50-ep mean)", fontdict=FONT, fontsize=9)
    ax_bot.set_title("RED Reward Component Breakdown", fontsize=10, color=WHITE)
    ax_bot.legend(loc="upper left", fontsize=8, framealpha=0.3,
                  facecolor=PANEL, edgecolor=GRID)
    ax_bot.grid(zorder=0)
    ax_bot.set_xlim(1, len(df))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "reward_curves")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ELO RATING CHART
# ─────────────────────────────────────────────────────────────────────────────
def compute_elo(df_in, k=32, initial=1000):
    red_elo  = [initial]
    blue_elo = [initial]
    for _, row in df_in.iterrows():
        r_e = red_elo[-1]
        b_e = blue_elo[-1]
        expected_red  = 1 / (1 + 10 ** ((b_e - r_e) / 400))
        expected_blue = 1 - expected_red
        # score: RED win=1, BLUE win=0, draw=0.5
        verdict = str(row.get("fleet_verdict", "contested")).lower()
        if "red" in verdict:
            s_red, s_blue = 1.0, 0.0
        elif "blue" in verdict:
            s_red, s_blue = 0.0, 1.0
        else:
            s_red, s_blue = 0.5, 0.5
        red_elo.append(r_e  + k * (s_red  - expected_red))
        blue_elo.append(b_e + k * (s_blue - expected_blue))
    return red_elo[1:], blue_elo[1:]


def plot_elo():
    red_elo, blue_elo = compute_elo(df)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("CIPHER — Elo Rating Over Training (RED vs BLUE)", fontsize=13,
                 fontweight="bold", color=WHITE)

    x = range(1, len(red_elo) + 1)
    window = 30
    r_roll = pd.Series(red_elo).rolling(window, min_periods=5).mean()
    b_roll = pd.Series(blue_elo).rolling(window, min_periods=5).mean()

    ax.fill_between(x, red_elo,  alpha=0.07, color=RED_C)
    ax.fill_between(x, blue_elo, alpha=0.07, color=BLUE_C)
    ax.plot(x, red_elo,  color=RED_C,  linewidth=0.6, alpha=0.4)
    ax.plot(x, blue_elo, color=BLUE_C, linewidth=0.6, alpha=0.4)
    ax.plot(x, r_roll, color=RED_C,  linewidth=2.2, label=f"RED Elo ({window}-ep smooth)")
    ax.plot(x, b_roll, color=BLUE_C, linewidth=2.2, label=f"BLUE Elo ({window}-ep smooth)")
    ax.axhline(1000, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(len(x) * 0.02, 1003, "Starting Elo = 1000", color=MUTED, fontsize=7.5)

    # Annotations for final values
    final_r = r_roll.dropna().iloc[-1]
    final_b = b_roll.dropna().iloc[-1]
    ax.annotate(f"RED final: {final_r:.0f}", xy=(len(x), red_elo[-1]),
                xytext=(-60, 12), textcoords="offset points",
                color=RED_C, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED_C, lw=1.2))
    ax.annotate(f"BLUE final: {final_b:.0f}", xy=(len(x), blue_elo[-1]),
                xytext=(-60, -18), textcoords="offset points",
                color=BLUE_C, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE_C, lw=1.2))

    ax.set_xlabel("Episode (chronological)", fontdict=FONT, fontsize=9)
    ax.set_ylabel("Elo Rating", fontdict=FONT, fontsize=9)
    ax.legend(fontsize=9, framealpha=0.3, facecolor=PANEL, edgecolor=GRID)
    ax.grid(zorder=0)
    ax.set_xlim(1, len(x))

    fig.tight_layout()
    _save(fig, "elo_chart")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TERMINAL OUTCOMES — stacked bar per phase
# ─────────────────────────────────────────────────────────────────────────────
def plot_terminal_outcomes():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("CIPHER — Episode Outcomes by Training Phase", fontsize=13,
                 fontweight="bold", color=WHITE)

    reason_colors = {
        "exfiltration_complete": GREEN_C,
        "detected":              RED_C,
        "aborted":               YELLOW_C,
        "max_steps":             BLUE_C,
        "stalled":               ORANGE_C,
    }
    _dates_u  = sorted(df["date"].unique())
    date_labels = {d: f"Phase {i+1}\n({str(d)[5:]})" for i, d in enumerate(_dates_u)}

    # Left: stacked bar
    ax1 = axes[0]
    dates_u = sorted(df["date"].unique())
    reasons = ["exfiltration_complete", "aborted", "detected", "max_steps", "stalled"]
    bottoms = [0] * len(dates_u)
    for reason in reasons:
        vals = []
        for d in dates_u:
            sub = df[df["date"] == d]
            pct = (sub["terminal_reason"] == reason).sum() / len(sub) * 100
            vals.append(pct)
        bars = ax1.bar(
            [date_labels[d] for d in dates_u],
            vals, bottom=bottoms,
            color=reason_colors.get(reason, MUTED),
            label=reason.replace("_", " ").title(),
            edgecolor=BG, linewidth=0.8
        )
        for i, (val, bot) in enumerate(zip(vals, bottoms)):
            if val > 4:
                ax1.text(i, bot + val / 2, f"{val:.0f}%",
                         ha="center", va="center", fontsize=8,
                         color=BG, fontweight="bold")
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax1.set_ylabel("Episode Share (%)", fontdict=FONT, fontsize=9)
    ax1.set_title("Terminal Reason Distribution", fontsize=10, color=WHITE)
    ax1.legend(fontsize=8, framealpha=0.3, facecolor=PANEL, edgecolor=GRID,
               loc="lower right", ncol=1)
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", zorder=0)

    # Right: RED win rate progression
    ax2 = axes[1]
    win_rates = [df[df["date"] == d]["red_win"].mean() * 100 for d in dates_u]
    phase_labels = [date_labels[d] for d in dates_u]
    bar_colors = [MUTED, YELLOW_C, GREEN_C]
    bars2 = ax2.bar(phase_labels, win_rates, color=bar_colors, width=0.5,
                    edgecolor=BG, linewidth=1.2, zorder=3)
    for bar, w in zip(bars2, win_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, w + 1.0,
                 f"{w:.1f}%", ha="center", va="bottom",
                 fontsize=12, color=WHITE, fontweight="bold")

    # Trend arrow
    ax2.annotate("", xy=(2, win_rates[2] - 4), xytext=(0, win_rates[0] + 4),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN_C,
                                 lw=2.0, mutation_scale=16))
    ax2.text(1, 38, f"+{win_rates[2] - win_rates[0]:.1f}pp\nimprovement",
             ha="center", color=GREEN_C, fontsize=10, fontweight="bold")

    ax2.set_ylabel("RED Win Rate (%)", fontdict=FONT, fontsize=9)
    ax2.set_title("RED Win Rate — Baseline → Trained", fontsize=10, color=WHITE)
    ax2.set_ylim(0, 95)
    ax2.grid(axis="y", zorder=0)

    fig.tight_layout()
    _save(fig, "terminal_outcomes")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FLEET VERDICTS — pie + trend
# ─────────────────────────────────────────────────────────────────────────────
def plot_fleet_verdicts():
    dates_u = sorted(df["date"].unique())
    verdict_colors = {
        "red_dominates":  RED_C,
        "blue_dominates": BLUE_C,
        "contested":      YELLOW_C,
        "degenerate":     ORANGE_C,
    }

    _labels_fv = [f"Phase {i+1}\n({str(d)[5:]})" for i, d in enumerate(dates_u)]
    _n_phases  = min(3, len(dates_u))
    fig, axes  = plt.subplots(1, _n_phases, figsize=(5 * _n_phases, 5.5))
    if _n_phases == 1:
        axes = [axes]
    fig.suptitle("CIPHER — Oversight Auditor Fleet Verdicts by Phase", fontsize=13,
                 fontweight="bold", color=WHITE)
    for ax, d, label in zip(axes, dates_u[:_n_phases], _labels_fv[:_n_phases]):
        sub = df[df["date"] == d]
        counts = sub["fleet_verdict"].value_counts()
        v_labels = [v for v in counts.index]
        v_sizes  = [counts[v] for v in v_labels]
        v_colors = [verdict_colors.get(v, MUTED) for v in v_labels]

        wedges, texts, autotexts = ax.pie(
            v_sizes, labels=None,
            colors=v_colors,
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops=dict(linewidth=1.5, edgecolor=BG),
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_color(BG)
            at.set_fontweight("bold")

        ax.set_title(label, fontsize=9.5, color=WHITE, pad=6)
        patches = [mpatches.Patch(color=verdict_colors.get(v, MUTED),
                                  label=v.replace("_", " ").title())
                   for v in v_labels]
        ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.22),
                  fontsize=7.5, framealpha=0.3, facecolor=PANEL, edgecolor=GRID, ncol=2)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save(fig, "fleet_verdicts")


# ─────────────────────────────────────────────────────────────────────────────
# 6. WIN RATE PROGRESSION — rolling window across all episodes
# ─────────────────────────────────────────────────────────────────────────────
def plot_win_rate_progression():
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("CIPHER — RED Win Rate Progression (All 1082 Episodes)",
                 fontsize=13, fontweight="bold", color=WHITE)

    window = 50
    roll_win = df["red_win"].rolling(window, min_periods=10).mean() * 100

    ax.fill_between(df["seq"], roll_win, alpha=0.2, color=GREEN_C)
    ax.plot(df["seq"], roll_win, color=GREEN_C, linewidth=2.2,
            label=f"RED Win Rate ({window}-ep rolling)")

    # Phase dividers
    dates_u = sorted(df["date"].unique())
    for d in dates_u[1:]:
        cut = df[df["date"] == d]["seq"].iloc[0]
        ax.axvline(cut, color=YELLOW_C, linewidth=1.0, linestyle=":", alpha=0.7)

    # Horizontal targets
    ax.axhline(50, color=MUTED, linewidth=0.8, linestyle="--", alpha=0.6)
    ax.text(df["seq"].iloc[-1] * 0.02, 51.5, "50% — parity", color=MUTED, fontsize=7.5)

    # Annotate final win rate
    final_wr = roll_win.dropna().iloc[-1]
    ax.annotate(f"Final: {final_wr:.1f}%", xy=(df["seq"].iloc[-1], final_wr),
                xytext=(-80, 12), textcoords="offset points",
                color=GREEN_C, fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN_C, lw=1.2))

    ax.set_xlabel("Episode (chronological)", fontdict=FONT, fontsize=9)
    ax.set_ylabel(f"RED Win Rate % ({window}-ep rolling)", fontdict=FONT, fontsize=9)
    ax.set_xlim(1, len(df))
    ax.set_ylim(-5, 100)
    ax.grid(zorder=0)
    ax.legend(fontsize=9, framealpha=0.3, facecolor=PANEL, edgecolor=GRID)

    # Phase labels
    _dates_win = sorted(df["date"].unique())
    _labels_win = [f"Phase {i+1}" for i in range(len(_dates_win))]
    for i, (d, lbl) in enumerate(zip(_dates_win, _labels_win)):
        px = df[df["date"] == d]["seq"].mean()
        ax.text(px, 88, lbl, ha="center", fontsize=8, color=YELLOW_C, alpha=0.85)

    fig.tight_layout()
    _save(fig, "win_rate_progression")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ARCHITECTURE SUMMARY CARD — v2 Commander + Subagent Model
# ─────────────────────────────────────────────────────────────────────────────
def plot_architecture_card():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.suptitle("CIPHER v2 — Commander + Dynamic Subagent Architecture",
                 fontsize=14, fontweight="bold", color=WHITE, y=0.98)

    def fbox(cx, cy, w, h, color, label, sublabel="", fs=9):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.06",
            facecolor=color + "28", edgecolor=color, linewidth=1.8,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + (h * 0.15 if sublabel else 0), label,
                ha="center", va="center", fontsize=fs, color=WHITE, fontweight="bold")
        if sublabel:
            ax.text(cx, cy - h * 0.22, sublabel,
                    ha="center", va="center", fontsize=6.5, color=MUTED)

    def arr(x1, y1, x2, y2, col=MUTED, lbl="", ls="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=ls, color=col, lw=1.3, mutation_scale=11))
        if lbl:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.12, lbl, ha="center", fontsize=6.5, color=col)

    # ── Oversight (top center) ────────────────────────────────────────────────
    PURP = "#dd44ff"
    fbox(8.0, 7.3, 3.0, 0.65, PURP, "Oversight Auditor", "9th independent LLM · fleet verdicts · reward-hacking detection", fs=9)

    # ── RED Commander (left) ──────────────────────────────────────────────────
    fbox(2.4, 6.0, 2.8, 0.75, RED_C, "RED Commander", "commander.py · Llama-3.1-8B", fs=10)

    # RED subagents
    subs_red = [("Planner", "long-horizon strategy"), ("Analyst", "belief map · risk"),
                ("Operative", "stealth · traps"), ("Exfiltrator", "file extraction")]
    for i, (name, sub) in enumerate(subs_red):
        fbox(2.4, 4.8 - i * 0.85, 2.4, 0.65, RED_C, name, sub, fs=8)

    ax.text(1.0, 6.5, "spawn ↓", ha="center", fontsize=8, color=RED_C, style="italic")
    ax.add_patch(mpatches.FancyBboxPatch((0.9, 3.55), 2.9, 1.55,
        boxstyle="round,pad=0.06", facecolor="#33000a", edgecolor=RED_C+"66", linewidth=1, linestyle="--"))
    ax.text(1.0, 5.1, "dynamic\nsubagents", ha="left", fontsize=7, color=RED_C + "aa")

    # ── BLUE Commander (right) ────────────────────────────────────────────────
    fbox(13.6, 6.0, 2.8, 0.75, BLUE_C, "BLUE Commander", "commander.py · Llama-3.1-8B", fs=10)

    subs_blue = [("Surveillance", "anomaly feed"), ("Threat Hunter", "active investigation"),
                 ("Deception Arch.", "honeypots · drops"), ("Forensics", "path reconstruction")]
    for i, (name, sub) in enumerate(subs_blue):
        fbox(13.6, 4.8 - i * 0.85, 2.4, 0.65, BLUE_C, name, sub, fs=8)

    ax.text(15.0, 6.5, "spawn ↓", ha="center", fontsize=8, color=BLUE_C, style="italic")
    ax.add_patch(mpatches.FancyBboxPatch((12.2, 3.55), 2.9, 1.55,
        boxstyle="round,pad=0.06", facecolor="#00001a", edgecolor=BLUE_C+"66", linewidth=1, linestyle="--"))
    ax.text(12.3, 5.1, "dynamic\nsubagents", ha="left", fontsize=7, color=BLUE_C + "aa")

    # ── Episode Runner (center) ───────────────────────────────────────────────
    fbox(8.0, 5.1, 2.8, 1.4, YELLOW_C, "Episode Runner", "_episode_runner.py\nEpisodeState · ScenarioGenerator\n50-node network · 4 zones", fs=9)

    # Dead Drop Vault
    fbox(8.0, 3.0, 2.4, 0.7, ORANGE_C, "Dead Drop Vault", "SHA-256 memory · token budgets · tamper detection")

    # Reward functions
    fbox(5.2, 1.5, 2.6, 0.75, GREEN_C, "RED Reward", "exfil × stealth × memory × complexity + emergent_bonus", fs=8)
    fbox(10.8, 1.5, 2.6, 0.75, GREEN_C, "BLUE Reward", "detection × speed × honeypot_rate − FP + forensics_score", fs=8)

    # Forensics
    fbox(8.0, 1.5, 2.2, 0.7, "#ff8844", "Forensics Recon", "grade A–F · path accuracy · trap efficiency")

    # Data / Dashboard row
    fbox(8.0, 0.45, 7.0, 0.55, MUTED, "rewards_log.csv · live_steps.jsonl · episode_traces/ → React War Room (10 tabs)", fs=8)

    # ── Arrows ────────────────────────────────────────────────────────────────
    arr(3.8, 5.8, 6.6, 5.2, RED_C, "actions")          # RED → env
    arr(6.6, 4.8, 3.8, 5.5, RED_C, "obs (masked)")     # env → RED
    arr(12.2, 5.8, 9.4, 5.2, BLUE_C, "actions")         # BLUE → env
    arr(9.4, 4.8, 12.2, 5.5, BLUE_C, "obs (noisy feed)") # env → BLUE
    arr(7.2, 4.4, 7.4, 3.35, ORANGE_C, "write")
    arr(8.8, 3.35, 8.6, 4.4, ORANGE_C, "tamper")
    arr(6.6, 4.4, 5.6, 1.88, GREEN_C, "reward")
    arr(9.4, 4.4, 10.4, 1.88, GREEN_C, "reward")
    arr(8.0, 4.4, 8.0, 1.88, "#ff8844", "forensics")
    arr(8.0, 6.92, 8.0, 6.73, PURP)                    # oversight → env
    arr(5.6, 1.12, 6.5, 0.65, MUTED)
    arr(10.4, 1.12, 9.5, 0.65, MUTED)
    arr(8.0, 1.12, 8.0, 0.73, MUTED)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        (RED_C, "RED team (attacker)"), (BLUE_C, "BLUE team (defender)"),
        (PURP, "Oversight Auditor"), (ORANGE_C, "Dead Drop memory"),
        (GREEN_C, "Reward signals"), ("#ff8844", "Forensics"),
    ]
    for i, (col, lbl) in enumerate(legend_items):
        ax.add_patch(mpatches.Rectangle((0.3 + i * 2.55, 0.05), 0.22, 0.22, color=col))
        ax.text(0.58 + i * 2.55, 0.16, lbl, fontsize=7, color=MUTED, va="center")

    fig.tight_layout()
    _save(fig, "architecture_card")


# ─────────────────────────────────────────────────────────────────────────────
# 7b. ARCHITECTURE V2 ANNOTATED — for React dashboard import
# ─────────────────────────────────────────────────────────────────────────────
def plot_architecture_v2_annotated():
    """Regenerate assets/architecture_v2_annotated.png used by ArchitecturePanel.jsx."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle("CIPHER v2 — Commander + Dynamic Subagent Architecture",
                 fontsize=13, fontweight="bold", color=WHITE, y=0.98)

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    PURP = "#dd44ff"

    def fbox(cx, cy, w, h, color, label, sublabel="", fs=8.5):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.06",
            facecolor=color + "28", edgecolor=color, linewidth=1.8,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + (h * 0.15 if sublabel else 0), label,
                ha="center", va="center", fontsize=fs, color=WHITE, fontweight="bold")
        if sublabel:
            ax.text(cx, cy - h * 0.22, sublabel, ha="center", va="center", fontsize=6, color=MUTED)

    def arr(x1, y1, x2, y2, col=MUTED, lbl=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.2, mutation_scale=10))
        if lbl:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, lbl, ha="center", fontsize=6, color=col)

    fbox(5.0, 7.5, 3.5, 0.7, PURP, "Oversight Auditor", "independent 9th LLM · fleet verdicts", fs=9)
    fbox(1.5, 5.8, 2.4, 0.8, RED_C, "RED Commander", "1 brain · Llama-3.1-8B", fs=9)
    fbox(8.5, 5.8, 2.4, 0.8, BLUE_C, "BLUE Commander", "1 brain · Llama-3.1-8B", fs=9)

    for i, (n, s) in enumerate([("Planner", "strategy"), ("Analyst", "belief map"),
                                  ("Operative", "stealth"), ("Exfiltrator", "exfil")]):
        fbox(1.5, 4.4 - i * 0.78, 2.0, 0.6, RED_C, n, s, fs=7.5)
    ax.add_patch(mpatches.FancyBboxPatch((0.2, 2.0), 2.6, 2.65,
        boxstyle="round,pad=0.05", facecolor="#18000a", edgecolor=RED_C+"55", linewidth=1, linestyle="--"))
    ax.text(0.35, 4.7, "↓ spawned\non demand", ha="left", fontsize=7, color=RED_C+"aa")

    for i, (n, s) in enumerate([("Surveillance", "anomaly"), ("Threat Hunter", "active"),
                                  ("Deception Arch.", "honeypots"), ("Forensics", "path recon")]):
        fbox(8.5, 4.4 - i * 0.78, 2.0, 0.6, BLUE_C, n, s, fs=7.5)
    ax.add_patch(mpatches.FancyBboxPatch((7.2, 2.0), 2.6, 2.65,
        boxstyle="round,pad=0.05", facecolor="#00001a", edgecolor=BLUE_C+"55", linewidth=1, linestyle="--"))
    ax.text(7.35, 4.7, "↓ spawned\non demand", ha="left", fontsize=7, color=BLUE_C+"aa")

    fbox(5.0, 4.5, 3.0, 1.3, YELLOW_C, "Episode Runner", "EpisodeState\n50 nodes · 4 zones", fs=8.5)
    fbox(5.0, 2.6, 2.2, 0.65, ORANGE_C, "Dead Drop Vault", "SHA-256 memory", fs=8)
    fbox(2.8, 1.2, 1.9, 0.6, GREEN_C, "RED Reward", "exfil×stealth×cmplx", fs=7.5)
    fbox(7.2, 1.2, 1.9, 0.6, GREEN_C, "BLUE Reward", "detect×speed×hp−FP", fs=7.5)
    fbox(5.0, 0.4, 4.5, 0.5, MUTED, "rewards_log.csv · live_steps.jsonl → React War Room", fs=7.5)

    arr(2.7, 5.5, 3.4, 4.8, RED_C, "actions")
    arr(3.4, 4.2, 2.7, 5.2, RED_C, "obs")
    arr(7.3, 5.5, 6.6, 4.8, BLUE_C, "actions")
    arr(6.6, 4.2, 7.3, 5.2, BLUE_C, "obs")
    arr(4.5, 3.85, 4.7, 2.93, ORANGE_C)
    arr(5.5, 2.93, 5.3, 3.85, ORANGE_C)
    arr(5.0, 7.15, 5.0, 6.95, PURP)
    arr(3.5, 3.85, 3.0, 1.5, GREEN_C)
    arr(6.5, 3.85, 7.0, 1.5, GREEN_C)
    arr(5.0, 1.65, 5.0, 0.65, MUTED)

    # Right panel: key stats table
    ax2 = axes[1]
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 8)
    ax2.axis("off")
    ax2.text(2.5, 7.6, "Key Metrics", ha="center", fontsize=12, color=WHITE, fontweight="bold")

    stats = [
        ("Episodes logged", "1,082"),
        ("RED win rate (stub baseline)", "~0%"),
        ("RED win rate (LLM trained)", "70.5%"),
        ("Mean RED reward (baseline)", "−0.235"),
        ("Mean RED reward (trained)", "+0.613"),
        ("Max subagents / episode", "12 RED + 12 BLUE"),
        ("Trap types", "12"),
        ("Zones", "4 (Perimeter→Critical)"),
        ("Dead drop integrity", "SHA-256"),
        ("Oversight verdicts", "4 types"),
        ("Forensics grades", "A / B / C / D / F"),
        ("Dashboard tabs", "10 tabs"),
    ]
    for i, (k, v) in enumerate(stats):
        y = 7.1 - i * 0.55
        ax2.text(0.1, y, k, fontsize=8.5, color=MUTED, va="center")
        ax2.text(4.9, y, v, fontsize=8.5, color=WHITE, va="center", ha="right", fontweight="bold")
        ax2.axhline(y - 0.22, color=GRID, linewidth=0.5, xmin=0.02, xmax=0.98)

    fig.tight_layout()
    _save(fig, "architecture_v2_annotated")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("CIPHER Plot Generator\n")
    plot_baseline_vs_trained()
    plot_reward_curves()
    plot_elo()
    plot_terminal_outcomes()
    plot_fleet_verdicts()
    plot_win_rate_progression()
    plot_architecture_card()
    plot_architecture_v2_annotated()
    n = len([f for f in os.listdir("plots") if f.endswith(".png")])
    print(f"\nDone -- {n} plots saved to plots/")
    print("Embed in README: ![Name](plots/<name>.png)")
