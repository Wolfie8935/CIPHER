"""
Generate all static plot assets for the CIPHER hackathon submission.
Outputs PNG files to assets/ directory.

Run:  python generate_plots.py
"""

import os
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

# ── Palette ──────────────────────────────────────────────────────────────────
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

os.makedirs("assets", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = pd.read_csv("rewards_log.csv")
df = df_raw[df_raw["timestamp"].str.match(r"^\d{4}-\d{2}-\d{2}", na=False)].copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
df["seq"] = range(1, len(df) + 1)
df["date"] = df["timestamp"].dt.date
df["date_str"] = df["timestamp"].dt.strftime("%b %d")
df["red_win"] = (df["red_total"] > 0).astype(int)

print(f"Loaded {len(df)} clean episodes")

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
    labels = ["Day 1\nStub Baseline\n(April 22)", "Day 2\nLLM Training\n(April 23)", "Day 3\nFull LLM\n(April 24)"]
    colors_bar = [MUTED, YELLOW_C, GREEN_C]

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
    phase_labels_x = [25, (phase_cuts[0] + phase_cuts[1]) // 2 if len(phase_cuts) > 1 else phase_cuts[0] + 50,
                      phase_cuts[-1] + 15 if len(phase_cuts) >= 1 else len(df) - 30]
    phase_text     = ["Stub Baseline", "LLM Training", "Full LLM"]
    ylim_top = ax3.get_ylim()[1] if ax3.get_ylim()[1] > 0.5 else 1.5
    for px, pt in zip(phase_labels_x, phase_text):
        ax3.text(px, 1.2, pt, ha="center", fontsize=7.5,
                 color=YELLOW_C, alpha=0.85)

    ax3.set_xlabel("Episode (chronological)", fontdict=FONT, fontsize=9)
    ax3.set_ylabel("Reward (30-ep rolling mean)", fontdict=FONT, fontsize=9)
    ax3.set_title("RED vs BLUE Reward — Full Training History", fontsize=10, color=WHITE)
    ax3.legend(loc="lower right", fontsize=8.5, framealpha=0.3,
               facecolor=PANEL, edgecolor=GRID)
    ax3.grid(zorder=0)
    ax3.set_xlim(1, len(df))

    fig.savefig("assets/baseline_vs_trained.png")
    plt.close(fig)
    print("[OK] assets/baseline_vs_trained.png")


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
    fig.savefig("assets/reward_curves.png")
    plt.close(fig)
    print("[OK] assets/reward_curves.png")


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
    fig.savefig("assets/elo_chart.png")
    plt.close(fig)
    print("[OK] assets/elo_chart.png")


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
    date_labels = {
        list(df["date"].unique())[0]: "Stub Baseline\n(Apr 22)",
        list(df["date"].unique())[1]: "LLM Training\n(Apr 23)",
        list(df["date"].unique())[2]: "Full LLM\n(Apr 24)",
    }

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
    fig.savefig("assets/terminal_outcomes.png")
    plt.close(fig)
    print("[OK] assets/terminal_outcomes.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FLEET VERDICTS — pie + trend
# ─────────────────────────────────────────────────────────────────────────────
def plot_fleet_verdicts():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.suptitle("CIPHER — Oversight Auditor Fleet Verdicts by Phase", fontsize=13,
                 fontweight="bold", color=WHITE)

    dates_u = sorted(df["date"].unique())
    date_labels_short = ["Stub Baseline\n(Apr 22)", "LLM Training\n(Apr 23)", "Full LLM\n(Apr 24)"]
    verdict_colors = {
        "red_dominates":  RED_C,
        "blue_dominates": BLUE_C,
        "contested":      YELLOW_C,
        "degenerate":     ORANGE_C,
    }

    for ax, d, label in zip(axes, dates_u, date_labels_short):
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
    fig.savefig("assets/fleet_verdicts.png")
    plt.close(fig)
    print("[OK] assets/fleet_verdicts.png")


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
    phase_info = [
        (25, "Stub Baseline"),
        (df[df["date"] == dates_u[1]]["seq"].mean(), "LLM Training"),
        (df[df["date"] == dates_u[2]]["seq"].mean(), "Full LLM"),
    ]
    for px, pt in phase_info:
        ax.text(px, 88, pt, ha="center", fontsize=8, color=YELLOW_C, alpha=0.85)

    fig.tight_layout()
    fig.savefig("assets/win_rate_progression.png")
    plt.close(fig)
    print("[OK] assets/win_rate_progression.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ARCHITECTURE SUMMARY CARD — visual overview
# ─────────────────────────────────────────────────────────────────────────────
def plot_architecture_card():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.suptitle("CIPHER — Multi-Agent Architecture Overview",
                 fontsize=14, fontweight="bold", color=WHITE, y=0.97)

    def box(cx, cy, w, h, color, label, sublabel="", fontsize=9):
        rect = mpatches.FancyBboxPatch(
            (cx - w/2, cy - h/2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color + "33",
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + (h*0.15 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize,
                color=WHITE, fontweight="bold")
        if sublabel:
            ax.text(cx, cy - h*0.25, sublabel,
                    ha="center", va="center", fontsize=7,
                    color=MUTED)

    def arrow(x1, y1, x2, y2, color=MUTED, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                   mutation_scale=12))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.13, label, ha="center", fontsize=6.5, color=color)

    # RED Team
    box(2.0, 5.5, 1.6, 0.55, RED_C, "Planner", "long-horizon strategy")
    box(2.0, 4.7, 1.6, 0.55, RED_C, "Analyst",  "belief map · risk")
    box(2.0, 3.9, 1.6, 0.55, RED_C, "Operative", "stealth · traps")
    box(2.0, 3.1, 1.6, 0.55, RED_C, "Exfiltrator", "file extraction")
    ax.text(2.0, 6.3, "RED TEAM", ha="center", fontsize=11, color=RED_C, fontweight="bold")

    # BLUE Team
    box(12.0, 5.5, 1.6, 0.55, BLUE_C, "Surveillance", "anomaly feed")
    box(12.0, 4.7, 1.6, 0.55, BLUE_C, "Threat Hunter", "active investigation")
    box(12.0, 3.9, 1.6, 0.55, BLUE_C, "Deception Arch.", "honeypots · drops")
    box(12.0, 3.1, 1.6, 0.55, BLUE_C, "Forensics", "op-graph reconstruction")
    ax.text(12.0, 6.3, "BLUE TEAM", ha="center", fontsize=11, color=BLUE_C, fontweight="bold")

    # Environment
    box(7.0, 4.5, 2.4, 1.3, YELLOW_C, "50-Node Network", "4 security zones\nasymmetric observations")

    # Dead Drops
    box(7.0, 2.4, 1.8, 0.65, ORANGE_C, "Dead Drop Vault", "RED inter-agent memory")

    # Oversight
    box(7.0, 6.2, 2.0, 0.65, "#ff00ff", "Oversight Auditor", "9th LLM · fleet verdicts")

    # Rewards
    box(4.5, 1.2, 1.8, 0.65, GREEN_C, "RED Reward", "exfil×stealth×memory×complexity")
    box(9.5, 1.2, 1.8, 0.65, GREEN_C, "BLUE Reward", "detect×speed×honeypot-FP")

    # Dashboard
    box(7.0, 0.45, 3.2, 0.55, MUTED, "Live Dashboard + Analytics", "9 tabs · Elo · Heatmap · Telemetry")

    # Arrows
    arrow(2.9, 4.5, 5.7, 4.5, RED_C, "actions")
    arrow(8.3, 4.5, 11.1, 4.5, BLUE_C, "actions")
    arrow(5.7, 4.2, 3.0, 4.2, RED_C, "obs (masked)")
    arrow(8.3, 4.2, 11.0, 4.2, BLUE_C, "obs (noisy)")
    arrow(6.5, 3.9, 6.3, 2.7, ORANGE_C, "write drops")
    arrow(7.7, 2.7, 7.5, 3.9, ORANGE_C, "tamper")
    arrow(5.5, 3.9, 4.7, 1.55, GREEN_C, "reward")
    arrow(8.5, 3.9, 9.3, 1.55, GREEN_C, "reward")
    arrow(7.0, 5.75, 7.0, 5.85, "#ff00ff", "")
    arrow(5.5, 0.9, 6.35, 0.65, MUTED, "")
    arrow(8.5, 0.9, 7.65, 0.65, MUTED, "")

    fig.tight_layout()
    fig.savefig("assets/architecture_card.png")
    plt.close(fig)
    print("[OK] assets/architecture_card.png")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating CIPHER submission plots...\n")
    plot_baseline_vs_trained()
    plot_reward_curves()
    plot_elo()
    plot_terminal_outcomes()
    plot_fleet_verdicts()
    plot_win_rate_progression()
    plot_architecture_card()
    print(f"\nDone — {len(os.listdir('assets'))} files in assets/")
