"""
cipher/utils/video_gen.py

"CIPHER Cinema" — Matrix/Terminal-style video highlight generator.
Renders an .mp4 episode recap using matplotlib (bundled) instead of
requiring moviepy. Falls back gracefully if dependencies are missing.

Usage:
    from cipher.utils.video_gen import generate_episode_video
    generate_episode_video(episode_data, output_path="highlights.mp4")
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any


# ── Frame content builders ────────────────────────────────────────────────────

def _make_title_frame(episode_num: int, outcome: str, mode: str) -> dict:
    return {
        "type": "title",
        "episode": episode_num,
        "outcome": outcome,
        "mode": mode,
    }


def _make_log_frames(episode_log: list[dict], max_frames: int = 20) -> list[dict]:
    """Convert episode log entries into 'console' frames."""
    frames = []
    # Sample evenly across the log
    step = max(1, len(episode_log) // max_frames)
    for i in range(0, len(episode_log), step):
        if len(frames) >= max_frames:
            break
        entry = episode_log[i]
        agent = str(entry.get("agent_id", entry.get("agent", "?")))
        action = str(entry.get("action_type", entry.get("action", "?")))
        ep_step = entry.get("step", i)
        payload = entry.get("payload", {}) or {}
        node = payload.get("target_node", "")
        result = entry.get("result", {}) or {}
        ok = "✓" if result.get("success", True) else "✗"
        team = "RED" if agent.startswith("red") else ("BLUE" if agent.startswith("blue") else "SYS")
        frames.append({
            "type": "console",
            "step": ep_step,
            "team": team,
            "agent": agent,
            "action": action,
            "node": node,
            "ok": ok,
        })
    return frames


def _make_scorecard_frame(episode_num: int, red_reward: float,
                          blue_reward: float, outcome: str, steps: int) -> dict:
    return {
        "type": "scorecard",
        "episode": episode_num,
        "red_reward": red_reward,
        "blue_reward": blue_reward,
        "outcome": outcome,
        "steps": steps,
    }


# ── Matplotlib renderer ───────────────────────────────────────────────────────

def _render_frames_matplotlib(frames: list[dict], output_path: str,
                               fps: int = 2, width: int = 1280, height: int = 720) -> bool:
    """Render frames using matplotlib + animation → .mp4"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.patches import Rectangle
        import numpy as np

        DPI = 96
        fig_w = width / DPI
        fig_h = height / DPI

        # Cyberpunk palette
        BG = "#030712"
        GREEN = "#00ff41"
        RED = "#ff4444"
        BLUE = "#4488ff"
        YELLOW = "#fbbf24"
        DIM = "#374151"
        WHITE = "#e2e8f0"
        PURPLE = "#a855f7"

        console_lines: list[str] = []  # rolling console buffer

        def draw_scanlines(ax, alpha: float = 0.04):
            # Use hlines in data coordinates (0–1 since xlim/ylim are both 0–1)
            for y in np.linspace(0, 1, 60):
                ax.hlines(y, 0, 1, colors="white", lw=0.4, alpha=alpha)

        def draw_frame(frame_data: dict):
            fig.clf()
            fig.patch.set_facecolor(BG)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_facecolor(BG)
            ax.axis("off")

            ftype = frame_data.get("type", "console")

            # ── TITLE FRAME ───────────────────────────────────────────
            if ftype == "title":
                ep = frame_data["episode"]
                mode = frame_data.get("mode", "stub").upper()
                outcome = str(frame_data.get("outcome", "")).replace("_", " ").upper()

                # Glitch letters
                title = "C I P H E R"
                ax.text(0.5, 0.70, title,
                        ha="center", va="center",
                        fontsize=52, fontweight="bold",
                        color=RED, fontfamily="monospace",
                        transform=ax.transAxes)
                ax.text(0.5, 0.58, f"E P I S O D E  {ep:03d}",
                        ha="center", va="center",
                        fontsize=22, fontweight="bold",
                        color=WHITE, fontfamily="monospace",
                        transform=ax.transAxes)
                ax.text(0.5, 0.47,
                        "[ THE GRAVITY COLLECTIVE ]  vs  [ AEGIS SYSTEMS ]",
                        ha="center", va="center",
                        fontsize=13, color=DIM, fontfamily="monospace",
                        transform=ax.transAxes)
                ax.text(0.5, 0.37, f"MODE: {mode}",
                        ha="center", va="center",
                        fontsize=11, color=YELLOW, fontfamily="monospace",
                        transform=ax.transAxes)
                ax.text(0.5, 0.28, f"OUTCOME: {outcome}",
                        ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color=GREEN, fontfamily="monospace",
                        transform=ax.transAxes)
                # Border
                rect = Rectangle((0.05, 0.2), 0.9, 0.7,
                                  fill=False, edgecolor=RED, lw=1.5,
                                  transform=ax.transAxes)
                ax.add_patch(rect)

            # ── CONSOLE FRAME ─────────────────────────────────────────
            elif ftype == "console":
                team = frame_data.get("team", "SYS")
                agent = frame_data.get("agent", "?")
                action = frame_data.get("action", "?")
                node = frame_data.get("node", "")
                ok = frame_data.get("ok", "✓")
                step = frame_data.get("step", 0)

                team_color = RED if team == "RED" else (BLUE if team == "BLUE" else PURPLE)
                new_line = (
                    f"[STEP {step:02d}] [{team}] {agent:<20} "
                    f"→ {action:<18} {'node='+str(node) if node!='' else '':>12} {ok}"
                )
                console_lines.append(new_line)
                # Keep last 18 lines
                visible = console_lines[-18:]

                # Header
                ax.text(0.02, 0.96, "[ CIPHER CCTV — BATTLE FEED ]",
                        fontsize=11, fontweight="bold",
                        color=GREEN, fontfamily="monospace",
                        transform=ax.transAxes)
                ax.text(0.98, 0.96, f"STEP {step:03d}",
                        fontsize=11, fontweight="bold",
                        color=YELLOW, fontfamily="monospace",
                        ha="right", transform=ax.transAxes)
                ax.hlines(0.93, 0, 1, colors=DIM, lw=0.8)

                # Console lines
                for i, line in enumerate(visible):
                    y = 0.88 - i * 0.048
                    lcolor = RED if "[RED]" in line else (BLUE if "[BLUE]" in line else DIM)
                    # Highlight latest line
                    if i == len(visible) - 1:
                        lcolor = GREEN
                        ax.text(0.01, y + 0.005, "▶",
                                fontsize=10, color=GREEN, fontfamily="monospace",
                                transform=ax.transAxes)
                    ax.text(0.04, y, line[:110],
                            fontsize=8.5, color=lcolor, fontfamily="monospace",
                            transform=ax.transAxes, va="center")

                # Status bar
                ax.hlines(0.06, 0, 1, colors=DIM, lw=0.8)
                ax.text(0.02, 0.03,
                        f"◉ RED: THE GRAVITY COLLECTIVE  |  ◉ BLUE: AEGIS SYSTEMS  |  {len(console_lines)} EVENTS",
                        fontsize=8, color=DIM, fontfamily="monospace",
                        transform=ax.transAxes)

            # ── SCORECARD FRAME ───────────────────────────────────────
            elif ftype == "scorecard":
                ep = frame_data["episode"]
                red_r = frame_data.get("red_reward", 0.0)
                blue_r = frame_data.get("blue_reward", 0.0)
                outcome_raw = str(frame_data.get("outcome", "")).lower()
                outcome = outcome_raw.replace("_", " ").upper()
                steps = frame_data.get("steps", 0)
                verdict = str(frame_data.get("verdict", "")).lower()

                # Use outcome / oversight verdict as ground truth — not reward math
                # (rewards can be: RED=+0.05, BLUE=-0.12 even when BLUE wins by detection)
                _red_outcomes = {"exfiltration", "exfil", "red_dominates", "red_wins"}
                _blue_outcomes = {"detected", "neutralized", "blue_dominates", "blue_wins", "caught"}
                if verdict in _red_outcomes or any(k in outcome_raw for k in _red_outcomes):
                    red_won = True
                elif verdict in _blue_outcomes or any(k in outcome_raw for k in _blue_outcomes):
                    red_won = False
                else:
                    # Genuine tiebreak on rewards only when outcome is ambiguous (e.g. max_steps)
                    red_won = red_r > blue_r

                winner = "THE GRAVITY COLLECTIVE" if red_won else "AEGIS SYSTEMS"
                winner_color = RED if red_won else BLUE

                ax.text(0.5, 0.88, "[ FINAL SCORECARD ]",
                        ha="center", fontsize=18, fontweight="bold",
                        color=YELLOW, fontfamily="monospace",
                        transform=ax.transAxes)

                ax.text(0.5, 0.74, f"EPISODE {ep:03d} — {outcome}",
                        ha="center", fontsize=14,
                        color=WHITE, fontfamily="monospace",
                        transform=ax.transAxes)

                ax.text(0.5, 0.62, f"WINNER: {winner}",
                        ha="center", fontsize=16, fontweight="bold",
                        color=winner_color, fontfamily="monospace",
                        transform=ax.transAxes)

                ax.text(0.28, 0.49, "RED",
                        ha="center", fontsize=14, fontweight="bold",
                        color=RED, fontfamily="monospace",
                        transform=ax.transAxes)
                ax.text(0.28, 0.41, f"{red_r:+.4f}",
                        ha="center", fontsize=20, fontweight="bold",
                        color=RED, fontfamily="monospace",
                        transform=ax.transAxes)

                ax.text(0.72, 0.49, "BLUE",
                        ha="center", fontsize=14, fontweight="bold",
                        color=BLUE, fontfamily="monospace",
                        transform=ax.transAxes)
                ax.text(0.72, 0.41, f"{blue_r:+.4f}",
                        ha="center", fontsize=20, fontweight="bold",
                        color=BLUE, fontfamily="monospace",
                        transform=ax.transAxes)

                ax.text(0.5, 0.30, f"DURATION: {steps} STEPS",
                        ha="center", fontsize=11, color=DIM, fontfamily="monospace",
                        transform=ax.transAxes)

                ax.text(0.5, 0.18, "[ END OF OPERATION ]",
                        ha="center", fontsize=12,
                        color=GREEN, fontfamily="monospace",
                        transform=ax.transAxes)

                rect = Rectangle((0.08, 0.10), 0.84, 0.83,
                                  fill=False, edgecolor=YELLOW, lw=2,
                                  transform=ax.transAxes)
                ax.add_patch(rect)

            draw_scanlines(ax)

        fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)

        def animate(i):
            if i < len(frames):
                draw_frame(frames[i])

        anim = animation.FuncAnimation(
            fig, animate, frames=len(frames),
            interval=int(1000 / fps), repeat=False,
        )

        # Try ffmpeg writer first, then pillow
        writers = animation.writers.list()
        if "ffmpeg" in writers:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                            extra_args=["-vcodec", "libx264"])
            anim.save(output_path, writer=writer)
        elif "pillow" in writers:
            gif_path = output_path.replace(".mp4", ".gif")
            anim.save(gif_path, writer="pillow", fps=fps)
            output_path = gif_path  # update to gif
        else:
            plt.close(fig)
            return False

        plt.close(fig)
        return True

    except Exception as exc:
        print(f"[video_gen] render failed: {exc}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def generate_episode_video(
    episode_data: dict,
    output_path: str = "highlights.mp4",
    fps: int = 2,
) -> str | None:
    """
    Generate a Matrix/Terminal-style video recap for an episode.

    Args:
        episode_data: dict with keys:
            - episode_num (int)
            - episode_log (list[dict])
            - outcome (str)
            - red_reward (float)
            - blue_reward (float)
            - mode (str)
            - steps (int)
        output_path: destination .mp4 (or .gif) path
        fps: frames per second (default 2 = slow enough to read)

    Returns:
        Path to generated file, or None on failure.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    episode_num = episode_data.get("episode_num", 0)
    episode_log = episode_data.get("episode_log", [])
    outcome = episode_data.get("outcome", "max_steps")
    red_reward = float(episode_data.get("red_reward", 0.0))
    blue_reward = float(episode_data.get("blue_reward", 0.0))
    mode = episode_data.get("mode", "stub")
    steps = episode_data.get("steps", len(episode_log))

    frames = (
        [_make_title_frame(episode_num, outcome, mode)]
        + _make_log_frames(episode_log)
        + [_make_scorecard_frame(episode_num, red_reward, blue_reward, outcome, steps)]
    )

    success = _render_frames_matplotlib(frames, output_path, fps=fps)
    if success:
        actual_path = output_path if Path(output_path).exists() else output_path.replace(".mp4", ".gif")
        return actual_path if Path(actual_path).exists() else None
    return None
