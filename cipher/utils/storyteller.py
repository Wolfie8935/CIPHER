"""
cipher/utils/storyteller.py

"The Daily Breach" — AI-powered narrative post-mortems.
After an episode ends, feeds the compressed episode_log to an LLM
and writes a dramatic 3-paragraph cyber-warfare news report.

Saved as: episode_X_report.md  (in episode_reports/)
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

REPORTS_DIR = Path("episode_reports")


def _compress_log(episode_log: list[dict], max_entries: int = 40) -> str:
    """Compress episode log to a readable summary for LLM context."""
    if not episode_log:
        return "No events recorded."

    lines = []
    for entry in episode_log[:max_entries]:
        step = entry.get("step", "?")
        agent = entry.get("agent_id", entry.get("agent", "?"))
        action = entry.get("action_type", entry.get("action", "?"))
        payload = entry.get("payload", {}) or {}
        result = entry.get("result", {}) or {}

        node = payload.get("target_node", "")
        file_ = payload.get("target_file", "")
        success = result.get("success", "")
        reason = result.get("reason", "")

        detail = f"node={node}" if node != "" else ""
        if file_:
            detail += f" file={file_}"
        if success != "":
            detail += f" ok={success}"
        if reason:
            detail += f" ({reason})"

        lines.append(f"  Step {step:>2} | {str(agent):<22} | {str(action):<20} | {detail}")

    if len(episode_log) > max_entries:
        lines.append(f"  ... ({len(episode_log) - max_entries} more entries omitted)")

    return "\n".join(lines)


def _build_prompt(episode_num: int, episode_log: list[dict], outcome: str,
                  red_reward: float, blue_reward: float) -> str:
    log_text = _compress_log(episode_log)
    return f"""You are a cyber-warfare correspondent for "The Daily Breach", a classified intelligence newsletter.

Write a 3-paragraph dramatic news report for Episode {episode_num} of Operation CIPHER.
- RED team is called "The Gravity Collective" — elite APT group
- BLUE team is called "Aegis Systems" — corporate cyber defense unit
- Outcome: {outcome}  (RED reward: {red_reward:+.3f}, BLUE reward: {blue_reward:+.3f})

Use the event log below to identify the turning point. Highlight any trap triggers, exfiltration success, or detection events.

Event Log:
{log_text}

Requirements:
1. Paragraph 1: Set the scene — describe the network infiltration attempt dramatically.
2. Paragraph 2: The turning point — what was the decisive moment? Quote specific steps/nodes if present.
3. Paragraph 3: Outcome & aftermath — who won and what it means strategically.

Write in a gripping, present-tense journalistic style. Max 300 words total. No headers, just three paragraphs.
"""


def generate_report(
    episode_num: int,
    episode_log: list[dict],
    outcome: str = "max_steps",
    red_reward: float = 0.0,
    blue_reward: float = 0.0,
    save: bool = True,
) -> str:
    """
    Generate a dramatic narrative post-mortem for an episode.

    Returns the report text. Saves to episode_reports/episode_X_report.md if save=True.
    Falls back to a template if LLM is unavailable.
    """
    prompt = _build_prompt(episode_num, episode_log, outcome, red_reward, blue_reward)

    report_text = _call_llm(prompt)
    if not report_text:
        report_text = _fallback_report(episode_num, outcome, red_reward, blue_reward, episode_log)

    if save:
        _save_report(episode_num, report_text, outcome, red_reward, blue_reward)

    return report_text


def _call_llm(prompt: str) -> str | None:
    """Try to call the configured LLM via OpenAI-compatible API."""
    try:
        from openai import OpenAI

        api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("HF_BASE_URL", "https://api-inference.huggingface.co/v1/")

        if not api_key:
            return None

        client = OpenAI(api_key=api_key, base_url=base_url)
        model = os.getenv("STORYTELLER_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.85,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return None


def _fallback_report(episode_num: int, outcome: str, red_reward: float,
                     blue_reward: float, episode_log: list[dict]) -> str:
    """Generate a template-based report when LLM is unavailable."""
    outcome_upper = outcome.replace("_", " ").upper()

    # Count key events
    exfils = sum(1 for e in episode_log if str(e.get("action_type", "")).lower() == "exfiltrate")
    traps = sum(1 for e in episode_log if str(e.get("action_type", "")).lower() in
                ("place_honeypot", "plant_breadcrumb", "trigger_false_escalation"))
    max_step = max((e.get("step", 0) for e in episode_log), default=0)

    winner = "The Gravity Collective (RED)" if red_reward > blue_reward else "Aegis Systems (BLUE)"
    loser = "Aegis Systems (BLUE)" if red_reward > blue_reward else "The Gravity Collective (RED)"

    return (
        f"In the shadow of a 50-node enterprise network, Episode {episode_num} saw "
        f"The Gravity Collective launch a calculated intrusion across four security zones. "
        f"Over {max_step} tense steps, RED agents probed the perimeter, mapping vulnerabilities "
        f"while Aegis Systems' analysts scrambled to triangulate the threat signature.\n\n"
        f"The turning point arrived mid-engagement. "
        f"{'Gravity Collective executed ' + str(exfils) + ' exfiltration attempt(s) while ' if exfils else ''}"
        f"Aegis Systems deployed {traps} deception trap(s) — "
        f"{'a honeypot successfully lured RED into a detection corridor' if traps > 0 else 'but none found their mark today'}. "
        f"The network's critical zone held its secrets tightly, each node a potential kill-switch.\n\n"
        f"Final verdict: **{outcome_upper}**. {winner} claimed the strategic advantage "
        f"(reward differential: {red_reward - blue_reward:+.3f}). "
        f"{loser} retreats to recalibrate. "
        f"In the zero-sum theater of cyber warfare, today's margin is tomorrow's lesson."
    )


def _save_report(episode_num: int, text: str, outcome: str,
                 red_reward: float, blue_reward: float) -> Path:
    """Save the report as a markdown file."""
    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = REPORTS_DIR / f"episode_{episode_num:03d}_report.md"

    header = (
        f"# The Daily Breach — Episode {episode_num} After-Action Report\n\n"
        f"> **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        f"> **Outcome:** `{outcome}`  \n"
        f"> **RED Reward:** `{red_reward:+.3f}`  |  **BLUE Reward:** `{blue_reward:+.3f}`\n\n"
        f"---\n\n"
    )

    filename.write_text(header + text + "\n", encoding="utf-8")
    return filename


def load_reports() -> list[dict]:
    """Load all episode reports from disk. Returns list of {episode, path, preview, mtime}."""
    REPORTS_DIR.mkdir(exist_ok=True)
    reports = []
    for p in sorted(REPORTS_DIR.glob("episode_*_report.md"), reverse=True):
        try:
            text = p.read_text(encoding="utf-8")
            # Extract episode number from filename
            parts = p.stem.split("_")
            ep_num = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            reports.append({
                "episode": ep_num,
                "path": str(p),
                "filename": p.name,
                "text": text,
                "preview": text[:200].replace("\n", " ").strip() + "…",
                "mtime": p.stat().st_mtime,
            })
        except Exception:
            continue
    return reports
