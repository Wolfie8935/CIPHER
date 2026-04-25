"""
Cross-episode performance memory for CIPHER agents.

Persists summarized episode outcomes across a training run. Agents receive
this history in their system prompt so they can learn from past mistakes and
understand what strategies (including emergent actions) correlate with wins.

Architecture: module-level in-memory list (same pattern as coordination.py).
Thread-safe for concurrent agent reads; only the episode runner writes to it.
"""
from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.Lock()
_EPISODE_HISTORY: list[dict[str, Any]] = []
_MAX_HISTORY = 20  # keep last N episodes


def record_episode(summary: dict[str, Any]) -> None:
    """Append a summarized episode outcome to the in-memory history.

    Expected keys in summary:
        episode_number (int), winner (str: 'red'|'blue'|'draw'),
        terminal_reason (str), steps (int),
        red_total_reward (float), blue_total_reward (float),
        red_emergent_used (bool), red_emergent_intents (list[str]),
        blue_emergent_used (bool), blue_emergent_intents (list[str]),
        red_exfiltrated (int), red_suspicion_final (float),
        blue_detection_final (float), zone_stall_occurred (bool),
    """
    with _LOCK:
        _EPISODE_HISTORY.append(summary)
        if len(_EPISODE_HISTORY) > _MAX_HISTORY:
            _EPISODE_HISTORY.pop(0)


def get_recent_summary(n: int = 3) -> str:
    """Return a formatted string of the last N episodes for injection into system prompts.

    Returns an empty string if no history exists yet (first episode).
    """
    with _LOCK:
        recent = _EPISODE_HISTORY[-n:] if _EPISODE_HISTORY else []

    if not recent:
        return ""

    lines = [f"RECENT EPISODES (last {len(recent)}):"]
    for ep in recent:
        ep_num = ep.get("episode_number", "?")
        winner = ep.get("winner", "?").upper()
        reason = ep.get("terminal_reason", "?")
        steps = ep.get("steps", "?")
        red_r = ep.get("red_total_reward", 0.0)
        blue_r = ep.get("blue_total_reward", 0.0)
        red_em = ep.get("red_emergent_intents", [])
        blue_em = ep.get("blue_emergent_intents", [])
        stall = ep.get("zone_stall_occurred", False)
        exfil = ep.get("red_exfiltrated", 0)

        ep_line = (
            f"  Ep {ep_num}: {winner} WIN in {steps} steps "
            f"({reason}). "
            f"RED={red_r:+.2f}, BLUE={blue_r:+.2f}. "
            f"Exfil={exfil}."
        )
        if red_em:
            ep_line += f" RED emergent: {', '.join(red_em)}."
        if blue_em:
            ep_line += f" BLUE emergent: {', '.join(blue_em)}."
        if stall:
            ep_line += " Zone stall occurred."
        lines.append(ep_line)

    # Derive patterns from recent episodes
    patterns = _derive_patterns(recent)
    if patterns:
        lines.append("PATTERNS:")
        for p in patterns:
            lines.append(f"  {p}")

    return "\n".join(lines)


def _derive_patterns(recent: list[dict[str, Any]]) -> list[str]:
    """Derive actionable patterns from recent episode summaries."""
    patterns: list[str] = []

    if len(recent) < 2:
        return patterns

    # Emergent action correlation with wins
    em_wins = [
        ep for ep in recent
        if ep.get("winner") == "red" and ep.get("red_emergent_used")
    ]
    em_losses = [
        ep for ep in recent
        if ep.get("winner") != "red" and not ep.get("red_emergent_used")
    ]
    if len(em_wins) >= 2 and len(em_wins) > len(em_losses):
        patterns.append(
            "PATTERN: Emergent actions correlated with RED wins — use them when stuck."
        )

    # Zone stall correlation with losses
    stall_losses = [
        ep for ep in recent
        if ep.get("zone_stall_occurred") and ep.get("winner") != "red"
    ]
    if len(stall_losses) >= 2:
        patterns.append(
            "PATTERN: Zone stalling correlated with RED losses — move through zones faster."
        )

    # Consecutive losses — trigger exploration directive
    winners = [ep.get("winner") for ep in recent[-3:]]
    if all(w != "red" for w in winners) and len(winners) >= 3:
        patterns.append(
            "PATTERN: RED lost last 3 episodes. Previous strategy is failing. "
            "Try a fundamentally different approach this episode."
        )

    # Consecutive blue losses
    if all(w != "blue" for w in winners) and len(winners) >= 3:
        patterns.append(
            "PATTERN: BLUE failed to detect RED in last 3 episodes. "
            "Standard investigation is insufficient — try emergent detection methods."
        )

    return patterns


def count_consecutive_losses(team: str = "red") -> int:
    """Return the number of consecutive recent losses for the given team.

    Used by Change 6 to trigger adaptive exploration pressure.
    """
    with _LOCK:
        history = list(_EPISODE_HISTORY)

    losses = 0
    for ep in reversed(history):
        if ep.get("winner") != team:
            losses += 1
        else:
            break
    return losses


def clear_history() -> None:
    """Clear all episode history. Called at the start of a fresh training run."""
    with _LOCK:
        _EPISODE_HISTORY.clear()
