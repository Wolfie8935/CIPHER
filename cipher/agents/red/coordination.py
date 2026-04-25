"""
cipher/agents/red/coordination.py

Shared RED team danger map.

All four RED agents (Planner, Analyst, Operative, Exfiltrator) read and write
this module-level dict so they can collectively avoid nodes that caused
suspicion spikes in the current episode.

Usage:
    from cipher.agents.red.coordination import mark_danger, get_danger_score, clear_danger_map

    # When RED detects a suspicion spike at a node:
    mark_danger(node_id, spike_size)

    # When choosing a path, check danger score:
    if get_danger_score(node_id) > 0.3:
        # avoid this node

    # Clear at the start of each episode (called from episode runner):
    clear_danger_map()
"""
from __future__ import annotations

_RED_DANGER_MAP: dict[int, float] = {}


def mark_danger(node_id: int, score: float) -> None:
    """Record or update the danger score for a node.

    Takes the maximum of the new score and any existing score so that
    multiple agents independently observing the same node converge on
    the worst-case assessment.
    """
    _RED_DANGER_MAP[node_id] = max(_RED_DANGER_MAP.get(node_id, 0.0), float(score))


def get_danger_score(node_id: int) -> float:
    """Return the current danger score for a node (0.0 if unknown)."""
    return _RED_DANGER_MAP.get(node_id, 0.0)


def get_danger_nodes(threshold: float = 0.3) -> set[int]:
    """Return the set of nodes whose danger score exceeds the threshold."""
    return {n for n, s in _RED_DANGER_MAP.items() if s > threshold}


def clear_danger_map() -> None:
    """Reset all danger scores. Must be called at the start of each episode."""
    _RED_DANGER_MAP.clear()
