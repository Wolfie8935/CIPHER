"""
Post-episode forensic analysis engine.
Produces a structured crime-scene reconstruction from episode data (Task 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ForensicsReconstruction:
    # Paths
    actual_red_path: list[int] = field(default_factory=list)
    suspected_red_path: list[int] = field(default_factory=list)
    path_accuracy: float = 0.0  # Jaccard similarity

    # Detection quality
    correctly_identified_nodes: list[int] = field(default_factory=list)
    missed_nodes: list[int] = field(default_factory=list)
    false_positive_nodes: list[int] = field(default_factory=list)

    # Trap analysis
    traps_triggered: list[dict] = field(default_factory=list)
    traps_wasted: list[dict] = field(default_factory=list)
    trap_efficiency: float = 0.0

    # Dead drop analysis
    drops_used: list[dict] = field(default_factory=list)
    drops_tampered: list[dict] = field(default_factory=list)
    drop_integrity_rate: float = 1.0

    # Timeline (one entry per step)
    timeline: list[dict] = field(default_factory=list)

    # Summary
    investigation_grade: str = "C"   # A / B / C / D / F
    summary_text: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def reconstruct_crime_scene(
    state: Any,
    episode_log: list[dict],
    forensics_agent: Any = None,
) -> ForensicsReconstruction:
    """Build a complete forensic reconstruction from episode data."""
    recon = ForensicsReconstruction()

    # 1. RED's actual path
    recon.actual_red_path = list(getattr(state, "red_path_history", [])
                                 or getattr(state, "red_visited_nodes", []))

    # 2. BLUE's suspected path (from forensics agent operation graph)
    if forensics_agent and hasattr(forensics_agent, "_operation_graph"):
        for entry in forensics_agent._operation_graph:
            node_id = entry.get("node_id")
            if node_id is not None and node_id not in recon.suspected_red_path:
                recon.suspected_red_path.append(node_id)

    # Fallback: derive suspected path from BLUE investigate_node actions
    if not recon.suspected_red_path:
        for entry in episode_log:
            if (str(entry.get("agent_id", "")).startswith("blue_")
                    and entry.get("action_type") == "investigate_node"):
                node = (entry.get("payload") or {}).get("target_node")
                if node is not None and node not in recon.suspected_red_path:
                    recon.suspected_red_path.append(node)

    # 3. Path accuracy (Jaccard)
    actual_set = set(recon.actual_red_path)
    suspected_set = set(recon.suspected_red_path)
    union = actual_set | suspected_set
    intersection = actual_set & suspected_set
    recon.path_accuracy = len(intersection) / max(1, len(union))

    # 4. Detection quality
    recon.correctly_identified_nodes = sorted(intersection)
    recon.missed_nodes = sorted(actual_set - suspected_set)
    recon.false_positive_nodes = sorted(suspected_set - actual_set)

    # 5. Trap analysis
    registry = getattr(state, "trap_registry", None)
    if registry:
        for trap in getattr(registry, "active_blue_traps", []):
            trap_info = {
                "type": getattr(trap.trap_type, "value", str(trap.trap_type)),
                "node": getattr(trap, "target_node", None),
            }
            if getattr(trap, "is_triggered", False):
                recon.traps_triggered.append(trap_info)
            else:
                recon.traps_wasted.append(trap_info)
        total_traps = len(recon.traps_triggered) + len(recon.traps_wasted)
        recon.trap_efficiency = len(recon.traps_triggered) / max(1, total_traps)

    # 6. Dead drop analysis (from episode log)
    drop_writes = 0
    drop_tampers = 0
    for entry in episode_log:
        atype = entry.get("action_type", "")
        if atype == "write_dead_drop":
            drop_writes += 1
            payload = entry.get("payload") or {}
            recon.drops_used.append({
                "step": entry.get("step"),
                "agent": entry.get("agent_id"),
                "node": payload.get("target_node"),
            })
        elif atype == "tamper_dead_drop":
            drop_tampers += 1
            payload = entry.get("payload") or {}
            recon.drops_tampered.append({
                "step": entry.get("step"),
                "agent": entry.get("agent_id"),
                "node": payload.get("target_node"),
            })
    recon.drop_integrity_rate = 1.0 - (drop_tampers / max(1, drop_writes + drop_tampers))

    # 7. Timeline
    for entry in sorted(episode_log, key=lambda e: (e.get("step", 0), e.get("agent_id", ""))):
        agent_id = str(entry.get("agent_id", ""))
        team = "red" if agent_id.startswith("red_") else "blue" if agent_id.startswith("blue_") else "oversight"
        recon.timeline.append({
            "step": entry.get("step", 0),
            "team": team,
            "agent": agent_id,
            "action": str(entry.get("action_type", "")),
            "details": str((entry.get("payload") or {}))[:100],
        })

    # 8. Investigation grade
    acc = recon.path_accuracy
    if acc >= 0.80:
        recon.investigation_grade = "A"
    elif acc >= 0.60:
        recon.investigation_grade = "B"
    elif acc >= 0.40:
        recon.investigation_grade = "C"
    elif acc >= 0.20:
        recon.investigation_grade = "D"
    else:
        recon.investigation_grade = "F"

    recon.summary_text = (
        f"BLUE reconstructed {len(recon.correctly_identified_nodes)}/{max(1, len(actual_set))} "
        f"of RED's nodes (accuracy {recon.path_accuracy:.0%}). "
        f"Missed {len(recon.missed_nodes)} nodes. "
        f"{len(recon.false_positive_nodes)} false positive(s). "
        f"Trap efficiency {recon.trap_efficiency:.0%}. "
        f"Grade: {recon.investigation_grade}"
    )
    return recon
