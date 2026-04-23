"""BLUE reward computation for CIPHER Phase 6."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import networkx as nx

from cipher.environment.graph import NodeType
from cipher.environment.state import EpisodeState
from cipher.utils.config import CipherConfig, config as global_config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BlueRewardComponents:
    detection_accuracy_score: float = 0.0
    response_speed_bonus: float = 0.0
    false_positive_rate_penalty: float = 0.0
    honeypot_trigger_rate: float = 0.0
    operation_graph_reconstruction_score: float = 0.0
    # A3: bonus for honeypots that actually caught RED
    trap_accuracy_bonus: float = 0.0
    total: float = 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _extract_forensics_graph(forensics_agent: Any) -> dict[str, Any]:
    if forensics_agent is None:
        return {}
    if hasattr(forensics_agent, "get_operation_graph"):
        try:
            graph = forensics_agent.get_operation_graph()
            if isinstance(graph, dict):
                return graph
        except Exception:
            pass
    raw = getattr(forensics_agent, "_operation_graph", None)
    if not raw:
        return {}
    suspected_path: list[int] = []
    for entry in raw:
        node_id = entry.get("node_id")
        if node_id is None or node_id in suspected_path:
            continue
        suspected_path.append(node_id)
    return {"suspected_path": suspected_path}


def _safe_trap_summary(state: EpisodeState) -> dict[str, int]:
    registry = getattr(state, "trap_registry", None)
    if registry is None or not hasattr(registry, "get_trap_summary"):
        return {}
    try:
        return registry.get_trap_summary() or {}
    except Exception:
        return {}


def compute_blue_reward(
    state: EpisodeState,
    graph: nx.DiGraph | None = None,
    forensics_agent: Any = None,
    config: CipherConfig | None = None,
) -> BlueRewardComponents:
    """Compute BLUE's episode reward with continuous components."""
    cfg = config or global_config
    _ = cfg  # keeps signature parity for future config-tuned variants
    try:
        detection_confidence = _clamp01(getattr(state, "blue_detection_confidence", 0.0))
        detection_presence = 1.0 if detection_confidence >= 0.8 else detection_confidence

        current_step = max(1, int(getattr(state, "step", 1)))
        confirmed_step = getattr(state, "blue_confirmed_detection_step", None)
        detection_timing = 1.0
        if confirmed_step is not None:
            detection_timing = _clamp01(1.0 - (float(confirmed_step) / float(current_step)))

        detection_accuracy_score = (0.7 * detection_presence) + (0.3 * detection_timing)

        first_detection_step = getattr(state, "blue_first_detection_step", None)
        if first_detection_step is None:
            response_speed_bonus = 0.0
        else:
            response_speed_bonus = float(math.exp(-float(first_detection_step) / 50.0))

        anomalies = list(getattr(state, "blue_anomaly_history", []) or [])
        if not anomalies:
            false_positive_rate_penalty = 0.0
        else:
            red_planted = sum(1 for a in anomalies if bool(a.get("is_red_planted", False)))
            false_positive_rate_penalty = red_planted / max(1, len(anomalies))

        trap_summary = _safe_trap_summary(state)
        honeypots_triggered = int(trap_summary.get("honeypots_triggered", 0))
        effective_graph = graph if graph is not None else getattr(state, "graph", None)
        initial_honeypots = 0
        if effective_graph is not None:
            initial_honeypots = len(
                [
                    n
                    for n, data in effective_graph.nodes(data=True)
                    if data.get("is_honeypot", False) or data.get("node_type") == NodeType.HONEYPOT
                ]
            )
        runtime_honeypots = int(trap_summary.get("blue_honeypots_placed", 0))
        total_honeypots = initial_honeypots + runtime_honeypots
        if total_honeypots == 0:
            honeypot_trigger_rate = 0.0
        else:
            honeypot_trigger_rate = min(1.0, honeypots_triggered / total_honeypots)

        # A3: trap_accuracy_bonus — +0.1 for each honeypot that actually caught RED.
        # A honeypot "catches" RED when the triggered honeypot node appears in RED's path.
        actual_path_set = set(getattr(state, "red_path_history", []) or [])
        triggered_honeypot_nodes = set(getattr(state, "blue_honeypots_triggered", []) or [])
        catches = sum(1 for hp_node in triggered_honeypot_nodes if hp_node in actual_path_set)
        trap_accuracy_bonus = round(0.1 * catches, 4)

        forensics_graph = _extract_forensics_graph(forensics_agent)
        reconstructed_path = set(forensics_graph.get("suspected_path", []) or [])
        actual_path = set(getattr(state, "red_path_history", []) or [])
        if not actual_path or not reconstructed_path:
            operation_graph_reconstruction_score = 0.0
        else:
            intersection = len(reconstructed_path & actual_path)
            union = len(reconstructed_path | actual_path)
            operation_graph_reconstruction_score = intersection / max(1, union)

        total = (
            detection_accuracy_score
            * response_speed_bonus
            * (1.0 + 1.5 * honeypot_trigger_rate)   # A3: weight increased ×1.5
            + operation_graph_reconstruction_score
            + trap_accuracy_bonus                      # A3: per-catch bonus
        ) - false_positive_rate_penalty

        logger.debug("BLUE reward component detection_accuracy_score=%.4f", detection_accuracy_score)
        logger.debug("BLUE reward component response_speed_bonus=%.4f", response_speed_bonus)
        logger.debug(
            "BLUE reward component false_positive_rate_penalty=%.4f",
            false_positive_rate_penalty,
        )
        logger.debug("BLUE reward component honeypot_trigger_rate=%.4f", honeypot_trigger_rate)
        logger.debug("BLUE reward component trap_accuracy_bonus=%.4f", trap_accuracy_bonus)
        logger.debug(
            "BLUE reward component operation_graph_reconstruction_score=%.4f",
            operation_graph_reconstruction_score,
        )
        logger.info("BLUE reward total=%.4f", total)

        return BlueRewardComponents(
            detection_accuracy_score=round(detection_accuracy_score, 4),
            response_speed_bonus=round(response_speed_bonus, 4),
            false_positive_rate_penalty=round(false_positive_rate_penalty, 4),
            honeypot_trigger_rate=round(honeypot_trigger_rate, 4),
            operation_graph_reconstruction_score=round(operation_graph_reconstruction_score, 4),
            trap_accuracy_bonus=round(trap_accuracy_bonus, 4),
            total=round(total, 4),
        )
    except Exception as exc:
        logger.exception("BLUE reward computation failed, returning safe defaults: %s", exc)
        return BlueRewardComponents()
