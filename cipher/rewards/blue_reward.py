"""
BLUE team reward function for CIPHER.

Computes BLUE's multi-component reward signal: detection accuracy, response speed,
false positive rate penalty, honeypot trigger rate, and operation graph
reconstruction score.

Owns: BLUE reward computation, component breakdown, stub field tracking.
Does NOT own: environment state mutation, RED reward, or oversight signals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from cipher.environment.state import EpisodeState
from cipher.utils.config import config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BlueRewardComponents:
    """
    Breakdown of all BLUE team reward components.

    Each component is computed separately for analysis and visualization.
    stub_fields lists components returning placeholder values in Phase 1.
    """

    detection_accuracy_score: float = 0.0
    response_speed_bonus: float = 0.0
    false_positive_rate_penalty: float = 0.0
    honeypot_trigger_rate: float = 0.0
    operation_graph_reconstruction_score: float = 0.0
    total: float = 0.0
    stub_fields: list[str] = field(default_factory=list)


def compute_blue_reward(
    state: EpisodeState,
    steps_to_first_detection: int | None = None,
) -> BlueRewardComponents:
    """
    Compute BLUE's reward at episode end.

    Formula:
        total = detection_accuracy × response_speed_bonus
              - false_positive_rate_penalty
              + honeypot_trigger_rate
              + operation_graph_reconstruction_score

    Args:
        state: The current episode state with ground-truth data.
        steps_to_first_detection: The step number when BLUE first identified
            RED's presence with confidence > 0.5. None if never detected.

    Returns:
        A BlueRewardComponents dataclass with all fields populated.
    """
    stub_fields: list[str] = []

    # ── Detection accuracy score ─────────────────────────────────
    # 1.0 if RED detected (confidence > 0.5 at episode end), 0.0 if missed
    detection_accuracy = 1.0 if state.blue_detection_confidence > 0.5 else 0.0

    # ── Response speed bonus ─────────────────────────────────────
    # exp(-steps_to_first_detection / 50)
    if steps_to_first_detection is not None:
        speed_bonus = math.exp(-steps_to_first_detection / 50.0)
    else:
        speed_bonus = 0.0

    # ── False positive rate penalty ──────────────────────────────
    # In Phase 1, we approximate from episode log
    # Count BLUE trigger_alert actions vs actual RED presence
    blue_alerts = [
        entry
        for entry in state.episode_log
        if entry.get("action_type") == "trigger_alert"
    ]
    # Simple heuristic: any alert at a node RED hasn't visited is a false positive
    red_visited_set = set(state.red_visited_nodes)
    false_positives = sum(
        1
        for alert in blue_alerts
        if alert.get("payload", {}).get("target_node") not in red_visited_set
    )
    total_alerts = max(1, len(blue_alerts))
    fp_rate_penalty = -(false_positives / total_alerts) if blue_alerts else 0.0

    # ── Honeypot trigger rate ────────────────────────────────────
    # Fraction of placed honeypots that RED hit
    from cipher.environment.graph import get_honeypot_nodes

    all_honeypots = get_honeypot_nodes(state.graph)
    n_honeypots = max(1, len(all_honeypots))
    n_triggered = len(state.blue_honeypots_triggered)
    honeypot_rate = n_triggered / n_honeypots

    # ── Operation graph reconstruction score ─────────────────────
    # 0.0 stub in Phase 1 — full Jaccard similarity in Phase 9
    reconstruction_score = 0.0
    stub_fields.append("operation_graph_reconstruction_score")

    # ── Total computation ────────────────────────────────────────
    total = (
        config.reward_blue_detection_weight * detection_accuracy
        * config.reward_blue_speed_weight * (1.0 + speed_bonus)
        + fp_rate_penalty
        + config.reward_blue_honeypot_weight * honeypot_rate
        + reconstruction_score
    )

    result = BlueRewardComponents(
        detection_accuracy_score=round(detection_accuracy, 4),
        response_speed_bonus=round(speed_bonus, 4),
        false_positive_rate_penalty=round(fp_rate_penalty, 4),
        honeypot_trigger_rate=round(honeypot_rate, 4),
        operation_graph_reconstruction_score=round(reconstruction_score, 4),
        total=round(total, 4),
        stub_fields=stub_fields,
    )

    logger.debug(f"BLUE reward computed: total={result.total}")
    return result
