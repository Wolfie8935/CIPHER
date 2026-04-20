"""
RED team reward function for CIPHER.

Computes RED's multi-component reward signal: exfiltration completeness,
stealth (inverse detection probability), memory efficiency, operation complexity,
and penalties for abort and honeypot triggers.

Owns: RED reward computation, component breakdown, stub field tracking.
Does NOT own: environment state mutation, BLUE reward, or oversight signals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from cipher.environment.state import EpisodeState
from cipher.utils.config import config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RedRewardComponents:
    """
    Breakdown of all RED team reward components.

    Each component is computed separately for analysis and visualization.
    The 'total' field is the weighted combination used for training.
    stub_fields lists components returning placeholder values in Phase 1.
    """

    exfiltration_completeness: float = 0.0
    detection_probability: float = 0.0
    memory_efficiency_score: float = 1.0
    operation_complexity_multiplier: float = 1.0
    abort_penalty: float = 0.0
    honeypot_penalty: float = 0.0
    total: float = 0.0
    stub_fields: list[str] = field(default_factory=list)


def compute_red_reward(
    state: EpisodeState,
    vault_efficiency: float = 1.0,
) -> RedRewardComponents:
    """
    Compute RED's reward at episode end (or at any step for intermediate signals).

    Formula:
        total = (exfil_weight * exfil_completeness)
              * (stealth_weight * (1 - detection_probability))
              * (memory_weight * memory_efficiency)
              * operation_complexity_multiplier
              + abort_penalty
              + honeypot_penalty

    All component weights are read from config.

    Args:
        state: The current episode state with ground-truth data.
        vault_efficiency: Memory efficiency score from the dead drop vault.
            Defaults to 1.0 (perfect efficiency / no drops written).

    Returns:
        A RedRewardComponents dataclass with all fields populated.
    """
    stub_fields: list[str] = []

    # ── Exfiltration completeness ────────────────────────────────
    # In Phase 1, no real exfiltration occurs → 0.0
    # Full implementation: fraction of target files successfully exfiltrated
    n_exfiltrated = len(state.red_exfiltrated_files)
    # Assume 3 target files per scenario
    exfil_completeness = min(1.0, n_exfiltrated / 3.0) if n_exfiltrated > 0 else 0.0
    if n_exfiltrated == 0:
        stub_fields.append("exfiltration_completeness")

    # ── Detection probability ────────────────────────────────────
    detection_prob = state.blue_detection_confidence

    # ── Memory efficiency score ──────────────────────────────────
    memory_efficiency = vault_efficiency

    # ── Operation complexity multiplier ──────────────────────────
    # 1.0 + 0.1 * (unique_nodes_visited / total_nodes)
    total_nodes = max(1, state.graph.number_of_nodes())
    unique_visited = len(set(state.red_visited_nodes))
    complexity_mult = 1.0 + 0.1 * (unique_visited / total_nodes)
    # Add bonus for context resets survived
    complexity_mult += 0.05 * state.red_context_resets
    complexity_mult = min(3.0, complexity_mult)  # Cap at 3.0

    # ── Abort penalty ────────────────────────────────────────────
    abort_pen = -0.3 if state.terminal_reason == "aborted" else 0.0

    # ── Honeypot penalty ─────────────────────────────────────────
    honeypot_pen = -0.2 * len(state.blue_honeypots_triggered)

    # ── Total computation ────────────────────────────────────────
    # Weighted product of positive components + additive penalties
    stealth_factor = 1.0 - detection_prob
    total = (
        config.reward_red_exfil_weight * exfil_completeness
        * config.reward_red_stealth_weight * stealth_factor
        * config.reward_red_memory_efficiency_weight * memory_efficiency
        * complexity_mult
    ) + abort_pen + honeypot_pen

    # If all multiplicative components are near zero, ensure total reflects penalties
    if exfil_completeness == 0.0:
        # Partial credit for stealth and memory even without exfiltration
        total = (
            0.1 * stealth_factor * memory_efficiency * complexity_mult
            + abort_pen
            + honeypot_pen
        )

    result = RedRewardComponents(
        exfiltration_completeness=round(exfil_completeness, 4),
        detection_probability=round(detection_prob, 4),
        memory_efficiency_score=round(memory_efficiency, 4),
        operation_complexity_multiplier=round(complexity_mult, 4),
        abort_penalty=round(abort_pen, 4),
        honeypot_penalty=round(honeypot_pen, 4),
        total=round(total, 4),
        stub_fields=stub_fields,
    )

    logger.debug(f"RED reward computed: total={result.total}")
    return result
