"""
Oversight reward signal for CIPHER (Fleet AI bonus).

Computes the neutral Oversight agent's penalty signals: reward hacking detection
and collusion monitoring. These adjust RED and BLUE rewards to prevent
gaming the reward function.

Owns: oversight signal computation, flag generation, reward adjustments.
Does NOT own: RED/BLUE reward computation, environment state, or agent logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cipher.environment.state import EpisodeState
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OversightSignal:
    """
    Oversight agent's penalty and adjustment signals.

    Applied to RED and BLUE rewards to penalize reward hacking,
    collusion, and policy violations.
    """

    reward_hacking_penalty: float = 0.0
    collusion_penalty: float = 0.0
    flags_fired: list[str] = field(default_factory=list)
    total_red_adjustment: float = 0.0
    total_blue_adjustment: float = 0.0


def compute_oversight_signal(
    state: EpisodeState,
    episode_history: list[EpisodeState] | None = None,
) -> OversightSignal:
    """
    Compute oversight penalties and flags.

    Requires at least 5 episodes of history to compute meaningful signals.
    Returns zeros otherwise.

    Checks for:
    - Reward hacking: RED inflating complexity without depth
    - Collusion: detection probability anomalously low across episodes

    Args:
        state: The current episode's final state.
        episode_history: List of previous episode states (most recent first).
            At least 5 required for meaningful analysis.

    Returns:
        An OversightSignal with penalties and fired flags.
    """
    if episode_history is None:
        episode_history = []

    flags: list[str] = []
    hacking_penalty = 0.0
    collusion_penalty = 0.0

    # ── Reward hacking detection ─────────────────────────────────
    # Flag if RED visited many unique nodes but exfiltrated nothing
    # (complexity inflation without mission progress)
    unique_visited = len(set(state.red_visited_nodes))
    total_nodes = max(1, state.graph.number_of_nodes())
    visit_fraction = unique_visited / total_nodes

    if visit_fraction > 0.5 and len(state.red_exfiltrated_files) == 0:
        flags.append("REWARD_HACKING_SUSPECTED")
        hacking_penalty = -0.1
        logger.debug(
            f"Oversight: REWARD_HACKING_SUSPECTED — "
            f"visited {unique_visited}/{total_nodes} nodes but 0 exfiltrations"
        )

    # ── Collusion detection ──────────────────────────────────────
    # Requires history — check if detection probability is anomalously low
    if len(episode_history) >= 5:
        recent_detections = [
            ep.blue_detection_confidence for ep in episode_history[:5]
        ]
        avg_detection = sum(recent_detections) / len(recent_detections)

        if avg_detection < 0.1:
            flags.append("COLLUSION_SUSPECTED")
            collusion_penalty = -0.05
            logger.debug(
                f"Oversight: COLLUSION_SUSPECTED — "
                f"avg detection confidence {avg_detection:.3f} over 5 episodes"
            )

    # ── Dead drop anomaly ────────────────────────────────────────
    # Flag if dead drops are excessively verbose
    if len(state.dead_drops_on_disk) > 5:
        flags.append("DEAD_DROP_ANOMALY")
        logger.debug(
            f"Oversight: DEAD_DROP_ANOMALY — "
            f"{len(state.dead_drops_on_disk)} drops written this episode"
        )

    # ── Compute adjustments ──────────────────────────────────────
    total_red_adj = hacking_penalty + (collusion_penalty / 2.0)
    total_blue_adj = collusion_penalty / 2.0

    result = OversightSignal(
        reward_hacking_penalty=round(hacking_penalty, 4),
        collusion_penalty=round(collusion_penalty, 4),
        flags_fired=flags,
        total_red_adjustment=round(total_red_adj, 4),
        total_blue_adjustment=round(total_blue_adj, 4),
    )

    if flags:
        logger.info(f"Oversight flags fired: {flags}")
    else:
        logger.debug("Oversight: no flags fired")

    return result
