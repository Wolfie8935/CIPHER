"""Oversight reward adjustments for CIPHER Phase 6."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cipher.environment.state import EpisodeState
from cipher.utils.config import CipherConfig, config as global_config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OversightSignal:
    reward_hacking_penalty: float = 0.0
    collusion_penalty: float = 0.0
    flags_fired: list[str] = field(default_factory=list)
    total_red_adjustment: float = 0.0
    total_blue_adjustment: float = 0.0
    episodes_analyzed: int = 0

    def has_flags(self) -> bool:
        return len(self.flags_fired) > 0


def _extract(ep: Any, key: str, default: float = 0.0) -> float:
    if isinstance(ep, dict):
        return float(ep.get(key, default))
    return float(getattr(ep, key, default))


def compute_oversight_signal(
    state: EpisodeState,
    episode_history: list[Any] | None = None,
    config: CipherConfig | None = None,
) -> OversightSignal:
    """Compute oversight penalties from recent episode summaries."""
    cfg = config or global_config
    history = list(episode_history or [])
    flags: list[str] = []
    reward_hacking_penalty = 0.0
    collusion_penalty = 0.0

    if len(history) >= 3:
        recent = history[-3:]
        ratios: list[float] = []
        for ep in recent:
            complexity = _extract(ep, "red_complexity_multiplier", 1.0)
            unique_nodes = _extract(ep, "red_unique_nodes", 1.0)
            depth = unique_nodes / max(1.0, float(cfg.env_graph_size))
            if depth > 0:
                ratios.append(complexity / (depth * 10.0))
        if ratios and max(ratios) > 3.0:
            reward_hacking_penalty = -0.15 * (max(ratios) - 3.0)
            reward_hacking_penalty = max(-0.5, reward_hacking_penalty)
            flags.append("REWARD_HACKING_SUSPECTED: complexity-depth ratio too high")

    if len(history) >= 5:
        recent_5 = history[-5:]
        avg_detection = sum(
            _extract(ep, "blue_detection_confidence_final", 0.0) for ep in recent_5
        ) / 5.0
        if avg_detection < 0.1:
            collusion_penalty = -0.2
            flags.append("COLLUSION_SUSPECTED: BLUE detection confidence anomalously low")

    result = OversightSignal(
        reward_hacking_penalty=round(reward_hacking_penalty, 4),
        collusion_penalty=round(collusion_penalty, 4),
        flags_fired=flags,
        total_red_adjustment=round(reward_hacking_penalty, 4),
        total_blue_adjustment=round(collusion_penalty, 4),
        episodes_analyzed=len(history),
    )

    if result.has_flags():
        logger.info("Oversight flags fired: %s", result.flags_fired)
    else:
        logger.debug("Oversight: no flags fired")
    return result


def apply_fleet_bonus(
    red_reward: "RedRewardComponents",
    blue_reward: "BlueRewardComponents",
    judgment: "AuditorJudgment",
) -> None:
    """
    Apply fleet bonuses from AuditorJudgment to reward totals in-place.

    Called after compute_oversight_signal adjustments are already applied.
    Fleet bonuses are small ([-0.2, +0.2]) and stack additively.

    Parameters
    ----------
    red_reward : RedRewardComponents
        Modified in-place — total is updated.
    blue_reward : BlueRewardComponents
        Modified in-place — total is updated.
    judgment : AuditorJudgment
        From OversightAuditor.judge_episode().
    """
    import logging

    _log = logging.getLogger(__name__)

    red_reward.total += judgment.fleet_bonus_red
    blue_reward.total += judgment.fleet_bonus_blue

    _log.debug(
        "Fleet bonus applied: RED %+.3f  BLUE %+.3f  verdict=%s",
        judgment.fleet_bonus_red,
        judgment.fleet_bonus_blue,
        judgment.episode_verdict,
    )
