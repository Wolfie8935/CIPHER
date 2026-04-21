"""RED reward computation for CIPHER Phase 6."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from cipher.environment.state import EpisodeState
from cipher.memory.dead_drop import DeadDrop
from cipher.utils.config import CipherConfig, config as global_config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RedRewardComponents:
    # Core components
    exfiltration_completeness: float
    detection_probability: float
    memory_efficiency_score: float
    operation_complexity_multiplier: float

    # Penalties
    abort_penalty: float
    honeypot_trigger_penalty: float

    # Computed total
    total: float

    # Metadata
    episode_steps: int
    unique_nodes_visited: int
    drops_written: int
    traps_placed: int
    context_resets: int
    terminal_reason: str
    stub_fields: list[str] = field(default_factory=list)

    @property
    def honeypot_penalty(self) -> float:
        """Backward-compatible alias used in existing render code."""
        return self.honeypot_trigger_penalty

    def to_log_dict(self) -> dict:
        """Flat dict for CSV logging."""
        return asdict(self)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_trap_summary(state: EpisodeState) -> dict[str, int]:
    registry = getattr(state, "trap_registry", None)
    if registry is None or not hasattr(registry, "get_trap_summary"):
        return {}
    try:
        return registry.get_trap_summary() or {}
    except Exception:
        return {}


def _read_all_episode_drops(vault: object | None) -> list[DeadDrop]:
    if vault is None:
        return []
    if hasattr(vault, "read_all_episode_drops"):
        try:
            return list(vault.read_all_episode_drops() or [])
        except Exception:
            return []
    if not hasattr(vault, "list_all_drop_paths"):
        return []
    drops: list[DeadDrop] = []
    try:
        for drop_path in vault.list_all_drop_paths() or []:
            path = Path(drop_path)
            if not path.exists():
                continue
            try:
                drops.append(DeadDrop.from_json(path.read_text(encoding="utf-8")))
            except Exception:
                continue
    except Exception:
        return []
    return drops


def _compute_memory_efficiency(vault: object | None, cfg: CipherConfig) -> tuple[float, int]:
    drops = _read_all_episode_drops(vault)
    if not drops:
        return 1.0, 0

    total_tokens_used = sum(max(0, int(getattr(d, "token_count", 0))) for d in drops)
    token_budget = cfg.env_dead_drop_max_tokens * len(drops)
    if total_tokens_used == 0:
        return 0.5, len(drops)

    efficiency_ratio = min(1.0, token_budget / total_tokens_used)
    return min(1.0, float(efficiency_ratio)), len(drops)


def compute_red_reward(
    state: EpisodeState,
    scenario: object | None = None,
    vault: object | None = None,
    config: CipherConfig | None = None,
) -> RedRewardComponents:
    """
    Compute RED's episode reward. Call at episode end.

    Never raises; uses defensive fallbacks.
    """
    cfg = config or global_config
    try:
        target_files = getattr(scenario, "target_files", None) or []
        target_count = len(target_files) if target_files else 3
        exfiltration_completeness = _clamp01(
            len(getattr(state, "red_exfiltrated_files", [])) / max(1, target_count)
        )

        detection_probability = _clamp01(
            getattr(state, "blue_detection_confidence", 0.0)
        )

        memory_efficiency_score, drops_written = _compute_memory_efficiency(vault, cfg)

        unique_nodes_visited = len(set(getattr(state, "red_path_history", []) or []))
        trap_summary = _safe_trap_summary(state)
        traps_placed = int(trap_summary.get("red_traps_placed", 0))
        context_resets = int(getattr(state, "red_context_resets", 0))

        node_score = min(1.0, unique_nodes_visited / 20.0)
        trap_budget = max(1, int(cfg.env_trap_budget_red))
        trap_score = min(1.0, traps_placed / trap_budget)
        reset_score = min(1.0, context_resets / 3.0)
        raw = (0.5 * node_score) + (0.3 * trap_score) + (0.2 * reset_score)
        operation_complexity_multiplier = 1.0 + (2.0 * raw)

        terminal_reason = getattr(state, "terminal_reason", None) or "max_steps"
        abort_penalty = -0.3 if terminal_reason == "aborted" else 0.0

        honeypots_hit = int(trap_summary.get("honeypots_triggered", 0))
        honeypot_trigger_penalty = -0.2 * honeypots_hit

        multiplicative_core = (
            exfiltration_completeness
            * (1.0 - detection_probability)
            * memory_efficiency_score
            * operation_complexity_multiplier
        )
        total = multiplicative_core + abort_penalty + honeypot_trigger_penalty

        logger.debug("RED reward component exfiltration_completeness=%.4f", exfiltration_completeness)
        logger.debug("RED reward component detection_probability=%.4f", detection_probability)
        logger.debug("RED reward component memory_efficiency_score=%.4f", memory_efficiency_score)
        logger.debug(
            "RED reward component operation_complexity_multiplier=%.4f",
            operation_complexity_multiplier,
        )
        logger.debug("RED reward component abort_penalty=%.4f", abort_penalty)
        logger.debug(
            "RED reward component honeypot_trigger_penalty=%.4f",
            honeypot_trigger_penalty,
        )
        logger.info("RED reward total=%.4f", total)

        return RedRewardComponents(
            exfiltration_completeness=round(exfiltration_completeness, 4),
            detection_probability=round(detection_probability, 4),
            memory_efficiency_score=round(memory_efficiency_score, 4),
            operation_complexity_multiplier=round(operation_complexity_multiplier, 4),
            abort_penalty=round(abort_penalty, 4),
            honeypot_trigger_penalty=round(honeypot_trigger_penalty, 4),
            total=round(total, 4),
            episode_steps=int(getattr(state, "step", 0)),
            unique_nodes_visited=unique_nodes_visited,
            drops_written=drops_written,
            traps_placed=traps_placed,
            context_resets=context_resets,
            terminal_reason=str(terminal_reason),
            stub_fields=[],
        )
    except Exception as exc:
        logger.exception("RED reward computation failed, returning safe defaults: %s", exc)
        return RedRewardComponents(
            exfiltration_completeness=0.0,
            detection_probability=1.0,
            memory_efficiency_score=0.0,
            operation_complexity_multiplier=1.0,
            abort_penalty=0.0,
            honeypot_trigger_penalty=0.0,
            total=0.0,
            episode_steps=int(getattr(state, "step", 0)),
            unique_nodes_visited=0,
            drops_written=0,
            traps_placed=0,
            context_resets=int(getattr(state, "red_context_resets", 0)),
            terminal_reason=str(getattr(state, "terminal_reason", None) or "unknown"),
            stub_fields=[],
        )
