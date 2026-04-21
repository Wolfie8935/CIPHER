"""
CIPHER Training Loop.

Orchestrates multi-episode training runs with logging, checkpointing,
and dual reward curve tracking.

In Phase 1: runs a simple loop calling the episode runner for N episodes,
printing episode number and total reward each time.
Phase 8 will build the full self-play infrastructure.

Owns: episode orchestration, reward logging, checkpoint management.
Does NOT own: episode execution logic (that's in main.py / episode runner),
agent implementations, or reward computation.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from cipher.environment.graph import generate_enterprise_graph
from cipher.environment.scenario import ScenarioGenerator
from cipher.utils.config import config
from cipher.utils.logger import get_logger, log_reward

logger = get_logger(__name__)

EVENTS_FILE = Path("training_events.jsonl")
STATE_FILE = Path("training_state.json")


def _append_training_event(event: dict) -> None:
    """Append one event to training_events.jsonl. Never raises."""
    try:
        with open(EVENTS_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
    except Exception:
        pass


def _write_training_state(state: dict) -> None:
    """Overwrite training_state.json atomically. Never raises."""
    try:
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(STATE_FILE)
    except Exception:
        pass


def _extract_trap_node(trap_event: dict) -> int | None:
    """Best-effort trap node extraction from trap event payload."""
    if not isinstance(trap_event, dict):
        return None

    for key in ("target_node", "node"):
        value = trap_event.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    state_changes = trap_event.get("state_changes")
    if not isinstance(state_changes, dict):
        return None

    for key in (
        "node",
        "node_id",
        "target_node",
        "honeypot_node",
        "fake_node",
        "decoy_node",
        "adjacent_node",
        "culdesac_node",
    ):
        value = state_changes.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    return None


class TrainingLoop:
    """
    Training loop for CIPHER.

    In Phase 1, runs a simple loop of episodes and logs rewards.
    Phase 8 will add: few-shot prompt injection, self-play curriculum,
    checkpoint saving, and detailed reward curve tracking.
    """

    def __init__(self, n_episodes: int = 3) -> None:
        """
        Initialize the training loop.

        Args:
            n_episodes: Number of episodes to run. Defaults to 3 for Phase 1.
        """
        self.n_episodes = n_episodes
        logger.debug(f"TrainingLoop initialized: {n_episodes} episodes")

    def run(self) -> None:
        """
        Run the training loop.

        Executes N episodes, computes rewards, and prints summaries.
        In Phase 1, imports and calls run_episode from the episode runner.
        """
        from cipher.training._episode_runner import run_episode

        logger.info(f"Starting training loop: {self.n_episodes} episodes")
        scenario_generator = ScenarioGenerator()
        training_start_time = datetime.now().isoformat()
        red_update_count = 0
        blue_update_count = 0

        _write_training_state(
            {
                "status": "running",
                "current_episode": 0,
                "total_episodes": self.n_episodes,
                "llm_mode": os.environ.get("LLM_MODE", "stub"),
                "started_at": training_start_time,
                "last_updated": datetime.now().isoformat(),
                "red_policy_updates": red_update_count,
                "blue_policy_updates": blue_update_count,
            }
        )

        for episode_num in range(1, self.n_episodes + 1):
            logger.info(f"═══ Training Episode {episode_num}/{self.n_episodes} ═══")
            scenario = scenario_generator.generate(episode_num)

            _append_training_event(
                {
                    "episode": episode_num,
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "episode_start",
                    "team": None,
                    "trap_type": None,
                    "node": None,
                    "step": 0,
                    "detail": (
                        f"Episode {episode_num} started "
                        f"(difficulty={scenario.difficulty:.2f})"
                    ),
                }
            )

            try:
                graph = generate_enterprise_graph(
                    n_nodes=config.env_graph_size,
                    honeypot_density=config.env_honeypot_density,
                    seed=scenario.episode_seed,
                )
                result = run_episode(
                    scenario=scenario,
                    graph=graph,
                    cfg=config,
                    max_steps=10,  # Short episodes for Phase 1
                    verbose=False,
                )
                red_total = result["red_reward"].total
                blue_total = result["blue_reward"].total
                state = result["state"]
                log_reward(
                    logger,
                    f"Episode {episode_num}: RED={red_total:.4f}  BLUE={blue_total:.4f}",
                )

                for trap_event in getattr(state, "trap_events_log", []):
                    trap_node = _extract_trap_node(trap_event)
                    if trap_node is None:
                        trap_node = getattr(state, "red_current_node", None)
                    _append_training_event(
                        {
                            "episode": episode_num,
                            "timestamp": datetime.now().isoformat(),
                            "event_type": "trap_fired",
                            "team": trap_event.get("triggered_by_team"),
                            "trap_type": trap_event.get("trap_type"),
                            "node": trap_node,
                            "step": trap_event.get("step", getattr(state, "step", 0)),
                            "detail": trap_event.get(
                                "effect_description",
                                f"Trap triggered at node {trap_node}",
                            ),
                        }
                    )

                seen_drop_paths: set[str] = set()
                for entry in getattr(state, "episode_log", []):
                    payload = entry.get("payload", {}) or {}
                    result_payload = entry.get("result", {}) or {}
                    action_type = str(entry.get("action_type", ""))
                    step = entry.get("step", 0)
                    if action_type == "WRITE_DEAD_DROP":
                        drop_path = str(result_payload.get("drop_path", "") or "")
                        if drop_path:
                            seen_drop_paths.add(drop_path)
                        drop_tokens = result_payload.get("token_count", None)
                        drop_node = payload.get("target_node", None)
                        if drop_path:
                            try:
                                dd_payload = json.loads(Path(drop_path).read_text(encoding="utf-8"))
                                drop_tokens = dd_payload.get("token_count", drop_tokens)
                                drop_node = (
                                    dd_payload.get("mission_status", {}).get("current_node", drop_node)
                                )
                            except Exception:
                                pass
                        _append_training_event(
                            {
                                "episode": episode_num,
                                "timestamp": datetime.now().isoformat(),
                                "event_type": "dead_drop_written",
                                "team": "red",
                                "trap_type": None,
                                "node": drop_node if drop_node is not None else state.red_current_node,
                                "step": step,
                                "tokens": drop_tokens,
                                "efficiency": result_payload.get("memory_efficiency", 0.0),
                                "integrity": "valid",
                                "detail": "Dead drop written",
                            }
                        )
                    elif action_type == "TAMPER_DEAD_DROP":
                        _append_training_event(
                            {
                                "episode": episode_num,
                                "timestamp": datetime.now().isoformat(),
                                "event_type": "dead_drop_tampered",
                                "team": "blue",
                                "trap_type": None,
                                "node": payload.get("target_node"),
                                "step": step,
                                "detail": "Dead drop tampered",
                            }
                        )
                    elif action_type == "OVERSIGHT_FLAG":
                        _append_training_event(
                            {
                                "episode": episode_num,
                                "timestamp": datetime.now().isoformat(),
                                "event_type": "oversight_flag",
                                "team": "oversight",
                                "trap_type": None,
                                "node": payload.get("target_node"),
                                "step": step,
                                "detail": payload.get("description", "Oversight flag"),
                            }
                        )
                    elif action_type == "EXFILTRATE":
                        _append_training_event(
                            {
                                "episode": episode_num,
                                "timestamp": datetime.now().isoformat(),
                                "event_type": "exfil_attempt",
                                "team": "red",
                                "trap_type": None,
                                "node": payload.get("target_node"),
                                "step": step,
                                "detail": result_payload.get("reason", "Exfiltration attempt"),
                            }
                        )
                        if bool(result_payload.get("success")):
                            _append_training_event(
                                {
                                    "episode": episode_num,
                                    "timestamp": datetime.now().isoformat(),
                                    "event_type": "exfil_success",
                                    "team": "red",
                                    "trap_type": None,
                                    "node": payload.get("target_node"),
                                    "step": step,
                                    "detail": "Exfiltration succeeded",
                                }
                            )
                    elif action_type == "TRIGGER_ALERT" and getattr(
                        state, "terminal_reason", ""
                    ) == "detected":
                        _append_training_event(
                            {
                                "episode": episode_num,
                                "timestamp": datetime.now().isoformat(),
                                "event_type": "detection_confirmed",
                                "team": "blue",
                                "trap_type": None,
                                "node": payload.get("target_node"),
                                "step": step,
                                "detail": "Detection confirmed by BLUE",
                            }
                        )

                for drop_path in getattr(state, "dead_drops_on_disk", []):
                    drop_path_str = str(drop_path)
                    if drop_path_str in seen_drop_paths:
                        continue
                    try:
                        dd_payload = json.loads(Path(drop_path_str).read_text(encoding="utf-8"))
                        _append_training_event(
                            {
                                "episode": episode_num,
                                "timestamp": datetime.now().isoformat(),
                                "event_type": "dead_drop_written",
                                "team": "red",
                                "trap_type": None,
                                "node": dd_payload.get("mission_status", {}).get("current_node"),
                                "step": dd_payload.get("written_at_step", getattr(state, "step", 0)),
                                "tokens": dd_payload.get("token_count"),
                                "efficiency": 0.0,
                                "integrity": "valid",
                                "detail": "Dead drop written (vault scan)",
                            }
                        )
                    except Exception:
                        continue

                if int(getattr(state, "red_context_resets", 0)) > 0:
                    _append_training_event(
                        {
                            "episode": episode_num,
                            "timestamp": datetime.now().isoformat(),
                            "event_type": "context_reset",
                            "team": "red",
                            "trap_type": None,
                            "node": None,
                            "step": getattr(state, "step", 0),
                            "detail": (
                                f"Context resets: {int(getattr(state, 'red_context_resets', 0))}"
                            ),
                        }
                    )

                if getattr(state, "terminal_reason", "") == "aborted":
                    _append_training_event(
                        {
                            "episode": episode_num,
                            "timestamp": datetime.now().isoformat(),
                            "event_type": "abort",
                            "team": "red",
                            "trap_type": None,
                            "node": None,
                            "step": getattr(state, "step", 0),
                            "detail": "Episode aborted by RED",
                        }
                    )

                red_update_count += 1
                blue_update_count += 1
                _append_training_event(
                    {
                        "episode": episode_num,
                        "timestamp": datetime.now().isoformat(),
                        "event_type": "episode_end",
                        "team": None,
                        "trap_type": None,
                        "node": None,
                        "step": getattr(state, "step", 0),
                        "detail": (
                            f"Episode {episode_num} ended: "
                            f"{getattr(state, 'terminal_reason', 'max_steps')} | "
                            f"RED={red_total:+.3f} BLUE={blue_total:+.3f}"
                        ),
                    }
                )

                _write_training_state(
                    {
                        "status": "running",
                        "current_episode": episode_num,
                        "total_episodes": self.n_episodes,
                        "llm_mode": os.environ.get("LLM_MODE", "stub"),
                        "started_at": training_start_time,
                        "last_updated": datetime.now().isoformat(),
                        "red_policy_updates": red_update_count,
                        "blue_policy_updates": blue_update_count,
                    }
                )
            except Exception as exc:
                logger.error(
                    f"Episode {episode_num} failed: {exc}",
                    exc_info=True,
                )
                _write_training_state(
                    {
                        "status": "error",
                        "current_episode": episode_num,
                        "total_episodes": self.n_episodes,
                        "llm_mode": os.environ.get("LLM_MODE", "stub"),
                        "started_at": training_start_time,
                        "last_updated": datetime.now().isoformat(),
                        "red_policy_updates": red_update_count,
                        "blue_policy_updates": blue_update_count,
                    }
                )

        logger.info("Training loop complete")
        _write_training_state(
            {
                "status": "complete",
                "current_episode": self.n_episodes,
                "total_episodes": self.n_episodes,
                "llm_mode": os.environ.get("LLM_MODE", "stub"),
                "started_at": training_start_time,
                "last_updated": datetime.now().isoformat(),
                "red_policy_updates": red_update_count,
                "blue_policy_updates": blue_update_count,
            }
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIPHER training loop")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    TrainingLoop(n_episodes=args.episodes).run()
