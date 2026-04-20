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

from cipher.utils.config import config
from cipher.utils.logger import get_logger, log_reward

logger = get_logger(__name__)


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

        for episode_num in range(1, self.n_episodes + 1):
            logger.info(f"═══ Training Episode {episode_num}/{self.n_episodes} ═══")

            try:
                red_total, blue_total = run_episode(
                    episode_number=episode_num,
                    max_steps=10,  # Short episodes for Phase 1
                    verbose=False,
                )
                log_reward(
                    logger,
                    f"Episode {episode_num}: RED={red_total:.4f}  BLUE={blue_total:.4f}",
                )
            except Exception as exc:
                logger.error(
                    f"Episode {episode_num} failed: {exc}",
                    exc_info=True,
                )

        logger.info("Training loop complete")
