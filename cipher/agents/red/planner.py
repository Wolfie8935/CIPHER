"""
RED Planner agent for CIPHER.

The strategic mind behind the infiltration operation. Designs the overall plan,
sets objectives, manages risk exposure, and decides when to abort.

In Phase 1: selects random valid actions and occasionally writes dead drops.
Phase 4 will connect this to the NVIDIA LLM backend.

Owns: mission planning, abort decisions, dead drop scheduling.
Does NOT own: environment traversal execution (Operative), risk analysis (Analyst),
or data extraction (Exfiltrator).
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import BlueObservation, RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger, log_red

logger = get_logger(__name__)


class RedPlanner(BaseAgent):
    """
    RED Planner — mission architect and strategic decision-maker.

    In Phase 1, selects random valid RED actions with a 20% probability
    of emitting a WRITE_DEAD_DROP action each step.
    """

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(
            agent_id=agent_id,
            team="red",
            role="planner",
            config=config,
        )

    def observe(self, observation: RedObservation | BlueObservation) -> None:
        """Store the latest RED observation for use in act()."""
        self._current_observation = observation
        self.step_count += 1

    def act(self) -> Action:
        """
        Select a random valid RED action.

        With 20% probability, emits WRITE_DEAD_DROP instead of a tactical action.

        Returns:
            An Action representing the planner's decision.
        """
        obs = self._current_observation
        if not isinstance(obs, RedObservation):
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.WAIT,
                reasoning="No valid observation available",
                step=self.step_count,
            )

        # 20% chance to write a dead drop
        if random.random() < 0.20:
            action = Action(
                agent_id=self.agent_id,
                action_type=ActionType.WRITE_DEAD_DROP,
                reasoning="Strategic dead drop — preserving mission state",
                step=self.step_count,
            )
            self.action_history.append(action)
            return action

        # Read dead drop if available and just reset
        if obs.context_reset_this_step and obs.dead_drops_available:
            action = Action(
                agent_id=self.agent_id,
                action_type=ActionType.READ_DEAD_DROP,
                reasoning="Context reset detected — reading dead drop for continuity",
                step=self.step_count,
            )
            self.action_history.append(action)
            return action

        # Select random valid action
        action = self._random_valid_action(obs)
        self.action_history.append(action)
        return action

    def _random_valid_action(self, obs: RedObservation) -> Action:
        """Select a random valid action given the current observation."""
        candidates: list[Action] = []

        # MOVE — only to adjacent nodes
        if obs.adjacent_nodes:
            target = random.choice(obs.adjacent_nodes)
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.MOVE,
                    target_node=target,
                    reasoning=f"Moving to node {target}",
                    step=self.step_count,
                )
            )

        # READ_FILE — only if files exist
        if obs.files_at_current_node:
            target_file = random.choice(obs.files_at_current_node)
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.READ_FILE,
                    target_file=target_file,
                    reasoning=f"Reading file {target_file}",
                    step=self.step_count,
                )
            )

        # WAIT — always valid
        candidates.append(
            Action(
                agent_id=self.agent_id,
                action_type=ActionType.WAIT,
                reasoning="Observing situation",
                step=self.step_count,
            )
        )

        return random.choice(candidates)
