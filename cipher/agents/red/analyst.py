"""
RED Analyst agent for CIPHER.

The intelligence officer of the infiltration team. Maps the environment,
identifies high-value targets, estimates detection risk at each node, and
maintains a running threat assessment.

In Phase 1: selects random valid actions from the RED action vocabulary.
Phase 4 will connect this to the NVIDIA LLM backend for Bayesian reasoning.

Owns: risk estimation, environment mapping, honeypot suspicion tracking.
Does NOT own: mission strategy (Planner), physical execution (Operative),
or data extraction (Exfiltrator).
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import BlueObservation, RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class RedAnalyst(BaseAgent):
    """
    RED Analyst — environmental analysis and risk assessment.

    In Phase 1, selects random valid RED actions focusing on
    information-gathering (READ_FILE, MOVE) over aggressive actions.
    """

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(
            agent_id=agent_id,
            team="red",
            role="analyst",
            config=config,
        )

    def observe(self, observation: RedObservation | BlueObservation) -> None:
        """Store the latest RED observation."""
        self._current_observation = observation
        self.step_count += 1

    def act(self) -> Action:
        """Select a random valid RED action biased toward information gathering."""
        obs = self._current_observation
        if not isinstance(obs, RedObservation):
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.WAIT,
                reasoning="No valid observation available",
                step=self.step_count,
            )

        candidates: list[Action] = []

        # MOVE
        if obs.adjacent_nodes:
            target = random.choice(obs.adjacent_nodes)
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.MOVE,
                    target_node=target,
                    reasoning=f"Scouting node {target} for risk assessment",
                    step=self.step_count,
                )
            )

        # READ_FILE — analyst prefers reading
        if obs.files_at_current_node:
            target_file = random.choice(obs.files_at_current_node)
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.READ_FILE,
                    target_file=target_file,
                    reasoning=f"Analyzing file {target_file} for intelligence value",
                    step=self.step_count,
                )
            )

        # WAIT
        candidates.append(
            Action(
                agent_id=self.agent_id,
                action_type=ActionType.WAIT,
                reasoning="Passive observation — gathering behavioral baseline",
                step=self.step_count,
            )
        )

        action = random.choice(candidates)
        self.action_history.append(action)
        return action
