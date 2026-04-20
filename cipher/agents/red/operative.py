"""
RED Operative agent for CIPHER.

The hands of the infiltration team. Executes in-environment actions: movement,
file access, trap placement, and maintaining stealth through careful action
sequencing.

In Phase 1: selects random valid actions with stealth awareness (suspicion check).
Phase 4 will connect this to the NVIDIA LLM backend.

Owns: stealth execution, traversal, file reads, trap placement.
Does NOT own: mission strategy (Planner), risk analysis (Analyst),
or data extraction (Exfiltrator).
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import BlueObservation, RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class RedOperative(BaseAgent):
    """
    RED Operative — stealth execution specialist.

    In Phase 1, selects random valid RED actions with basic suspicion awareness:
    if suspicion > 0.65, defaults to WAIT.
    """

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(
            agent_id=agent_id,
            team="red",
            role="operative",
            config=config,
        )

    def observe(self, observation: RedObservation | BlueObservation) -> None:
        """Store the latest RED observation."""
        self._current_observation = observation
        self.step_count += 1

    def act(self) -> Action:
        """
        Select a random valid action with basic stealth awareness.

        If estimated suspicion > 0.65, always WAIT or WRITE_DEAD_DROP.
        """
        obs = self._current_observation
        if not isinstance(obs, RedObservation):
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.WAIT,
                reasoning="No valid observation available",
                step=self.step_count,
            )

        # Stealth check — high suspicion triggers defensive mode
        if obs.estimated_suspicion > 0.65:
            action = Action(
                agent_id=self.agent_id,
                action_type=ActionType.WAIT,
                reasoning=f"Suspicion high ({obs.estimated_suspicion:.2f}) — covert hold",
                step=self.step_count,
            )
            self.action_history.append(action)
            return action

        candidates: list[Action] = []

        # MOVE
        if obs.adjacent_nodes:
            target = random.choice(obs.adjacent_nodes)
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.MOVE,
                    target_node=target,
                    reasoning=f"Advancing to node {target}",
                    step=self.step_count,
                )
            )

        # READ_FILE
        if obs.files_at_current_node:
            target_file = random.choice(obs.files_at_current_node)
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.READ_FILE,
                    target_file=target_file,
                    reasoning=f"Accessing file {target_file}",
                    step=self.step_count,
                )
            )

        # WAIT
        candidates.append(
            Action(
                agent_id=self.agent_id,
                action_type=ActionType.WAIT,
                reasoning="Maintaining covert posture",
                step=self.step_count,
            )
        )

        action = random.choice(candidates)
        self.action_history.append(action)
        return action
