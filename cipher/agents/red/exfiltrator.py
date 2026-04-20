"""
RED Exfiltrator agent for CIPHER.

The extraction specialist. Packages target data, executes the final exfiltration
at the high-value target, and manages the exit sequence without triggering alarms.

In Phase 1: selects random valid actions. EXFILTRATE is chosen when at HVT node.
Phase 4 will connect this to the NVIDIA LLM backend.

Owns: data exfiltration, exit sequencing, terminal-phase decisions.
Does NOT own: mission strategy (Planner), risk analysis (Analyst),
or traversal execution (Operative).
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.graph import NodeType
from cipher.environment.observation import BlueObservation, RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class RedExfiltrator(BaseAgent):
    """
    RED Exfiltrator — terminal-phase data extraction specialist.

    In Phase 1, selects random valid RED actions. If at a HIGH_VALUE_TARGET
    node with files, attempts EXFILTRATE.
    """

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(
            agent_id=agent_id,
            team="red",
            role="exfiltrator",
            config=config,
        )

    def observe(self, observation: RedObservation | BlueObservation) -> None:
        """Store the latest RED observation."""
        self._current_observation = observation
        self.step_count += 1

    def act(self) -> Action:
        """
        Select action. Attempts EXFILTRATE when at HIGH_VALUE_TARGET.

        Returns:
            An Action for this step.
        """
        obs = self._current_observation
        if not isinstance(obs, RedObservation):
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.WAIT,
                reasoning="No valid observation available",
                step=self.step_count,
            )

        # If at HVT node with files, attempt exfiltration
        if (
            obs.current_node_type == NodeType.HIGH_VALUE_TARGET
            and obs.files_at_current_node
        ):
            target_file = random.choice(obs.files_at_current_node)
            action = Action(
                agent_id=self.agent_id,
                action_type=ActionType.EXFILTRATE,
                target_file=target_file,
                reasoning=f"Exfiltrating {target_file} from HVT",
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
                    reasoning=f"Moving toward extraction point via node {target}",
                    step=self.step_count,
                )
            )

        # WAIT
        candidates.append(
            Action(
                agent_id=self.agent_id,
                action_type=ActionType.WAIT,
                reasoning="Awaiting extraction window",
                step=self.step_count,
            )
        )

        action = random.choice(candidates)
        self.action_history.append(action)
        return action
