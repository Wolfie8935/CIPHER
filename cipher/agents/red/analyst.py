"""
RED Analyst agent for CIPHER — Phase 3 LLM Integration.

The Analyst is the RED team's intelligence specialist: maps the environment,
identifies honeypots, estimates risk at each node, and feeds intelligence
back to the team via dead drops.
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class RedAnalyst(BaseAgent):
    """RED team Analyst — intelligence gatherer and risk estimator."""

    _model_env_key = "nvidia_model_red_analyst"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "red", "analyst", config)

    def observe(self, observation: RedObservation) -> None:
        self._current_observation = observation

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                         reasoning="No observation — waiting.")

        # Read dead drops if available
        if obs.dead_drops_available:
            return Action(agent_id=self.agent_id, action_type=ActionType.READ_DEAD_DROP,
                         reasoning="Reading dead drop for prior team intel.")

        # Read files to gather intelligence
        if obs.files_at_current_node:
            target_file = random.choice(obs.files_at_current_node)
            return Action(agent_id=self.agent_id, action_type=ActionType.READ_FILE,
                         target_file=target_file,
                         reasoning=f"Analyzing file: {target_file}")

        # Move to explore
        if obs.adjacent_nodes:
            target = random.choice(obs.adjacent_nodes)
            return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                         target_node=target,
                         reasoning=f"Mapping unexplored node {target}.")

        return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                     reasoning="Analyzing current environment — no movement needed.")

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, RedObservation):
            intel_context = (
                f"\n\nINTELLIGENCE CONTEXT:\n"
                f"Files analyzed: {len([a for a in self.action_history if a.action_type == ActionType.READ_FILE])}\n"
                f"Dead drops consumed: {len([a for a in self.action_history if a.action_type == ActionType.READ_DEAD_DROP])}\n"
                f"Current suspicion: {obs.estimated_suspicion:.2f}"
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + intel_context
        return messages
