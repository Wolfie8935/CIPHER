"""
RED Operative agent for CIPHER — Phase 3 LLM Integration.

The Operative is the RED team's field executor: handles stealth movement,
protocol selection, trap avoidance, and real-time suspicion management.
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class RedOperative(BaseAgent):
    """RED team Operative — stealth executor and field agent."""

    _model_env_key = "nvidia_model_red_operative"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "red", "operative", config)

    def observe(self, observation: RedObservation) -> None:
        self._current_observation = observation

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                         reasoning="No observation — maintaining position.")

        # Wait if suspicion is elevated (stealth-first)
        if obs.estimated_suspicion > 0.6:
            return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                         reasoning=f"Suspicion at {obs.estimated_suspicion:.2f} — going dark.")

        # Periodically seed deception so Phase 5 trap layer is exercised in stub runs.
        if self.step_count % 4 == 0:
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.PLANT_FALSE_TRAIL,
                target_node=obs.current_node,
                trap_params={"n_fake_events": 3},
                reasoning="Planting a false trail to pull BLUE away from true route.",
            )

        # Move to adjacent node (stealth priority - avoid honeypots but we can't tell)
        if obs.adjacent_nodes:
            target = random.choice(obs.adjacent_nodes)
            hostname = obs.adjacent_hostnames[obs.adjacent_nodes.index(target)] if target in obs.adjacent_nodes and obs.adjacent_hostnames else f"node-{target}"
            return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                         target_node=target,
                         reasoning=f"Stealth move to {hostname}.")

        # Read files opportunistically
        if obs.files_at_current_node:
            target_file = random.choice(obs.files_at_current_node)
            return Action(agent_id=self.agent_id, action_type=ActionType.READ_FILE,
                         target_file=target_file,
                         reasoning=f"Opportunistic file read: {target_file}")

        return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                     reasoning="No safe movement — holding position.")

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, RedObservation):
            stealth_context = (
                f"\n\nSTEALTH STATUS:\n"
                f"Current suspicion: {obs.estimated_suspicion:.2f}\n"
                f"Suspicion threshold (abort): 0.85\n"
                f"Suspicion threshold (caution): 0.60\n"
                f"Priority: MINIMIZE suspicion delta on every action"
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + stealth_context
        return messages
