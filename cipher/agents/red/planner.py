"""
RED Planner agent for CIPHER — Phase 3 LLM Integration.

The Planner is the RED team's strategist: designs multi-phase attack plans,
prioritizes targets, decides abort vs. continue, and delineates zone-transition
strategy. Maintains a high-level strategic view while operatives handle execution.

In LIVE mode: uses LLM to generate strategic decisions.
In STUB mode: uses weighted random action selection matching Phase 1 behavior.
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class RedPlanner(BaseAgent):
    """
    RED team Planner — mission architect and abort authority.
    """

    _model_env_key = "nvidia_model_red_planner"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "red", "planner", config)

    def observe(self, observation: RedObservation) -> None:
        self._current_observation = observation

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                         reasoning="No observation — waiting.")

        # Abort if suspicion is dangerously high
        if obs.estimated_suspicion > 0.85:
            return Action(agent_id=self.agent_id, action_type=ActionType.ABORT,
                         reasoning=f"Suspicion critical ({obs.estimated_suspicion:.2f}). Aborting.")

        # Write dead drop on context reset or periodically
        if obs.context_reset_this_step or (self.step_count % 8 == 0 and self.step_count > 0):
            return Action(agent_id=self.agent_id, action_type=ActionType.WRITE_DEAD_DROP,
                         reasoning="Writing operational continuity dead drop.")

        # Strategic movement — prefer moving toward higher zones
        if obs.adjacent_nodes:
            # Get zone info for adjacent nodes
            deeper = []
            same = []
            for i, node_id in enumerate(obs.adjacent_nodes):
                adj_type = obs.adjacent_node_types[i] if i < len(obs.adjacent_node_types) else None
                # We don't have zone info per adjacent node directly,
                # but zone_boundary_ahead tells us if deeper zones are accessible
                same.append(node_id)

            if obs.zone_boundary_ahead and obs.estimated_suspicion < 0.5:
                # Move to a random adjacent (might cross zone)
                target = random.choice(obs.adjacent_nodes)
                return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                             target_node=target,
                             reasoning=f"Advancing toward zone boundary via node {target}.")

            target = random.choice(obs.adjacent_nodes)
            return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                         target_node=target,
                         reasoning=f"Lateral movement within zone {obs.current_zone}.")

        # Exfiltrate if at HVT
        node_type_val = obs.current_node_type.value if hasattr(obs.current_node_type, 'value') else str(obs.current_node_type)
        if node_type_val == "high_value_target" and obs.files_at_current_node:
            target_file = obs.files_at_current_node[0]
            return Action(agent_id=self.agent_id, action_type=ActionType.EXFILTRATE,
                         target_file=target_file,
                         reasoning=f"At HVT — exfiltrating {target_file}.")

        # Read files if available
        if obs.files_at_current_node:
            target_file = random.choice(obs.files_at_current_node)
            return Action(agent_id=self.agent_id, action_type=ActionType.READ_FILE,
                         target_file=target_file,
                         reasoning=f"Reading {target_file} for intel.")

        return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                     reasoning="Holding position — assessing situation.")

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, RedObservation):
            phase = "RECON" if obs.current_zone <= 1 else "POSITIONING" if obs.current_zone == 2 else "EXFILTRATION"
            strategic_context = (
                f"\n\nCURRENT STRATEGIC CONTEXT:\n"
                f"Phase: {phase}\n"
                f"Steps taken: {self.step_count}\n"
                f"Actions in memory: {len(self.action_history)}\n"
                f"Suspicion trend: {'RISING' if obs.estimated_suspicion > 0.3 else 'STABLE'}"
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + strategic_context
        return messages
