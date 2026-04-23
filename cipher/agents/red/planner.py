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

        # PRIORITY 1: Exfiltrate if at HVT — must check BEFORE abort logic
        node_type_val = obs.current_node_type.value if hasattr(obs.current_node_type, 'value') else str(obs.current_node_type)
        if node_type_val == "high_value_target" and obs.files_at_current_node:
            already_attempted = {
                a.target_file for a in self.action_history
                if a.action_type == ActionType.EXFILTRATE and a.target_file
            }
            remaining = [f for f in obs.files_at_current_node if f not in already_attempted]
            target_file = remaining[0] if remaining else obs.files_at_current_node[0]
            return Action(agent_id=self.agent_id, action_type=ActionType.EXFILTRATE,
                         target_file=target_file,
                         reasoning=f"At HVT — exfiltrating {target_file}.")

        # PRIORITY 2: Move toward HVT if adjacent
        adj_zones = getattr(obs, 'adjacent_node_zones', [])
        adj_types = obs.adjacent_node_types or []
        if obs.adjacent_nodes:
            # Find HVT-type adjacent nodes
            hvt_neighbors = [
                obs.adjacent_nodes[i]
                for i, t in enumerate(adj_types)
                if (t.value if hasattr(t, 'value') else str(t)) == "high_value_target"
            ]
            if hvt_neighbors:
                target = hvt_neighbors[0]
                return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                             target_node=target,
                             reasoning=f"HVT detected adjacent — moving to node {target}.")

            # PRIORITY 3: Advance to deeper zone when possible
            if adj_zones:
                deeper = [
                    obs.adjacent_nodes[i]
                    for i, z in enumerate(adj_zones)
                    if z > obs.current_zone
                ]
                if deeper and obs.estimated_suspicion < 0.70:
                    target = random.choice(deeper)
                    return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                                 target_node=target,
                                 reasoning=f"Advancing to Zone {obs.current_zone + 1} via node {target}.")
            elif obs.zone_boundary_ahead and obs.estimated_suspicion < 0.70:
                target = random.choice(obs.adjacent_nodes)
                return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                             target_node=target,
                             reasoning=f"Crossing zone boundary from Zone {obs.current_zone}.")

            # PRIORITY 4: Move within zone (continue exploration)
            target = random.choice(obs.adjacent_nodes)
            return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                         target_node=target,
                         reasoning=f"Lateral movement in Zone {obs.current_zone}.")

        # Abort if suspicion is dangerously high and NOT at HVT
        if obs.estimated_suspicion > 0.90:
            return Action(agent_id=self.agent_id, action_type=ActionType.ABORT,
                         reasoning=f"Suspicion critical ({obs.estimated_suspicion:.2f}). Aborting.")

        # Write dead drop on context reset
        if obs.context_reset_this_step:
            return Action(agent_id=self.agent_id, action_type=ActionType.WRITE_DEAD_DROP,
                         reasoning="Writing operational continuity dead drop after context reset.")

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
