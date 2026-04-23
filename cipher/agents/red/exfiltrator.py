"""
RED Exfiltrator agent for CIPHER — Phase 3 LLM Integration.

The Exfiltrator is the RED team's extraction specialist: packages target files,
sequences the exit plan, and manages the critical final phase.
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class RedExfiltrator(BaseAgent):
    """RED team Exfiltrator — data extraction and exit specialist."""

    _model_env_key = "nvidia_model_red_exfil"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "red", "exfiltrator", config)

    def observe(self, observation: RedObservation) -> None:
        self._current_observation = observation

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                         reasoning="No observation — waiting.")

        # Priority 1: Exfiltrate if at HVT with target files (cycle through unattempted)
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
                         reasoning=f"Exfiltrating target file: {target_file}")

        # Priority 2: Move toward HVT (deeper zones via boundary)
        if obs.adjacent_nodes:
            if obs.zone_boundary_ahead:
                target = random.choice(obs.adjacent_nodes)
                return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                             target_node=target,
                             reasoning=f"Moving toward HVT — crossing zone boundary.")

            target = random.choice(obs.adjacent_nodes)
            return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                         target_node=target,
                         reasoning=f"Lateral search for HVT in zone {obs.current_zone}.")

        # Read dead drops for extraction intel
        if obs.dead_drops_available:
            return Action(agent_id=self.agent_id, action_type=ActionType.READ_DEAD_DROP,
                         reasoning="Reading dead drop for extraction coordinates.")

        # Write dead drop with current status
        if self.step_count % 10 == 0 and self.step_count > 0:
            return Action(agent_id=self.agent_id, action_type=ActionType.WRITE_DEAD_DROP,
                         reasoning="Leaving extraction status update in dead drop.")

        return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                     reasoning="Awaiting extraction opportunity.")

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, RedObservation):
            node_type_val = obs.current_node_type.value if hasattr(obs.current_node_type, 'value') else str(obs.current_node_type)
            exfil_context = (
                f"\n\nEXFILTRATION STATUS:\n"
                f"At HVT: {'YES' if node_type_val == 'high_value_target' else 'NO'}\n"
                f"Files at current node: {len(obs.files_at_current_node)}\n"
                f"Suspicion: {obs.estimated_suspicion:.2f}\n"
                f"Priority: REACH HVT → EXFILTRATE ALL TARGET FILES → EXIT"
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + exfil_context
        return messages
