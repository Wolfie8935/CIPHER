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

        node_type_val = obs.current_node_type.value if hasattr(obs.current_node_type, 'value') else str(obs.current_node_type)
        honeypots = set(getattr(obs, 'honeypot_nodes_nearby', []) or [])
        breadcrumbs = set(getattr(obs, 'breadcrumb_nodes_nearby', []) or [])
        adj_types = getattr(obs, 'adjacent_node_types', [])
        adj_zones = getattr(obs, 'adjacent_node_zones', [])

        # Priority 1: Exfiltrate at HVT
        if node_type_val == "high_value_target" and obs.files_at_current_node:
            already_done = set(getattr(obs, 'exfiltrated_files', []) or [])
            already_attempted = {
                a.target_file for a in self.action_history
                if a.action_type == ActionType.EXFILTRATE and a.target_file
            }
            already_done = already_done | already_attempted
            remaining = [f for f in obs.files_at_current_node if f not in already_done]
            if not remaining:
                return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                             reasoning="All target files exfiltrated — mission complete.")
            return Action(agent_id=self.agent_id, action_type=ActionType.EXFILTRATE,
                         target_file=remaining[0],
                         reasoning=f"At HVT — exfiltrating: {remaining[0]}")

        if obs.adjacent_nodes:
            forbidden = honeypots | breadcrumbs
            safe = [n for n in obs.adjacent_nodes if n not in forbidden]
            pool = safe if safe else obs.adjacent_nodes

            # Priority 2: HVT is adjacent — move there now
            for i, node_id in enumerate(obs.adjacent_nodes):
                if node_id in pool and i < len(adj_types):
                    nt = adj_types[i].value if hasattr(adj_types[i], 'value') else str(adj_types[i])
                    if nt == "high_value_target":
                        return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                                     target_node=node_id,
                                     reasoning=f"HVT adjacent — moving to node {node_id} for exfiltration.")

            # Priority 3: Cross zone boundary or advance zone
            deeper = []
            if adj_zones and len(adj_zones) == len(obs.adjacent_nodes):
                deeper = [
                    obs.adjacent_nodes[i] for i in range(len(obs.adjacent_nodes))
                    if obs.adjacent_nodes[i] in pool and i < len(adj_zones)
                    and int(adj_zones[i]) > int(obs.current_zone)
                ]
            if deeper:
                target = random.choice(self._avoid_recent(deeper))
                return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                             target_node=target,
                             reasoning=f"Advancing to Zone {obs.current_zone + 1} toward HVT.")

            # Priority 4: Stay in Zone 3 and search if already there
            if obs.current_zone >= 3:
                target = random.choice(self._avoid_recent(pool))
                return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                             target_node=target,
                             reasoning=f"Searching Zone 3 for HVT — exploring node {target}.")

            # General movement toward deeper zone
            target = random.choice(self._avoid_recent(pool))
            return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                         target_node=target,
                         reasoning=f"Moving toward HVT — Zone {obs.current_zone}→deeper.")

        # Read dead drop only after context reset
        if obs.context_reset_this_step and obs.dead_drops_available:
            return Action(agent_id=self.agent_id, action_type=ActionType.READ_DEAD_DROP,
                         reasoning="Context reset — reading dead drop for HVT coordinates.")

        return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                     reasoning="No movement available — waiting.")

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, RedObservation):
            node_type_val = obs.current_node_type.value if hasattr(obs.current_node_type, 'value') else str(obs.current_node_type)
            at_hvt = node_type_val == 'high_value_target'
            steps_left = max(0, getattr(obs, 'max_steps', 30) - self.step_count)
            zone_msg = ""
            if not at_hvt:
                zone_msg = f"\nYou are in Zone {obs.current_zone}. Your ONLY goal right now is to MOVE toward Zone 3 (Critical/HVT). Do not stop to read files or write dead drops."
            exfil_context = (
                f"\n\nEXFILTRATION STATUS:\n"
                f"At HVT (Zone 3): {'YES — EXFILTRATE NOW' if at_hvt else 'NO'}\n"
                f"Current zone: {obs.current_zone} | Steps remaining: {steps_left}\n"
                f"Files at current node: {len(obs.files_at_current_node)}\n"
                f"Suspicion: {obs.estimated_suspicion:.2f} (safe to exfil if < 0.78)\n"
                f"Priority: REACH HVT → EXFILTRATE ALL TARGET FILES → EXIT"
                f"{zone_msg}"
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + exfil_context
        return messages
