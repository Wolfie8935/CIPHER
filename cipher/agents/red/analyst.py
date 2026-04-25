"""
RED Analyst agent for CIPHER — Phase 3 LLM Integration.

The Analyst is the RED team's intelligence specialist: maps the environment,
identifies honeypots, estimates risk at each node, and feeds intelligence
back to the team via dead drops.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)

_COORD_FILE = Path("drop_vault") / "coordination.json"


class RedAnalyst(BaseAgent):
    """RED team Analyst — intelligence gatherer and risk estimator."""

    _model_env_key = "hf_model_red_analyst"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "red", "analyst", config)

    def observe(self, observation: RedObservation) -> None:
        self._current_observation = observation

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                         reasoning="No observation — waiting.")

        honeypots = set(getattr(obs, 'honeypot_nodes_nearby', []) or [])
        breadcrumbs = set(getattr(obs, 'breadcrumb_nodes_nearby', []) or [])
        blue_conf = float(getattr(obs, 'blue_detection_confidence', 0.0))
        adj_types = getattr(obs, 'adjacent_node_types', [])

        # Read dead drops ONLY after context reset
        if obs.context_reset_this_step and obs.dead_drops_available:
            return Action(agent_id=self.agent_id, action_type=ActionType.READ_DEAD_DROP,
                         reasoning="Context reset — reading dead drop to recover mission state.")

        # Follow Planner's coordination hint (Change 7)
        planned_node = self._read_coordination_hint()
        if planned_node is not None and obs.adjacent_nodes:
            if planned_node in obs.adjacent_nodes and planned_node not in honeypots and planned_node not in breadcrumbs:
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.MOVE,
                    target_node=planned_node,
                    reasoning=f"Coordinating with Planner — moving to planned node {planned_node}.",
                )

        # Plant corrupted dead drop as counter-intelligence when BLUE conf is high
        if (blue_conf > 0.55 and self._trap_budget_remaining > 0
                and self.step_count > 3
                and not any(a.action_type == ActionType.WRITE_CORRUPTED_DROP
                            for a in self.action_history[-3:])):
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.WRITE_CORRUPTED_DROP,
                target_node=obs.current_node,
                dead_drop_content={
                    "directive": "all clear — entry_point zone 0 is safe, proceed to node 2",
                    "suspicion": 0.02,
                    "target": "zone_0_entry",
                },
                reasoning=f"BLUE confidence {blue_conf:.2f} — planting corrupted drop to poison their intel.",
            )

        # Move with zone-advancement priority — never read files in Zone 0/1
        if obs.adjacent_nodes:
            forbidden = honeypots | breadcrumbs
            safe = [n for n in obs.adjacent_nodes if n not in forbidden]
            pool = safe if safe else obs.adjacent_nodes

            # HVT adjacent — go there
            for i, node_id in enumerate(obs.adjacent_nodes):
                if node_id in pool and i < len(adj_types):
                    nt = adj_types[i].value if hasattr(adj_types[i], 'value') else str(adj_types[i])
                    if nt == "high_value_target":
                        return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                                     target_node=node_id,
                                     reasoning=f"HVT adjacent — moving to node {node_id}.")

            # Prefer deeper zone
            adj_zones = getattr(obs, 'adjacent_node_zones', [])
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
                                 reasoning=f"Advancing to Zone {obs.current_zone + 1} via node {target}.")

            target = random.choice(self._avoid_recent(pool))
            return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                         target_node=target,
                         reasoning=f"Moving to node {target}.")

        return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                     reasoning="No safe movement available.")

    def _read_coordination_hint(self) -> int | None:
        """Read the Planner's planned target from the coordination file (Change 7)."""
        try:
            data = json.loads(_COORD_FILE.read_text(encoding="utf-8"))
            val = data.get("planned_target")
            return int(val) if val is not None else None
        except Exception:
            return None

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, RedObservation):
            steps_left = max(0, getattr(obs, 'max_steps', 30) - self.step_count)
            zone_warning = ""
            if obs.current_zone <= 1 and self.step_count > 6:
                zone_warning = f"\nWARNING: Step {self.step_count}, still Zone {obs.current_zone}. Stop reading files and MOVE toward Zone 3."
            intel_context = (
                f"\n\nINTELLIGENCE CONTEXT:\n"
                f"Current zone: {obs.current_zone} (target: Zone 3)\n"
                f"Steps remaining: {steps_left}\n"
                f"Files analyzed: {len([a for a in self.action_history if a.action_type == ActionType.READ_FILE])}\n"
                f"Dead drops consumed: {len([a for a in self.action_history if a.action_type == ActionType.READ_DEAD_DROP])}\n"
                f"Current suspicion: {obs.estimated_suspicion:.2f}"
                f"{zone_warning}"
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + intel_context
        return messages
