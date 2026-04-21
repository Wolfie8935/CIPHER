"""
BLUE Surveillance agent for CIPHER — Phase 3 LLM Integration.

The Surveillance agent is the BLUE team's real-time monitor: watches the anomaly
feed, correlates events across zones, and flags suspicious patterns.
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import BlueObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class BlueSurveillance(BaseAgent):
    """BLUE team Surveillance — real-time anomaly monitor and correlator."""

    _model_env_key = "nvidia_model_blue_surv"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "blue", "surveillance", config)

    def observe(self, observation: BlueObservation) -> None:
        self._current_observation = observation

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.STAND_DOWN,
                         reasoning="No observation — standing down.")

        # If honeypots triggered — high confidence alert
        if obs.honeypots_triggered_this_step:
            target = obs.honeypots_triggered_this_step[0]
            return Action(agent_id=self.agent_id, action_type=ActionType.TRIGGER_ALERT,
                         target_node=target,
                         reasoning=f"Honeypot {target} triggered — issuing alert.")

        # Analyze anomalies if present
        if obs.anomaly_feed:
            worst = max(obs.anomaly_feed, key=lambda a: a.severity)
            return Action(agent_id=self.agent_id, action_type=ActionType.ANALYZE_ANOMALY,
                         target_node=worst.node_id,
                         reasoning=f"Analyzing anomaly at node {worst.node_id} "
                                  f"(severity={worst.severity:.2f}).")

        # Investigate honeypot nodes periodically
        if obs.active_honeypots and self.step_count % 5 == 0:
            target = random.choice(obs.active_honeypots)
            return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                         target_node=target,
                         reasoning=f"Routine check on honeypot node {target}.")

        return Action(agent_id=self.agent_id, action_type=ActionType.STAND_DOWN,
                     reasoning="No anomalies detected — maintaining passive surveillance.")

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, BlueObservation):
            surv_context = (
                f"\n\nSURVEILLANCE STATUS:\n"
                f"Total anomalies this step: {len(obs.anomaly_feed)}\n"
                f"Honeypots triggered: {len(obs.honeypots_triggered_this_step)}\n"
                f"Detection confidence: {obs.current_detection_confidence:.2f}\n"
                f"Active honeypots: {len(obs.active_honeypots)}"
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + surv_context
        return messages
