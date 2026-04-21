"""
BLUE Threat Hunter agent for CIPHER — Phase 3 LLM Integration.

The Threat Hunter is the BLUE team's proactive investigator: forms hypotheses
about RED's location and intent, conducts targeted node investigations.
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import BlueObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class BlueThreatHunter(BaseAgent):
    """BLUE team Threat Hunter — hypothesis-driven investigator."""

    _model_env_key = "nvidia_model_blue_hunter"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "blue", "threat_hunter", config)
        self._hypothesis_nodes: list[int] = []

    def observe(self, observation: BlueObservation) -> None:
        self._current_observation = observation
        if observation.anomaly_feed:
            self._hypothesis_nodes = list(set(
                a.node_id for a in observation.anomaly_feed if a.node_id is not None
            ))

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.STAND_DOWN,
                         reasoning="No observation — standing down.")

        # High confidence? Trigger alert
        if obs.current_detection_confidence > 0.7 and self._hypothesis_nodes:
            target = self._hypothesis_nodes[0]
            return Action(agent_id=self.agent_id, action_type=ActionType.TRIGGER_ALERT,
                         target_node=target,
                         reasoning=f"High confidence ({obs.current_detection_confidence:.2f}) — "
                                  f"alerting on node {target}.")

        # Investigate hypothesis nodes
        if self._hypothesis_nodes:
            target = random.choice(self._hypothesis_nodes)
            return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                         target_node=target,
                         reasoning=f"Investigating hypothesis target: node {target}.")

        # Analyze anomalies
        if obs.anomaly_feed:
            worst = max(obs.anomaly_feed, key=lambda a: a.severity)
            return Action(agent_id=self.agent_id, action_type=ActionType.ANALYZE_ANOMALY,
                         target_node=worst.node_id,
                         reasoning=f"Analyzing anomaly at node {worst.node_id}.")

        # Proactive sweep
        if obs.active_honeypots:
            target = random.choice(obs.active_honeypots)
            return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                         target_node=target,
                         reasoning=f"Proactive sweep of honeypot node {target}.")

        return Action(agent_id=self.agent_id, action_type=ActionType.STAND_DOWN,
                     reasoning="No leads — maintaining watch posture.")

    def reset(self) -> None:
        super().reset()
        self._hypothesis_nodes = []

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, BlueObservation):
            hunt_context = (
                f"\n\nTHREAT HUNTING STATUS:\n"
                f"Active hypotheses: {len(self._hypothesis_nodes)} nodes\n"
                f"Hypothesis targets: {self._hypothesis_nodes[:5]}\n"
                f"Investigations completed: {len([a for a in self.action_history if a.action_type == ActionType.INVESTIGATE_NODE])}\n"
                f"Detection confidence: {obs.current_detection_confidence:.2f}"
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + hunt_context
        return messages
