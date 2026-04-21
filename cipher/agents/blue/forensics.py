"""
BLUE Forensics agent for CIPHER — Phase 3 LLM Integration.

The Forensics agent is the BLUE team's evidence analyst: reconstructs RED's
operation from behavioral fragments and builds operation graphs.
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import BlueObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class BlueForensics(BaseAgent):
    """BLUE team Forensics — operation graph reconstructor."""

    _model_env_key = "nvidia_model_blue_forensics"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "blue", "forensics", config)
        self._operation_graph: list[dict] = []

    def observe(self, observation: BlueObservation) -> None:
        self._current_observation = observation
        if observation.anomaly_feed:
            for anomaly in observation.anomaly_feed:
                self._operation_graph.append({
                    "step": anomaly.step,
                    "node_id": anomaly.node_id,
                    "zone": anomaly.zone,
                    "type": anomaly.event_type,
                    "severity": anomaly.severity,
                })

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.STAND_DOWN,
                         reasoning="No observation — maintaining analysis posture.")

        # Reconstruct path when enough evidence
        if len(self._operation_graph) >= 3:
            return Action(agent_id=self.agent_id, action_type=ActionType.RECONSTRUCT_PATH,
                         reasoning=f"Updating operation graph with "
                                  f"{len(self._operation_graph)} evidence points.")

        # Analyze fresh anomalies
        if obs.anomaly_feed:
            worst = max(obs.anomaly_feed, key=lambda a: a.severity)
            return Action(agent_id=self.agent_id, action_type=ActionType.ANALYZE_ANOMALY,
                         target_node=worst.node_id,
                         reasoning=f"Deep-analyzing anomaly at node {worst.node_id} "
                                  f"(severity={worst.severity:.2f}).")

        # Investigate from graph
        if self._operation_graph:
            recent = self._operation_graph[-1]
            return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                         target_node=recent["node_id"],
                         reasoning=f"Investigating node {recent['node_id']} "
                                  f"(suspected at step {recent['step']}).")

        # Trigger alert if high confidence
        if obs.current_detection_confidence > 0.75 and self._operation_graph:
            latest = self._operation_graph[-1]
            return Action(agent_id=self.agent_id, action_type=ActionType.TRIGGER_ALERT,
                         target_node=latest["node_id"],
                         reasoning=f"Forensic reconstruction confidence high — "
                                  f"alerting on node {latest['node_id']}.")

        return Action(agent_id=self.agent_id, action_type=ActionType.STAND_DOWN,
                     reasoning="Insufficient evidence — collecting more data points.")

    def reset(self) -> None:
        super().reset()
        self._operation_graph = []

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, BlueObservation):
            graph_summary = "None" if not self._operation_graph else str(
                self._operation_graph[-5:]
            )
            forensic_context = (
                f"\n\nFORENSIC STATUS:\n"
                f"Operation graph entries: {len(self._operation_graph)}\n"
                f"Recent graph: {graph_summary}\n"
                f"Detection confidence: {obs.current_detection_confidence:.2f}\n"
                f"Evidence quality: {'LOW' if len(self._operation_graph) < 3 else 'MODERATE' if len(self._operation_graph) < 8 else 'HIGH'}"
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + forensic_context
        return messages
