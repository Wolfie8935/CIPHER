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

    _model_env_key = "hf_model_blue_forensics"

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

        # High-zone entries are more significant
        high_zone_entries = [e for e in self._operation_graph if e.get("zone", 0) >= 2]

        # Escalate to TRIGGER_ALERT when graph is rich enough and confidence is elevated
        if len(self._operation_graph) >= 5 and obs.current_detection_confidence >= 0.45 and high_zone_entries:
            # Pick the most recent high-zone node as the alert target
            latest_high = max(high_zone_entries, key=lambda e: (e["zone"], e["step"]))
            return Action(agent_id=self.agent_id, action_type=ActionType.TRIGGER_ALERT,
                         target_node=latest_high["node_id"],
                         reasoning=(
                             f"Forensic graph has {len(self._operation_graph)} entries — "
                             f"RED signature confirmed at zone {latest_high['zone']} node {latest_high['node_id']}. Alerting."
                         ))

        # Investigate the latest high-zone node if any
        if high_zone_entries:
            latest = max(high_zone_entries, key=lambda e: e["step"])
            return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                         target_node=latest["node_id"],
                         reasoning=f"Investigating zone {latest['zone']} node {latest['node_id']} from op graph.")

        # Reconstruct path at most once every 4 steps (avoid infinite loop)
        last_reconstruct = getattr(self, "_last_reconstruct_step", -99)
        if len(self._operation_graph) >= 3 and (obs.step - last_reconstruct) >= 4:
            self._last_reconstruct_step = obs.step
            return Action(agent_id=self.agent_id, action_type=ActionType.RECONSTRUCT_PATH,
                         reasoning=f"Updating op graph with {len(self._operation_graph)} evidence points.")

        # Analyze fresh anomalies
        if obs.anomaly_feed:
            worst = max(obs.anomaly_feed, key=lambda a: a.severity)
            return Action(agent_id=self.agent_id, action_type=ActionType.ANALYZE_ANOMALY,
                         target_node=worst.node_id,
                         reasoning=f"Deep-analyzing anomaly at node {worst.node_id} (severity={worst.severity:.2f}).")

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
