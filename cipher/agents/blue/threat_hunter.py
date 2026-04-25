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

    _model_env_key = "hf_model_blue_hunter"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "blue", "threat_hunter", config)
        self._hypothesis_nodes: list[int] = []
        self._candidate_alert_node: int | None = None
        self._candidate_streak: int = 0
        self._last_alert_step: int = -999
        self._heat_map: dict[int, float] = {}  # node → accumulated suspicion score
        self._investigated: set[int] = set()   # nodes already investigated this episode

    def observe(self, observation: BlueObservation) -> None:
        self._current_observation = observation
        if observation.anomaly_feed:
            self._hypothesis_nodes = list(set(
                a.node_id for a in observation.anomaly_feed if a.node_id is not None
            ))
            # Build heat map: accumulate severity across steps — high-severity
            # anomalies on the same nodes across steps build conviction
            for a in observation.anomaly_feed:
                if a.node_id is not None:
                    # Weight by zone (higher zones = more important) and severity
                    zone_weight = 1.0 + (a.zone or 0) * 0.5
                    self._heat_map[a.node_id] = (
                        self._heat_map.get(a.node_id, 0.0) + a.severity * zone_weight
                    )
            # Decay old entries slightly so stale signals fade
            for nid in list(self._heat_map):
                self._heat_map[nid] *= 0.85
                if self._heat_map[nid] < 0.01:
                    del self._heat_map[nid]

            strongest = max(
                (a for a in observation.anomaly_feed if a.node_id is not None),
                key=lambda a: a.severity,
                default=None,
            )
            if strongest is not None:
                if strongest.node_id == self._candidate_alert_node:
                    self._candidate_streak += 1
                else:
                    self._candidate_alert_node = strongest.node_id
                    self._candidate_streak = 1
        else:
            self._candidate_alert_node = None
            self._candidate_streak = 0

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.STAND_DOWN,
                         reasoning="No observation — standing down.")

        # Determine which zone is most active (where RED likely is)
        hottest_zone = 0
        if obs.zone_alert_levels:
            hottest_zone = max(obs.zone_alert_levels, key=lambda z: obs.zone_alert_levels[z])

        # Zone 3 anomaly — only alert when confidence is strong AND multiple z3 signals
        zone3_anomalies = [a for a in obs.anomaly_feed if a.zone == 3]
        if len(zone3_anomalies) >= 2 and obs.current_detection_confidence >= 0.55:
            target = max(zone3_anomalies, key=lambda a: a.severity).node_id
            self._last_alert_step = obs.step
            return Action(agent_id=self.agent_id, action_type=ActionType.TRIGGER_ALERT,
                         target_node=target,
                         reasoning=(
                             f"Multiple zone-3 anomalies ({len(zone3_anomalies)}) at node {target} "
                             f"with confidence {obs.current_detection_confidence:.2f} — alerting."
                         ))

        # High confidence + repeated hypothesis + cooldown — investigate first, alert second
        can_alert_now = (obs.step - self._last_alert_step) >= 5
        alert_threshold = 0.65 if hottest_zone >= 2 else 0.75
        if (
            obs.current_detection_confidence > alert_threshold
            and self._candidate_alert_node is not None
            and self._candidate_streak >= 3
            and can_alert_now
        ):
            target = self._candidate_alert_node
            self._last_alert_step = obs.step
            return Action(agent_id=self.agent_id, action_type=ActionType.TRIGGER_ALERT,
                         target_node=target,
                         reasoning=(
                             f"Strong hypothesis: node {target} confirmed {self._candidate_streak} times "
                             f"(conf={obs.current_detection_confidence:.2f}) — alerting."
                         ))

        # Use heat map to pick the hottest UN-investigated node for investigation
        if self._heat_map:
            # Sort by accumulated heat, prefer nodes not yet investigated
            candidates = sorted(
                self._heat_map.items(),
                key=lambda kv: (kv[0] not in self._investigated, kv[1]),
                reverse=True,
            )
            target = candidates[0][0]
            self._investigated.add(target)
            return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                         target_node=target,
                         reasoning=f"Investigating hottest node {target} (heat={candidates[0][1]:.2f}) — triangulating RED.")

        # Fallback: zone-based anomaly investigation
        zone_anomalies = [a for a in obs.anomaly_feed if a.zone == hottest_zone and a.node_id is not None]
        if zone_anomalies:
            target = max(zone_anomalies, key=lambda a: a.severity).node_id
            self._investigated.add(target)
            return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                         target_node=target,
                         reasoning=f"Investigating zone {hottest_zone} anomaly at node {target}.")

        if obs.anomaly_feed:
            worst = max(obs.anomaly_feed, key=lambda a: a.severity)
            return Action(agent_id=self.agent_id, action_type=ActionType.ANALYZE_ANOMALY,
                         target_node=worst.node_id,
                         reasoning=f"Analyzing highest-severity anomaly at node {worst.node_id}.")

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
        self._candidate_alert_node = None
        self._candidate_streak = 0
        self._last_alert_step = -999
        self._heat_map = {}
        self._investigated = set()

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
