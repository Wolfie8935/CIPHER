"""
BLUE Threat Hunter agent for CIPHER.

The proactive investigator. Generates hypotheses about RED's position and intent,
investigates high-probability locations, and discovers hidden RED artifacts
like dead drops.

In Phase 1: selects random valid BLUE actions with investigation focus.
Phase 4 will connect this to the NVIDIA LLM backend.

Owns: proactive investigation, dead drop discovery, hypothesis generation.
Does NOT own: passive monitoring (Surveillance), trap design (Deception Architect),
or evidence synthesis (Forensics).
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import BlueObservation, RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class BlueThreatHunter(BaseAgent):
    """
    BLUE Threat Hunter — proactive hypothesis-driven investigator.

    In Phase 1, selects random valid BLUE actions with a bias toward
    INVESTIGATE_NODE.
    """

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(
            agent_id=agent_id,
            team="blue",
            role="threat_hunter",
            config=config,
        )

    def observe(self, observation: RedObservation | BlueObservation) -> None:
        """Store the latest BLUE observation."""
        self._current_observation = observation
        self.step_count += 1

    def act(self) -> Action:
        """Select a random valid BLUE action biased toward investigation."""
        obs = self._current_observation
        if not isinstance(obs, BlueObservation):
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.STAND_DOWN,
                reasoning="No valid observation available",
                step=self.step_count,
            )

        candidates: list[Action] = []

        # INVESTIGATE_NODE — primary action, pick from anomaly hints
        if obs.anomaly_feed:
            # Investigate nodes mentioned in anomalies
            anomaly_nodes = [
                a.node_id for a in obs.anomaly_feed if a.node_id is not None
            ]
            if anomaly_nodes:
                target = random.choice(anomaly_nodes)
                candidates.append(
                    Action(
                        agent_id=self.agent_id,
                        action_type=ActionType.INVESTIGATE_NODE,
                        target_node=target,
                        reasoning=f"Hunting — investigating anomaly at node {target}",
                        step=self.step_count,
                    )
                )

        # ANALYZE_ANOMALY
        if obs.anomaly_feed:
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.ANALYZE_ANOMALY,
                    reasoning="Deep analysis of anomaly pattern",
                    step=self.step_count,
                )
            )

        # TRIGGER_ALERT — only when detection confidence is high
        if obs.current_detection_confidence > 0.5 and obs.anomaly_feed:
            anomaly_nodes = [
                a.node_id for a in obs.anomaly_feed if a.node_id is not None
            ]
            if anomaly_nodes:
                target = random.choice(anomaly_nodes)
                candidates.append(
                    Action(
                        agent_id=self.agent_id,
                        action_type=ActionType.TRIGGER_ALERT,
                        target_node=target,
                        reasoning=f"High confidence — triggering alert on node {target}",
                        step=self.step_count,
                    )
                )

        # STAND_DOWN
        candidates.append(
            Action(
                agent_id=self.agent_id,
                action_type=ActionType.STAND_DOWN,
                reasoning="Reformulating hunting hypothesis",
                step=self.step_count,
            )
        )

        action = random.choice(candidates)
        self.action_history.append(action)
        return action
