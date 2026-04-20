"""
BLUE Surveillance agent for CIPHER.

The sensor network of the defensive team. Monitors the environment in real-time,
processes the anomaly feed, filters false positives, and maintains a running
detection confidence.

In Phase 1: selects random valid BLUE actions from the surveillance vocabulary.
Phase 4 will connect this to the NVIDIA LLM backend.

Owns: anomaly feed processing, false positive filtering, confidence updates.
Does NOT own: active hunting (Threat Hunter), trap design (Deception Architect),
or evidence reconstruction (Forensics).
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import BlueObservation, RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class BlueSurveillance(BaseAgent):
    """
    BLUE Surveillance — continuous anomaly monitoring.

    In Phase 1, selects random valid BLUE surveillance actions.
    Primarily ANALYZE_ANOMALY and STAND_DOWN.
    """

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(
            agent_id=agent_id,
            team="blue",
            role="surveillance",
            config=config,
        )

    def observe(self, observation: RedObservation | BlueObservation) -> None:
        """Store the latest BLUE observation."""
        self._current_observation = observation
        self.step_count += 1

    def act(self) -> Action:
        """Select a random valid BLUE surveillance action."""
        obs = self._current_observation
        if not isinstance(obs, BlueObservation):
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.STAND_DOWN,
                reasoning="No valid observation available",
                step=self.step_count,
            )

        candidates: list[Action] = []

        # ANALYZE_ANOMALY — primary action when anomalies exist
        if obs.anomaly_feed:
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.ANALYZE_ANOMALY,
                    reasoning=f"Analyzing {len(obs.anomaly_feed)} anomaly event(s)",
                    step=self.step_count,
                )
            )

        # INVESTIGATE_NODE — if honeypots triggered
        if obs.honeypots_triggered_this_step:
            target = obs.honeypots_triggered_this_step[0]
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.INVESTIGATE_NODE,
                    target_node=target,
                    reasoning=f"Honeypot triggered at node {target} — investigating",
                    step=self.step_count,
                )
            )

        # STAND_DOWN — always valid
        candidates.append(
            Action(
                agent_id=self.agent_id,
                action_type=ActionType.STAND_DOWN,
                reasoning="Maintaining passive monitoring",
                step=self.step_count,
            )
        )

        action = random.choice(candidates)
        self.action_history.append(action)
        return action
