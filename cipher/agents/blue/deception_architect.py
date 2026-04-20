"""
BLUE Deception Architect agent for CIPHER.

The trap designer. Plants honeypots, engineers false environments, sets
cognitive traps for RED, and exploits RED's dead drop system when discovered.

In Phase 1: selects random valid BLUE actions. Trap placement is stubbed.
Phase 4 will connect this to the NVIDIA LLM backend.

Owns: honeypot placement, trap design, dead drop tampering, false escalation.
Does NOT own: passive monitoring (Surveillance), active hunting (Threat Hunter),
or evidence reconstruction (Forensics).
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import BlueObservation, RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class BlueDeceptionArchitect(BaseAgent):
    """
    BLUE Deception Architect — adversarial environment designer.

    In Phase 1, selects random valid BLUE actions. Trap and honeypot
    actions are stubbed (Phase 5).
    """

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(
            agent_id=agent_id,
            team="blue",
            role="deception_architect",
            config=config,
        )

    def observe(self, observation: RedObservation | BlueObservation) -> None:
        """Store the latest BLUE observation."""
        self._current_observation = observation
        self.step_count += 1

    def act(self) -> Action:
        """Select a random valid BLUE deception action."""
        obs = self._current_observation
        if not isinstance(obs, BlueObservation):
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.STAND_DOWN,
                reasoning="No valid observation available",
                step=self.step_count,
            )

        candidates: list[Action] = []

        # ANALYZE_ANOMALY — study patterns for optimal trap placement
        if obs.anomaly_feed:
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.ANALYZE_ANOMALY,
                    reasoning="Studying anomaly patterns for trap placement strategy",
                    step=self.step_count,
                )
            )

        # INVESTIGATE_NODE — assess suitability for trap placement
        if obs.active_honeypots:
            target = random.choice(obs.active_honeypots)
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.INVESTIGATE_NODE,
                    target_node=target,
                    reasoning=f"Assessing honeypot effectiveness at node {target}",
                    step=self.step_count,
                )
            )

        # STAND_DOWN — maintain current trap configuration
        candidates.append(
            Action(
                agent_id=self.agent_id,
                action_type=ActionType.STAND_DOWN,
                reasoning="Maintaining current deception posture",
                step=self.step_count,
            )
        )

        action = random.choice(candidates)
        self.action_history.append(action)
        return action
