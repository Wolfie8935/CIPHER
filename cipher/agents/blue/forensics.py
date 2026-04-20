"""
BLUE Forensics agent for CIPHER.

The evidence analyst and operation reconstructor. Reconstructs RED's full
operation from behavioral fragments, detects context reset signatures, and
builds a coherent operation graph from sparse, noisy evidence.

In Phase 1: selects random valid BLUE actions with reconstruction focus.
Phase 4 will connect this to the NVIDIA LLM backend.

Owns: operation graph reconstruction, handoff detection, evidence synthesis.
Does NOT own: passive monitoring (Surveillance), active hunting (Threat Hunter),
or trap design (Deception Architect).
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import BlueObservation, RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class BlueForensics(BaseAgent):
    """
    BLUE Forensics — behavioral forensics and operation reconstruction.

    In Phase 1, selects random valid BLUE actions with a strong bias
    toward RECONSTRUCT_PATH and ANALYZE_ANOMALY.
    """

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(
            agent_id=agent_id,
            team="blue",
            role="forensics",
            config=config,
        )

    def observe(self, observation: RedObservation | BlueObservation) -> None:
        """Store the latest BLUE observation."""
        self._current_observation = observation
        self.step_count += 1

    def act(self) -> Action:
        """Select a random valid BLUE action biased toward reconstruction."""
        obs = self._current_observation
        if not isinstance(obs, BlueObservation):
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.STAND_DOWN,
                reasoning="No valid observation available",
                step=self.step_count,
            )

        candidates: list[Action] = []

        # RECONSTRUCT_PATH — primary action
        candidates.append(
            Action(
                agent_id=self.agent_id,
                action_type=ActionType.RECONSTRUCT_PATH,
                reasoning="Updating operation graph hypothesis from accumulated evidence",
                step=self.step_count,
            )
        )

        # ANALYZE_ANOMALY — deep analysis
        if obs.anomaly_feed:
            candidates.append(
                Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.ANALYZE_ANOMALY,
                    reasoning=f"Forensic analysis of {len(obs.anomaly_feed)} anomaly event(s)",
                    step=self.step_count,
                )
            )

        # INVESTIGATE_NODE — targeted investigation from reconstruction
        if obs.anomaly_feed:
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
                        reasoning=f"Investigating node {target} based on reconstructed path",
                        step=self.step_count,
                    )
                )

        # STAND_DOWN
        candidates.append(
            Action(
                agent_id=self.agent_id,
                action_type=ActionType.STAND_DOWN,
                reasoning="Accumulating evidence before next reconstruction update",
                step=self.step_count,
            )
        )

        action = random.choice(candidates)
        self.action_history.append(action)
        return action
