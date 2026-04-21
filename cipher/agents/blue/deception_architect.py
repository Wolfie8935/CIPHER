"""
BLUE Deception Architect agent for CIPHER — Phase 3 LLM Integration.

The Deception Architect is the BLUE team's trap designer: places honeypots,
engineers false environments, and sets cognitive traps to misdirect RED.
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import BlueObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class BlueDeceptionArchitect(BaseAgent):
    """BLUE team Deception Architect — honeypot and trap designer."""

    _model_env_key = "nvidia_model_blue_deceiver"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "blue", "deception_architect", config)

    def observe(self, observation: BlueObservation) -> None:
        self._current_observation = observation

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.STAND_DOWN,
                         reasoning="No observation — maintaining current traps.")

        # Periodically trigger strategic deception to exercise Phase 5 trap layer.
        if self.step_count % 5 == 0 and obs.anomaly_feed:
            target = obs.anomaly_feed[0].node_id if obs.anomaly_feed[0].node_id is not None else 0
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.TRIGGER_FALSE_ESCALATION,
                target_node=target,
                trap_params={"fake_node": target},
                reasoning=f"Injecting false escalation signal near node {target} to force RED reaction.",
            )

        # Opportunistically place runtime honeypots while budget remains.
        if self.step_count % 3 == 0 and obs.anomaly_feed:
            anomaly_nodes = [a.node_id for a in obs.anomaly_feed if a.node_id is not None]
            if anomaly_nodes:
                target = random.choice(anomaly_nodes)
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.PLACE_HONEYPOT,
                    target_node=target,
                    reasoning=f"Placing runtime honeypot at likely RED path node {target}.",
                )

        # If honeypots triggered — analyze the trigger point
        if obs.honeypots_triggered_this_step:
            target = obs.honeypots_triggered_this_step[0]
            return Action(agent_id=self.agent_id, action_type=ActionType.ANALYZE_ANOMALY,
                         target_node=target,
                         reasoning=f"Honeypot {target} triggered — analyzing RED contact point.")

        # Analyze anomalies for trap placement
        if obs.anomaly_feed:
            anomaly_nodes = [a.node_id for a in obs.anomaly_feed if a.node_id is not None]
            if anomaly_nodes:
                target = random.choice(anomaly_nodes)
                return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                             target_node=target,
                             reasoning=f"Investigating node {target} for trap placement viability.")

        # Periodic honeypot monitoring
        if obs.active_honeypots and self.step_count % 3 == 0:
            target = random.choice(obs.active_honeypots)
            return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                         target_node=target,
                         reasoning=f"Checking honeypot {target} integrity.")

        return Action(agent_id=self.agent_id, action_type=ActionType.STAND_DOWN,
                     reasoning="Trap configuration stable — no changes needed.")

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, BlueObservation):
            deception_context = (
                f"\n\nDECEPTION STATUS:\n"
                f"Active honeypots: {len(obs.active_honeypots)}\n"
                f"Honeypots triggered: {len(obs.honeypots_triggered_this_step)}\n"
                f"Detection confidence: {obs.current_detection_confidence:.2f}"
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + deception_context
        return messages
