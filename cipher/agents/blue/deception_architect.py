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

    _model_env_key = "hf_model_blue_deceiver"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "blue", "deception_architect", config)

    def observe(self, observation: BlueObservation) -> None:
        self._current_observation = observation

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.STAND_DOWN,
                         reasoning="No observation — maintaining current traps.")

        # ── Honeypot trigger response (highest priority) ──────────
        if obs.honeypots_triggered_this_step:
            target = obs.honeypots_triggered_this_step[0]
            return Action(agent_id=self.agent_id, action_type=ActionType.ANALYZE_ANOMALY,
                         target_node=target,
                         reasoning=f"Honeypot {target} triggered — analyzing RED contact point.")

        # ── Determine RED's likely current zone from alert levels ─
        # RED must pass through zones 0→1→2→3 in order.
        # We detect which zone has the hottest activity = RED's current zone.
        # Then we place honeypots in zone+1 (where RED is going NEXT).
        red_zone_estimate = 0
        if obs.zone_alert_levels:
            red_zone_estimate = max(obs.zone_alert_levels, key=lambda z: obs.zone_alert_levels[z])
        target_zone = min(3, red_zone_estimate + 1)  # place traps ONE zone ahead of RED

        # ── Build candidate honeypot nodes for the target zone ────
        # Priority 1: anomaly nodes already in the target zone (highest quality)
        next_zone_nodes = [
            a.node_id for a in obs.anomaly_feed
            if a.node_id is not None and a.zone == target_zone
        ]

        # Priority 2: anomaly nodes in RED's current estimated zone (still useful)
        current_zone_nodes = [
            a.node_id for a in obs.anomaly_feed
            if a.node_id is not None and a.zone == red_zone_estimate
        ]

        # Priority 3: any anomaly node (fallback)
        any_anomaly_nodes = [a.node_id for a in obs.anomaly_feed if a.node_id is not None]

        # ── Honeypot placement every 3 steps (predictive, not reactive) ──
        if self.step_count % 3 == 0:
            if next_zone_nodes:
                target = random.choice(next_zone_nodes)
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.PLACE_HONEYPOT,
                    target_node=target,
                    reasoning=(
                        f"Placing predictive honeypot at zone {target_zone} node {target} — "
                        f"RED is in zone {red_zone_estimate}, intercepting its next zone crossing."
                    ),
                )
            elif current_zone_nodes:
                target = random.choice(current_zone_nodes)
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.PLACE_HONEYPOT,
                    target_node=target,
                    reasoning=(
                        f"Placing honeypot at zone {red_zone_estimate} node {target} — "
                        f"no next-zone feed available; baiting RED's current area."
                    ),
                )
            elif any_anomaly_nodes:
                target = random.choice(any_anomaly_nodes)
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.PLACE_HONEYPOT,
                    target_node=target,
                    reasoning=f"Fallback honeypot at anomaly node {target} — limited zone intelligence.",
                )

        # ── False escalation every 5 steps to force RED to wait ───
        if self.step_count % 5 == 0 and any_anomaly_nodes:
            target = any_anomaly_nodes[0]
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.TRIGGER_FALSE_ESCALATION,
                target_node=target,
                trap_params={"fake_node": target},
                reasoning=f"Injecting false alert at node {target} — forcing RED to slow down.",
            )

        # ── Investigate anomaly nodes in target zone ───────────────
        if next_zone_nodes:
            target = random.choice(next_zone_nodes)
            return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                         target_node=target,
                         reasoning=f"Investigating ahead-of-RED zone {target_zone} node {target}.")

        if any_anomaly_nodes:
            target = random.choice(any_anomaly_nodes)
            return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                         target_node=target,
                         reasoning=f"Investigating anomaly node {target} for trap placement.")

        # ── Periodic honeypot integrity check ─────────────────────
        if obs.active_honeypots and self.step_count % 4 == 0:
            target = random.choice(obs.active_honeypots)
            return Action(agent_id=self.agent_id, action_type=ActionType.INVESTIGATE_NODE,
                         target_node=target,
                         reasoning=f"Checking honeypot {target} integrity.")

        return Action(agent_id=self.agent_id, action_type=ActionType.STAND_DOWN,
                     reasoning="No anomaly activity — trap configuration stable.")


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
