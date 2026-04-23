"""
RED Planner agent for CIPHER — Phase 3 LLM Integration.

The Planner is the RED team's strategist: designs multi-phase attack plans,
prioritizes targets, decides abort vs. continue, and delineates zone-transition
strategy. Maintains a high-level strategic view while operatives handle execution.

In LIVE mode: uses LLM to generate strategic decisions.
In STUB mode: uses weighted random action selection matching Phase 1 behavior.
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class RedPlanner(BaseAgent):
    """
    RED team Planner — mission architect and abort authority.
    """

    _model_env_key = "nvidia_model_red_planner"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "red", "planner", config)
        # Runtime stability tuning: avoid single-step noisy abort flips.
        self._abort_gate_streak = 0

    def observe(self, observation: RedObservation) -> None:
        self._current_observation = observation

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                         reasoning="No observation — waiting.")
        if obs.context_reset_this_step:
            self._abort_gate_streak = 0

        # PRIORITY 1: Exfiltrate if at HVT — must check BEFORE abort logic
        node_type_val = obs.current_node_type.value if hasattr(obs.current_node_type, 'value') else str(obs.current_node_type)
        if node_type_val == "high_value_target" and obs.files_at_current_node:
            already_attempted = {
                a.target_file for a in self.action_history
                if a.action_type == ActionType.EXFILTRATE and a.target_file
            }
            remaining = [f for f in obs.files_at_current_node if f not in already_attempted]
            target_file = remaining[0] if remaining else obs.files_at_current_node[0]
            return Action(agent_id=self.agent_id, action_type=ActionType.EXFILTRATE,
                         target_file=target_file,
                         reasoning=f"At HVT — exfiltrating {target_file}.")

        # PRIORITY 2: Abort ONLY when suspicion is critically high AND BLUE has confirmed detection.
        # Threshold: suspicion > 0.80 AND blue_detection_confidence > 0.60.
        # Do NOT abort on suspicion alone — the noisy estimate is unreliable below 0.80.
        blue_confidence = getattr(obs, 'blue_detection_confidence', 0.0)
        abort_gate_active = obs.estimated_suspicion > 0.80 and blue_confidence > 0.60
        if abort_gate_active:
            self._abort_gate_streak += 1
            # Require persistence across two consecutive planner decisions to reduce
            # premature aborts from single noisy spikes (Track A1 spec).
            if self._abort_gate_streak >= 2:
                return Action(agent_id=self.agent_id, action_type=ActionType.ABORT,
                             reasoning=(
                                 f"Persistent critical risk: suspicion {obs.estimated_suspicion:.2f}, "
                                 f"BLUE confidence {blue_confidence:.2f} for {self._abort_gate_streak} steps — aborting."
                             ))
        else:
            self._abort_gate_streak = 0

        # PRIORITY 3: Write dead drop on context reset
        if obs.context_reset_this_step:
            return Action(agent_id=self.agent_id, action_type=ActionType.WRITE_DEAD_DROP,
                         reasoning="Writing operational continuity dead drop after context reset.")

        # PRIORITY 4: Path strategy (A2) — use available_paths and avoid known honeypots.
        available_paths = getattr(obs, "available_paths", []) or []
        honeypot_nodes = set(getattr(obs, "honeypot_nodes_nearby", []) or [])
        if available_paths:
            safe_paths = [
                p for p in available_paths
                if p.get("node") in set(obs.adjacent_nodes or []) and p.get("node") not in honeypot_nodes
            ]
            if not safe_paths:
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.WAIT,
                    reasoning="All adjacent paths are flagged as honeypots — waiting to avoid trap entry.",
                )

            # If all safe options are noisy under elevated suspicion, hold position.
            if (
                obs.estimated_suspicion > 0.70
                and all(float(p.get("suspicion_delta", 0.0)) > 0.15 for p in safe_paths)
            ):
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.WAIT,
                    reasoning="Elevated suspicion with only high-cost paths — waiting to reduce exposure.",
                )

            deeper_safe = [
                p for p in safe_paths if int(p.get("zone", obs.current_zone)) > int(obs.current_zone)
            ]
            pool = deeper_safe if deeper_safe and obs.estimated_suspicion < 0.70 else safe_paths
            best = min(
                pool,
                key=lambda p: (float(p.get("suspicion_delta", 0.0)), int(p.get("zone", obs.current_zone))),
            )
            target = int(best["node"])
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.MOVE,
                target_node=target,
                reasoning=(
                    f"Selecting node {target} via minimum-risk path "
                    f"(delta={float(best.get('suspicion_delta', 0.0)):.3f}) while avoiding honeypots."
                ),
            )

        # PRIORITY 5: Fallback movement heuristics when path metadata is unavailable.
        adj_zones = getattr(obs, 'adjacent_node_zones', [])
        adj_types = obs.adjacent_node_types or []
        if obs.adjacent_nodes:
            hvt_neighbors = [
                obs.adjacent_nodes[i]
                for i, t in enumerate(adj_types)
                if (t.value if hasattr(t, 'value') else str(t)) == "high_value_target"
            ]
            if hvt_neighbors:
                target = hvt_neighbors[0]
                return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                             target_node=target,
                             reasoning=f"HVT detected adjacent — moving to node {target}.")

            if adj_zones:
                deeper = [
                    obs.adjacent_nodes[i]
                    for i, z in enumerate(adj_zones)
                    if z > obs.current_zone
                ]
                if deeper and obs.estimated_suspicion < 0.70:
                    target = random.choice(deeper)
                    return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                                 target_node=target,
                                 reasoning=f"Advancing to Zone {obs.current_zone + 1} via node {target}.")
            elif obs.zone_boundary_ahead and obs.estimated_suspicion < 0.70:
                target = random.choice(obs.adjacent_nodes)
                return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                             target_node=target,
                             reasoning=f"Crossing zone boundary from Zone {obs.current_zone}.")

            target = random.choice(obs.adjacent_nodes)
            return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                         target_node=target,
                         reasoning=f"Lateral movement in Zone {obs.current_zone}.")

        return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                     reasoning="Holding position — assessing situation.")

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, RedObservation):
            phase = "RECON" if obs.current_zone <= 1 else "POSITIONING" if obs.current_zone == 2 else "EXFILTRATION"
            blue_confidence = float(getattr(obs, "blue_detection_confidence", 0.0))

            # Suspicion-decay hint derived from BLUE confidence trend (not RED action history).
            prev_blue = getattr(self, "_prev_blue_confidence", None)
            no_recent_alerts = (
                blue_confidence < 0.40
                and (prev_blue is None or blue_confidence <= float(prev_blue) + 0.02)
            )
            self._prev_blue_confidence = blue_confidence
            decay_hint = (
                "[SUSPICION DECAY ACTIVE] No BLUE alerts in last 3 steps — "
                "effective suspicion is likely lower than the noisy reading. "
                "Do NOT abort unless suspicion > 0.80 AND blue_confidence > 0.60."
                if no_recent_alerts
                else "[BLUE ACTIVE] Recent alerts detected — treat suspicion as reliable."
            )

            # Available paths and honeypot avoidance context
            available_paths = getattr(obs, 'available_paths', [])
            honeypot_nodes_nearby = getattr(obs, 'honeypot_nodes_nearby', [])

            path_lines = []
            if available_paths:
                path_lines.append("Available paths (prefer lowest suspicion_delta):")
                for p in available_paths:
                    hp_flag = " ⚠ HONEYPOT" if p.get('node') in honeypot_nodes_nearby else ""
                    path_lines.append(
                        f"  node {p.get('node')} | zone {p.get('zone')} | "
                        f"host={p.get('hostname')} | suspicion_delta={p.get('suspicion_delta', 0):.3f}{hp_flag}"
                    )
            if honeypot_nodes_nearby:
                path_lines.append(f"HONEYPOT NODES NEARBY (AVOID): {honeypot_nodes_nearby}")

            strategic_context = (
                f"\n\nCURRENT STRATEGIC CONTEXT:\n"
                f"Phase: {phase}\n"
                f"Steps taken: {self.step_count}\n"
                f"Actions in memory: {len(self.action_history)}\n"
                f"Suspicion trend: {'RISING' if obs.estimated_suspicion > 0.3 else 'STABLE'}\n"
                f"BLUE confidence estimate: {blue_confidence:.2f}\n"
                f"Abort gate streak: {self._abort_gate_streak}\n"
                f"{decay_hint}\n"
                + ("\n".join(path_lines) if path_lines else "No path data available.")
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + strategic_context
        return messages
