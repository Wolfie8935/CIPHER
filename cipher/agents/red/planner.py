"""
RED Planner agent for CIPHER — Phase 3 LLM Integration.

The Planner is the RED team's strategist: designs multi-phase attack plans,
prioritizes targets, decides abort vs. continue, and delineates zone-transition
strategy. Maintains a high-level strategic view while operatives handle execution.

In LIVE mode: uses LLM to generate strategic decisions.
In STUB mode: uses weighted random action selection matching Phase 1 behavior.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)

_COORD_FILE = Path("drop_vault") / "coordination.json"


class RedPlanner(BaseAgent):
    """
    RED team Planner — mission architect and abort authority.
    """

    _model_env_key = "hf_model_red_planner"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "red", "planner", config)
        self._abort_gate_streak = 0
        # Change 3: rate-of-change abort histories
        self._abort_gate_streak = 0
        self._suspicion_history: list[float] = []
        self._blue_conf_history: list[float] = []
        # Change 4: track nodes that caused suspicion spikes
        self._nodes_with_suspicion_spikes: set = set()
        # Anti-loop: track how many consecutive steps we've been in the same zone
        self._zone_stall_steps: int = 0
        self._last_zone: int = -1

    def reset(self) -> None:
        super().reset()
        self._abort_gate_streak = 0
        self._suspicion_history = []
        self._blue_conf_history = []
        self._nodes_with_suspicion_spikes = set()
        self._zone_stall_steps = 0
        self._last_zone = -1

    def observe(self, observation: RedObservation) -> None:
        self._current_observation = observation

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                         reasoning="No observation — waiting.")
        if obs.context_reset_this_step:
            self._abort_gate_streak = 0

        # ── Change 3 & 4: Track suspicion/confidence history ─────────
        blue_confidence = float(getattr(obs, 'blue_detection_confidence', 0.0))
        self._suspicion_history.append(obs.estimated_suspicion)
        self._blue_conf_history.append(blue_confidence)
        if len(self._suspicion_history) > 5:
            self._suspicion_history.pop(0)
            self._blue_conf_history.pop(0)

        # Change 4: record nodes that caused suspicion spikes
        if len(self._suspicion_history) >= 2:
            if (self._suspicion_history[-1] - self._suspicion_history[-2] > 0.10
                    and obs.estimated_suspicion > 0.55):
                self._nodes_with_suspicion_spikes.add(obs.current_node)

        # Anti-loop: track zone stall
        if obs.current_zone == self._last_zone:
            self._zone_stall_steps += 1
        else:
            self._zone_stall_steps = 0
            self._last_zone = obs.current_zone
        zone_stalled = self._zone_stall_steps >= 4

        # PRIORITY 0: If in Zone 3 but not at HVT, hunt for HVT node in adjacent
        node_type_val = obs.current_node_type.value if hasattr(obs.current_node_type, 'value') else str(obs.current_node_type)
        if obs.current_zone >= 3 and node_type_val != "high_value_target":
            adj_types = getattr(obs, 'adjacent_node_types', [])
            for i, node_id in enumerate(obs.adjacent_nodes or []):
                nt = adj_types[i].value if i < len(adj_types) and hasattr(adj_types[i], 'value') else ""
                if nt == "high_value_target":
                    self._write_coordination_hint(node_id)
                    return Action(
                        agent_id=self.agent_id,
                        action_type=ActionType.MOVE,
                        target_node=node_id,
                        reasoning=f"In Zone 3 but not at HVT — moving to adjacent HVT node {node_id}.",
                    )
            # HVT not adjacent — explore Zone 3 to find it
            honeypot_nodes = set(getattr(obs, 'honeypot_nodes_nearby', []) or [])
            breadcrumb_nodes = set(getattr(obs, 'breadcrumb_nodes_nearby', []) or [])
            safe_adj = [n for n in (obs.adjacent_nodes or [])
                        if n not in honeypot_nodes and n not in breadcrumb_nodes]
            if safe_adj:
                target = random.choice(self._avoid_recent(safe_adj))
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.MOVE,
                    target_node=target,
                    reasoning=f"Searching Zone 3 for HVT node — exploring node {target}.",
                )

        # PRIORITY 1: Exfiltrate if at HVT — must check BEFORE abort logic
        if node_type_val == "high_value_target" and obs.files_at_current_node:
            # Change 5: cross-reference with state-level exfiltrated files
            already_done = set(getattr(obs, 'exfiltrated_files', []) or [])
            already_attempted = {
                a.target_file for a in self.action_history
                if a.action_type == ActionType.EXFILTRATE and a.target_file
            }
            already_done = already_done | already_attempted
            remaining = [f for f in obs.files_at_current_node if f not in already_done]
            if not remaining:
                return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                             reasoning="All target files already exfiltrated — waiting.")
            target_file = remaining[0]
            return Action(agent_id=self.agent_id, action_type=ActionType.EXFILTRATE,
                         target_file=target_file,
                         reasoning=f"At HVT — exfiltrating {target_file}.")

        # Early abort only if suspicion is both high AND rising very fast
        if len(self._suspicion_history) >= 3:
            recent_delta = self._suspicion_history[-1] - self._suspicion_history[-3]
            blue_delta = self._blue_conf_history[-1] - self._blue_conf_history[-3]
            # Raised threshold: 0.60 → 0.78. Don't abort just because suspicion is "rising"
            if recent_delta > 0.20 and blue_delta > 0.18 and obs.estimated_suspicion > 0.78:
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.ABORT,
                    reasoning=(
                        f"Early abort: suspicion rising fast (+{recent_delta:.2f} in 3 steps), "
                        f"blue_conf rising (+{blue_delta:.2f}). Aborting before confirmed detection."
                    ),
                )

        # PRIORITY 2: Abort when suspicion is critically high AND BLUE has high confidence.
        abort_gate_active = obs.estimated_suspicion > 0.85 and blue_confidence > 0.70
        if abort_gate_active:
            self._abort_gate_streak += 1
            if self._abort_gate_streak >= 2:
                return Action(agent_id=self.agent_id, action_type=ActionType.ABORT,
                             reasoning=(
                                 f"Persistent critical risk: suspicion {obs.estimated_suspicion:.2f}, "
                                 f"BLUE confidence {blue_confidence:.2f} for {self._abort_gate_streak} steps."
                             ))
        else:
            self._abort_gate_streak = 0

        # PRIORITY 3: Write dead drop on context reset
        if obs.context_reset_this_step:
            return Action(agent_id=self.agent_id, action_type=ActionType.WRITE_DEAD_DROP,
                         reasoning="Writing operational continuity dead drop after context reset.")

        # PRIORITY 4: Path strategy — avoid honeypots AND breadcrumb traps (Changes 4 & 6)
        available_paths = getattr(obs, "available_paths", []) or []
        honeypot_nodes = set(getattr(obs, "honeypot_nodes_nearby", []) or [])
        breadcrumb_nodes = set(getattr(obs, "breadcrumb_nodes_nearby", []) or [])

        if available_paths:
            safe_paths = [
                p for p in available_paths
                if p.get("node") in set(obs.adjacent_nodes or [])
                and p.get("node") not in honeypot_nodes
                and p.get("node") not in breadcrumb_nodes  # Change 6
            ]
            if not safe_paths:
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.WAIT,
                    reasoning="All adjacent paths are flagged as traps — waiting to avoid trap entry.",
                )

            # Hold position only when suspicion is critical AND all paths are very costly
            if (
                obs.estimated_suspicion > 0.82
                and all(float(p.get("suspicion_delta", 0.0)) > 0.20 for p in safe_paths)
            ):
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.WAIT,
                    reasoning="Critical suspicion with only high-cost paths — waiting 1 step.",
                )

            deeper_safe = [
                p for p in safe_paths if int(p.get("zone", obs.current_zone)) > int(obs.current_zone)
            ]
            # Always try to advance zones if suspicion < 0.82 (was 0.70 — too conservative)
            pool = deeper_safe if deeper_safe and obs.estimated_suspicion < 0.82 else safe_paths

            # Use long lookback to break 6-node cycles (25→28→29→25…)
            pool_deduped = self._avoid_recent_long(pool)

            # Change 4: trap-aware path scoring with stronger zone-advancement bias
            spike_nodes = self._nodes_with_suspicion_spikes

            def _path_score(p: dict) -> tuple:
                node_id = int(p.get("node", 0))
                base_delta = float(p.get("suspicion_delta", 0.0))
                zone_val = int(p.get("zone", obs.current_zone))
                trap_penalty = 0.50 if node_id in spike_nodes else 0.0
                # Strong zone bonus: advancing zones is worth much more than minor suspicion cost
                zone_bonus = -0.30 if zone_val > obs.current_zone else 0.0
                # When stalled 4+ steps in same zone, dramatically boost zone-advance priority
                if zone_stalled and zone_val > obs.current_zone:
                    zone_bonus = -2.0  # overwhelms any suspicion_delta concern
                return (base_delta + trap_penalty + zone_bonus, -zone_val)

            best = min(pool_deduped, key=_path_score)
            target = int(best["node"])
            self._write_coordination_hint(target)  # Change 7
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.MOVE,
                target_node=target,
                reasoning=(
                    f"Selecting node {target} via minimum-risk path "
                    f"(delta={float(best.get('suspicion_delta', 0.0)):.3f}, zone_stall={self._zone_stall_steps}) — advancing."
                ),
            )

        # PRIORITY 5: Fallback movement heuristics when path metadata is unavailable.
        adj_zones = getattr(obs, 'adjacent_node_zones', [])
        adj_types = obs.adjacent_node_types or []
        forbidden = honeypot_nodes | breadcrumb_nodes | self._nodes_with_suspicion_spikes
        if obs.adjacent_nodes:
            hvt_neighbors = [
                obs.adjacent_nodes[i]
                for i, t in enumerate(adj_types)
                if (t.value if hasattr(t, 'value') else str(t)) == "high_value_target"
            ]
            if hvt_neighbors:
                target = hvt_neighbors[0]
                self._write_coordination_hint(target)
                return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                             target_node=target,
                             reasoning=f"HVT detected adjacent — moving to node {target}.")

            if adj_zones:
                deeper = [
                    obs.adjacent_nodes[i]
                    for i, z in enumerate(adj_zones)
                    if z > obs.current_zone and obs.adjacent_nodes[i] not in forbidden
                ]
                if deeper and obs.estimated_suspicion < 0.70:
                    target = random.choice(self._avoid_recent(deeper))
                    self._write_coordination_hint(target)
                    return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                                 target_node=target,
                                 reasoning=f"Advancing to Zone {obs.current_zone + 1} via node {target}.")
            elif obs.zone_boundary_ahead and obs.estimated_suspicion < 0.70:
                candidates = [n for n in obs.adjacent_nodes if n not in forbidden]
                if candidates:
                    target = random.choice(candidates)
                    self._write_coordination_hint(target)
                    return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                                 target_node=target,
                                 reasoning=f"Crossing zone boundary from Zone {obs.current_zone}.")

            safe_adj = [n for n in obs.adjacent_nodes if n not in forbidden]
            pool = self._avoid_recent(safe_adj if safe_adj else obs.adjacent_nodes)
            target = random.choice(pool)
            self._write_coordination_hint(target)
            return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                         target_node=target,
                         reasoning=f"Lateral movement in Zone {obs.current_zone}.")

        return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                     reasoning="Holding position — assessing situation.")

    def _write_coordination_hint(self, target_node: int) -> None:
        """Write planned target node to shared coordination file (Change 7)."""
        try:
            _COORD_FILE.parent.mkdir(parents=True, exist_ok=True)
            _COORD_FILE.write_text(
                json.dumps({"planned_target": target_node}), encoding="utf-8"
            )
        except Exception:
            pass

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, RedObservation):
            # Zone 0 = INITIAL, Zone 1 = must advance, Zone 2 = close, Zone 3 = exfil now
            phase = "INITIAL-RECON" if obs.current_zone == 0 else "ADVANCE-TO-ZONE2" if obs.current_zone == 1 else "POSITIONING" if obs.current_zone == 2 else "EXFILTRATION"
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

            # Available paths, honeypot, and breadcrumb avoidance context
            available_paths = getattr(obs, 'available_paths', [])
            honeypot_nodes_nearby = getattr(obs, 'honeypot_nodes_nearby', [])
            breadcrumb_nodes_nearby = getattr(obs, 'breadcrumb_nodes_nearby', [])

            path_lines = []

            # Change 9: detection spike warning at top
            if getattr(obs, 'detection_spike_this_step', False):
                path_lines.append(
                    "WARNING: BLUE detection confidence spiked suddenly — "
                    "likely TRIGGER_FALSE_ESCALATION trap. Do NOT abort on this signal alone."
                )

            if available_paths:
                path_lines.append("Available paths (prefer lowest suspicion_delta):")
                for p in available_paths:
                    flags = ""
                    if p.get('node') in (honeypot_nodes_nearby or []):
                        flags += " HONEYPOT-AVOID"
                    if p.get('node') in (breadcrumb_nodes_nearby or []):
                        flags += " BREADCRUMB-AVOID"
                    if p.get('node') in self._nodes_with_suspicion_spikes:
                        flags += " SPIKE-NODE-AVOID"
                    path_lines.append(
                        f"  node {p.get('node')} | zone {p.get('zone')} | "
                        f"host={p.get('hostname')} | suspicion_delta={p.get('suspicion_delta', 0):.3f}{flags}"
                    )
            if honeypot_nodes_nearby:
                path_lines.append(f"HONEYPOT NODES NEARBY (AVOID): {honeypot_nodes_nearby}")
            # Change 6: breadcrumb warning
            if breadcrumb_nodes_nearby:
                path_lines.append(f"BREADCRUMB TRAPS NEARBY (AVOID): {breadcrumb_nodes_nearby}")

            # Urgency: how many steps remain and how far behind are we?
            steps_left = max(0, getattr(obs, 'max_steps', 30) - self.step_count)
            zone_urgency = ""
            if obs.current_zone < 3 and steps_left < 15:
                zone_urgency = f"\n⚠ URGENCY: Only {steps_left} steps left and you are in Zone {obs.current_zone}! Must advance NOW."
            elif obs.current_zone <= 1 and self.step_count > 8:
                zone_urgency = f"\nWARNING: Step {self.step_count} and still in Zone {obs.current_zone} — you are falling behind. MOVE toward Zone {obs.current_zone + 1} immediately."

            strategic_context = (
                f"\n\nCURRENT STRATEGIC CONTEXT:\n"
                f"Phase: {phase}\n"
                f"Steps taken: {self.step_count} | Steps remaining: {steps_left}\n"
                f"Actions in memory: {len(self.action_history)}\n"
                f"Suspicion trend: {'CRITICAL' if obs.estimated_suspicion > 0.80 else 'HIGH' if obs.estimated_suspicion > 0.72 else 'ELEVATED' if obs.estimated_suspicion > 0.55 else 'NORMAL'}\n"
                f"BLUE confidence estimate: {blue_confidence:.2f}\n"
                f"Abort gate streak: {self._abort_gate_streak}\n"
                f"{decay_hint}\n"
                f"{zone_urgency}\n"
                + ("\n".join(path_lines) if path_lines else "No path data available.")
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + strategic_context
        return messages
