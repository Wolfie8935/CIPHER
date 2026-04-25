"""
RED Operative agent for CIPHER — Phase 3 LLM Integration.

The Operative is the RED team's field executor: handles stealth movement,
protocol selection, counter-trap placement, and real-time suspicion management.
"""
from __future__ import annotations

import random

from cipher.agents.base_agent import Action, ActionType, BaseAgent
from cipher.environment.observation import RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class RedOperative(BaseAgent):
    """RED team Operative — stealth executor and counter-trap specialist."""

    _model_env_key = "hf_model_red_operative"

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(agent_id, "red", "operative", config)
        self._last_suspicion: float = 0.0
        self._spike_nodes: set[int] = set()
        self._counter_trap_cooldown: int = 0

    def reset(self) -> None:
        super().reset()
        self._last_suspicion = 0.0
        self._spike_nodes = set()
        self._counter_trap_cooldown = 0

    def observe(self, observation: RedObservation) -> None:
        self._current_observation = observation

    def _choose_counter_trap(self, obs: RedObservation, blue_conf: float) -> Action | None:
        """
        Select the best counter-trap based on current threat level.

        Returns an Action or None if no trap should be placed.
        """
        if self._trap_budget_remaining <= 0:
            return None
        if self._counter_trap_cooldown > 0:
            self._counter_trap_cooldown -= 1
            return None

        n_moves = sum(1 for a in self.action_history if a.action_type == ActionType.MOVE)

        # Suspicion spike this step — reactive counter-trap
        susp_delta = obs.estimated_suspicion - self._last_suspicion
        if susp_delta > 0.12 and obs.estimated_suspicion > 0.50 and n_moves >= 2:
            # Plant false trail at current node to send BLUE to a ghost location
            self._counter_trap_cooldown = 2  # don't plant again for 2 steps
            self._trap_budget_remaining = max(0, self._trap_budget_remaining - 1)
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.PLANT_FALSE_TRAIL,
                target_node=obs.current_node,
                trap_params={"n_fake_events": 4, "ghost_zone": max(0, obs.current_zone - 1)},
                reasoning=(
                    f"Suspicion spiked +{susp_delta:.2f} this step. "
                    f"Planting false trail to redirect BLUE investigation."
                ),
            )

        # BLUE actively hunting (high conf) AND we have moves to burn
        if obs.estimated_suspicion > 0.60 and blue_conf > 0.50 and n_moves >= 3:
            if obs.current_zone >= 2:
                # Zone 2/3: use temporal decoy — noisy burst to lock BLUE into old position
                trap_type = ActionType.PLANT_TEMPORAL_DECOY
                params = {"burst_steps": 3}
                reason = f"Zone {obs.current_zone} approach — temporal decoy buys 3 steps of misdirection."
            else:
                # Zone 0/1: simpler false trail
                trap_type = ActionType.PLANT_FALSE_TRAIL
                params = {"n_fake_events": 3}
                reason = f"BLUE converging (conf={blue_conf:.2f}) — redirecting with false trail."

            self._counter_trap_cooldown = 3
            self._trap_budget_remaining = max(0, self._trap_budget_remaining - 1)
            return Action(
                agent_id=self.agent_id,
                action_type=trap_type,
                target_node=obs.current_node,
                trap_params=params,
                reasoning=reason,
            )

        # Honeypot detected adjacent — try to poison it to confuse BLUE
        honeypots = set(getattr(obs, 'honeypot_nodes_nearby', []) or [])
        if honeypots and obs.estimated_suspicion < 0.55 and self._trap_budget_remaining >= 2:
            hp_node = next(iter(honeypots))
            self._trap_budget_remaining = max(0, self._trap_budget_remaining - 1)
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.PLANT_HONEYPOT_POISON,
                target_node=hp_node,
                trap_params={"fake_intel": "entry_point_safe"},
                reasoning=f"Poisoning BLUE honeypot at node {hp_node} to feed false intel.",
            )

        return None

    def _stub_act(self) -> Action:
        obs = self._current_observation
        if obs is None:
            return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                         reasoning="No observation — maintaining position.")

        blue_conf = float(getattr(obs, 'blue_detection_confidence', 0.0))
        honeypots = set(getattr(obs, 'honeypot_nodes_nearby', []) or [])
        breadcrumbs = set(getattr(obs, 'breadcrumb_nodes_nearby', []) or [])
        susp_delta = obs.estimated_suspicion - self._last_suspicion
        self._last_suspicion = obs.estimated_suspicion

        # Track nodes where suspicion spiked (avoid them)
        if susp_delta > 0.12:
            self._spike_nodes.add(obs.current_node)

        # Only wait at critical suspicion
        if obs.estimated_suspicion > 0.82:
            return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                         reasoning=f"Critical suspicion {obs.estimated_suspicion:.2f} — going dark 1 step.")

        # Counter-trap decision
        counter = self._choose_counter_trap(obs, blue_conf)
        if counter is not None:
            return counter

        # Move with zone-advancement + danger avoidance priority
        if obs.adjacent_nodes:
            adj_zones = getattr(obs, 'adjacent_node_zones', [])
            adj_types = getattr(obs, 'adjacent_node_types', [])
            forbidden = honeypots | breadcrumbs | self._spike_nodes

            safe = [n for n in obs.adjacent_nodes if n not in forbidden]
            pool = safe if safe else obs.adjacent_nodes

            # HVT adjacent? Go there immediately
            for i, node_id in enumerate(obs.adjacent_nodes):
                if node_id in pool and i < len(adj_types):
                    nt = adj_types[i].value if hasattr(adj_types[i], 'value') else str(adj_types[i])
                    if nt == "high_value_target":
                        return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                                     target_node=node_id,
                                     reasoning=f"HVT detected adjacent — moving to node {node_id}.")

            # Prefer deeper zones
            if adj_zones and len(adj_zones) == len(obs.adjacent_nodes):
                deeper = [
                    obs.adjacent_nodes[i] for i in range(len(obs.adjacent_nodes))
                    if obs.adjacent_nodes[i] in pool and i < len(adj_zones)
                    and int(adj_zones[i]) > int(obs.current_zone)
                ]
                if deeper:
                    target = random.choice(self._avoid_recent(deeper))
                    return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                                 target_node=target,
                                 reasoning=f"Advancing to Zone {obs.current_zone + 1} via node {target}.")

            target = random.choice(self._avoid_recent(pool))
            hostname = (obs.adjacent_hostnames[obs.adjacent_nodes.index(target)]
                        if obs.adjacent_hostnames and target in obs.adjacent_nodes else f"node-{target}")
            return Action(agent_id=self.agent_id, action_type=ActionType.MOVE,
                         target_node=target,
                         reasoning=f"Stealth move to {hostname}.")

        return Action(agent_id=self.agent_id, action_type=ActionType.WAIT,
                     reasoning="No safe movement available — holding position.")

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        obs = self._current_observation
        if obs and isinstance(obs, RedObservation):
            blue_conf = float(getattr(obs, 'blue_detection_confidence', 0.0))
            susp_delta = obs.estimated_suspicion - self._last_suspicion
            steps_left = max(0, getattr(obs, 'max_steps', 30) - self.step_count)
            threat_level = (
                "CRITICAL — plant counter-trap then move" if obs.estimated_suspicion > 0.75
                else "HIGH — consider false trail before next move" if obs.estimated_suspicion > 0.60
                else "NORMAL — keep moving"
            )
            stealth_context = (
                f"\n\nSTEALTH STATUS:\n"
                f"Suspicion: {obs.estimated_suspicion:.2f} (delta this step: {susp_delta:+.2f})\n"
                f"BLUE confidence: {blue_conf:.2f}\n"
                f"Threat level: {threat_level}\n"
                f"Trap budget remaining: {self._trap_budget_remaining}\n"
                f"Steps remaining: {steps_left}\n"
                f"Counter-trap rules:\n"
                f"  - Suspicion spiked > 0.12 this step → PLANT_FALSE_TRAIL at current node\n"
                f"  - Suspicion > 0.60 AND blue_conf > 0.50 → PLANT_TEMPORAL_DECOY (zone 2+) or PLANT_FALSE_TRAIL\n"
                f"  - Honeypot adjacent AND suspicion < 0.55 → PLANT_HONEYPOT_POISON to feed BLUE false data\n"
                f"  - After planting, MOVE immediately — do not wait\n"
                f"Priority: MOVE toward Zone 3 > plant counter-trap > wait"
            )
            messages[0] = dict(messages[0])
            messages[0]["content"] = messages[0]["content"] + stealth_context
        return messages
