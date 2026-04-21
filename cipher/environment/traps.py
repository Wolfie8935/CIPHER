"""
cipher/environment/traps.py

Trap registry and execution engine for CIPHER.

Traps are cognitive weapons — they are designed to manipulate the opposing
agent's REASONING, not just their position. A honeypot doesn't block movement;
it makes RED believe a node is safe when it isn't. A false trail doesn't
stop BLUE from finding RED; it makes BLUE waste its action budget chasing ghosts.
"""
from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cipher.environment.graph import NodeType
from cipher.utils.logger import get_logger

if TYPE_CHECKING:
    import networkx as nx

    from cipher.agents.base_agent import Action
    from cipher.environment.state import EpisodeState
    from cipher.memory.dead_drop import DeadDropVault
    from cipher.utils.config import CipherConfig

logger = get_logger(__name__)


class RedTrapType(str, Enum):
    FALSE_TRAIL = "false_trail"
    TEMPORAL_DECOY = "temporal_decoy"
    HONEYPOT_POISON = "honeypot_poison"
    DEAD_DROP_CORRUPTION = "dead_drop_corruption"


class BlueTrapType(str, Enum):
    HONEYPOT = "honeypot"
    BREADCRUMB = "breadcrumb"
    FALSE_ESCALATION = "false_escalation"
    DEAD_DROP_TAMPER = "dead_drop_tamper"


@dataclass
class RedTrap:
    trap_id: str
    trap_type: RedTrapType
    placed_by: str
    placed_at_step: int
    target_node: int
    duration_steps: int
    params: dict
    is_triggered: bool = False
    triggered_at_step: int | None = None
    is_expired: bool = False


@dataclass
class BlueTrap:
    trap_id: str
    trap_type: BlueTrapType
    placed_by: str
    placed_at_step: int
    target_node: int
    duration_steps: int
    params: dict
    is_triggered: bool = False
    triggered_at_step: int | None = None
    is_expired: bool = False


@dataclass
class TrapEvent:
    event_id: str
    step: int
    trap_id: str
    trap_type: str
    triggered_by_team: str
    effect_description: str
    state_changes: dict


class TrapRegistry:
    """Manages active RED/BLUE traps for one episode."""

    def __init__(self, config: CipherConfig):
        self.red_trap_budget: int = config.env_trap_budget_red
        self.blue_trap_budget: int = config.env_trap_budget_blue
        self.active_red_traps: list[RedTrap] = []
        self.active_blue_traps: list[BlueTrap] = []
        self.triggered_traps: list[dict] = []
        self.expired_traps: list[dict] = []
        self._summary: dict[str, int] = {
            "red_traps_placed": 0,
            "red_traps_triggered": 0,
            "blue_traps_placed": 0,
            "blue_traps_triggered": 0,
            "false_trails_effective": 0,
            "honeypots_triggered": 0,
            "dead_drops_tampered": 0,
            "dead_drops_tamper_detected": 0,
        }

    def place_red_trap(
        self,
        trap_type: RedTrapType,
        placed_by: str,
        target_node: int,
        step: int,
        params: dict,
        state: EpisodeState | None = None,
    ) -> tuple[bool, str]:
        if self.red_trap_budget <= 0:
            return False, "RED trap budget exhausted"
        if trap_type == RedTrapType.HONEYPOT_POISON and state is not None:
            if state.last_honeypot_trigger_step is None:
                return False, "HONEYPOT_POISON requires prior honeypot trigger"
        if trap_type == RedTrapType.DEAD_DROP_CORRUPTION and state is not None:
            if not state.blue_discovered_drop_paths:
                return False, "DEAD_DROP_CORRUPTION requires discovered drop evidence"

        duration = int(params.get("duration_steps", self._default_duration_red(trap_type)))
        trap = RedTrap(
            trap_id=str(uuid.uuid4()),
            trap_type=trap_type,
            placed_by=placed_by,
            placed_at_step=step,
            target_node=target_node,
            duration_steps=duration,
            params=dict(params or {}),
        )
        self.active_red_traps.append(trap)
        self.red_trap_budget -= 1
        self._summary["red_traps_placed"] += 1
        return True, "placed"

    def place_blue_trap(
        self,
        trap_type: BlueTrapType,
        placed_by: str,
        target_node: int,
        step: int,
        params: dict,
        state: EpisodeState | None = None,
    ) -> tuple[bool, str]:
        if self.blue_trap_budget <= 0:
            return False, "BLUE trap budget exhausted"
        if trap_type == BlueTrapType.DEAD_DROP_TAMPER and state is not None:
            if not state.blue_discovered_drop_paths:
                return False, "DEAD_DROP_TAMPER requires discovered drop path"

        duration = int(params.get("duration_steps", self._default_duration_blue(trap_type)))
        trap = BlueTrap(
            trap_id=str(uuid.uuid4()),
            trap_type=trap_type,
            placed_by=placed_by,
            placed_at_step=step,
            target_node=target_node,
            duration_steps=duration,
            params=dict(params or {}),
        )
        self.active_blue_traps.append(trap)
        self.blue_trap_budget -= 1
        self._summary["blue_traps_placed"] += 1
        return True, "placed"

    def evaluate_step(
        self,
        step: int,
        last_red_action: Action | None,
        last_blue_actions: list[Action],
        state: EpisodeState,
        graph: nx.DiGraph,
        vault: DeadDropVault,
    ) -> list[TrapEvent]:
        events: list[TrapEvent] = []
        for trap in self.active_red_traps:
            if trap.is_expired or trap.is_triggered:
                continue
            event = self._evaluate_red_trap(trap, step, last_red_action, state, graph, vault)
            if event:
                events.append(event)
            self._expire_if_needed(trap, step)

        for trap in self.active_blue_traps:
            if trap.is_expired or trap.is_triggered:
                continue
            event = self._evaluate_blue_trap(
                trap, step, last_red_action, last_blue_actions, state, graph, vault
            )
            if event:
                events.append(event)
            self._expire_if_needed(trap, step)
        return events

    def apply_false_trail(self, trap: RedTrap, state: EpisodeState) -> dict:
        history = [n for n in state.red_path_history if n != state.red_current_node]
        if not history:
            return {"injected": 0}
        n_events = int(trap.params.get("n_fake_events", random.randint(3, 5)))
        n_events = max(3, min(5, n_events))
        injected = []
        for _ in range(n_events):
            node_id = random.choice(history)
            injected.append(
                {
                    "event_type": random.choice(["unusual_traversal", "file_access_out_of_hours"]),
                    "node_id": node_id,
                    "severity": round(random.uniform(0.35, 0.75), 3),
                    "step": state.step,
                    "zone": state.get_zone_for_node(node_id),
                    "is_red_planted": True,
                    "is_noise": False,
                }
            )
        state.blue_anomaly_history.extend(injected)
        state.anomaly_log.extend(injected)
        self._summary["false_trails_effective"] += 1
        return {"injected": len(injected)}

    def apply_temporal_decoy(self, trap: RedTrap, state: EpisodeState) -> dict:
        decoy_node = int(trap.params.get("decoy_node", trap.target_node))
        anomalies = [
            {
                "event_type": "AUTH_GATEWAY_STRESS",
                "node_id": decoy_node,
                "severity": 0.85,
                "step": state.step,
                "zone": state.get_zone_for_node(decoy_node),
                "is_red_planted": True,
                "is_noise": False,
            },
            {
                "event_type": "RAPID_LATERAL_MOVEMENT",
                "node_id": decoy_node,
                "severity": 0.9,
                "step": state.step,
                "zone": state.get_zone_for_node(decoy_node),
                "is_red_planted": True,
                "is_noise": False,
            },
        ]
        state.blue_anomaly_history.extend(anomalies)
        state.anomaly_log.extend(anomalies)
        state.blue_suspected_zone = state.get_zone_for_node(decoy_node)
        return {"decoy_node": decoy_node}

    def apply_honeypot_poison(
        self, trap: RedTrap, state: EpisodeState, vault: DeadDropVault
    ) -> dict:
        adjacent_node = int(trap.params.get("adjacent_node", trap.target_node))
        dead_end_node = int(trap.params.get("dead_end_node", adjacent_node))
        directive = f"High-value asset detected at node {dead_end_node}. Proceed immediately."
        # Keep existing schema by writing a normal dead drop and only varying directive.
        from cipher.environment.observation import generate_red_observation
        from cipher.memory.dead_drop import build_dead_drop_from_state

        obs = generate_red_observation(state)
        drop = build_dead_drop_from_state(
            trap.placed_by,
            state,
            obs,
            directive,
        )
        path, _ = vault.write(drop, adjacent_node)
        state.blue_detection_confidence = max(0.0, state.blue_detection_confidence - 0.15)
        return {"poison_drop_path": path, "adjacent_node": adjacent_node, "dead_end_node": dead_end_node}

    def apply_breadcrumb(
        self, trap: BlueTrap, state: EpisodeState, graph: nx.DiGraph
    ) -> dict:
        culdesac = int(trap.params.get("culdesac_node", trap.target_node))
        path_nodes = trap.params.get("path_nodes", [])
        if not path_nodes:
            path_nodes = [n for n in list(graph.nodes)[:3] if n != culdesac]
        for idx, node in enumerate(path_nodes):
            files = list(graph.nodes[node].get("files", []))
            files.append(f"CONFIDENTIAL_DATA_{idx}.enc")
            graph.nodes[node]["files"] = files
        return {"breadcrumb_nodes": path_nodes, "culdesac_node": culdesac}

    def apply_false_escalation(self, trap: BlueTrap, state: EpisodeState) -> dict:
        fake_zone = trap.params.get("fake_zone")
        if fake_zone is None:
            zones = [z for z in [0, 1, 2, 3] if z != state.red_current_zone]
            fake_zone = random.choice(zones) if zones else 0
        fake_node = int(trap.params.get("fake_node", trap.target_node))
        anomaly = {
            "event_type": "HONEYPOT_TRIGGERED",
            "node_id": fake_node,
            "severity": 0.9,
            "step": state.step,
            "zone": int(fake_zone),
            "is_red_planted": False,
            "is_noise": False,
        }
        state.blue_anomaly_history.append(anomaly)
        state.anomaly_log.append(anomaly)
        state.blue_traps_active.append(
            {"pending_false_escalation_eval": {"placed_zone": int(fake_zone), "placed_step": state.step}}
        )
        return {"fake_zone": int(fake_zone), "fake_node": fake_node}

    def apply_dead_drop_tamper(
        self, trap: BlueTrap, vault: DeadDropVault, drop_path: str
    ) -> bool:
        path = Path(drop_path)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        payload["continuation_directive"] = trap.params.get(
            "new_directive",
            "Proceed immediately to nearest high-value file server.",
        )
        # Intentionally do not update integrity_hash so verify() fails.
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._summary["dead_drops_tampered"] += 1
        return True

    def get_trap_summary(self) -> dict:
        return dict(self._summary)

    def _evaluate_red_trap(
        self,
        trap: RedTrap,
        step: int,
        last_red_action: Action | None,
        state: EpisodeState,
        graph: nx.DiGraph,
        vault: DeadDropVault,
    ) -> TrapEvent | None:
        changes: dict[str, Any] | None = None
        if trap.trap_type == RedTrapType.FALSE_TRAIL:
            changes = self.apply_false_trail(trap, state)
        elif trap.trap_type == RedTrapType.TEMPORAL_DECOY:
            changes = self.apply_temporal_decoy(trap, state)
            state.update_suspicion(0.08)
        elif trap.trap_type == RedTrapType.HONEYPOT_POISON:
            if state.last_honeypot_trigger_step is None:
                return None
            changes = self.apply_honeypot_poison(trap, state, vault)
        elif trap.trap_type == RedTrapType.DEAD_DROP_CORRUPTION:
            if not state.blue_discovered_drop_paths:
                return None
            changes = {"ready": True}
        if changes is None:
            return None
        trap.is_triggered = True
        trap.triggered_at_step = step
        self._summary["red_traps_triggered"] += 1
        event = TrapEvent(
            event_id=str(uuid.uuid4()),
            step=step,
            trap_id=trap.trap_id,
            trap_type=trap.trap_type.value,
            triggered_by_team="red",
            effect_description=f"RED trap {trap.trap_type.value} triggered",
            state_changes=changes,
        )
        self.triggered_traps.append({"event": event.event_id, "trap_id": trap.trap_id})
        return event

    def _evaluate_blue_trap(
        self,
        trap: BlueTrap,
        step: int,
        last_red_action: Action | None,
        last_blue_actions: list[Action],
        state: EpisodeState,
        graph: nx.DiGraph,
        vault: DeadDropVault,
    ) -> TrapEvent | None:
        changes: dict[str, Any] | None = None
        if trap.trap_type == BlueTrapType.HONEYPOT:
            if state.red_current_node != trap.target_node:
                return None
            state.update_suspicion(0.40)
            state.last_honeypot_trigger_step = step
            state.blue_honeypots_triggered.append(trap.target_node)
            self._summary["honeypots_triggered"] += 1
            anomaly = {
                "event_type": "HONEYPOT_TRIGGERED",
                "node_id": trap.target_node,
                "severity": 1.0,
                "step": step,
                "zone": state.get_zone_for_node(trap.target_node),
                "is_red_planted": False,
                "is_noise": False,
            }
            state.blue_anomaly_history.append(anomaly)
            state.anomaly_log.append(anomaly)
            changes = {"honeypot_node": trap.target_node}
        elif trap.trap_type == BlueTrapType.BREADCRUMB:
            changes = self.apply_breadcrumb(trap, state, graph)
        elif trap.trap_type == BlueTrapType.FALSE_ESCALATION:
            changes = self.apply_false_escalation(trap, state)
            if last_red_action is not None and last_red_action.action_type.value in {"wait", "move"}:
                if last_red_action.action_type.value == "wait":
                    state.blue_detection_confidence = min(1.0, state.blue_detection_confidence + 0.10)
                elif last_red_action.target_node is not None:
                    target_zone = state.get_zone_for_node(last_red_action.target_node)
                    if target_zone != state.red_current_zone:
                        state.blue_detection_confidence = min(1.0, state.blue_detection_confidence + 0.10)
        elif trap.trap_type == BlueTrapType.DEAD_DROP_TAMPER:
            if not state.blue_discovered_drop_paths:
                return None
            ok = self.apply_dead_drop_tamper(trap, vault, state.blue_discovered_drop_paths[0])
            if not ok:
                return None
            changes = {"tampered_path": state.blue_discovered_drop_paths[0]}
        if changes is None:
            return None
        trap.is_triggered = True
        trap.triggered_at_step = step
        self._summary["blue_traps_triggered"] += 1
        event = TrapEvent(
            event_id=str(uuid.uuid4()),
            step=step,
            trap_id=trap.trap_id,
            trap_type=trap.trap_type.value,
            triggered_by_team="blue",
            effect_description=f"BLUE trap {trap.trap_type.value} triggered",
            state_changes=changes,
        )
        self.triggered_traps.append({"event": event.event_id, "trap_id": trap.trap_id})
        return event

    def _expire_if_needed(self, trap: RedTrap | BlueTrap, step: int) -> None:
        if trap.is_expired:
            return
        if step >= trap.placed_at_step + trap.duration_steps:
            trap.is_expired = True
            self.expired_traps.append({"trap_id": trap.trap_id, "step": step})

    @staticmethod
    def _default_duration_red(trap_type: RedTrapType) -> int:
        return {
            RedTrapType.FALSE_TRAIL: 5,
            RedTrapType.TEMPORAL_DECOY: 3,
            RedTrapType.HONEYPOT_POISON: 1,
            RedTrapType.DEAD_DROP_CORRUPTION: 1000,
        }[trap_type]

    @staticmethod
    def _default_duration_blue(trap_type: BlueTrapType) -> int:
        return {
            BlueTrapType.HONEYPOT: 1000,
            BlueTrapType.BREADCRUMB: 10,
            BlueTrapType.FALSE_ESCALATION: 3,
            BlueTrapType.DEAD_DROP_TAMPER: 1,
        }[trap_type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "red_trap_budget": self.red_trap_budget,
            "blue_trap_budget": self.blue_trap_budget,
            "active_red_traps": [t.__dict__ for t in self.active_red_traps],
            "active_blue_traps": [t.__dict__ for t in self.active_blue_traps],
            "summary": self.get_trap_summary(),
        }

