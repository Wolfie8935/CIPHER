"""
Episode state for CIPHER — Phase 2.

Holds the complete ground-truth state of a single episode: RED position,
movement history, zone tracking, credential state, BLUE knowledge, suspicion
scores (global and per-zone), dead drops, anomaly log, and the full action log.

Owns: episode state representation, state mutation methods, serialization.
Does NOT own: observation generation (observation.py), reward computation,
or agent logic.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import networkx as nx
from networkx.readwrite import json_graph

from cipher.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EpisodeState:
    """
    Complete ground-truth state of one CIPHER episode.

    This object is the single source of truth. Observations are derived from it.
    Rewards are computed from it. The episode log records every action taken.
    """

    # ── Identity ─────────────────────────────────────────────────
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step: int = 0

    # ── Graph ────────────────────────────────────────────────────
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    # ── RED state ────────────────────────────────────────────────
    red_current_node: int = 0
    red_visited_nodes: list[int] = field(default_factory=list)
    red_exfiltrated_files: list[str] = field(default_factory=list)
    red_suspicion_score: float = 0.0
    red_traps_placed: list[dict[str, Any]] = field(default_factory=list)
    red_context_resets: int = 0

    # ── RED Phase 2 fields ───────────────────────────────────────
    red_movement_history: list[dict[str, Any]] = field(default_factory=list)
    red_current_zone: int = 0
    red_privilege_level: int = 0  # 0=guest, 1=user, 2=elevated, 3=admin
    red_credentials_acquired: list[str] = field(default_factory=list)

    # ── BLUE state ───────────────────────────────────────────────
    blue_honeypots_triggered: list[int] = field(default_factory=list)
    blue_detection_confidence: float = 0.0
    blue_current_investigation_node: int | None = None
    blue_traps_active: list[dict[str, Any]] = field(default_factory=list)

    # ── BLUE Phase 2 fields ──────────────────────────────────────
    blue_investigated_nodes: list[int] = field(default_factory=list)
    blue_alerts_issued: list[dict[str, Any]] = field(default_factory=list)
    blue_false_positives: int = 0

    # ── Shared ───────────────────────────────────────────────────
    dead_drops_on_disk: list[str] = field(default_factory=list)
    episode_log: list[dict[str, Any]] = field(default_factory=list)
    is_terminal: bool = False
    terminal_reason: str | None = None
    # Valid terminal reasons:
    # "exfiltration_complete" | "detected" | "max_steps" | "aborted"

    # ── Phase 2: anomaly and zone tracking ───────────────────────
    anomaly_log: list[dict[str, Any]] = field(default_factory=list)
    zone_suspicion_scores: dict[int, float] = field(
        default_factory=lambda: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    )

    # ── Action logging ───────────────────────────────────────────

    def log_action(
        self,
        agent_id: str,
        action_type: str,
        action_payload: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """
        Append an action record to the episode log.

        Args:
            agent_id: The acting agent's identifier.
            action_type: The type of action taken (ActionType value).
            action_payload: The full action parameters.
            result: The outcome of the action.
        """
        entry = {
            "step": self.step,
            "agent_id": agent_id,
            "action_type": action_type,
            "payload": action_payload,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.episode_log.append(entry)

    # ── Suspicion mechanics ──────────────────────────────────────

    def update_suspicion(self, delta: float) -> None:
        """
        Add a suspicion delta to RED's global score, clamped to [0, 1].

        Args:
            delta: Amount to add (positive increases suspicion).
        """
        previous = self.red_suspicion_score
        self.red_suspicion_score = max(
            0.0, min(1.0, self.red_suspicion_score + delta)
        )
        if abs(self.red_suspicion_score - previous) > 0.01:
            logger.debug(
                f"Suspicion updated: {previous:.3f} -> {self.red_suspicion_score:.3f} "
                f"(delta={delta:+.3f})"
            )

    def update_suspicion_from_action(self, action_type: str, target_node: int, graph: Any) -> float:
        """Update suspicion dynamically based on action and graph properties."""
        from cipher.environment.graph import NodeType
        delta = 0.05
        if graph.nodes[target_node].get("node_type") == NodeType.HONEYPOT:
            delta = 0.50
        self.update_suspicion(delta)
        return delta

    @classmethod
    def create_from_scenario(cls, scenario: Any, graph: Any) -> "EpisodeState":
        """Factory for generating initial state from scenario."""
        from cipher.environment.graph import get_entry_points
        eps = get_entry_points(graph)
        start_node = eps[0] if eps else 0
        return cls(graph=graph, red_current_node=start_node)

    def update_zone_suspicion(self, zone: int, delta: float) -> None:
        """
        Update the suspicion score for a specific zone, clamped to [0, 1].

        Args:
            zone: Zone index (0-3).
            delta: Amount to add.
        """
        if zone not in self.zone_suspicion_scores:
            self.zone_suspicion_scores[zone] = 0.0
        previous = self.zone_suspicion_scores[zone]
        self.zone_suspicion_scores[zone] = max(
            0.0, min(1.0, previous + delta)
        )

    # ── Phase 2: movement tracking ───────────────────────────────

    def record_movement(
        self,
        from_node: int,
        to_node: int,
        protocol: str = "ssh",
        suspicion_cost: float = 0.0,
    ) -> None:
        """
        Record a RED movement between nodes with protocol and cost info.

        Updates movement history, visited nodes, zone tracking, and
        per-zone suspicion.

        Args:
            from_node: Source node ID.
            to_node: Destination node ID.
            protocol: Protocol used for the traversal.
            suspicion_cost: The suspicion delta incurred.
        """
        record = {
            "step": self.step,
            "from_node": from_node,
            "to_node": to_node,
            "protocol": protocol,
            "suspicion_cost": round(suspicion_cost, 4),
        }
        self.red_movement_history.append(record)

        # Update current node and visited list
        self.red_current_node = to_node
        if to_node not in self.red_visited_nodes:
            self.red_visited_nodes.append(to_node)

        # Update zone tracking
        to_zone = self.get_zone_for_node(to_node)
        if to_zone is not None:
            self.red_current_zone = to_zone
            self.update_zone_suspicion(to_zone, suspicion_cost)

        # Update global suspicion
        self.update_suspicion(suspicion_cost)

    # ── Phase 2: credential system ───────────────────────────────

    def acquire_credential(self, credential_id: str) -> None:
        """
        Add a credential to RED's inventory and check for privilege escalation.

        Credential IDs encode their level: 'cred_zone_{N}_{suffix}'.
        Acquiring a credential for a higher zone raises privilege_level.

        Args:
            credential_id: The credential token string.
        """
        if credential_id not in self.red_credentials_acquired:
            self.red_credentials_acquired.append(credential_id)
            logger.debug(
                f"Credential acquired: {credential_id} "
                f"(total: {len(self.red_credentials_acquired)})"
            )

            # Check if this escalates privilege
            # Convention: credential IDs contain 'zone_N'
            for i in range(3, -1, -1):
                if f"zone_{i}" in credential_id:
                    if i > self.red_privilege_level:
                        self.red_privilege_level = i
                        logger.debug(
                            f"Privilege escalated to level {i}"
                        )
                    break

    # ── Phase 2: zone lookup ─────────────────────────────────────

    def get_zone_for_node(self, node_id: int) -> int | None:
        """
        Return the zone index for a given node.

        Args:
            node_id: The node to look up.

        Returns:
            Zone index (0-3), or None if node not in graph.
        """
        if node_id in self.graph.nodes:
            zone = self.graph.nodes[node_id].get("zone")
            if zone is not None:
                return zone.value if hasattr(zone, "value") else int(zone)
        return None

    # ── Phase 2: anomaly logging ─────────────────────────────────

    def record_anomaly(self, anomaly_event: dict[str, Any]) -> None:
        """
        Append an anomaly event to the episode anomaly log.

        Args:
            anomaly_event: Dict with keys like event_type, node_id, severity, step.
        """
        self.anomaly_log.append(anomaly_event)

    # ── Phase 2: BLUE alert tracking ─────────────────────────────

    def issue_blue_alert(self, node_id: int, confidence: float) -> None:
        """
        Record a BLUE alert, checking if it's a true or false positive.

        If the alerted node is RED's current node, it's a true positive.
        Otherwise, it's a false positive.

        Args:
            node_id: The node BLUE is alerting on.
            confidence: BLUE's confidence level (0-1).
        """
        is_correct = (node_id == self.red_current_node)
        alert_record = {
            "step": self.step,
            "node": node_id,
            "confidence": round(confidence, 3),
            "correct": is_correct,
        }
        self.blue_alerts_issued.append(alert_record)

        if not is_correct:
            self.blue_false_positives += 1
            logger.debug(
                f"BLUE false positive alert at node {node_id} "
                f"(RED is at {self.red_current_node})"
            )

    # ── Terminal check ───────────────────────────────────────────

    def is_done(self) -> bool:
        """Return True if the episode is in a terminal state."""
        return self.is_terminal

    # ── Serialization ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the episode state to a JSON-compatible dictionary.

        The NetworkX graph is serialized using node-link format.
        Enum values are converted to their primitive representations.

        Returns:
            A dict that can be passed to json.dumps().
        """
        graph_data = json_graph.node_link_data(self.graph)

        # Convert enums in graph data to their values
        for node in graph_data.get("nodes", []):
            if "node_type" in node and hasattr(node["node_type"], "value"):
                node["node_type"] = node["node_type"].value
            if "zone" in node and hasattr(node["zone"], "value"):
                node["zone"] = node["zone"].value

        # Convert zone_suspicion_scores keys to strings for JSON
        zone_susp = {str(k): v for k, v in self.zone_suspicion_scores.items()}

        return {
            "episode_id": self.episode_id,
            "step": self.step,
            "graph": graph_data,
            # RED state
            "red_current_node": self.red_current_node,
            "red_visited_nodes": self.red_visited_nodes,
            "red_exfiltrated_files": self.red_exfiltrated_files,
            "red_suspicion_score": self.red_suspicion_score,
            "red_traps_placed": self.red_traps_placed,
            "red_context_resets": self.red_context_resets,
            "red_movement_history": self.red_movement_history,
            "red_current_zone": self.red_current_zone,
            "red_privilege_level": self.red_privilege_level,
            "red_credentials_acquired": self.red_credentials_acquired,
            # BLUE state
            "blue_honeypots_triggered": self.blue_honeypots_triggered,
            "blue_detection_confidence": self.blue_detection_confidence,
            "blue_current_investigation_node": self.blue_current_investigation_node,
            "blue_traps_active": self.blue_traps_active,
            "blue_investigated_nodes": self.blue_investigated_nodes,
            "blue_alerts_issued": self.blue_alerts_issued,
            "blue_false_positives": self.blue_false_positives,
            # Shared
            "dead_drops_on_disk": self.dead_drops_on_disk,
            "episode_log": self.episode_log,
            "is_terminal": self.is_terminal,
            "terminal_reason": self.terminal_reason,
            "anomaly_log": self.anomaly_log,
            "zone_suspicion_scores": zone_susp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeState:
        """
        Deserialize an EpisodeState from a dictionary.

        Args:
            data: A dict previously produced by to_dict().

        Returns:
            A reconstructed EpisodeState instance.
        """
        graph = json_graph.node_link_graph(data.get("graph", {}))

        # Restore zone_suspicion_scores integer keys
        zone_susp_raw = data.get("zone_suspicion_scores", {})
        zone_susp = {int(k): float(v) for k, v in zone_susp_raw.items()}
        if not zone_susp:
            zone_susp = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}

        return cls(
            episode_id=data.get("episode_id", str(uuid.uuid4())),
            step=data.get("step", 0),
            graph=graph,
            # RED state
            red_current_node=data.get("red_current_node", 0),
            red_visited_nodes=data.get("red_visited_nodes", []),
            red_exfiltrated_files=data.get("red_exfiltrated_files", []),
            red_suspicion_score=data.get("red_suspicion_score", 0.0),
            red_traps_placed=data.get("red_traps_placed", []),
            red_context_resets=data.get("red_context_resets", 0),
            red_movement_history=data.get("red_movement_history", []),
            red_current_zone=data.get("red_current_zone", 0),
            red_privilege_level=data.get("red_privilege_level", 0),
            red_credentials_acquired=data.get("red_credentials_acquired", []),
            # BLUE state
            blue_honeypots_triggered=data.get("blue_honeypots_triggered", []),
            blue_detection_confidence=data.get("blue_detection_confidence", 0.0),
            blue_current_investigation_node=data.get(
                "blue_current_investigation_node"
            ),
            blue_traps_active=data.get("blue_traps_active", []),
            blue_investigated_nodes=data.get("blue_investigated_nodes", []),
            blue_alerts_issued=data.get("blue_alerts_issued", []),
            blue_false_positives=data.get("blue_false_positives", 0),
            # Shared
            dead_drops_on_disk=data.get("dead_drops_on_disk", []),
            episode_log=data.get("episode_log", []),
            is_terminal=data.get("is_terminal", False),
            terminal_reason=data.get("terminal_reason"),
            anomaly_log=data.get("anomaly_log", []),
            zone_suspicion_scores=zone_susp,
        )
