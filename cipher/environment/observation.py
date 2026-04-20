"""
Asymmetric observation engine for CIPHER — Phase 2.

This is one of the most important files in the project. The entire asymmetry
of CIPHER — RED sees the world differently from BLUE — is enforced here and
only here.

RED receives: current position with zone/hostname/services, adjacent nodes
(with honeypots MASKED as file servers), noisy suspicion estimate, privilege
level, credential state, dead drop availability, zone boundary warnings.

BLUE receives: anomaly feed (zone-aware statistical anomalies), per-zone alert
levels, honeypot health, traversal pattern summaries, investigation budget.

Phase 2 additions:
- Zone-aware anomaly generation (Critical zone → higher severity)
- Protocol-aware classification (SSH brute force vs HTTP anomaly vs SMB lateral)
- Temporal correlation (rapid actions → compound anomalies)
- New anomaly types: privilege_escalation_attempt, lateral_movement_burst,
  data_staging, credential_harvesting

Owns: observation generation, honeypot masking, anomaly event creation.
Does NOT own: state mutation, agent logic, or reward computation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from cipher.environment.graph import NetworkZone, NodeType
from cipher.environment.state import EpisodeState
from cipher.utils.config import config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


# ── Anomaly event types ──────────────────────────────────────────

ANOMALY_TYPES = [
    "unusual_traversal",
    "file_access_out_of_hours",
    "repeated_auth_attempt",
    "anomalous_data_volume",
    "privilege_escalation_attempt",
    "lateral_movement_burst",
    "data_staging",
    "credential_harvesting",
]


# ── Data classes ─────────────────────────────────────────────────


@dataclass
class AnomalyEvent:
    """
    A single anomaly event in BLUE's anomaly feed.

    These are statistical anomalies derived from RED's actions (or injected
    as noise). BLUE never sees RED directly — only these exhaust signatures.
    """

    event_type: str
    node_id: int | None  # Which node the anomaly was detected at
    severity: float  # 0-1
    is_red_planted: bool  # GROUND TRUTH — BLUE agents must never see this field
    step: int
    zone: int | None = None  # Zone where anomaly occurred
    protocol: str | None = None  # Protocol involved if applicable
    description: str = ""  # Human-readable anomaly description


@dataclass
class RedObservation:
    """
    What RED agents see each step — Phase 2 expanded.

    CRITICAL: honeypot nodes appear as FILE_SERVER in this view.
    RED cannot distinguish honeypots from real file servers.
    """

    # ── Core position ────────────────────────────────────────────
    current_node: int
    current_node_type: NodeType  # NEVER NodeType.HONEYPOT — masked
    adjacent_nodes: list[int] = field(default_factory=list)
    adjacent_node_types: list[NodeType] = field(default_factory=list)
    files_at_current_node: list[str] = field(default_factory=list)
    estimated_suspicion: float = 0.0  # noisy — true + Gaussian(0, 0.1)
    dead_drops_available: list[str] = field(default_factory=list)
    step: int = 0
    context_reset_this_step: bool = False

    # ── Phase 2: zone awareness ──────────────────────────────────
    current_zone: int = 0
    current_hostname: str = ""
    current_services: list[str] = field(default_factory=list)
    adjacent_hostnames: list[str] = field(default_factory=list)
    edge_protocols: list[str] = field(default_factory=list)

    # ── Phase 2: privilege and credentials ────────────────────────
    current_privilege_level: int = 0
    required_privileges: list[int] = field(default_factory=list)
    credentials_held: list[str] = field(default_factory=list)

    # ── Phase 2: tactical awareness ──────────────────────────────
    zone_boundary_ahead: bool = False
    steps_since_last_move: int = 0


@dataclass
class BlueObservation:
    """
    What BLUE agents see each step — Phase 2 expanded.

    BLUE never sees RED's position directly. Only the anomaly feed,
    their own honeypot state, zone-level aggregations, and investigation budget.
    """

    # ── Core feeds ───────────────────────────────────────────────
    anomaly_feed: list[AnomalyEvent] = field(default_factory=list)
    current_detection_confidence: float = 0.0
    active_honeypots: list[int] = field(default_factory=list)
    honeypots_triggered_this_step: list[int] = field(default_factory=list)
    step: int = 0

    # ── Phase 2: zone-level intelligence ─────────────────────────
    zone_alert_levels: dict[int, float] = field(default_factory=dict)
    network_anomaly_rate: float = 0.0
    honeypot_health: dict[int, str] = field(default_factory=dict)

    # ── Phase 2: pattern analysis ────────────────────────────────
    recent_traversal_patterns: list[dict[str, Any]] = field(default_factory=list)
    investigation_budget_remaining: int = 10
    total_alerts_issued: int = 0


# ── Masking ──────────────────────────────────────────────────────


def _mask_node_type(nt: NodeType) -> NodeType:
    """
    Mask node types for RED's observation.

    Honeypots and decoy assets appear as file servers to RED.
    All other types are shown as-is.
    """
    if nt == NodeType.HONEYPOT:
        return NodeType.FILE_SERVER
    if nt == NodeType.DECOY_ASSET:
        return NodeType.FILE_SERVER
    return nt


# ── RED observation generator ────────────────────────────────────


def generate_red_observation(
    state: EpisodeState,
    *args,
    dead_drop_paths: list[str] | None = None,
    context_reset_this_step: bool = False,
    **kwargs,
) -> RedObservation:
    """
    Generate RED's observation from the current episode state.

    Implements:
    - Honeypot masking: HONEYPOT/DECOY nodes appear as FILE_SERVER
    - Noisy suspicion: Gaussian noise (sigma=0.1) added to true suspicion
    - Dead drop availability: only drops at or near the current node
    - Zone awareness: current zone, hostname, services, boundary detection
    - Privilege tracking: current level and requirements for adjacent nodes
    - Edge protocol info: what protocols connect to adjacent nodes
    - Inactivity tracking: steps since last move action

    Args:
        state: The current ground-truth episode state.
        dead_drop_paths: Paths to dead drops available at RED's current node.
        context_reset_this_step: Whether a context reset is happening this step.

    Returns:
        A RedObservation with masked information.
    """
    graph = state.graph
    current = state.red_current_node

    # Current node data
    node_data = graph.nodes[current]
    raw_type = node_data.get("node_type", NodeType.STANDARD_NODE)
    masked_type = _mask_node_type(raw_type)

    # Adjacent nodes and their masked types
    successors = list(graph.successors(current))
    adj_types = [
        _mask_node_type(graph.nodes[n].get("node_type", NodeType.STANDARD_NODE))
        for n in successors
    ]

    # Files at current node
    files = list(node_data.get("files", []))

    # Noisy suspicion estimate
    noise = random.gauss(0.0, 0.1)
    noisy_suspicion = max(0.0, min(1.0, state.red_suspicion_score + noise))

    # Zone info
    current_zone = node_data.get("zone")
    zone_int = current_zone.value if hasattr(current_zone, "value") else (current_zone if current_zone is not None else 0)
    hostname = node_data.get("hostname", f"node-{current}")
    services = list(node_data.get("services", []))

    # Adjacent hostnames and edge protocols
    adj_hostnames = [
        graph.nodes[n].get("hostname", f"node-{n}") for n in successors
    ]
    edge_protos = []
    for n in successors:
        edge_data = graph.edges.get((current, n), {})
        edge_protos.append(edge_data.get("protocol", "ssh"))

    # Required privileges for adjacent nodes
    req_privs = [
        graph.nodes[n].get("privilege_level", 0) for n in successors
    ]

    # Zone boundary detection
    zone_boundary = any(
        graph.nodes[n].get("zone") != current_zone
        for n in successors
    ) if successors else False

    # Steps since last move
    last_move_step = 0
    for record in reversed(state.red_movement_history):
        last_move_step = record.get("step", 0)
        break
    steps_since = max(0, state.step - last_move_step) if last_move_step > 0 else 0

    return RedObservation(
        current_node=current,
        current_node_type=masked_type,
        adjacent_nodes=successors,
        adjacent_node_types=adj_types,
        files_at_current_node=files,
        estimated_suspicion=round(noisy_suspicion, 3),
        dead_drops_available=dead_drop_paths or [],
        step=state.step,
        context_reset_this_step=context_reset_this_step,
        # Phase 2 fields
        current_zone=zone_int,
        current_hostname=hostname,
        current_services=services,
        adjacent_hostnames=adj_hostnames,
        edge_protocols=edge_protos,
        current_privilege_level=state.red_privilege_level,
        required_privileges=req_privs,
        credentials_held=list(state.red_credentials_acquired),
        zone_boundary_ahead=zone_boundary,
        steps_since_last_move=steps_since,
    )


# ── BLUE observation generator ───────────────────────────────────


def generate_blue_observation(
    state: EpisodeState,
    recent_anomalies: list[AnomalyEvent] | None = None,
) -> BlueObservation:
    """
    Generate BLUE's observation from the current episode state.

    BLUE sees:
    - An anomaly feed with real RED-caused anomalies AND false positives
    - Per-zone alert levels aggregated from recent anomalies
    - Their own detection confidence
    - Honeypot positions, status, and trigger history
    - Recent traversal pattern summaries (anonymized)
    - Investigation budget remaining

    BLUE does NOT see:
    - RED's position
    - RED's suspicion score
    - Dead drop contents
    - RED's credential state

    Args:
        state: The current ground-truth episode state.
        recent_anomalies: Pre-generated anomaly events for this step.

    Returns:
        A BlueObservation.
    """
    from cipher.environment.graph import get_honeypot_nodes

    # Generate anomalies if not provided
    if recent_anomalies is None:
        recent_anomalies = _generate_step_anomalies(state)

    # Active honeypots
    active_honeypots = get_honeypot_nodes(state.graph)

    # Honeypots triggered this step
    triggered_this_step = [
        hp
        for hp in state.blue_honeypots_triggered
        if hp not in getattr(state, "_prev_honeypots_triggered", [])
    ]

    # Zone alert levels — aggregate from zone_suspicion_scores + recent anomalies
    zone_alerts = dict(state.zone_suspicion_scores)
    for anomaly in recent_anomalies:
        if anomaly.zone is not None:
            zone_alerts[anomaly.zone] = min(
                1.0,
                zone_alerts.get(anomaly.zone, 0.0) + anomaly.severity * 0.05,
            )

    # Network anomaly rate — rolling average over last 10 steps
    recent_window = [
        e for e in state.anomaly_log
        if e.get("step", 0) >= max(0, state.step - 10)
    ]
    anomaly_rate = len(recent_window) / 10.0 if state.step > 0 else 0.0

    # Honeypot health — map each honeypot to its status
    honeypot_health: dict[int, str] = {}
    for hp in active_honeypots:
        if hp in state.blue_honeypots_triggered:
            honeypot_health[hp] = "triggered"
        else:
            honeypot_health[hp] = "active"

    # Recent traversal patterns — anonymized edge usage
    recent_traversals = _compute_traversal_patterns(state)

    # Investigation budget
    n_investigations = len(state.blue_investigated_nodes)
    budget_remaining = max(0, 10 - n_investigations)

    # Total alerts issued
    total_alerts = len(state.blue_alerts_issued)

    return BlueObservation(
        anomaly_feed=recent_anomalies,
        current_detection_confidence=state.blue_detection_confidence,
        active_honeypots=active_honeypots,
        honeypots_triggered_this_step=triggered_this_step,
        step=state.step,
        # Phase 2 fields
        zone_alert_levels=zone_alerts,
        network_anomaly_rate=round(anomaly_rate, 3),
        honeypot_health=honeypot_health,
        recent_traversal_patterns=recent_traversals,
        investigation_budget_remaining=budget_remaining,
        total_alerts_issued=total_alerts,
    )


# ── Anomaly generation ───────────────────────────────────────────


def _generate_step_anomalies(state: EpisodeState) -> list[AnomalyEvent]:
    """
    Generate anomaly events for the current step.

    Produces real anomalies from RED's recent actions plus false positives
    (noise) according to env_anomaly_feed_noise. Phase 2 adds zone-aware
    severity scaling and temporal correlation.

    Args:
        state: Current episode state.

    Returns:
        List of AnomalyEvent objects for this step.
    """
    anomalies: list[AnomalyEvent] = []

    # Extract RED's most recent actions from episode log
    recent_actions = [
        entry
        for entry in state.episode_log
        if entry.get("step") == state.step
        and entry.get("agent_id", "").startswith("red_")
    ]

    # Generate real anomalies from RED actions
    for action_entry in recent_actions:
        anomaly = generate_anomaly_from_action(action_entry, state)
        if anomaly is not None:
            anomalies.append(anomaly)
            # Record in state anomaly log
            state.record_anomaly({
                "step": state.step,
                "event_type": anomaly.event_type,
                "node_id": anomaly.node_id,
                "severity": anomaly.severity,
                "zone": anomaly.zone,
                "is_real": True,
            })

    # Check for compound anomalies (rapid successive actions)
    if len(recent_actions) >= 3:
        compound = _generate_compound_anomaly(recent_actions, state)
        if compound is not None:
            anomalies.append(compound)

    # Inject false positives (noise)
    noise_rate = config.env_anomaly_feed_noise
    n_noise = max(
        1,
        int(len(anomalies) * noise_rate / (1.0 - noise_rate + 0.001)),
    )
    # Occasional noise even when no RED actions generated anomalies
    if not anomalies and random.random() < noise_rate:
        n_noise = 1

    for _ in range(n_noise):
        noise_node = (
            random.choice(list(state.graph.nodes))
            if state.graph.number_of_nodes() > 0
            else None
        )
        noise_zone = None
        if noise_node is not None:
            z = state.graph.nodes[noise_node].get("zone")
            noise_zone = z.value if hasattr(z, "value") else z

        anomalies.append(
            AnomalyEvent(
                event_type=random.choice(ANOMALY_TYPES[:4]),  # noise uses base types
                node_id=noise_node,
                severity=round(random.uniform(0.05, 0.35), 3),
                is_red_planted=False,
                step=state.step,
                zone=noise_zone,
                description="Background network noise",
            )
        )

    # Shuffle so BLUE can't distinguish real from noise by position
    random.shuffle(anomalies)
    return anomalies


def generate_anomaly_from_action(
    action_entry: dict[str, Any],
    state: EpisodeState,
) -> AnomalyEvent | None:
    """
    Convert a RED action into a BLUE-visible anomaly event.

    Phase 2 expansion:
    - Zone-aware severity: Critical zone actions generate higher severity
    - Protocol-aware classification: different protocols → different anomaly types
    - Privilege escalation detection: auth attempts generate distinct events
    - Suspicion-scaled detection probability

    Args:
        action_entry: A dict from the episode log representing one action.
        state: Current episode state.

    Returns:
        An AnomalyEvent if the action generates a visible signature, else None.
    """
    action_type = action_entry.get("action_type", "")

    # Base detection probabilities per action type
    detection_prob_base = {
        "move": 0.25,
        "read_file": 0.35,
        "exfiltrate": 0.70,
        "write_dead_drop": 0.15,
        "read_dead_drop": 0.10,
        "plant_trap": 0.20,
        "wait": 0.05,
        "abort": 0.10,
    }

    base_prob = detection_prob_base.get(action_type, 0.20)

    # Suspicion scaling — higher suspicion = higher detection
    adjusted_prob = min(1.0, base_prob + state.red_suspicion_score * 0.25)

    # Zone scaling — Critical zone has elevated monitoring
    current_zone = state.get_zone_for_node(state.red_current_node)
    if current_zone is not None:
        zone_monitor_bonus = {0: 0.0, 1: 0.02, 2: 0.05, 3: 0.10}
        adjusted_prob = min(
            1.0, adjusted_prob + zone_monitor_bonus.get(current_zone, 0.0)
        )

    # Roll detection check
    if random.random() > adjusted_prob:
        return None  # Action was not detected

    # ── Map action → anomaly type (Phase 2: protocol-aware) ──────
    payload = action_entry.get("payload", {})
    target_node = payload.get("target_node", state.red_current_node)

    event_type = _classify_anomaly_type(action_type, state, payload)

    # ── Severity computation (Phase 2: zone-scaled) ──────────────
    base_severity = base_prob + random.uniform(0.0, 0.2)
    zone_severity_mult = {0: 1.0, 1: 1.1, 2: 1.3, 3: 1.6}
    zone_mult = zone_severity_mult.get(current_zone or 0, 1.0)
    severity = round(min(1.0, base_severity * zone_mult), 3)

    # Description
    description = _anomaly_description(event_type, action_type, target_node)

    return AnomalyEvent(
        event_type=event_type,
        node_id=target_node,
        severity=severity,
        is_red_planted=True,
        step=state.step,
        zone=current_zone,
        protocol=payload.get("protocol"),
        description=description,
    )


def _classify_anomaly_type(
    action_type: str,
    state: EpisodeState,
    payload: dict[str, Any],
) -> str:
    """Classify the anomaly type based on action, zone, and protocol context."""
    # Protocol-specific classification
    protocol = payload.get("protocol", "")

    # Check for privilege escalation patterns first
    current_zone = state.get_zone_for_node(state.red_current_node)
    if current_zone is not None and current_zone >= 2:
        if action_type in ("move", "read_file"):
            if random.random() < 0.3:
                return "privilege_escalation_attempt"

    if action_type == "move":
        # Check if this is a lateral movement burst
        recent_moves = [
            r for r in state.red_movement_history
            if r.get("step", 0) >= max(0, state.step - 3)
        ]
        if len(recent_moves) >= 3:
            return "lateral_movement_burst"
        if protocol in ("smb", "rdp"):
            return "lateral_movement_burst"
        return "unusual_traversal"

    if action_type == "read_file":
        # Check for data staging pattern
        if len(state.red_exfiltrated_files) > 0:
            return "data_staging"
        return "file_access_out_of_hours"

    if action_type == "exfiltrate":
        return "anomalous_data_volume"

    if action_type in ("write_dead_drop", "read_dead_drop"):
        return "file_access_out_of_hours"

    if action_type == "plant_trap":
        return "unusual_traversal"

    return "unusual_traversal"


def _generate_compound_anomaly(
    recent_actions: list[dict[str, Any]],
    state: EpisodeState,
) -> AnomalyEvent | None:
    """
    Generate a compound anomaly when multiple RED actions occur in rapid
    succession (3+ actions in the same step).

    This catches burst behavior that individual anomaly checks might miss.
    """
    if random.random() > 0.6:
        return None  # Not always generated

    current_zone = state.get_zone_for_node(state.red_current_node)
    zone_mult = {0: 1.0, 1: 1.2, 2: 1.5, 3: 1.8}.get(current_zone or 0, 1.0)

    return AnomalyEvent(
        event_type="lateral_movement_burst",
        node_id=state.red_current_node,
        severity=round(min(1.0, 0.5 * zone_mult), 3),
        is_red_planted=True,
        step=state.step,
        zone=current_zone,
        description=f"Burst of {len(recent_actions)} actions detected in single step",
    )


def _anomaly_description(
    event_type: str,
    action_type: str,
    target_node: Any,
) -> str:
    """Generate a human-readable description for an anomaly event."""
    descriptions = {
        "unusual_traversal": f"Unusual network traversal detected near node {target_node}",
        "file_access_out_of_hours": f"File access outside normal hours at node {target_node}",
        "repeated_auth_attempt": f"Repeated authentication attempts at node {target_node}",
        "anomalous_data_volume": f"Anomalous data transfer volume at node {target_node}",
        "privilege_escalation_attempt": f"Potential privilege escalation at node {target_node}",
        "lateral_movement_burst": f"Rapid lateral movement pattern near node {target_node}",
        "data_staging": f"Possible data staging activity at node {target_node}",
        "credential_harvesting": f"Credential harvesting indicators at node {target_node}",
    }
    return descriptions.get(event_type, f"Unknown anomaly at node {target_node}")


def _compute_traversal_patterns(state: EpisodeState) -> list[dict[str, Any]]:
    """
    Compute anonymized traversal pattern summaries for BLUE.

    Returns aggregated edge usage statistics over the last 10 steps,
    without revealing RED's exact position.
    """
    patterns: list[dict[str, Any]] = []
    recent_movements = [
        m for m in state.red_movement_history
        if m.get("step", 0) >= max(0, state.step - 10)
    ]

    if not recent_movements:
        return patterns

    # Aggregate by zone transitions
    zone_transitions: dict[str, int] = {}
    for m in recent_movements:
        from_node = m.get("from_node", 0)
        to_node = m.get("to_node", 0)
        from_zone = state.get_zone_for_node(from_node)
        to_zone = state.get_zone_for_node(to_node)
        key = f"zone_{from_zone}_to_{to_zone}"
        zone_transitions[key] = zone_transitions.get(key, 0) + 1

    for transition, count in zone_transitions.items():
        patterns.append({
            "pattern_type": "zone_transition",
            "details": transition,
            "frequency": count,
            "window": "last_10_steps",
        })

    # Protocol usage summary
    protocol_counts: dict[str, int] = {}
    for m in recent_movements:
        proto = m.get("protocol", "unknown")
        protocol_counts[proto] = protocol_counts.get(proto, 0) + 1

    for proto, count in protocol_counts.items():
        patterns.append({
            "pattern_type": "protocol_usage",
            "details": proto,
            "frequency": count,
            "window": "last_10_steps",
        })

    return patterns
