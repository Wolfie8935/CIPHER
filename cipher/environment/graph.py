"""
Enterprise network graph generator for CIPHER — Phase 2.

Generates a directed graph representing a realistic enterprise network with
a four-zone topology (DMZ → Corporate → Restricted → Critical), typed nodes
(entry points, auth gateways, file servers, databases, honeypots, decoys,
high-value target), realistic hostnames, service inventories, and weighted
edges modeling traversal cost, suspicion impact, and protocol requirements.

Zone Architecture:
    Zone 0 — DMZ:        ~8 nodes  (entry points, web servers, mail gateways)
    Zone 1 — Corporate:  ~15 nodes (workstations, file servers, auth gateways)
    Zone 2 — Restricted: ~15 nodes (databases, app servers, internal APIs)
    Zone 3 — Critical:   ~12 nodes (HVT, core DBs, backup servers)

Owns: graph topology generation, node/edge attribute assignment, zone layout.
Does NOT own: observation masking (observation.py), state tracking (state.py),
or agent logic.
"""
from __future__ import annotations

import math
import random
from enum import Enum
from typing import Optional

import networkx as nx

from cipher.utils.config import config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


# ── Enums ────────────────────────────────────────────────────────


class NodeType(str, Enum):
    """Types of nodes in the enterprise network graph."""

    ENTRY_POINT = "entry_point"
    AUTH_GATEWAY = "auth_gateway"
    FILE_SERVER = "file_server"
    DATABASE = "database"
    HONEYPOT = "honeypot"
    DECOY_ASSET = "decoy_asset"
    HIGH_VALUE_TARGET = "high_value_target"
    STANDARD_NODE = "standard_node"


class NetworkZone(int, Enum):
    """Enterprise network zones, ordered by depth / sensitivity."""

    DMZ = 0
    CORPORATE = 1
    RESTRICTED = 2
    CRITICAL = 3


# ── Zone metadata ────────────────────────────────────────────────

# Target node counts per zone (for a 50-node graph).
# Fractions of total are used when n_nodes != 50.
_ZONE_FRACTIONS = {
    NetworkZone.DMZ: 0.16,        # ~8 nodes
    NetworkZone.CORPORATE: 0.30,  # ~15 nodes
    NetworkZone.RESTRICTED: 0.30, # ~15 nodes
    NetworkZone.CRITICAL: 0.24,   # ~12 nodes
}

# Hostname prefixes per zone
_ZONE_PREFIXES = {
    NetworkZone.DMZ: "dmz",
    NetworkZone.CORPORATE: "corp",
    NetworkZone.RESTRICTED: "rstr",
    NetworkZone.CRITICAL: "crit",
}

# Hostname fragments per node type
_TYPE_FRAGMENTS = {
    NodeType.ENTRY_POINT: "gw",
    NodeType.AUTH_GATEWAY: "auth",
    NodeType.FILE_SERVER: "fs",
    NodeType.DATABASE: "db",
    NodeType.HONEYPOT: "fs",          # looks like a file server
    NodeType.DECOY_ASSET: "db",       # looks like a database
    NodeType.HIGH_VALUE_TARGET: "core",
    NodeType.STANDARD_NODE: "srv",
}

# Services per node type
_TYPE_SERVICES: dict[NodeType, list[list[str]]] = {
    NodeType.ENTRY_POINT: [["http", "https"], ["ssh", "http"], ["smtp", "http"]],
    NodeType.AUTH_GATEWAY: [["ldap", "kerberos"], ["radius", "ldap"], ["ssh", "ldap"]],
    NodeType.FILE_SERVER: [["smb", "nfs"], ["smb", "ssh"], ["ftp", "ssh"]],
    NodeType.DATABASE: [["mysql", "ssh"], ["postgresql", "ssh"], ["mssql", "rdp"]],
    NodeType.HONEYPOT: [["smb", "ssh"], ["http", "ssh"], ["ftp", "ssh"]],
    NodeType.DECOY_ASSET: [["mysql", "ssh"], ["smb", "http"], ["postgresql", "ssh"]],
    NodeType.HIGH_VALUE_TARGET: [["ssh", "internal_api"], ["rdp", "smb"], ["ssh", "smb"]],
    NodeType.STANDARD_NODE: [["ssh"], ["rdp"], ["http"], ["ssh", "http"]],
}

# OS types per node type (weighted choices)
_TYPE_OS: dict[NodeType, list[str]] = {
    NodeType.ENTRY_POINT: ["linux", "appliance"],
    NodeType.AUTH_GATEWAY: ["linux", "appliance"],
    NodeType.FILE_SERVER: ["linux", "windows"],
    NodeType.DATABASE: ["linux", "linux", "windows"],
    NodeType.HONEYPOT: ["linux", "windows"],
    NodeType.DECOY_ASSET: ["linux", "windows"],
    NodeType.HIGH_VALUE_TARGET: ["linux"],
    NodeType.STANDARD_NODE: ["linux", "windows", "windows"],
}

# Protocols available on edges between zones
_EDGE_PROTOCOLS = {
    (NetworkZone.DMZ, NetworkZone.DMZ): ["http", "ssh", "smtp"],
    (NetworkZone.DMZ, NetworkZone.CORPORATE): ["ssh", "http"],
    (NetworkZone.CORPORATE, NetworkZone.CORPORATE): ["smb", "rdp", "ssh", "http"],
    (NetworkZone.CORPORATE, NetworkZone.RESTRICTED): ["ssh", "internal_api"],
    (NetworkZone.RESTRICTED, NetworkZone.RESTRICTED): ["ssh", "internal_api", "smb"],
    (NetworkZone.RESTRICTED, NetworkZone.CRITICAL): ["ssh", "internal_api"],
    (NetworkZone.CRITICAL, NetworkZone.CRITICAL): ["ssh", "internal_api", "smb"],
}

# Realistic file names per node type
_FILE_TEMPLATES: dict[NodeType, list[str]] = {
    NodeType.FILE_SERVER: [
        "shared_docs_{id}.xlsx", "project_plan_{id}.pdf", "budget_{id}.csv",
        "meeting_notes_{id}.docx", "org_chart_{id}.pdf",
    ],
    NodeType.DATABASE: [
        "customers_{id}.sql", "transactions_{id}.bak", "audit_log_{id}.db",
        "employee_records_{id}.sql", "inventory_{id}.csv",
    ],
    NodeType.HIGH_VALUE_TARGET: [
        "classified_roadmap_{id}.pdf", "merger_financials_{id}.xlsx",
        "board_minutes_{id}.docx", "patent_application_{id}.pdf",
        "executive_payroll_{id}.csv",
    ],
    NodeType.HONEYPOT: [
        "confidential_report_{id}.pdf", "salary_data_{id}.xlsx",
        "trade_secrets_{id}.docx",
    ],
    NodeType.DECOY_ASSET: [
        "sensitive_data_{id}.csv", "strategic_plan_{id}.pdf",
        "research_results_{id}.xlsx", "compliance_audit_{id}.docx",
    ],
    NodeType.STANDARD_NODE: [
        "readme_{id}.txt", "config_{id}.ini",
    ],
}


# ── Public API ───────────────────────────────────────────────────


def generate_enterprise_graph(
    n_nodes: int,
    honeypot_density: float = 0.2,
    seed: int | None = None,
) -> nx.DiGraph:
    """
    Generate a directed enterprise network graph with zone-based topology.

    Creates a connected directed graph with 4 security zones, typed nodes,
    realistic hostnames, service inventories, and weighted edges with protocol
    and firewall attributes.

    Args:
        n_nodes: Total number of nodes in the graph. Recommended: >= 15.
        honeypot_density: Fraction of nodes that are honeypots (0.0–1.0).
        seed: Random seed for reproducible generation. Each episode should
              use a different seed for unique topology.

    Returns:
        A connected nx.DiGraph with node and edge attributes set.
    """
    rng = random.Random(seed)

    if n_nodes < 10:
        logger.warning(
            f"Graph size {n_nodes} is very small. Minimum recommended: 10."
        )

    graph = nx.DiGraph()

    # ── Step 1: Compute zone sizes ───────────────────────────────
    zone_sizes = _compute_zone_sizes(n_nodes, rng)

    # ── Step 2: Assign node types within each zone ───────────────
    node_assignments = _assign_zone_node_types(
        zone_sizes, honeypot_density, rng
    )

    # ── Step 3: Create nodes with full attributes ────────────────
    _type_counters: dict[str, int] = {}
    for node_id, (zone, node_type) in enumerate(node_assignments):
        # Generate hostname
        type_key = f"{zone.value}_{node_type.value}"
        _type_counters[type_key] = _type_counters.get(type_key, 0) + 1
        hostname = generate_realistic_hostname(
            zone, node_type, _type_counters[type_key]
        )

        # Services
        service_options = _TYPE_SERVICES.get(node_type, [["ssh"]])
        services = list(rng.choice(service_options))

        # OS type
        os_options = _TYPE_OS.get(node_type, ["linux"])
        os_type = rng.choice(os_options)

        # Files
        files = _generate_node_files(node_id, node_type, rng)

        # Risk score
        risk_score = _base_risk_score(node_type, zone, rng)

        # Auth and privilege
        requires_auth = zone.value >= NetworkZone.RESTRICTED.value
        privilege_level = zone.value

        graph.add_node(
            node_id,
            node_id=node_id,
            node_type=node_type,
            zone=zone,
            hostname=hostname,
            os_type=os_type,
            services=services,
            files=files,
            risk_score=risk_score,
            requires_auth=requires_auth,
            privilege_level=privilege_level,
        )

    # ── Step 4: Generate edges ───────────────────────────────────
    _generate_edges(graph, n_nodes, zone_sizes, rng)

    # ── Step 5: Ensure connectivity ──────────────────────────────
    _ensure_connectivity(graph, n_nodes, rng)

    logger.debug(
        f"Generated graph: {n_nodes} nodes, {graph.number_of_edges()} edges, "
        f"zones={[zone_sizes[z] for z in NetworkZone]}, seed={seed}"
    )

    return graph


def get_entry_points(graph: nx.DiGraph) -> list[int]:
    """Return node IDs of all entry point nodes."""
    return [
        n
        for n, data in graph.nodes(data=True)
        if data.get("node_type") == NodeType.ENTRY_POINT
    ]


def get_high_value_target(graph: nx.DiGraph) -> int:
    """
    Return the node ID of the high-value target.

    Raises:
        ValueError: If no high-value target exists in the graph.
    """
    for n, data in graph.nodes(data=True):
        if data.get("node_type") == NodeType.HIGH_VALUE_TARGET:
            return n
    raise ValueError("No HIGH_VALUE_TARGET node found in graph")


def get_honeypot_nodes(graph: nx.DiGraph) -> list[int]:
    """Return node IDs of all honeypot nodes."""
    return [
        n
        for n, data in graph.nodes(data=True)
        if data.get("node_type") == NodeType.HONEYPOT
    ]


def get_nodes_by_zone(graph: nx.DiGraph, zone: NetworkZone) -> list[int]:
    """Return node IDs belonging to the specified zone."""
    return [
        n
        for n, data in graph.nodes(data=True)
        if data.get("zone") == zone
    ]


def get_zone_boundary_nodes(
    graph: nx.DiGraph,
    from_zone: NetworkZone,
    to_zone: NetworkZone,
) -> list[int]:
    """
    Return node IDs that serve as gateways between two zones.

    A boundary node is in `from_zone` and has at least one edge to a node
    in `to_zone`.

    Args:
        graph: The enterprise network graph.
        from_zone: The source zone.
        to_zone: The destination zone.

    Returns:
        List of node IDs in from_zone that connect to to_zone.
    """
    boundary: list[int] = []
    for n, data in graph.nodes(data=True):
        if data.get("zone") != from_zone:
            continue
        for successor in graph.successors(n):
            if graph.nodes[successor].get("zone") == to_zone:
                boundary.append(n)
                break
    return boundary


def get_lateral_movement_paths(
    graph: nx.DiGraph,
    from_node: int,
) -> list[int]:
    """
    Return nodes reachable from `from_node` within the same zone.

    Args:
        graph: The enterprise network graph.
        from_node: The starting node.

    Returns:
        List of node IDs reachable via single-hop within the same zone.
    """
    node_zone = graph.nodes[from_node].get("zone")
    return [
        n
        for n in graph.successors(from_node)
        if graph.nodes[n].get("zone") == node_zone
    ]


def generate_realistic_hostname(
    zone: NetworkZone,
    node_type: NodeType,
    index: int,
) -> str:
    """
    Generate a realistic enterprise hostname.

    Format: {zone_prefix}-{type_fragment}-{index:02d}
    Examples: dmz-gw-01, corp-fs-03, crit-core-01

    Args:
        zone: The network zone.
        node_type: The type of node.
        index: Sequential index within this type+zone combination.

    Returns:
        A realistic hostname string.
    """
    prefix = _ZONE_PREFIXES.get(zone, "srv")
    fragment = _TYPE_FRAGMENTS.get(node_type, "srv")
    return f"{prefix}-{fragment}-{index:02d}"


# ── Internal helpers ─────────────────────────────────────────────


def _compute_zone_sizes(
    n_nodes: int,
    rng: random.Random,
) -> dict[NetworkZone, int]:
    """
    Compute the number of nodes per zone, ensuring all zones are populated.

    Applies the target fractions with small random jitter, then adjusts
    to sum to exactly n_nodes.
    """
    raw_sizes: dict[NetworkZone, int] = {}
    total_assigned = 0

    for zone in NetworkZone:
        frac = _ZONE_FRACTIONS[zone]
        raw = max(2, round(n_nodes * frac + rng.uniform(-1, 1)))
        raw_sizes[zone] = raw
        total_assigned += raw

    # Adjust to match n_nodes exactly
    diff = n_nodes - total_assigned
    zones_list = list(NetworkZone)
    while diff != 0:
        zone = rng.choice(zones_list)
        if diff > 0:
            raw_sizes[zone] += 1
            diff -= 1
        elif diff < 0 and raw_sizes[zone] > 2:
            raw_sizes[zone] -= 1
            diff += 1

    return raw_sizes


def _assign_zone_node_types(
    zone_sizes: dict[NetworkZone, int],
    honeypot_density: float,
    rng: random.Random,
) -> list[tuple[NetworkZone, NodeType]]:
    """
    Assign (zone, node_type) pairs for every node index.

    Returns a list of (zone, node_type) tuples ordered by node_id.
    Zones are assigned contiguously: DMZ nodes first, then Corporate, etc.

    Node type distribution rules:
    - DMZ: 2 entry points, 1 auth gateway, rest standard/honeypots
    - Corporate: 2 auth gateways, file servers, standard, honeypots
    - Restricted: databases, file servers, standard, honeypots
    - Critical: 1 HVT, databases, standard, decoys
    """
    assignments: list[tuple[NetworkZone, NodeType]] = []

    for zone in NetworkZone:
        n = zone_sizes[zone]
        zone_types: list[NodeType] = [NodeType.STANDARD_NODE] * n

        if zone == NetworkZone.DMZ:
            # 2 entry points
            for i in range(min(2, n)):
                zone_types[i] = NodeType.ENTRY_POINT
            # 1 auth gateway at the border
            if n > 2:
                zone_types[2] = NodeType.AUTH_GATEWAY
            # Some honeypots
            n_hp = max(1, int(n * honeypot_density))
            for i in range(3, min(3 + n_hp, n)):
                zone_types[i] = NodeType.HONEYPOT

        elif zone == NetworkZone.CORPORATE:
            # 2 auth gateways (border to Restricted)
            for i in range(min(2, n)):
                zone_types[i] = NodeType.AUTH_GATEWAY
            # File servers
            n_fs = max(2, int(n * 0.25))
            for i in range(2, min(2 + n_fs, n)):
                zone_types[i] = NodeType.FILE_SERVER
            # Honeypots
            n_hp = max(1, int(n * honeypot_density))
            cursor = 2 + n_fs
            for i in range(cursor, min(cursor + n_hp, n)):
                zone_types[i] = NodeType.HONEYPOT

        elif zone == NetworkZone.RESTRICTED:
            # 1 auth gateway (border to Critical)
            zone_types[0] = NodeType.AUTH_GATEWAY
            # Databases
            n_db = max(2, int(n * 0.25))
            for i in range(1, min(1 + n_db, n)):
                zone_types[i] = NodeType.DATABASE
            # File servers
            n_fs = max(1, int(n * 0.15))
            cursor = 1 + n_db
            for i in range(cursor, min(cursor + n_fs, n)):
                zone_types[i] = NodeType.FILE_SERVER
            # Honeypots
            n_hp = max(1, int(n * honeypot_density))
            cursor = 1 + n_db + n_fs
            for i in range(cursor, min(cursor + n_hp, n)):
                zone_types[i] = NodeType.HONEYPOT

        elif zone == NetworkZone.CRITICAL:
            # 1 HVT
            zone_types[0] = NodeType.HIGH_VALUE_TARGET
            # Databases
            n_db = max(2, int(n * 0.30))
            for i in range(1, min(1 + n_db, n)):
                zone_types[i] = NodeType.DATABASE
            # Decoy assets (look like HVT)
            n_dec = max(1, int(n * 0.15))
            cursor = 1 + n_db
            for i in range(cursor, min(cursor + n_dec, n)):
                zone_types[i] = NodeType.DECOY_ASSET
            # Honeypots
            n_hp = max(1, int(n * honeypot_density))
            cursor = 1 + n_db + n_dec
            for i in range(cursor, min(cursor + n_hp, n)):
                zone_types[i] = NodeType.HONEYPOT

        # Shuffle within zone to prevent predictable layout
        rng.shuffle(zone_types)
        for nt in zone_types:
            assignments.append((zone, nt))

    return assignments


def _generate_node_files(
    node_id: int,
    node_type: NodeType,
    rng: random.Random,
) -> list[str]:
    """Generate realistic file names for a node based on its type."""
    templates = _FILE_TEMPLATES.get(node_type, [])
    if not templates:
        return []

    if node_type == NodeType.HIGH_VALUE_TARGET:
        # HVT always has 5 files (3 are targets, 2 are noise)
        return [t.format(id=f"{node_id:03d}") for t in templates[:5]]

    if node_type in (NodeType.FILE_SERVER, NodeType.DATABASE):
        n_files = rng.randint(2, 4)
    elif node_type in (NodeType.HONEYPOT, NodeType.DECOY_ASSET):
        n_files = rng.randint(2, 3)
    else:
        n_files = rng.randint(0, 2)

    selected = rng.sample(templates, min(n_files, len(templates)))
    return [t.format(id=f"{node_id:03d}") for t in selected]


def _base_risk_score(
    nt: NodeType,
    zone: NetworkZone,
    rng: random.Random,
) -> float:
    """Compute base risk score factoring in both node type and zone depth."""
    type_base = {
        NodeType.ENTRY_POINT: 0.10,
        NodeType.AUTH_GATEWAY: 0.50,
        NodeType.FILE_SERVER: 0.30,
        NodeType.DATABASE: 0.60,
        NodeType.HONEYPOT: 0.20,       # Designed to look safe
        NodeType.DECOY_ASSET: 0.45,
        NodeType.HIGH_VALUE_TARGET: 0.90,
        NodeType.STANDARD_NODE: 0.15,
    }
    base = type_base.get(nt, 0.20)
    # Deeper zones are inherently riskier
    zone_bonus = zone.value * 0.05
    jitter = rng.uniform(-0.03, 0.03)
    return max(0.0, min(1.0, base + zone_bonus + jitter))


def _generate_edges(
    graph: nx.DiGraph,
    n_nodes: int,
    zone_sizes: dict[NetworkZone, int],
    rng: random.Random,
) -> None:
    """
    Generate directed edges respecting zone topology.

    Edge generation strategy:
    1. Intra-zone: dense lateral connectivity (2-4 edges per node)
    2. Cross-zone forward: controlled chokepoints through auth gateways
    3. Cross-zone backward: rare (represents realistic firewall rules)
    """
    # Build zone index
    zone_nodes: dict[NetworkZone, list[int]] = {z: [] for z in NetworkZone}
    for node_id in range(n_nodes):
        zone = graph.nodes[node_id].get("zone")
        if zone is not None:
            zone_nodes[zone].append(node_id)

    # ── Intra-zone lateral edges ─────────────────────────────────
    for zone in NetworkZone:
        nodes = zone_nodes[zone]
        if len(nodes) < 2:
            continue
        for node_id in nodes:
            others = [n for n in nodes if n != node_id]
            n_edges = rng.randint(1, min(4, len(others)))
            targets = rng.sample(others, n_edges)
            for t in targets:
                _add_edge(graph, node_id, t, zone, zone, rng)

    # ── Cross-zone forward edges ─────────────────────────────────
    zone_list = list(NetworkZone)
    for i in range(len(zone_list) - 1):
        from_zone = zone_list[i]
        to_zone = zone_list[i + 1]
        from_nodes = zone_nodes[from_zone]
        to_nodes = zone_nodes[to_zone]

        if not from_nodes or not to_nodes:
            continue

        # Find gateway nodes in the destination zone (or from zone)
        # Auth gateways and entry points are the chokepoints
        from_gateways = [
            n for n in from_nodes
            if graph.nodes[n].get("node_type") in (
                NodeType.AUTH_GATEWAY, NodeType.ENTRY_POINT
            )
        ]
        to_gateways = [
            n for n in to_nodes
            if graph.nodes[n].get("node_type") == NodeType.AUTH_GATEWAY
        ]

        # If no gateways, use random nodes
        if not from_gateways:
            from_gateways = rng.sample(from_nodes, min(2, len(from_nodes)))
        if not to_gateways:
            to_gateways = rng.sample(to_nodes, min(2, len(to_nodes)))

        # DMZ → Corporate: 2-3 chokepoint edges
        n_cross = rng.randint(2, min(3, len(from_gateways) * len(to_gateways)))
        cross_pairs = [
            (src, dst)
            for src in from_gateways
            for dst in to_gateways
        ]
        rng.shuffle(cross_pairs)
        for src, dst in cross_pairs[:n_cross]:
            _add_edge(graph, src, dst, from_zone, to_zone, rng, cross_zone=True)

        # Also connect some non-gateway from_nodes to to_gateways
        # for alternative paths (1-2 extra edges)
        non_gw_from = [n for n in from_nodes if n not in from_gateways]
        if non_gw_from and to_gateways:
            n_extra = rng.randint(0, min(2, len(non_gw_from)))
            for src in rng.sample(non_gw_from, n_extra):
                dst = rng.choice(to_gateways)
                _add_edge(graph, src, dst, from_zone, to_zone, rng, cross_zone=True)

    # ── Sparse backward edges (realistic return paths) ───────────
    for i in range(1, len(zone_list)):
        from_zone = zone_list[i]
        to_zone = zone_list[i - 1]
        from_nodes = zone_nodes[from_zone]
        to_nodes = zone_nodes[to_zone]
        if not from_nodes or not to_nodes:
            continue
        # 0-1 backward edge per zone pair
        if rng.random() < 0.4:
            src = rng.choice(from_nodes)
            dst = rng.choice(to_nodes)
            _add_edge(graph, src, dst, from_zone, to_zone, rng, cross_zone=True)


def _add_edge(
    graph: nx.DiGraph,
    src: int,
    dst: int,
    src_zone: NetworkZone,
    dst_zone: NetworkZone,
    rng: random.Random,
    cross_zone: bool = False,
) -> None:
    """Add a directed edge with traversal cost, suspicion, protocol, and firewall."""
    if graph.has_edge(src, dst) or src == dst:
        return

    # Traversal cost — higher for cross-zone edges
    base_cost = rng.uniform(0.2, 0.6) if not cross_zone else rng.uniform(0.5, 0.9)
    traversal_cost = round(base_cost, 3)

    # Suspicion delta — scaled by destination risk + zone depth
    dst_risk = graph.nodes[dst].get("risk_score", 0.2)
    zone_factor = 1.0 + dst_zone.value * 0.15
    suspicion_delta = round(
        (rng.uniform(0.01, 0.08) + dst_risk * 0.05) * zone_factor,
        4,
    )

    # Protocol
    zone_pair = (src_zone, dst_zone)
    available_protocols = _EDGE_PROTOCOLS.get(
        zone_pair, _EDGE_PROTOCOLS.get((src_zone, src_zone), ["ssh"])
    )
    protocol = rng.choice(available_protocols)

    # Credential requirement — cross-zone edges and restricted+ destinations
    requires_credential = cross_zone or dst_zone.value >= NetworkZone.RESTRICTED.value

    # Firewall rule
    if cross_zone:
        firewall_rule = "restrict"
    elif dst_zone.value >= NetworkZone.RESTRICTED.value:
        firewall_rule = "monitor"
    else:
        firewall_rule = "allow"

    graph.add_edge(
        src,
        dst,
        traversal_cost=traversal_cost,
        suspicion_delta=suspicion_delta,
        protocol=protocol,
        requires_credential=requires_credential,
        firewall_rule=firewall_rule,
    )


def _ensure_connectivity(
    graph: nx.DiGraph,
    n_nodes: int,
    rng: random.Random,
) -> None:
    """
    Ensure every node is reachable from every entry point.

    Adds directed edges to connect disconnected components, respecting
    zone ordering where possible.
    """
    # Make weakly connected first
    undirected = graph.to_undirected()
    components = list(nx.connected_components(undirected))

    if len(components) > 1:
        for i in range(len(components) - 1):
            src = rng.choice(list(components[i]))
            dst = rng.choice(list(components[i + 1]))
            src_zone = graph.nodes[src].get("zone", NetworkZone.DMZ)
            dst_zone = graph.nodes[dst].get("zone", NetworkZone.DMZ)
            _add_edge(graph, src, dst, src_zone, dst_zone, rng)
            _add_edge(graph, dst, src, dst_zone, src_zone, rng)

    # Verify entry point reachability
    entry_points = get_entry_points(graph)
    all_nodes = set(range(n_nodes))

    for ep in entry_points:
        reachable = set(nx.descendants(graph, ep)) | {ep}
        unreachable = all_nodes - reachable

        for node in unreachable:
            # Find a reachable node to bridge through
            if len(reachable) > 1:
                bridge = rng.choice(list(reachable - {ep}))
            else:
                bridge = ep
            bridge_zone = graph.nodes[bridge].get("zone", NetworkZone.DMZ)
            node_zone = graph.nodes[node].get("zone", NetworkZone.DMZ)
            _add_edge(graph, bridge, node, bridge_zone, node_zone, rng)
            reachable.add(node)


def print_graph_summary(graph: nx.DiGraph) -> None:
    """Print an ASCII summary of the graph topology and node type distribution."""
    node_types: dict[str, int] = {}
    zone_counts: dict[int, int] = {}
    
    for n, data in graph.nodes(data=True):
        nt = data.get("node_type")
        z = data.get("zone")
        if nt:
            val = nt.value if hasattr(nt, 'value') else str(nt)
            node_types[val] = node_types.get(val, 0) + 1
        if z is not None:
            val = z.value if hasattr(z, 'value') else int(z)
            zone_counts[val] = zone_counts.get(val, 0) + 1

    print("┌─────────────────────────────────────────┐")
    print("│         CIPHER Network Graph            │")
    print(f"│         Nodes: {graph.number_of_nodes():<25}│")
    print("├─────────────────────┬───────────────────┤")
    print("│ Node Type           │ Count             │")
    print("├─────────────────────┼───────────────────┤")
    for nt, count in sorted(node_types.items()):
        print(f"│ {nt:<19} │ {count:<17} │")
    print("├─────────────────────┼───────────────────┤")
    for z, count in sorted(zone_counts.items()):
        print(f"│ Zone {z:<15} │ {str(count)+' nodes':<17} │")
    print("└─────────────────────┴───────────────────┘")
