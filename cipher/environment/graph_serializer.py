"""Serialize a NetworkX enterprise graph to JSON for the React war room dashboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def serialize_graph_for_dashboard(
    graph: Any,
    output_path: str = "logs/network_graph.json",
) -> dict:
    """Convert a NetworkX graph to {nodes, edges} JSON and write it to disk.

    Args:
        graph:       NetworkX graph produced by generate_enterprise_graph().
        output_path: Destination file; parent directory is created if missing.

    Returns:
        The serialised dict.
    """
    nodes: list[dict] = []
    for node_id, data in graph.nodes(data=True):
        zone = data.get("zone") or data.get("zone_id", 0)
        if hasattr(zone, "value"):
            zone_val = zone.value
        else:
            try:
                zone_val = int(zone)
            except (TypeError, ValueError):
                zone_val = 0

        node_type = data.get("node_type") or data.get("type", "server")
        if hasattr(node_type, "value"):
            type_val = node_type.value
        elif hasattr(node_type, "name"):
            type_val = node_type.name.lower()
        else:
            type_val = str(node_type)

        nodes.append({
            "id":         int(node_id),
            "hostname":   str(data.get("hostname", f"node_{node_id}")),
            "zone":       zone_val,
            "type":       type_val,
            "files":      list(data.get("files", []) or []),
            "services":   list(data.get("services", []) or []),
            "is_honeypot":bool(data.get("is_honeypot", False)),
            "is_entry":   bool(data.get("is_entry", False)),
            "is_hvt":     bool(data.get("is_hvt", False)),
        })

    edges: list[dict] = []
    for u, v, edata in graph.edges(data=True):
        edges.append({
            "source":    int(u),
            "target":    int(v),
            "protocol":  str(edata.get("protocol", "tcp")),
            "bandwidth": int(edata.get("bandwidth", 100)),
        })

    result = {"nodes": nodes, "edges": edges}
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
