"""
CIPHER RED team agents.

Contains the four RED team specialists:
- Planner: mission design and abort logic
- Analyst: environment mapping and risk estimation
- Operative: stealth execution and trap planting
- Exfiltrator: data packaging and exit sequencing
"""
from __future__ import annotations

from cipher.agents.red.coordination import (  # noqa: F401
    clear_danger_map,
    get_danger_nodes,
    get_danger_score,
    mark_danger,
)
