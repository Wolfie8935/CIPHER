"""
Subagent role profiles for CIPHER's v2 commander+subagent architecture.

A SubagentRoleProfile is the declarative spec for a subagent role:
    - role_name (e.g. 'planner', 'scout', 'anomaly_triager')
    - team ('red' | 'blue')
    - default_lifespan (steps before auto-dismiss)
    - allowed_actions (whitelist subset of ActionType values; empty = team default)
    - prompt_filename (relative to cipher/agents/prompts/)
    - lora_env_key (optional .env variable name for hybrid mode)
    - lora_default_path (optional fallback adapter path)
    - heuristic_fn_name (the name of the function on the legacy class
      whose `_stub_act` we call to preserve heuristic behaviour.
      None → falls back to the generic SubagentHeuristics router)

This module is intentionally pure data + tiny lookup helpers.  Adding a new
subagent role = adding a profile here (+ optional prompt + heuristic mapping).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SubagentRoleProfile:
    """Declarative spec for one subagent role."""

    role_name: str
    team: str  # 'red' | 'blue'
    description: str
    default_lifespan: int = 5
    allowed_actions: tuple[str, ...] = ()
    prompt_filename: Optional[str] = None
    lora_env_key: Optional[str] = None
    lora_default_path: Optional[str] = None
    legacy_class_path: Optional[str] = None  # e.g. "cipher.agents.red.planner:RedPlanner"


# ── Built-in role profiles ────────────────────────────────────────
# The first 8 are the legacy 4+4 specialists, retained as roles the commander
# can spawn on demand. The rest are new emergent roles.

_PROFILES: dict[str, SubagentRoleProfile] = {}


def _register(profile: SubagentRoleProfile) -> None:
    _PROFILES[profile.role_name] = profile


# ── RED roles (legacy specialists) ─────────────────────────────────
_register(SubagentRoleProfile(
    role_name="planner",
    team="red",
    description="Long-horizon strategist; sets path, decides abort.",
    default_lifespan=8,
    prompt_filename="red_planner.txt",
    lora_env_key="RED_PLANNER_LORA_PATH",
    lora_default_path="red trained/cipher-red-planner-v1",
    legacy_class_path="cipher.agents.red.planner:RedPlanner",
))
_register(SubagentRoleProfile(
    role_name="analyst",
    team="red",
    description="Intel and risk estimation; reads dead-drops; coordinates with planner.",
    default_lifespan=6,
    prompt_filename="red_analyst.txt",
    lora_env_key="RED_ANALYST_LORA_PATH",
    lora_default_path="red trained/cipher-red-analyst-v1",
    legacy_class_path="cipher.agents.red.analyst:RedAnalyst",
))
_register(SubagentRoleProfile(
    role_name="operative",
    team="red",
    description="Stealth executor and counter-trap planter.",
    default_lifespan=6,
    prompt_filename="red_operative.txt",
    legacy_class_path="cipher.agents.red.operative:RedOperative",
))
_register(SubagentRoleProfile(
    role_name="exfiltrator",
    team="red",
    description="HVT extraction specialist; only useful in zone 2/3.",
    default_lifespan=6,
    prompt_filename="red_exfiltrator.txt",
    legacy_class_path="cipher.agents.red.exfiltrator:RedExfiltrator",
))

# ── BLUE roles (legacy specialists) ────────────────────────────────
_register(SubagentRoleProfile(
    role_name="surveillance",
    team="blue",
    description="Real-time anomaly monitor and correlator.",
    default_lifespan=8,
    prompt_filename="blue_surveillance.txt",
    lora_env_key="BLUE_SURVEILLANCE_LORA_PATH",
    lora_default_path="blue trained/cipher-blue-surveillance-v1",
    legacy_class_path="cipher.agents.blue.surveillance:BlueSurveillance",
))
_register(SubagentRoleProfile(
    role_name="threat_hunter",
    team="blue",
    description="Hypothesis-driven investigator; triggers alerts on strong evidence.",
    default_lifespan=6,
    prompt_filename="blue_threat_hunter.txt",
    lora_env_key="BLUE_THREAT_HUNTER_LORA_PATH",
    lora_default_path="blue trained/cipher-blue-threat-hunter-v1",
    legacy_class_path="cipher.agents.blue.threat_hunter:BlueThreatHunter",
))
_register(SubagentRoleProfile(
    role_name="deception_architect",
    team="blue",
    description="Honeypot and breadcrumb designer; runs predictively ahead of RED.",
    default_lifespan=6,
    prompt_filename="blue_deception_architect.txt",
    legacy_class_path="cipher.agents.blue.deception_architect:BlueDeceptionArchitect",
))
_register(SubagentRoleProfile(
    role_name="forensics",
    team="blue",
    description="Reconstructs RED's operation graph from behavioural fragments.",
    default_lifespan=8,
    prompt_filename="blue_forensics.txt",
    legacy_class_path="cipher.agents.blue.forensics:BlueForensics",
))

# ── New emergent roles (no legacy class — Subagent uses heuristic router) ─
_register(SubagentRoleProfile(
    role_name="scout",
    team="red",
    description=(
        "Lightweight, short-lived recon worker. Walks unexplored adjacent nodes "
        "without engaging or planting traps. Useful for mapping early zones."
    ),
    default_lifespan=3,
    allowed_actions=("move", "wait"),
    prompt_filename="red_scout.txt",
))
_register(SubagentRoleProfile(
    role_name="dead_drop_courier",
    team="red",
    description=(
        "Memory custodian. Writes and reads dead drops on the commander's behalf "
        "to preserve mission continuity across context resets."
    ),
    default_lifespan=4,
    allowed_actions=("write_dead_drop", "read_dead_drop", "wait"),
    prompt_filename="red_dead_drop_courier.txt",
))
_register(SubagentRoleProfile(
    role_name="anomaly_triager",
    team="blue",
    description=(
        "Drains the anomaly feed, deciding which signals are worth investigating "
        "and which are noise. Cheaper than the full surveillance role."
    ),
    default_lifespan=4,
    allowed_actions=("analyze_anomaly", "investigate_node", "stand_down"),
    prompt_filename="blue_anomaly_triager.txt",
))
_register(SubagentRoleProfile(
    role_name="alert_judge",
    team="blue",
    description=(
        "Specialist that decides WHEN to trigger an alert. Conservative: only "
        "alerts on multi-evidence convergence."
    ),
    default_lifespan=3,
    allowed_actions=("trigger_alert", "stand_down"),
    prompt_filename="blue_alert_judge.txt",
))


def get_profile(role_name: str) -> SubagentRoleProfile | None:
    """Return the profile for `role_name`, or None if not registered."""
    return _PROFILES.get(role_name)


def list_profiles(team: str | None = None) -> list[SubagentRoleProfile]:
    """Return all registered profiles, optionally filtered by team."""
    if team is None:
        return list(_PROFILES.values())
    return [p for p in _PROFILES.values() if p.team == team]


def list_role_names(team: str | None = None) -> list[str]:
    """Return role names available for spawning."""
    return [p.role_name for p in list_profiles(team)]
