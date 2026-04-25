"""Tests for the SubagentRegistry / role profile / Subagent stack (v2 architecture)."""
from __future__ import annotations

import pytest

from cipher.agents.base_agent import Action, ActionType, SubagentSpec
from cipher.agents.role_profiles import (
    SubagentRoleProfile,
    get_profile,
    list_profiles,
    list_role_names,
)
from cipher.agents.subagent import Subagent, build_subagent
from cipher.agents.subagent_registry import SubagentRegistry
from cipher.utils.config import config as global_config


# ── role_profiles ─────────────────────────────────────────────────


def test_legacy_specialists_have_profiles():
    for role in ("planner", "analyst", "operative", "exfiltrator"):
        p = get_profile(role)
        assert p is not None and p.team == "red", f"red role {role} missing"
    for role in ("surveillance", "threat_hunter", "deception_architect", "forensics"):
        p = get_profile(role)
        assert p is not None and p.team == "blue", f"blue role {role} missing"


def test_emergent_roles_present():
    for role in ("scout", "dead_drop_courier", "anomaly_triager", "alert_judge"):
        assert get_profile(role) is not None


def test_list_profiles_team_filter():
    red = list_role_names("red")
    blue = list_role_names("blue")
    assert "planner" in red and "scout" in red
    assert "surveillance" in blue and "alert_judge" in blue
    # No leakage
    assert "scout" not in blue
    assert "alert_judge" not in red


# ── Subagent build/ act ─────────────────────────────────────────────


def test_build_subagent_unknown_role_returns_none():
    spec = SubagentSpec(role_name="not_a_real_role", team="red")
    assert build_subagent(spec, global_config, agent_id="x") is None


def test_build_subagent_team_mismatch_returns_none():
    spec = SubagentSpec(role_name="planner", team="blue")
    assert build_subagent(spec, global_config, agent_id="x") is None


def test_subagent_meta_actions_are_blocked():
    """Subagents must NEVER emit SPAWN/DELEGATE/DISMISS even if they want to."""
    spec = SubagentSpec(role_name="scout", team="red", parent_id="red_commander_01")
    sub = build_subagent(spec, global_config, agent_id="red_scout_99")
    assert sub is not None

    # Inject a meta-action through the legacy proxy bypass: simulate a stub_act
    # returning a meta-action and confirm sanitizer kicks it out.
    forced = Action(
        agent_id="red_scout_99",
        action_type=ActionType.SPAWN_SUBAGENT,
        reasoning="Subagent should not be able to spawn.",
    )
    sanitized = sub._sanitize_action(forced)
    assert sanitized.action_type == ActionType.WAIT


def test_subagent_whitelist_enforced_for_scout():
    """Scout role only allows MOVE / WAIT — anything else falls back."""
    spec = SubagentSpec(role_name="scout", team="red", parent_id="red_commander_01")
    sub = build_subagent(spec, global_config, agent_id="red_scout_98")
    assert sub is not None
    bad = Action(
        agent_id="red_scout_98",
        action_type=ActionType.EXFILTRATE,
        target_file="x.pdf",
        reasoning="Scout should not exfiltrate.",
    )
    sanitized = sub._sanitize_action(bad)
    assert sanitized.action_type == ActionType.WAIT


# ── SubagentRegistry ────────────────────────────────────────────────


def test_registry_spawn_and_unique_ids():
    reg = SubagentRegistry(
        team="red",
        commander_id="red_commander_01",
        config=global_config,
        max_concurrent=4,
        spawn_budget=5,
    )
    spec = SubagentSpec(role_name="scout", team="red", parent_id="red_commander_01")
    a = reg.spawn(spec, step=1)
    b = reg.spawn(spec, step=1)
    assert a is not None and b is not None
    assert a.agent_id == "red_scout_01"
    assert b.agent_id == "red_scout_02"
    assert len(reg) == 2


def test_registry_enforces_max_concurrent():
    reg = SubagentRegistry(
        team="red",
        commander_id="red_commander_01",
        config=global_config,
        max_concurrent=2,
        spawn_budget=10,
    )
    spec = SubagentSpec(role_name="scout", team="red", parent_id="red_commander_01")
    assert reg.spawn(spec, step=1) is not None
    assert reg.spawn(spec, step=1) is not None
    assert reg.spawn(spec, step=1) is None  # cap reached
    assert len(reg) == 2


def test_registry_enforces_spawn_budget():
    reg = SubagentRegistry(
        team="red",
        commander_id="red_commander_01",
        config=global_config,
        max_concurrent=10,
        spawn_budget=2,
    )
    spec = SubagentSpec(role_name="scout", team="red", parent_id="red_commander_01")
    assert reg.spawn(spec, step=1) is not None
    assert reg.spawn(spec, step=2) is not None
    assert reg.spawn(spec, step=3) is None  # budget exhausted
    # Attempted-but-rejected spawn shows up as a 'reject' lifecycle event
    rejects = [e for e in reg.events if e.event_type == "reject"]
    assert any("spawn_budget_exhausted" in e.reason for e in rejects)


def test_registry_lifespan_auto_dismiss():
    reg = SubagentRegistry(
        team="red",
        commander_id="red_commander_01",
        config=global_config,
        max_concurrent=4,
        spawn_budget=4,
    )
    spec = SubagentSpec(role_name="scout", team="red", lifespan_steps=2,
                       parent_id="red_commander_01")
    sub = reg.spawn(spec, step=1)
    assert sub is not None
    reg.tick_step(1)  # -> 1 remaining
    assert sub.is_alive()
    reg.tick_step(2)  # -> 0 remaining → expired
    assert sub.agent_id not in reg
    expires = [e for e in reg.events if e.event_type == "expire"]
    assert len(expires) == 1 and expires[0].subagent_id == sub.agent_id


def test_registry_dismiss_works():
    reg = SubagentRegistry(
        team="red",
        commander_id="red_commander_01",
        config=global_config,
        max_concurrent=4,
        spawn_budget=4,
    )
    spec = SubagentSpec(role_name="scout", team="red", parent_id="red_commander_01")
    sub = reg.spawn(spec, step=1)
    assert reg.dismiss(sub.agent_id, step=2) is True
    assert sub.agent_id not in reg
    assert reg.dismiss("not_a_real_id", step=3) is False


def test_registry_thrash_detection():
    reg = SubagentRegistry(
        team="red",
        commander_id="red_commander_01",
        config=global_config,
        max_concurrent=10,
        spawn_budget=10,
    )
    spec = SubagentSpec(role_name="scout", team="red", parent_id="red_commander_01")
    reg.spawn(spec, step=1)
    reg.spawn(spec, step=2)
    reg.spawn(spec, step=2)
    thrash = reg.detect_thrash(step=2, window=2)
    assert "scout" in thrash


def test_registry_reset_clears_alive():
    reg = SubagentRegistry(
        team="red",
        commander_id="red_commander_01",
        config=global_config,
        max_concurrent=4,
        spawn_budget=4,
    )
    spec = SubagentSpec(role_name="scout", team="red", parent_id="red_commander_01")
    reg.spawn(spec, step=1)
    reg.reset_episode()
    assert len(reg) == 0
    assert reg.events == []
