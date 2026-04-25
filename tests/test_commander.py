"""Tests for RedCommander / BlueCommander (v2 architecture)."""
from __future__ import annotations

import pytest

from cipher.agents.base_agent import Action, ActionType
from cipher.agents.commander import BlueCommander, RedCommander
from cipher.environment.graph import NodeType
from cipher.environment.observation import (
    AnomalyEvent,
    BlueObservation,
    RedObservation,
)
from cipher.utils.config import config as global_config


def _empty_red_obs() -> RedObservation:
    return RedObservation(
        current_node=0,
        current_node_type=NodeType.STANDARD_NODE,
        adjacent_nodes=[1, 2],
        adjacent_node_types=[NodeType.STANDARD_NODE, NodeType.STANDARD_NODE],
        files_at_current_node=[],
        estimated_suspicion=0.0,
        blue_detection_confidence=0.0,
        dead_drops_available=[],
        step=1,
        context_reset_this_step=False,
    )


def _empty_blue_obs() -> BlueObservation:
    return BlueObservation(
        anomaly_feed=[],
        current_detection_confidence=0.0,
        step=1,
    )


def test_red_commander_spawns_default_roster_in_stub_mode():
    cmd = RedCommander("red_commander_01", global_config)
    cmd.reset_episode()
    cmd.observe(_empty_red_obs())
    # Each step the stub policy spawns one of the default roster roles.
    spawned_roles = []
    for step in range(1, 6):
        actions = cmd.act_step(step=step)
        for a in actions:
            if a.action_type == ActionType.SPAWN_SUBAGENT and a.subagent_spec:
                spawned_roles.append(a.subagent_spec.role_name)
    # First 4 calls each emit a SPAWN_SUBAGENT for a default roster role.
    assert {"planner", "analyst", "operative", "exfiltrator"}.issubset(
        {s.role for s in cmd.registry.alive()}
    )


def test_red_commander_emits_subagent_actions_after_roster_built():
    cmd = RedCommander("red_commander_01", global_config)
    cmd.reset_episode()
    cmd.observe(_empty_red_obs())
    # Run enough steps to build the roster.
    for step in range(1, 6):
        cmd.act_step(step=step)
    # Now the registry has subagents — next step should produce primitive actions
    # from at least the commander + every alive subagent (some may be in legacy
    # proxy form so they could emit valid primitives).
    cmd.observe(_empty_red_obs())
    actions = cmd.act_step(step=6)
    assert len(actions) >= 1
    agent_ids = {a.agent_id for a in actions}
    # Commander present
    assert "red_commander_01" in agent_ids
    # All alive subagents should also have produced an action
    for sub in cmd.registry.alive():
        assert sub.agent_id in agent_ids


def test_blue_commander_acts_aggressively_on_high_confidence():
    """At high detection confidence, BLUE commander should trigger an alert
    or investigate directly (not just spawn alert_judge and stand_down)."""
    cmd = BlueCommander("blue_commander_01", global_config)
    cmd.reset_episode()
    # First, build roster.
    for step in range(1, 6):
        cmd.observe(_empty_blue_obs())
        cmd.act_step(step=step)
    # Then simulate high confidence with anomaly feed.
    obs = BlueObservation(
        anomaly_feed=[AnomalyEvent(
            event_type="auth_anomaly",
            node_id=12,
            severity=0.8,
            is_red_planted=True,
            step=10,
        )],
        current_detection_confidence=0.7,
        step=10,
    )
    cmd.observe(obs)
    actions = cmd.act_step(step=10)
    # Commander must take an active action — trigger_alert or investigate_node.
    commander_actions = [a for a in actions if a.agent_id == cmd.agent_id]
    assert commander_actions, "Commander should have produced an action"
    active_types = {ActionType.TRIGGER_ALERT, ActionType.INVESTIGATE_NODE, ActionType.ANALYZE_ANOMALY}
    assert any(a.action_type in active_types for a in commander_actions), (
        f"Expected active detection action, got: {[a.action_type for a in commander_actions]}"
    )


def test_red_commander_meta_actions_consumed_not_returned():
    cmd = RedCommander("red_commander_01", global_config)
    cmd.reset_episode()
    cmd.observe(_empty_red_obs())
    actions = cmd.act_step(step=1)
    # Stub policy on step 1 emits SPAWN_SUBAGENT — it must NOT appear in
    # the returned action list (it's been consumed by the registry).
    assert all(
        a.action_type
        not in (ActionType.SPAWN_SUBAGENT, ActionType.DELEGATE_TASK, ActionType.DISMISS_SUBAGENT)
        for a in actions
    )


def test_commander_full_episode_reset_clears_roster():
    cmd = RedCommander("red_commander_01", global_config)
    cmd.reset_episode()
    cmd.observe(_empty_red_obs())
    for step in range(1, 6):
        cmd.act_step(step=step)
    assert len(cmd.registry) >= 1
    cmd.reset_episode()
    assert len(cmd.registry) == 0
