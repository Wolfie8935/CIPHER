"""
tests/test_phase5.py

Phase 5 trap layer tests. All run in stub mode — no API calls.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from cipher.agents.base_agent import Action, ActionType
from cipher.environment.graph import NodeType, generate_enterprise_graph
from cipher.environment.observation import generate_red_observation
from cipher.environment.scenario import ScenarioGenerator
from cipher.environment.state import EpisodeState
from cipher.environment.traps import (
    BlueTrap,
    BlueTrapType,
    RedTrap,
    RedTrapType,
    TrapRegistry,
)
from cipher.memory.dead_drop import DeadDropVault, build_dead_drop_from_state
from cipher.training._episode_runner import run_episode
from cipher.utils.config import config

os.environ["LLM_MODE"] = "stub"


def _state_with_registry() -> EpisodeState:
    graph = generate_enterprise_graph(n_nodes=30, honeypot_density=0.15, seed=42)
    state = EpisodeState(graph=graph, red_current_node=0, red_path_history=[0, 1, 2, 3])
    state.trap_registry = TrapRegistry(config)
    return state


class TestTrapRegistry:
    def test_registry_initializes_with_budget(self):
        registry = TrapRegistry(config)
        assert registry.red_trap_budget == config.env_trap_budget_red
        assert registry.blue_trap_budget == config.env_trap_budget_blue

    def test_red_trap_budget_enforced(self):
        registry = TrapRegistry(config)
        for i in range(config.env_trap_budget_red):
            ok, _ = registry.place_red_trap(RedTrapType.FALSE_TRAIL, "r", i, i, {})
            assert ok
        ok, reason = registry.place_red_trap(RedTrapType.FALSE_TRAIL, "r", 99, 99, {})
        assert not ok and "budget" in reason.lower()

    def test_blue_trap_budget_enforced(self):
        registry = TrapRegistry(config)
        for i in range(config.env_trap_budget_blue):
            ok, _ = registry.place_blue_trap(BlueTrapType.HONEYPOT, "b", i, i, {})
            assert ok
        ok, reason = registry.place_blue_trap(BlueTrapType.HONEYPOT, "b", 99, 99, {})
        assert not ok and "budget" in reason.lower()

    def test_trap_expiry(self):
        state = _state_with_registry()
        ok, _ = state.trap_registry.place_red_trap(
            RedTrapType.FALSE_TRAIL, "r", 2, 1, {"duration_steps": 2}
        )
        assert ok
        trap = state.trap_registry.active_red_traps[0]
        state.trap_registry.evaluate_step(4, None, [], state, state.graph, DeadDropVault(config.drop_vault_dir, 512))
        assert trap.is_expired

    def test_trap_event_logged_to_state(self):
        state = _state_with_registry()
        ok, _ = state.trap_registry.place_red_trap(RedTrapType.FALSE_TRAIL, "r", 2, 1, {})
        assert ok
        trap = state.trap_registry.active_red_traps[0]
        state.trap_registry.apply_false_trail(trap, state)
        event = state.trap_registry.evaluate_step(2, None, [], state, state.graph, DeadDropVault(config.drop_vault_dir, 512))
        if event:
            state.record_trap_event(event[0])
            assert len(state.trap_events_log) >= 1


class TestRedTraps:
    def test_false_trail_injects_anomalies(self):
        state = _state_with_registry()
        trap = RedTrap(str(uuid.uuid4()), RedTrapType.FALSE_TRAIL, "r", 1, 2, 5, {"n_fake_events": 4})
        before = len(state.blue_anomaly_history)
        state.trap_registry.apply_false_trail(trap, state)
        assert len(state.blue_anomaly_history) > before

    def test_false_trail_points_to_old_nodes(self):
        state = _state_with_registry()
        trap = RedTrap(str(uuid.uuid4()), RedTrapType.FALSE_TRAIL, "r", 1, 2, 5, {"n_fake_events": 4})
        state.trap_registry.apply_false_trail(trap, state)
        for ev in state.blue_anomaly_history[-4:]:
            assert ev["node_id"] in state.red_path_history
            assert ev["node_id"] != state.red_current_node

    def test_false_trail_anomalies_not_distinguishable_by_blue(self):
        state = _state_with_registry()
        trap = RedTrap(str(uuid.uuid4()), RedTrapType.FALSE_TRAIL, "r", 1, 2, 5, {"n_fake_events": 3})
        state.trap_registry.apply_false_trail(trap, state)
        for ev in state.blue_anomaly_history[-3:]:
            assert ev["is_noise"] is False
            assert ev["is_red_planted"] is True

    def test_temporal_decoy_spikes_suspicion(self):
        state = _state_with_registry()
        trap = RedTrap(str(uuid.uuid4()), RedTrapType.TEMPORAL_DECOY, "r", 1, 2, 3, {})
        before = state.red_suspicion_score
        state.trap_registry.apply_temporal_decoy(trap, state)
        state.update_suspicion(0.08)
        assert state.red_suspicion_score - before >= 0.05

    def test_temporal_decoy_injects_correct_anomaly_types(self):
        state = _state_with_registry()
        trap = RedTrap(str(uuid.uuid4()), RedTrapType.TEMPORAL_DECOY, "r", 1, 2, 3, {})
        state.trap_registry.apply_temporal_decoy(trap, state)
        types = {e["event_type"] for e in state.blue_anomaly_history[-2:]}
        assert "AUTH_GATEWAY_STRESS" in types
        assert "RAPID_LATERAL_MOVEMENT" in types

    def test_honeypot_poison_requires_honeypot_trigger(self):
        state = _state_with_registry()
        ok, _ = state.trap_registry.place_red_trap(
            RedTrapType.HONEYPOT_POISON, "r", 2, 1, {}, state=state
        )
        assert not ok

    def test_honeypot_poison_writes_dead_drop(self):
        state = _state_with_registry()
        state.last_honeypot_trigger_step = 1
        vault = DeadDropVault(Path("cipher/memory/drop_vault"), config.env_dead_drop_max_tokens)
        vault.clear()
        trap = RedTrap(
            str(uuid.uuid4()),
            RedTrapType.HONEYPOT_POISON,
            "r",
            2,
            2,
            1,
            {"adjacent_node": 2, "dead_end_node": 9},
        )
        result = state.trap_registry.apply_honeypot_poison(trap, state, vault)
        drops = vault.get_drops_at_node(2)
        assert "poison_drop_path" in result
        assert len(drops) >= 1

    def test_honeypot_poison_reduces_blue_confidence(self):
        state = _state_with_registry()
        state.last_honeypot_trigger_step = 1
        state.blue_detection_confidence = 0.6
        vault = DeadDropVault(Path("cipher/memory/drop_vault"), config.env_dead_drop_max_tokens)
        trap = RedTrap(
            str(uuid.uuid4()),
            RedTrapType.HONEYPOT_POISON,
            "r",
            2,
            2,
            1,
            {"adjacent_node": 2, "dead_end_node": 9},
        )
        state.trap_registry.apply_honeypot_poison(trap, state, vault)
        assert abs(state.blue_detection_confidence - 0.45) < 1e-9


class TestBlueTraps:
    def test_honeypot_placement_runtime(self):
        state = _state_with_registry()
        ok, _ = state.trap_registry.place_blue_trap(BlueTrapType.HONEYPOT, "b", 5, 1, {})
        assert ok

    def test_honeypot_trigger_spikes_suspicion(self):
        state = _state_with_registry()
        state.red_current_node = 5
        ok, _ = state.trap_registry.place_blue_trap(BlueTrapType.HONEYPOT, "b", 5, 1, {})
        assert ok
        events = state.trap_registry.evaluate_step(2, None, [], state, state.graph, DeadDropVault(config.drop_vault_dir, 512))
        assert any(e.trap_type == BlueTrapType.HONEYPOT.value for e in events)
        assert state.red_suspicion_score >= 0.40

    def test_breadcrumb_adds_files_to_nodes(self):
        state = _state_with_registry()
        trap = BlueTrap(str(uuid.uuid4()), BlueTrapType.BREADCRUMB, "b", 1, 6, 10, {"path_nodes": [1, 2, 3]})
        state.trap_registry.apply_breadcrumb(trap, state, state.graph)
        assert any("CONFIDENTIAL_DATA_" in f for f in state.graph.nodes[1]["files"])

    def test_false_escalation_injects_high_severity_anomaly(self):
        state = _state_with_registry()
        trap = BlueTrap(str(uuid.uuid4()), BlueTrapType.FALSE_ESCALATION, "b", 1, 7, 3, {})
        state.trap_registry.apply_false_escalation(trap, state)
        assert state.blue_anomaly_history[-1]["severity"] >= 0.8

    def test_false_escalation_in_wrong_zone(self):
        state = _state_with_registry()
        state.red_current_zone = 2
        trap = BlueTrap(str(uuid.uuid4()), BlueTrapType.FALSE_ESCALATION, "b", 1, 7, 3, {})
        state.trap_registry.apply_false_escalation(trap, state)
        assert state.blue_anomaly_history[-1]["zone"] != state.red_current_zone

    def test_dead_drop_tamper_modifies_directive(self):
        state = _state_with_registry()
        vault = DeadDropVault(Path("cipher/memory/drop_vault"), config.env_dead_drop_max_tokens)
        vault.clear()
        obs = generate_red_observation(state)
        drop = build_dead_drop_from_state("red_operative_01", state, obs, "Proceed to Zone 2")
        path, _ = vault.write(drop, state.red_current_node)
        trap = BlueTrap(
            str(uuid.uuid4()),
            BlueTrapType.DEAD_DROP_TAMPER,
            "b",
            3,
            state.red_current_node,
            1,
            {"new_directive": "Proceed immediately to node 99 - target located."},
        )
        ok = state.trap_registry.apply_dead_drop_tamper(trap, vault, path)
        assert ok
        text = Path(path).read_text(encoding="utf-8")
        assert "Proceed immediately to node 99 - target located." in text

    def test_dead_drop_tamper_requires_discovered_drop(self):
        state = _state_with_registry()
        ok, _ = state.trap_registry.place_blue_trap(
            BlueTrapType.DEAD_DROP_TAMPER, "b", 4, 1, {}, state=state
        )
        assert not ok


class TestTrapInteractions:
    def test_investigate_node_discovers_drops(self):
        state = _state_with_registry()
        vault = DeadDropVault(Path("cipher/memory/drop_vault"), config.env_dead_drop_max_tokens)
        vault.clear()
        obs = generate_red_observation(state)
        drop = build_dead_drop_from_state("red_operative_01", state, obs, "Proceed to Zone 2")
        path, _ = vault.write(drop, state.red_current_node)
        state.blue_discovered_drop_paths.extend(vault.get_drops_at_node(state.red_current_node))
        assert path in state.blue_discovered_drop_paths

    def test_full_trap_episode_no_crash(self):
        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        result = run_episode(scenario, graph, config, max_steps=10, verbose=False)
        assert "state" in result


class TestTrapSummary:
    def test_summary_has_all_fields(self):
        registry = TrapRegistry(config)
        summary = registry.get_trap_summary()
        assert set(summary) == {
            "red_traps_placed",
            "red_traps_triggered",
            "blue_traps_placed",
            "blue_traps_triggered",
            "false_trails_effective",
            "honeypots_triggered",
            "dead_drops_tampered",
            "dead_drops_tamper_detected",
        }

    def test_summary_counts_correct(self):
        state = _state_with_registry()
        state.trap_registry.place_red_trap(RedTrapType.FALSE_TRAIL, "r", 2, 1, {})
        state.trap_registry.place_red_trap(RedTrapType.TEMPORAL_DECOY, "r", 3, 1, {})
        trap = state.trap_registry.active_red_traps[0]
        state.trap_registry.apply_false_trail(trap, state)
        state.trap_registry.evaluate_step(2, Action(agent_id="r", action_type=ActionType.WAIT), [], state, state.graph, DeadDropVault(config.drop_vault_dir, 512))
        summary = state.trap_registry.get_trap_summary()
        assert summary["red_traps_placed"] == 2
