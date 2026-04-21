"""
tests/test_phase6.py

Phase 6 reward function tests.
All run in stub mode — no API calls.
"""
from __future__ import annotations

import csv
import os
import statistics
import tempfile
from pathlib import Path

import pytest

from cipher.agents.blue.forensics import BlueForensics
from cipher.environment.scenario import ScenarioGenerator
from cipher.environment.state import EpisodeState
from cipher.memory.dead_drop import DeadDrop, DeadDropVault
from cipher.rewards.blue_reward import compute_blue_reward
from cipher.rewards.oversight_reward import compute_oversight_signal
from cipher.rewards.red_reward import compute_red_reward
from cipher.rewards.reward_logger import RewardLogger
from cipher.training._episode_runner import run_episode
from cipher.utils.config import config

os.environ.setdefault("LLM_MODE", "stub")


class _TrapRegistryStub:
    def __init__(self, summary: dict):
        self._summary = summary

    def get_trap_summary(self) -> dict:
        return dict(self._summary)


class _VaultStub:
    def __init__(self, drops: list[DeadDrop]):
        self._drops = drops

    def read_all_episode_drops(self) -> list[DeadDrop]:
        return self._drops


def _scenario_state() -> tuple:
    gen = ScenarioGenerator()
    scenario, graph = gen.generate(1)
    state = EpisodeState.create_from_scenario(scenario, graph)
    state.red_path_history = [state.red_current_node]
    state.trap_registry = _TrapRegistryStub(
        {"red_traps_placed": 0, "honeypots_triggered": 0, "blue_honeypots_placed": 0}
    )
    state.step = 30
    return scenario, graph, state


class TestRedReward:
    def test_zero_exfil_gives_near_zero_total(self):
        scenario, _, state = _scenario_state()
        state.blue_detection_confidence = 0.2
        r = compute_red_reward(state, scenario, _VaultStub([]), config)
        assert r.exfiltration_completeness == 0.0
        assert abs(r.total) < 0.5

    def test_full_exfil_gives_positive_total(self):
        scenario, _, state = _scenario_state()
        state.red_exfiltrated_files = scenario.target_files.copy()
        state.blue_detection_confidence = 0.1
        r = compute_red_reward(state, scenario, _VaultStub([]), config)
        assert r.total > 0.0

    def test_abort_penalty_applied(self):
        scenario, _, state = _scenario_state()
        state.terminal_reason = "aborted"
        r = compute_red_reward(state, scenario, _VaultStub([]), config)
        assert r.abort_penalty == -0.3

    def test_no_abort_no_penalty(self):
        scenario, _, state = _scenario_state()
        state.terminal_reason = "max_steps"
        r = compute_red_reward(state, scenario, _VaultStub([]), config)
        assert r.abort_penalty == 0.0

    def test_honeypot_penalty_per_trigger(self):
        scenario, _, state = _scenario_state()
        state.trap_registry = _TrapRegistryStub({"red_traps_placed": 0, "honeypots_triggered": 2})
        r = compute_red_reward(state, scenario, _VaultStub([]), config)
        assert r.honeypot_trigger_penalty == -0.4

    def test_complexity_multiplier_minimum_one(self):
        scenario, _, state = _scenario_state()
        r = compute_red_reward(state, scenario, _VaultStub([]), config)
        assert r.operation_complexity_multiplier >= 1.0

    def test_complexity_multiplier_scales_with_nodes(self):
        scenario, _, state1 = _scenario_state()
        state2 = EpisodeState.from_dict(state1.to_dict())
        state2.trap_registry = _TrapRegistryStub({"red_traps_placed": 0, "honeypots_triggered": 0})
        state2.red_path_history = list(range(25))
        r1 = compute_red_reward(state1, scenario, _VaultStub([]), config)
        r2 = compute_red_reward(state2, scenario, _VaultStub([]), config)
        assert r2.operation_complexity_multiplier > r1.operation_complexity_multiplier

    def test_memory_efficiency_perfect_when_no_drops(self):
        scenario, _, state = _scenario_state()
        r = compute_red_reward(state, scenario, _VaultStub([]), config)
        assert r.memory_efficiency_score == 1.0

    def test_memory_efficiency_penalizes_over_budget(self):
        scenario, _, state = _scenario_state()
        drop = DeadDrop(written_by="red", token_count=config.env_dead_drop_max_tokens * 2)
        r = compute_red_reward(state, scenario, _VaultStub([drop]), config)
        assert r.memory_efficiency_score < 1.0

    def test_detection_probability_from_state(self):
        scenario, _, state = _scenario_state()
        state.blue_detection_confidence = 0.73
        r = compute_red_reward(state, scenario, _VaultStub([]), config)
        assert r.detection_probability == pytest.approx(0.73, abs=1e-4)

    def test_all_components_between_bounds(self):
        scenario, _, state = _scenario_state()
        r = compute_red_reward(state, scenario, _VaultStub([]), config)
        assert 0.0 <= r.exfiltration_completeness <= 1.0
        assert 0.0 <= r.detection_probability <= 1.0
        assert 0.0 <= r.memory_efficiency_score <= 1.0
        assert 1.0 <= r.operation_complexity_multiplier <= 3.0

    def test_total_is_multiplicative_core_plus_penalties(self):
        scenario, _, state = _scenario_state()
        state.red_exfiltrated_files = scenario.target_files[:2]
        state.blue_detection_confidence = 0.25
        state.terminal_reason = "aborted"
        state.trap_registry = _TrapRegistryStub({"red_traps_placed": 1, "honeypots_triggered": 1})
        r = compute_red_reward(state, scenario, _VaultStub([]), config)
        expected = (
            r.exfiltration_completeness
            * (1 - r.detection_probability)
            * r.memory_efficiency_score
            * r.operation_complexity_multiplier
        ) + r.abort_penalty + r.honeypot_trigger_penalty
        assert r.total == pytest.approx(expected, abs=1e-4)


class TestBlueReward:
    def test_no_detection_gives_zero_accuracy(self):
        _, graph, state = _scenario_state()
        state.blue_detection_confidence = 0.0
        b = compute_blue_reward(state, graph, None, config)
        assert b.detection_accuracy_score <= 0.31

    def test_full_detection_gives_high_accuracy(self):
        _, graph, state = _scenario_state()
        state.blue_detection_confidence = 0.9
        b = compute_blue_reward(state, graph, None, config)
        assert b.detection_accuracy_score >= 0.7

    def test_early_detection_gives_high_speed_bonus(self):
        _, graph, state = _scenario_state()
        state.blue_first_detection_step = 5
        b = compute_blue_reward(state, graph, None, config)
        assert b.response_speed_bonus > 0.85

    def test_late_detection_gives_low_speed_bonus(self):
        _, graph, state = _scenario_state()
        state.blue_first_detection_step = 150
        b = compute_blue_reward(state, graph, None, config)
        assert b.response_speed_bonus < 0.1

    def test_no_detection_gives_zero_speed_bonus(self):
        _, graph, state = _scenario_state()
        state.blue_first_detection_step = None
        b = compute_blue_reward(state, graph, None, config)
        assert b.response_speed_bonus == 0.0

    def test_false_positive_penalty_proportional(self):
        _, graph, state = _scenario_state()
        state.blue_anomaly_history = [
            {"is_red_planted": True},
            {"is_red_planted": True},
            {"is_red_planted": False},
            {"is_red_planted": False},
        ]
        b = compute_blue_reward(state, graph, None, config)
        assert b.false_positive_rate_penalty == pytest.approx(0.5, abs=1e-4)

    def test_no_false_positives_no_penalty(self):
        _, graph, state = _scenario_state()
        state.blue_anomaly_history = [{"is_red_planted": False}]
        b = compute_blue_reward(state, graph, None, config)
        assert b.false_positive_rate_penalty == 0.0

    def test_jaccard_perfect_reconstruction(self):
        _, graph, state = _scenario_state()
        state.red_path_history = [1, 2, 3]
        forensics = BlueForensics("bf", config)
        forensics._operation_graph = [{"node_id": 1}, {"node_id": 2}, {"node_id": 3}]
        b = compute_blue_reward(state, graph, forensics, config)
        assert b.operation_graph_reconstruction_score == 1.0

    def test_jaccard_zero_reconstruction(self):
        _, graph, state = _scenario_state()
        state.red_path_history = [1, 2, 3]
        forensics = BlueForensics("bf", config)
        forensics._operation_graph = [{"node_id": 10}, {"node_id": 11}]
        b = compute_blue_reward(state, graph, forensics, config)
        assert b.operation_graph_reconstruction_score == 0.0

    def test_jaccard_partial_reconstruction(self):
        _, graph, state = _scenario_state()
        state.red_path_history = [1, 2, 3, 4]
        forensics = BlueForensics("bf", config)
        forensics._operation_graph = [{"node_id": 3}, {"node_id": 4}, {"node_id": 5}, {"node_id": 6}]
        b = compute_blue_reward(state, graph, forensics, config)
        assert 0.2 <= b.operation_graph_reconstruction_score <= 0.5

    def test_honeypot_trigger_rate_zero_when_none_triggered(self):
        _, graph, state = _scenario_state()
        state.trap_registry = _TrapRegistryStub({"honeypots_triggered": 0, "blue_honeypots_placed": 0})
        b = compute_blue_reward(state, graph, None, config)
        assert b.honeypot_trigger_rate == 0.0


class TestOversightReward:
    def test_no_history_returns_zeros(self):
        _, _, state = _scenario_state()
        o = compute_oversight_signal(state, [], config)
        assert o.total_red_adjustment == 0.0
        assert o.total_blue_adjustment == 0.0

    def test_reward_hacking_flag_fires(self):
        _, _, state = _scenario_state()
        history = [
            {"red_complexity_multiplier": 3.0, "red_unique_nodes": 2, "blue_detection_confidence_final": 0.5},
            {"red_complexity_multiplier": 2.8, "red_unique_nodes": 2, "blue_detection_confidence_final": 0.5},
            {"red_complexity_multiplier": 2.9, "red_unique_nodes": 1, "blue_detection_confidence_final": 0.5},
        ]
        o = compute_oversight_signal(state, history, config)
        assert o.reward_hacking_penalty < 0.0
        assert o.has_flags()

    def test_collusion_flag_fires_on_low_detection(self):
        _, _, state = _scenario_state()
        history = [
            {"red_complexity_multiplier": 1.2, "red_unique_nodes": 10, "blue_detection_confidence_final": 0.01}
            for _ in range(5)
        ]
        o = compute_oversight_signal(state, history, config)
        assert o.collusion_penalty == -0.2

    def test_normal_play_no_flags(self):
        _, _, state = _scenario_state()
        history = [
            {"red_complexity_multiplier": 1.3, "red_unique_nodes": 12, "blue_detection_confidence_final": 0.5}
            for _ in range(5)
        ]
        o = compute_oversight_signal(state, history, config)
        assert not o.has_flags()

    def test_flags_list_populated_when_triggered(self):
        _, _, state = _scenario_state()
        history = [
            {"red_complexity_multiplier": 3.0, "red_unique_nodes": 1, "blue_detection_confidence_final": 0.01}
            for _ in range(5)
        ]
        o = compute_oversight_signal(state, history, config)
        assert len(o.flags_fired) > 0


class TestRewardLogger:
    def test_csv_created_on_first_log(self):
        with tempfile.TemporaryDirectory() as td:
            old = RewardLogger.LOG_FILE
            RewardLogger.LOG_FILE = Path(td) / "rewards_log.csv"
            try:
                logger = RewardLogger()
                assert logger.LOG_FILE.exists()
            finally:
                RewardLogger.LOG_FILE = old

    def test_row_appended_each_episode(self):
        with tempfile.TemporaryDirectory() as td:
            old = RewardLogger.LOG_FILE
            RewardLogger.LOG_FILE = Path(td) / "rewards_log.csv"
            try:
                logger = RewardLogger()
                red = compute_red_reward(_scenario_state()[2], _scenario_state()[0], _VaultStub([]), config)
                blue = compute_blue_reward(_scenario_state()[2], _scenario_state()[1], None, config)
                oversight = compute_oversight_signal(_scenario_state()[2], [], config)
                logger.log(1, 10, "max_steps", red, blue, oversight)
                logger.log(2, 12, "max_steps", red, blue, oversight)
                rows = list(csv.DictReader(RewardLogger.LOG_FILE.open("r", encoding="utf-8")))
                assert len(rows) == 2
            finally:
                RewardLogger.LOG_FILE = old

    def test_all_columns_present(self):
        with tempfile.TemporaryDirectory() as td:
            old = RewardLogger.LOG_FILE
            RewardLogger.LOG_FILE = Path(td) / "rewards_log.csv"
            try:
                logger = RewardLogger()
                scenario, graph, state = _scenario_state()
                red = compute_red_reward(state, scenario, _VaultStub([]), config)
                blue = compute_blue_reward(state, graph, None, config)
                oversight = compute_oversight_signal(state, [], config)
                logger.log(1, 10, "max_steps", red, blue, oversight)
                row = next(csv.DictReader(RewardLogger.LOG_FILE.open("r", encoding="utf-8")))
                for col in RewardLogger.COLUMNS:
                    assert col in row
            finally:
                RewardLogger.LOG_FILE = old

    def test_values_are_numeric(self):
        with tempfile.TemporaryDirectory() as td:
            old = RewardLogger.LOG_FILE
            RewardLogger.LOG_FILE = Path(td) / "rewards_log.csv"
            try:
                logger = RewardLogger()
                scenario, graph, state = _scenario_state()
                red = compute_red_reward(state, scenario, _VaultStub([]), config)
                blue = compute_blue_reward(state, graph, None, config)
                oversight = compute_oversight_signal(state, [], config)
                logger.log(1, 10, "max_steps", red, blue, oversight)
                row = next(csv.DictReader(RewardLogger.LOG_FILE.open("r", encoding="utf-8")))
                float(row["red_total"])
                float(row["blue_total"])
                float(row["oversight_red_adj"])
            finally:
                RewardLogger.LOG_FILE = old


class TestRewardVariance:
    def test_red_reward_varies_across_10_episodes(self):
        gen = ScenarioGenerator()
        red_totals = []
        for i in range(1, 11):
            scenario, graph = gen.generate(i)
            result = run_episode(scenario, graph, config, max_steps=30, verbose=False)
            red_totals.append(result["red_reward"].total)
        assert statistics.stdev(red_totals) > 0.01

    def test_blue_reward_varies_across_10_episodes(self):
        gen = ScenarioGenerator()
        blue_totals = []
        for i in range(1, 11):
            scenario, graph = gen.generate(i)
            result = run_episode(scenario, graph, config, max_steps=30, verbose=False)
            blue_totals.append(result["blue_reward"].total)
        assert statistics.stdev(blue_totals) > 0.01

    def test_high_exfil_episodes_score_higher_red(self):
        scenario, _, state0 = _scenario_state()
        state1 = EpisodeState.from_dict(state0.to_dict())
        state1.trap_registry = _TrapRegistryStub({"red_traps_placed": 0, "honeypots_triggered": 0})
        state1.red_exfiltrated_files = scenario.target_files.copy()
        r0 = compute_red_reward(state0, scenario, _VaultStub([]), config)
        r1 = compute_red_reward(state1, scenario, _VaultStub([]), config)
        assert r1.total > r0.total

    def test_early_detection_episodes_score_higher_blue(self):
        _, graph, state_early = _scenario_state()
        state_late = EpisodeState.from_dict(state_early.to_dict())
        state_late.trap_registry = _TrapRegistryStub({"honeypots_triggered": 0, "blue_honeypots_placed": 0})
        state_early.blue_first_detection_step = 5
        state_late.blue_first_detection_step = 150
        b1 = compute_blue_reward(state_early, graph, None, config)
        b2 = compute_blue_reward(state_late, graph, None, config)
        assert b1.response_speed_bonus > b2.response_speed_bonus
