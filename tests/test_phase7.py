"""
tests/test_phase7.py

Phase 7 — OversightAuditor tests.
All run in stub mode. No API calls.
"""
from __future__ import annotations

import csv
import json
import os

import networkx as nx
import pytest

os.environ.setdefault("LLM_MODE", "stub")

from cipher.agents.base_agent import Action, ActionType
from cipher.agents.blue.forensics import BlueForensics
from cipher.agents.oversight.auditor import AuditorJudgment, OversightAuditor
from cipher.environment.scenario import ScenarioGenerator
from cipher.environment.state import EpisodeState
from cipher.memory.dead_drop import DeadDropVault
from cipher.rewards.blue_reward import BlueRewardComponents, compute_blue_reward
from cipher.rewards.oversight_reward import apply_fleet_bonus, compute_oversight_signal
from cipher.rewards.red_reward import RedRewardComponents, compute_red_reward
from cipher.rewards.reward_logger import RewardLogger
from cipher.training._episode_runner import run_episode
from cipher.utils.config import config


@pytest.fixture
def auditor():
    return OversightAuditor(config)


def _minimal_state():
    gen = ScenarioGenerator()
    scenario, graph = gen.generate(1)
    return EpisodeState.create_from_scenario(scenario, graph)


def _red_log(n_steps: int = 5) -> list[dict]:
    return [
        {
            "step": i,
            "agent": "red_planner_01",
            "action": "MOVE",
            "target": f"node_{i+10}",
            "reasoning": f"Moving to node {i+10} for access.",
        }
        for i in range(1, n_steps + 1)
    ]


def _blue_log(n_steps: int = 5) -> list[dict]:
    return [
        {
            "step": i,
            "agent": "blue_surveillance_01",
            "action": "ANALYZE_ANOMALY",
            "target": None,
            "reasoning": f"Checking anomaly at step {i}.",
        }
        for i in range(1, n_steps + 1)
    ]


class TestAuditorJudgment:
    def test_to_log_dict_excludes_raw_response(self):
        j = AuditorJudgment(
            fleet_bonus_red=0.1,
            fleet_bonus_blue=-0.05,
            judgment_text="RED was aggressive.",
            notable_red_action="Exfiltrated salary_data",
            notable_blue_action="Placed honeypot at node 26",
            quality_score_red=0.75,
            quality_score_blue=0.6,
            episode_verdict="contested",
            raw_llm_response="<very long raw text>",
        )
        d = j.to_log_dict()
        assert "raw_llm_response" not in d

    def test_to_log_dict_contains_all_other_fields(self):
        j = AuditorJudgment(
            fleet_bonus_red=0.0,
            fleet_bonus_blue=0.0,
            judgment_text="Baseline.",
            notable_red_action="MOVE",
            notable_blue_action="STAND_DOWN",
            quality_score_red=0.5,
            quality_score_blue=0.5,
            episode_verdict="contested",
            raw_llm_response="",
        )
        d = j.to_log_dict()
        expected_keys = {
            "fleet_bonus_red",
            "fleet_bonus_blue",
            "judgment_text",
            "notable_red_action",
            "notable_blue_action",
            "quality_score_red",
            "quality_score_blue",
            "episode_verdict",
        }
        assert expected_keys.issubset(d.keys())


class TestAuditorInit:
    def test_auditor_initializes_without_error(self, auditor):
        assert auditor is not None

    def test_agent_id_correct(self, auditor):
        assert auditor.AGENT_ID == "oversight_auditor_01"

    def test_team_is_oversight(self, auditor):
        assert auditor.TEAM == "oversight"

    def test_role_is_oversight(self, auditor):
        assert auditor.ROLE == "oversight"

    def test_system_prompt_is_non_empty(self, auditor):
        assert len(auditor.SYSTEM_PROMPT) > 100


class TestJudgeEpisode:
    def test_returns_auditor_judgment(self, auditor):
        state = _minimal_state()
        result = auditor.judge_episode(state, _red_log(), _blue_log())
        assert isinstance(result, AuditorJudgment)

    def test_fleet_bonus_red_in_range(self, auditor):
        state = _minimal_state()
        j = auditor.judge_episode(state, _red_log(), _blue_log())
        assert -0.2 <= j.fleet_bonus_red <= 0.2

    def test_fleet_bonus_blue_in_range(self, auditor):
        state = _minimal_state()
        j = auditor.judge_episode(state, _red_log(), _blue_log())
        assert -0.2 <= j.fleet_bonus_blue <= 0.2

    def test_quality_scores_in_range(self, auditor):
        state = _minimal_state()
        j = auditor.judge_episode(state, _red_log(), _blue_log())
        assert 0.0 <= j.quality_score_red <= 1.0
        assert 0.0 <= j.quality_score_blue <= 1.0

    def test_judgment_text_is_string(self, auditor):
        state = _minimal_state()
        j = auditor.judge_episode(state, _red_log(), _blue_log())
        assert isinstance(j.judgment_text, str)
        assert len(j.judgment_text) > 0

    def test_episode_verdict_is_valid(self, auditor):
        valid = {"red_dominates", "blue_dominates", "contested", "degenerate"}
        state = _minimal_state()
        j = auditor.judge_episode(state, _red_log(), _blue_log())
        assert j.episode_verdict in valid

    def test_empty_action_logs_do_not_crash(self, auditor):
        state = _minimal_state()
        j = auditor.judge_episode(state, [], [])
        assert isinstance(j, AuditorJudgment)

    def test_never_raises_on_bad_state(self, auditor):
        j = auditor.judge_episode(object(), [], [])  # type: ignore[arg-type]
        assert isinstance(j, AuditorJudgment)
        assert j.fleet_bonus_red == 0.0

    def test_long_action_logs_truncated_safely(self, auditor):
        state = _minimal_state()
        big_log = _red_log(n_steps=100)
        j = auditor.judge_episode(state, big_log, _blue_log())
        assert isinstance(j, AuditorJudgment)


class TestParseResponse:
    def test_clean_json_parses_correctly(self, auditor):
        raw = json.dumps(
            {
                "fleet_bonus_red": 0.15,
                "fleet_bonus_blue": -0.1,
                "judgment_text": "RED showed strategic sophistication.",
                "notable_red_action": "Multi-hop path to zone 3",
                "notable_blue_action": "Honeypot placed at auth_gateway",
                "quality_score_red": 0.8,
                "quality_score_blue": 0.6,
                "episode_verdict": "red_dominates",
            }
        )
        j = auditor._parse_response(raw)
        assert j.fleet_bonus_red == pytest.approx(0.15)
        assert j.episode_verdict == "red_dominates"

    def test_markdown_fenced_json_parses(self, auditor):
        raw = (
            "```json\n"
            '{"fleet_bonus_red": 0.0, "fleet_bonus_blue": 0.0, '
            '"judgment_text": "Contested.", "notable_red_action": "MOVE", '
            '"notable_blue_action": "STAND_DOWN", "quality_score_red": 0.5, '
            '"quality_score_blue": 0.5, "episode_verdict": "contested"}\n'
            "```"
        )
        j = auditor._parse_response(raw)
        assert j.episode_verdict == "contested"

    def test_fleet_bonus_clamped_above_max(self, auditor):
        raw = json.dumps(
            {
                "fleet_bonus_red": 0.99,
                "fleet_bonus_blue": 0.0,
                "judgment_text": "test",
                "notable_red_action": "",
                "notable_blue_action": "",
                "quality_score_red": 0.5,
                "quality_score_blue": 0.5,
                "episode_verdict": "contested",
            }
        )
        j = auditor._parse_response(raw)
        assert j.fleet_bonus_red == pytest.approx(0.2)

    def test_fleet_bonus_clamped_below_min(self, auditor):
        raw = json.dumps(
            {
                "fleet_bonus_red": -5.0,
                "fleet_bonus_blue": 0.0,
                "judgment_text": "test",
                "notable_red_action": "",
                "notable_blue_action": "",
                "quality_score_red": 0.5,
                "quality_score_blue": 0.5,
                "episode_verdict": "contested",
            }
        )
        j = auditor._parse_response(raw)
        assert j.fleet_bonus_red == pytest.approx(-0.2)

    def test_missing_fields_use_defaults(self, auditor):
        raw = json.dumps({"episode_verdict": "degenerate"})
        j = auditor._parse_response(raw)
        assert j.episode_verdict == "degenerate"
        assert j.fleet_bonus_red == 0.0

    def test_invalid_json_triggers_exception(self, auditor):
        with pytest.raises(Exception):
            auditor._parse_response("not json at all {{ }}")


class TestApplyFleetBonus:
    def _make_rewards(self):
        red = RedRewardComponents(
            exfiltration_completeness=0.333,
            detection_probability=0.4,
            memory_efficiency_score=1.0,
            operation_complexity_multiplier=1.2,
            abort_penalty=0.0,
            honeypot_trigger_penalty=0.0,
            total=0.24,
            episode_steps=40,
            unique_nodes_visited=8,
            drops_written=1,
            traps_placed=1,
            context_resets=0,
            terminal_reason="max_steps",
        )
        blue = BlueRewardComponents(
            detection_accuracy_score=0.6,
            response_speed_bonus=0.3,
            false_positive_rate_penalty=0.0,
            honeypot_trigger_rate=0.0,
            operation_graph_reconstruction_score=0.0,
            total=0.18,
        )
        return red, blue

    def _make_judgment(self, bonus_red: float, bonus_blue: float) -> AuditorJudgment:
        return AuditorJudgment(
            fleet_bonus_red=bonus_red,
            fleet_bonus_blue=bonus_blue,
            judgment_text="test",
            notable_red_action="",
            notable_blue_action="",
            quality_score_red=0.5,
            quality_score_blue=0.5,
            episode_verdict="contested",
            raw_llm_response="",
        )

    def test_positive_bonus_increases_totals(self):
        red, blue = self._make_rewards()
        original_red = red.total
        original_blue = blue.total
        j = self._make_judgment(0.1, 0.15)
        apply_fleet_bonus(red, blue, j)
        assert red.total == pytest.approx(original_red + 0.1)
        assert blue.total == pytest.approx(original_blue + 0.15)

    def test_negative_bonus_decreases_totals(self):
        red, blue = self._make_rewards()
        original_red = red.total
        j = self._make_judgment(-0.1, 0.0)
        apply_fleet_bonus(red, blue, j)
        assert red.total == pytest.approx(original_red - 0.1)

    def test_zero_bonus_no_change(self):
        red, blue = self._make_rewards()
        original_red = red.total
        original_blue = blue.total
        j = self._make_judgment(0.0, 0.0)
        apply_fleet_bonus(red, blue, j)
        assert red.total == pytest.approx(original_red)
        assert blue.total == pytest.approx(original_blue)


class TestRewardLoggerPhase7:
    def test_fleet_verdict_column_present(self):
        assert "fleet_verdict" in RewardLogger.COLUMNS

    def test_fleet_judgment_column_present(self):
        assert "fleet_judgment" in RewardLogger.COLUMNS

    def test_log_with_judgment_creates_row(self, tmp_path, monkeypatch):
        monkeypatch.setattr(RewardLogger, "LOG_FILE", tmp_path / "test_rewards.csv")
        rl = RewardLogger()

        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        state = EpisodeState.create_from_scenario(scenario, graph)

        vault = DeadDropVault(tmp_path / "vault", config.env_dead_drop_max_tokens)
        red = compute_red_reward(state, scenario, vault, config)
        forensics = BlueForensics("blue_forensics_01", config)
        blue = compute_blue_reward(state, graph, forensics, config)
        oversight = compute_oversight_signal(state, [], config)

        judgment = AuditorJudgment(
            fleet_bonus_red=0.05,
            fleet_bonus_blue=-0.05,
            judgment_text="Test judgment.",
            notable_red_action="none",
            notable_blue_action="none",
            quality_score_red=0.5,
            quality_score_blue=0.5,
            episode_verdict="contested",
            raw_llm_response="",
        )

        rl.log(1, state.step, "max_steps", red, blue, oversight, judgment)
        rows = list(csv.DictReader(open(tmp_path / "test_rewards.csv")))
        assert len(rows) == 1
        assert rows[0]["fleet_verdict"] == "contested"
        assert "Test judgment" in rows[0]["fleet_judgment"]

    def test_log_without_judgment_defaults_to_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(RewardLogger, "LOG_FILE", tmp_path / "test_rewards2.csv")
        rl = RewardLogger()

        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        state = EpisodeState.create_from_scenario(scenario, graph)
        vault = DeadDropVault(tmp_path / "vault2", config.env_dead_drop_max_tokens)
        red = compute_red_reward(state, scenario, vault, config)
        forensics = BlueForensics("blue_forensics_01", config)
        blue = compute_blue_reward(state, graph, forensics, config)
        oversight = compute_oversight_signal(state, [], config)

        rl.log(1, state.step, "max_steps", red, blue, oversight)
        rows = list(csv.DictReader(open(tmp_path / "test_rewards2.csv")))
        assert rows[0]["fleet_verdict"] == "none"
        assert rows[0]["fleet_judgment"] == "none"


class TestEpisodeRunnerPhase7:
    def test_run_episode_returns_judgment(self):
        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        result = run_episode(scenario, graph, config, max_steps=10, verbose=False)
        assert "judgment" in result
        assert isinstance(result["judgment"], AuditorJudgment)

    def test_fleet_bonus_applied_to_totals(self):
        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        result = run_episode(scenario, graph, config, max_steps=10, verbose=False)
        assert result["red_reward"].total == result["red_reward"].total
        assert result["blue_reward"].total == result["blue_reward"].total

    def test_csv_includes_fleet_columns_after_run(self, tmp_path, monkeypatch):
        log_path = tmp_path / "phase7_rewards.csv"
        monkeypatch.setattr(RewardLogger, "LOG_FILE", log_path)

        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        run_episode(scenario, graph, config, max_steps=10, verbose=False)

        rows = list(csv.DictReader(open(log_path)))
        assert len(rows) >= 1
        assert "fleet_verdict" in rows[0]
        assert "fleet_judgment" in rows[0]


class TestPhase8Preflight:
    def test_oversight_step_flag_generation(self):
        auditor = OversightAuditor(config)
        state = _minimal_state()
        flags = auditor.evaluate_step(
            step=1,
            state=state,
            red_actions=[
                Action(
                    agent_id="red_operative_01",
                    action_type=ActionType.WRITE_DEAD_DROP,
                    reasoning="drop1",
                ),
                Action(
                    agent_id="red_planner_01",
                    action_type=ActionType.WRITE_DEAD_DROP,
                    reasoning="drop2",
                ),
            ],
            blue_actions=[
                Action(
                    agent_id="blue_deception_architect_01",
                    action_type=ActionType.STAND_DOWN,
                    reasoning="standing by",
                )
            ],
        )
        flag_types = {f.flag_type for f in flags}
        assert "REWARD_HACKING_SUSPECTED" in flag_types
        assert "BLUE_PASSIVITY" in flag_types

    def test_oversight_flags_persist_in_episode_log(self):
        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        result = run_episode(
            scenario,
            graph,
            config,
            max_steps=1,
            verbose=False,
            scripted_red_actions={
                1: [
                    Action(
                        agent_id="red_planner_01",
                        action_type=ActionType.WAIT,
                        reasoning="hold",
                    )
                ]
            },
            scripted_blue_actions={
                1: [
                    Action(
                        agent_id="blue_deception_architect_01",
                        action_type=ActionType.STAND_DOWN,
                        reasoning="hold while budget remains",
                    )
                ]
            },
        )
        oversight_entries = [
            e for e in result["state"].episode_log if e.get("action_type") == "OVERSIGHT_FLAG"
        ]
        assert len(oversight_entries) >= 1
        assert "applied_penalty_blue" in oversight_entries[0]["result"]
        assert "severity" in oversight_entries[0]["payload"]

    def test_oversight_flag_severity_penalty_applied(self):
        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        result = run_episode(
            scenario,
            graph,
            config,
            max_steps=1,
            verbose=False,
            scripted_red_actions={
                1: [
                    Action(
                        agent_id="red_planner_01",
                        action_type=ActionType.WAIT,
                        reasoning="hold",
                    )
                ]
            },
            scripted_blue_actions={
                1: [
                    Action(
                        agent_id="blue_deception_architect_01",
                        action_type=ActionType.STAND_DOWN,
                        reasoning="hold while budget remains",
                    )
                ]
            },
        )
        assert result["oversight_step_penalty_blue"] < 0.0
        assert len(result["oversight_flags"]) >= 1

    def test_exfil_parser_rejects_node_like_target(self):
        from cipher.agents.red.exfiltrator import RedExfiltrator

        agent = RedExfiltrator("red_exfiltrator_01", config)
        parsed = agent._parse_action_from_response(
            json.dumps(
                {
                    "action_type": "exfiltrate",
                    "target_node": 47,
                    "target_file": "47",
                    "reasoning": "Exfil node 47.",
                }
            )
        )
        assert parsed.action_type == ActionType.WAIT
        assert parsed.target_file is None

    def test_abort_stops_remaining_red_actions_same_step(self):
        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        scripted_red = {
            1: [
                Action(
                    agent_id="red_planner_01",
                    action_type=ActionType.ABORT,
                    reasoning="Abort now.",
                ),
                Action(
                    agent_id="red_exfiltrator_01",
                    action_type=ActionType.EXFILTRATE,
                    target_file="should_not_run.txt",
                    reasoning="Should not execute after abort.",
                ),
            ]
        }
        result = run_episode(
            scenario,
            graph,
            config,
            max_steps=5,
            verbose=False,
            scripted_red_actions=scripted_red,
            scripted_blue_actions={1: []},
        )
        assert result["state"].terminal_reason == "aborted"
        red_agent_actions = [
            e for e in result["state"].episode_log if str(e.get("agent_id", "")).startswith("red_")
        ]
        assert len(red_agent_actions) == 1
        assert red_agent_actions[0]["agent_id"] == "red_planner_01"

    def test_exfil_takes_precedence_over_abort_same_step(self):
        from cipher.environment.graph import get_high_value_target

        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        hvt = get_high_value_target(graph)
        valid_target = graph.nodes[hvt].get("files", [f"target_{hvt}_0"])[0]
        scripted_red = {
            1: [
                Action(
                    agent_id="red_exfiltrator_01",
                    action_type=ActionType.EXFILTRATE,
                    target_file=valid_target,
                    reasoning="Exfil first.",
                ),
                Action(
                    agent_id="red_planner_01",
                    action_type=ActionType.ABORT,
                    reasoning="Abort second.",
                ),
            ]
        }
        # Force a valid exfil state before the scripted step.
        result = run_episode(
            scenario,
            graph,
            config,
            max_steps=1,
            verbose=False,
            debug_force_exfil_sanity=True,
            scripted_red_actions=scripted_red,
            scripted_blue_actions={1: []},
        )
        assert result["state"].terminal_reason != "aborted"
        assert len(result["state"].red_exfiltrated_files) > 0

    def test_debug_force_exfil_sanity_guarantees_reachability(self):
        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        result = run_episode(
            scenario,
            graph,
            config,
            max_steps=1,
            verbose=False,
            debug_force_exfil_sanity=True,
            scripted_red_actions={
                1: [
                    Action(
                        agent_id="red_planner_01",
                        action_type=ActionType.WAIT,
                        reasoning="No-op after sanity check.",
                    )
                ]
            },
            scripted_blue_actions={1: []},
        )
        assert len(result["state"].red_exfiltrated_files) > 0
        assert result["red_reward"].exfiltration_completeness > 0.0

    def test_action_reason_mismatch_counter_increments(self):
        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        result = run_episode(
            scenario,
            graph,
            config,
            max_steps=1,
            verbose=False,
            scripted_red_actions={
                1: [
                    Action(
                        agent_id="red_planner_01",
                        action_type=ActionType.WAIT,
                        reasoning="idle",
                    )
                ]
            },
            scripted_blue_actions={
                1: [
                    Action(
                        agent_id="blue_deception_architect_01",
                        action_type=ActionType.STAND_DOWN,
                        reasoning="Placing a honeypot near target traffic.",
                    )
                ]
            },
        )
        assert result["action_reason_mismatch_count"] >= 1

    def test_oversight_prompt_file_loads_without_warning(self, caplog):
        caplog.clear()
        with caplog.at_level("WARNING"):
            OversightAuditor(config)
        assert "Prompt file not found" not in caplog.text

    def test_scripted_golden_episode_yields_positive_exfil_signal(self):
        from cipher.environment.graph import get_entry_points, get_high_value_target

        gen = ScenarioGenerator()
        scenario, graph = gen.generate(1)
        start = get_entry_points(graph)[0]
        hvt = get_high_value_target(graph)
        path = nx.shortest_path(graph, start, hvt)
        target_file = graph.nodes[hvt].get("files", [f"target_{hvt}_0"])[0]

        scripted_red: dict[int, list[Action]] = {}
        step_idx = 1
        for node in path[1:]:
            scripted_red[step_idx] = [
                Action(
                    agent_id="red_exfiltrator_01",
                    action_type=ActionType.MOVE,
                    target_node=node,
                    reasoning="Scripted move to HVT.",
                )
            ]
            step_idx += 1
        scripted_red[step_idx] = [
            Action(
                agent_id="red_exfiltrator_01",
                action_type=ActionType.READ_FILE,
                target_file=target_file,
                reasoning="Scripted read before exfil.",
            )
        ]
        step_idx += 1
        scripted_red[step_idx] = [
            Action(
                agent_id="red_exfiltrator_01",
                action_type=ActionType.EXFILTRATE,
                target_file=target_file,
                reasoning="Scripted successful exfil.",
            )
        ]

        result = run_episode(
            scenario,
            graph,
            config,
            max_steps=step_idx,
            verbose=False,
            scripted_red_actions=scripted_red,
            scripted_blue_actions={s: [] for s in range(1, step_idx + 1)},
        )
        assert len(result["state"].red_exfiltrated_files) > 0
        assert result["red_reward"].exfiltration_completeness > 0.0
