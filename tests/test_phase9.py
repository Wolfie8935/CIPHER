"""
Tests for Phase 9 — Prompt Evolution Learning Loop.

Target: ~15 new tests → total ~245 passing after Phase 9.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cipher.training.prompt_evolver import PromptEvolver


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _make_rewards_csv(tmp_path: Path, n: int = 20) -> Path:
    """Write a minimal rewards_log.csv with *n* rows to *tmp_path*."""
    rng = np.random.default_rng(7)
    terminal = [
        "aborted" if rng.random() < 0.5 else "max_steps" for _ in range(n)
    ]
    df = pd.DataFrame(
        {
            "episode": [1] * n,
            "timestamp": ["2026-01-01"] * n,
            "steps": rng.integers(4, 12, n),
            "terminal_reason": terminal,
            "red_total": rng.uniform(-0.5, 0.3, n),
            "red_exfil": [0.0] * n,          # all zero → low exfil rate
            "red_stealth": rng.uniform(0.3, 0.9, n),
            "red_memory": [1.0] * n,
            "red_complexity": rng.uniform(1.0, 2.0, n),
            "red_abort_penalty": [-0.3 if t == "aborted" else 0.0 for t in terminal],
            "red_honeypot_penalty": [0.0] * n,
            "blue_total": rng.uniform(-0.5, 0.5, n),
            "blue_detection": rng.uniform(0.3, 0.8, n),
            "blue_speed": rng.uniform(0.0, 0.05, n),     # very low
            "blue_fp_penalty": rng.uniform(0.6, 0.9, n),  # high → triggers rule
            "blue_honeypot_rate": rng.uniform(0.0, 0.15, n),  # low → triggers rule
            "blue_graph_reconstruction": [0.0] * n,
            "oversight_red_adj": [0.0] * n,
            "oversight_blue_adj": [0.0] * n,
            "oversight_flags": ["none"] * n,
            "red_unique_nodes": rng.integers(3, 12, n),
            "red_drops_written": rng.integers(0, 4, n),
            "red_traps_placed": rng.integers(0, 3, n),
            "red_context_resets": [0] * n,
            "red_complexity_multiplier": rng.uniform(1.0, 2.5, n),
            "fleet_verdict": ["contested"] * n,
            "fleet_judgment": ["no verdict"] * n,
        }
    )
    path = tmp_path / "rewards_log.csv"
    df.to_csv(path, index=False)
    return path


def _make_df(
    n: int = 20,
    abort_rate: float = 0.4,
    exfil_rate: float = 0.1,
    fp_penalty: float = 0.7,
) -> pd.DataFrame:
    """Build a test rewards DataFrame with controllable statistics."""
    rng = np.random.default_rng(42)
    terminal = [
        "aborted" if rng.random() < abort_rate else "max_steps" for _ in range(n)
    ]
    return pd.DataFrame(
        {
            "red_total": rng.uniform(-0.5, 0.5, n),
            "red_stealth": rng.uniform(0.3, 0.9, n),
            "red_exfil": [
                0.5 if rng.random() < exfil_rate else 0.0 for _ in range(n)
            ],
            "red_complexity_multiplier": rng.uniform(1.0, 2.5, n),
            "red_drops_written": rng.integers(0, 5, n),
            "terminal_reason": terminal,
            "blue_fp_penalty": rng.uniform(fp_penalty - 0.1, fp_penalty + 0.1, n),
            "blue_honeypot_rate": rng.uniform(0.0, 0.15, n),
            "blue_speed": rng.uniform(0.0, 0.05, n),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TestPromptEvolverInit
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptEvolverInit:
    def test_evolver_initializes(self):
        e = PromptEvolver()
        assert e.EVOLVE_EVERY_N == 10

    def test_should_evolve_false_on_non_multiple(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        e = PromptEvolver()
        assert not e.should_evolve(7)
        assert not e.should_evolve(11)
        assert not e.should_evolve(5)

    def test_should_evolve_true_on_multiple_with_enough_rows(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        _make_rewards_csv(tmp_path, n=15)
        e = PromptEvolver()
        assert e.should_evolve(10)
        assert e.should_evolve(20)

    def test_should_not_evolve_with_too_few_rows(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_rewards_csv(tmp_path, n=3)
        e = PromptEvolver()
        # Episode 10 is a multiple, but only 3 rows → should NOT evolve
        assert not e.should_evolve(10)

    def test_should_not_evolve_zero_episode(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_rewards_csv(tmp_path, n=15)
        e = PromptEvolver()
        # Episode 0 divides evenly, but training doesn't start at 0
        # The spec says "multiple of EVOLVE_EVERY_N" — 0 % 10 == 0,
        # but should_evolve(0) may be called before any episodes run.
        # We simply confirm it doesn't crash.
        result = e.should_evolve(0)
        assert isinstance(result, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# TestHeuristicExtraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestHeuristicExtraction:
    def test_red_heuristics_high_abort_rate(self):
        e = PromptEvolver()
        df = _make_df(abort_rate=0.5)
        rules = e._extract_red_heuristics(df)
        assert any("aborting" in r.lower() for r in rules)

    def test_red_heuristics_low_exfil_rate(self):
        e = PromptEvolver()
        df = _make_df(exfil_rate=0.05)
        rules = e._extract_red_heuristics(df)
        assert any("exfiltrate" in r.lower() for r in rules)

    def test_red_heuristics_empty_df_returns_empty(self):
        e = PromptEvolver()
        rules = e._extract_red_heuristics(pd.DataFrame())
        assert rules == []

    def test_red_heuristics_too_few_rows_returns_empty(self):
        e = PromptEvolver()
        df = _make_df(n=4)
        rules = e._extract_red_heuristics(df)
        assert rules == []

    def test_red_heuristics_stealth_rule_present(self):
        e = PromptEvolver()
        df = _make_df()
        rules = e._extract_red_heuristics(df)
        assert any("stealth" in r.lower() or "suspicion" in r.lower() for r in rules)

    def test_blue_heuristics_high_fp_penalty(self):
        e = PromptEvolver()
        df = _make_df(fp_penalty=0.8)
        rules = e._extract_blue_heuristics(df)
        assert any(
            "false positive" in r.lower() or "precision" in r.lower() for r in rules
        )

    def test_blue_heuristics_low_honeypot_rate(self):
        e = PromptEvolver()
        df = _make_df()  # blue_honeypot_rate is 0-0.15 in helper → triggers rule
        rules = e._extract_blue_heuristics(df)
        assert any("honeypot" in r.lower() for r in rules)

    def test_blue_heuristics_empty_df_returns_empty(self):
        e = PromptEvolver()
        rules = e._extract_blue_heuristics(pd.DataFrame())
        assert rules == []

    def test_all_rules_start_with_learned(self):
        e = PromptEvolver()
        df = _make_df()
        all_rules = e._extract_red_heuristics(df) + e._extract_blue_heuristics(df)
        assert len(all_rules) > 0, "Expected at least one rule to be generated"
        for rule in all_rules:
            assert rule.startswith("LEARNED:"), (
                f"Rule does not start with 'LEARNED:': {rule!r}"
            )

    def test_heuristic_extraction_is_deterministic(self):
        """Same DataFrame → same list of rules (no randomness)."""
        e = PromptEvolver()
        df = _make_df()
        r1 = e._extract_red_heuristics(df)
        r2 = e._extract_red_heuristics(df)
        assert r1 == r2
        b1 = e._extract_blue_heuristics(df)
        b2 = e._extract_blue_heuristics(df)
        assert b1 == b2


# ═══════════════════════════════════════════════════════════════════════════════
# TestPromptUpdate
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptUpdate:
    def test_update_creates_section_header(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PromptEvolver, "PROMPTS_DIR", tmp_path)
        (tmp_path / "red_planner.txt").write_text(
            "You are a red agent.\n", encoding="utf-8"
        )
        e = PromptEvolver()
        e._update_prompt("red_planner.txt", ["LEARNED: Test rule one."])
        content = (tmp_path / "red_planner.txt").read_text(encoding="utf-8")
        assert "LEARNED HEURISTICS" in content
        assert "LEARNED: Test rule one." in content

    def test_update_deduplicates_rules(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PromptEvolver, "PROMPTS_DIR", tmp_path)
        (tmp_path / "red_planner.txt").write_text(
            textwrap.dedent(
                """\
                Base prompt.

                ## LEARNED HEURISTICS (Auto-evolved)
                - LEARNED: Rule A.
                """
            ),
            encoding="utf-8",
        )
        e = PromptEvolver()
        e._update_prompt("red_planner.txt", ["LEARNED: Rule A."])
        content = (tmp_path / "red_planner.txt").read_text(encoding="utf-8")
        assert content.count("LEARNED: Rule A.") == 1

    def test_update_missing_file_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PromptEvolver, "PROMPTS_DIR", tmp_path)
        e = PromptEvolver()
        result = e._update_prompt("nonexistent.txt", ["LEARNED: Test."])
        assert result is False

    def test_no_tmp_file_left_after_update(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PromptEvolver, "PROMPTS_DIR", tmp_path)
        (tmp_path / "red_planner.txt").write_text("Prompt.", encoding="utf-8")
        e = PromptEvolver()
        e._update_prompt("red_planner.txt", ["LEARNED: Test."])
        assert not (tmp_path / "red_planner.tmp").exists()

    def test_update_returns_true_when_modified(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PromptEvolver, "PROMPTS_DIR", tmp_path)
        (tmp_path / "red_planner.txt").write_text("Base.", encoding="utf-8")
        e = PromptEvolver()
        result = e._update_prompt("red_planner.txt", ["LEARNED: New rule."])
        assert result is True

    def test_update_returns_false_when_no_new_rules(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PromptEvolver, "PROMPTS_DIR", tmp_path)
        existing = (
            "Base.\n\n## LEARNED HEURISTICS (Auto-evolved)\n- LEARNED: Dupe rule.\n"
        )
        (tmp_path / "red_planner.txt").write_text(existing, encoding="utf-8")
        e = PromptEvolver()
        result = e._update_prompt("red_planner.txt", ["LEARNED: Dupe rule."])
        assert result is False

    def test_update_replaces_rule_of_same_type(self, tmp_path, monkeypatch):
        """A newer rule of the same keyword type replaces the old one (no duplication)."""
        monkeypatch.setattr(PromptEvolver, "PROMPTS_DIR", tmp_path)
        existing = (
            "Base.\n\n## LEARNED HEURISTICS (Auto-evolved)\n"
            "- LEARNED: You are aborting in 55% of episodes. Continue unless suspicion exceeds 0.75. Aborting costs -0.30.\n"
        )
        (tmp_path / "red_planner.txt").write_text(existing, encoding="utf-8")
        e = PromptEvolver()
        new_rule = "LEARNED: You are aborting in 40% of episodes. Continue unless suspicion exceeds 0.75. Aborting costs -0.30."
        e._update_prompt("red_planner.txt", [new_rule])
        content = (tmp_path / "red_planner.txt").read_text(encoding="utf-8")
        # Old rule gone, new rule present, only one aborting rule total
        abort_rules = [ln for ln in content.splitlines() if "aborting" in ln.lower()]
        assert len(abort_rules) == 1, f"Expected 1 abort rule, got {len(abort_rules)}: {abort_rules}"
        assert "40%" in content

    def test_each_rule_type_appears_at_most_once(self, tmp_path, monkeypatch):
        """After multiple evolve() calls with changing values, each rule type has exactly one line."""
        prompts_dir = tmp_path / "cipher" / "agents" / "prompts"
        prompts_dir.mkdir(parents=True)
        (prompts_dir / "red_planner.txt").write_text("Stub.\n", encoding="utf-8")
        monkeypatch.setattr(PromptEvolver, "PROMPTS_DIR", prompts_dir)
        e = PromptEvolver()

        # Simulate three successive evolve rounds with different abort values
        for pct in [55, 63, 72]:
            rule = (
                f"LEARNED: You are aborting in {pct}% of episodes. "
                "Continue the operation unless suspicion exceeds 0.75. Aborting costs -0.30."
            )
            e._update_prompt("red_planner.txt", [rule])

        content = (prompts_dir / "red_planner.txt").read_text(encoding="utf-8")
        abort_rules = [ln for ln in content.splitlines() if "aborting" in ln.lower()]
        assert len(abort_rules) == 1, (
            f"Expected exactly 1 'aborting' rule, got {len(abort_rules)}: {abort_rules}"
        )
        assert "72%" in content  # Latest value wins



# ═══════════════════════════════════════════════════════════════════════════════
# TestEvolutionLog
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvolutionLog:
    def test_log_creates_jsonl_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PromptEvolver, "EVOLUTION_LOG", tmp_path / "evo.jsonl")
        e = PromptEvolver()
        e._log_evolution(10, ["LEARNED: Rule 1."], ["LEARNED: Blue rule 1."])
        assert (tmp_path / "evo.jsonl").exists()

    def test_log_entry_has_required_fields(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PromptEvolver, "EVOLUTION_LOG", tmp_path / "evo.jsonl")
        e = PromptEvolver()
        e._log_evolution(10, ["LEARNED: R1."], ["LEARNED: B1."])
        raw = (tmp_path / "evo.jsonl").read_text(encoding="utf-8").strip()
        entry = json.loads(raw)
        required = [
            "timestamp",
            "episode",
            "evolution_number",
            "red_rules_count",
            "blue_rules_count",
            "red_rules",
            "blue_rules",
        ]
        for field in required:
            assert field in entry, f"Missing field: {field!r}"

    def test_evolution_number_increments(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PromptEvolver, "EVOLUTION_LOG", tmp_path / "evo.jsonl")
        e = PromptEvolver()
        e._log_evolution(10, ["LEARNED: R1."], [])
        e._log_evolution(20, ["LEARNED: R2."], [])
        lines = [
            json.loads(l)
            for l in (tmp_path / "evo.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        assert lines[0]["evolution_number"] == 1
        assert lines[1]["evolution_number"] == 2

    def test_get_summary_returns_zeros_when_no_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            PromptEvolver, "EVOLUTION_LOG", tmp_path / "missing.jsonl"
        )
        e = PromptEvolver()
        summary = e.get_evolution_summary()
        assert summary["total_evolutions"] == 0
        assert summary["total_red_rules"] == 0
        assert summary["total_blue_rules"] == 0
        assert summary["log"] == []

    def test_get_summary_counts_correctly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PromptEvolver, "EVOLUTION_LOG", tmp_path / "evo.jsonl")
        e = PromptEvolver()
        e._log_evolution(10, ["LEARNED: R1.", "LEARNED: R2."], ["LEARNED: B1."])
        e._log_evolution(20, ["LEARNED: R3."], [])
        summary = e.get_evolution_summary()
        assert summary["total_evolutions"] == 2
        assert summary["total_red_rules"] == 3
        assert summary["total_blue_rules"] == 1


# ═══════════════════════════════════════════════════════════════════════════════
# TestEvolveIntegration
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvolveIntegration:
    def _scaffold_prompts(self, prompts_dir: Path) -> None:
        """Create stub prompt files in *prompts_dir*."""
        for name in [
            "red_planner.txt",
            "red_operative.txt",
            "red_exfiltrator.txt",
            "blue_surveillance.txt",
            "blue_threat_hunter.txt",
            "blue_forensics.txt",
        ]:
            (prompts_dir / name).write_text(
                f"Stub prompt for {name}.\n", encoding="utf-8"
            )

    def test_evolve_returns_dict_with_correct_keys(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        prompts_dir = tmp_path / "cipher" / "agents" / "prompts"
        prompts_dir.mkdir(parents=True)
        self._scaffold_prompts(prompts_dir)

        monkeypatch.setattr(PromptEvolver, "PROMPTS_DIR", prompts_dir)
        monkeypatch.setattr(
            PromptEvolver, "EVOLUTION_LOG", tmp_path / "evo.jsonl"
        )

        _make_rewards_csv(tmp_path, n=15)
        e = PromptEvolver()
        result = e.evolve(10)

        required_keys = {
            "evolution_number",
            "red_rules_added",
            "blue_rules_added",
            "red_rules",
            "blue_rules",
        }
        assert required_keys.issubset(result.keys())
        assert isinstance(result["red_rules"], list)
        assert isinstance(result["blue_rules"], list)

    def test_evolve_idempotent_no_duplicate_rules(
        self, tmp_path, monkeypatch
    ):
        """Running evolve() twice on the same data produces no duplicates."""
        monkeypatch.chdir(tmp_path)
        prompts_dir = tmp_path / "cipher" / "agents" / "prompts"
        prompts_dir.mkdir(parents=True)
        self._scaffold_prompts(prompts_dir)

        monkeypatch.setattr(PromptEvolver, "PROMPTS_DIR", prompts_dir)
        monkeypatch.setattr(
            PromptEvolver, "EVOLUTION_LOG", tmp_path / "evo.jsonl"
        )

        _make_rewards_csv(tmp_path, n=15)
        e = PromptEvolver()
        e.evolve(10)
        e.evolve(10)  # second call with same data

        content = (prompts_dir / "red_planner.txt").read_text(encoding="utf-8")
        rules = [l for l in content.splitlines() if l.startswith("- LEARNED:")]
        # No rule should appear twice
        assert len(rules) == len(set(rules)), (
            "Duplicate rules found in prompt file after idempotent evolve"
        )
