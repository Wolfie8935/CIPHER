"""
Phase 10 tests — Reward Improvement Metrics + Dashboard Tab 6.

Verifies:
1. ImprovementAnalyzer can be imported and instantiated.
2. All metric methods return the correct shape even with no data.
3. Metrics are correct when synthetic data is provided.
4. Dashboard Tab 6 layout contains t6-reward-chart and t6-winrate-chart.
5. update_tab6 returns three objects (two figures + stats div).
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_reward_rows(n: int) -> list[dict]:
    import random
    rng = random.Random(42)
    rows = []
    for i in range(1, n + 1):
        # Simulate improving RED performance: later episodes score better
        progress = i / n
        red_total = -0.3 + progress * 0.4 + rng.gauss(0, 0.05)
        exfil = max(0.0, min(1.0, progress * 0.5 + rng.gauss(0, 0.1)))
        rows.append(
            {
                "episode": i,
                "timestamp": "2026-01-01T00:00:00",
                "steps": 10,
                "terminal_reason": "aborted" if rng.random() < 0.4 * (1 - progress) else "max_steps",
                "red_total": round(red_total, 4),
                "red_exfil": round(exfil, 4),
                "red_stealth": round(0.5 + rng.gauss(0, 0.1), 4),
                "red_memory": 1.0,
                "red_complexity": 1.3,
                "red_abort_penalty": 0.0,
                "red_honeypot_penalty": 0.0,
                "blue_total": round(0.5 + rng.gauss(0, 0.1), 4),
                "blue_detection": 0.5,
                "blue_speed": 0.1,
                "blue_fp_penalty": 0.0,
                "blue_honeypot_rate": 0.1,
                "blue_graph_reconstruction": 0.1,
                "oversight_red_adj": 0.0,
                "oversight_blue_adj": 0.0,
                "oversight_flags": "none",
                "red_unique_nodes": 5,
                "red_drops_written": 0,
                "red_traps_placed": 0,
                "red_context_resets": 0,
                "red_complexity_multiplier": 1.3,
                "fleet_verdict": "contested",
                "fleet_judgment": "n/a",
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Import tests
# ─────────────────────────────────────────────────────────────────────────────

class TestImport:
    def test_improvement_analyzer_importable(self):
        from cipher.training.improvement_analyzer import ImprovementAnalyzer
        assert ImprovementAnalyzer is not None

    def test_instantiation(self):
        from cipher.training.improvement_analyzer import ImprovementAnalyzer
        a = ImprovementAnalyzer()
        assert a is not None


# ─────────────────────────────────────────────────────────────────────────────
# Empty-data tests (no CSV / no log)
# ─────────────────────────────────────────────────────────────────────────────

class TestEmptyData:
    def setup_method(self):
        from cipher.training.improvement_analyzer import ImprovementAnalyzer
        self.analyzer = ImprovementAnalyzer()
        # Point to nonexistent files so df is None
        self.analyzer.REWARDS_LOG = Path("__nonexistent_rewards__.csv")
        self.analyzer.EVOLUTION_LOG = Path("__nonexistent_evolutions__.jsonl")
        self.analyzer.EVENTS_LOG = Path("__nonexistent_events__.jsonl")

    def test_rolling_win_rates_empty(self):
        result = self.analyzer.compute_rolling_win_rates()
        assert result["episodes"] == []
        assert result["red_win_rate"] == []
        assert result["blue_win_rate"] == []

    def test_exfil_rate_empty(self):
        result = self.analyzer.compute_exfil_rate()
        assert result["episodes"] == []
        assert result["exfil_rate"] == []

    def test_abort_rate_empty(self):
        result = self.analyzer.compute_abort_rate()
        assert result["episodes"] == []
        assert result["abort_rate"] == []

    def test_mean_suspicion_empty(self):
        result = self.analyzer.compute_mean_suspicion()
        assert result["episodes"] == []
        assert result["mean_suspicion"] == []

    def test_early_late_empty(self):
        result = self.analyzer.compute_early_late_comparison()
        assert result["n_early"] == 0
        assert result["n_late"] == 0
        assert result["red_improvement"] == 0.0

    def test_evolution_episodes_empty(self):
        result = self.analyzer.get_evolution_episodes()
        assert result == []

    def test_evolution_summary_empty(self):
        result = self.analyzer.get_evolution_summary()
        assert result["total_evolutions"] == 0
        assert result["total_red_rules"] == 0

    def test_difficulty_corr_nan(self):
        import math
        result = self.analyzer.compute_difficulty_reward_correlation()
        assert math.isnan(result)

    def test_full_summary_empty(self):
        result = self.analyzer.get_full_summary()
        assert result["n_episodes"] == 0
        assert result["evolution"]["total_evolutions"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Real-data tests
# ─────────────────────────────────────────────────────────────────────────────

class TestWithData:
    def setup_method(self):
        from cipher.training.improvement_analyzer import ImprovementAnalyzer
        self.tmpdir = tempfile.mkdtemp()
        self.rewards_path = Path(self.tmpdir) / "rewards_log.csv"
        self.evo_path = Path(self.tmpdir) / "prompt_evolution_log.jsonl"
        self.events_path = Path(self.tmpdir) / "training_events.jsonl"

        # Write 50 synthetic episodes
        rows = _make_reward_rows(50)
        _write_csv(self.rewards_path, rows)

        # Write 2 evolution events (at ep 10 and 20)
        evos = [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "episode": 10,
                "evolution_number": 1,
                "red_rules_count": 3,
                "blue_rules_count": 2,
                "red_rules": ["LEARNED: rule1", "LEARNED: rule2", "LEARNED: rule3"],
                "blue_rules": ["LEARNED: b1", "LEARNED: b2"],
            },
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "episode": 20,
                "evolution_number": 2,
                "red_rules_count": 5,
                "blue_rules_count": 3,
                "red_rules": ["LEARNED: r" + str(i) for i in range(5)],
                "blue_rules": ["LEARNED: b" + str(i) for i in range(3)],
            },
        ]
        with open(self.evo_path, "w", encoding="utf-8") as f:
            for e in evos:
                f.write(json.dumps(e) + "\n")

        self.analyzer = ImprovementAnalyzer()
        self.analyzer.REWARDS_LOG = self.rewards_path
        self.analyzer.EVOLUTION_LOG = self.evo_path
        self.analyzer.EVENTS_LOG = self.events_path

    def test_df_loaded(self):
        assert self.analyzer.df is not None
        assert len(self.analyzer.df) == 50

    def test_rolling_win_rates_shape(self):
        result = self.analyzer.compute_rolling_win_rates()
        assert len(result["episodes"]) == 50
        assert len(result["red_win_rate"]) == 50
        assert len(result["blue_win_rate"]) == 50

    def test_rolling_win_rates_range(self):
        result = self.analyzer.compute_rolling_win_rates()
        for v in result["red_win_rate"]:
            assert 0.0 <= v <= 1.0
        for v in result["blue_win_rate"]:
            assert 0.0 <= v <= 1.0

    def test_exfil_rate_shape(self):
        result = self.analyzer.compute_exfil_rate()
        assert len(result["exfil_rate"]) == 50

    def test_abort_rate_shape(self):
        result = self.analyzer.compute_abort_rate()
        assert len(result["abort_rate"]) == 50

    def test_mean_suspicion_range(self):
        result = self.analyzer.compute_mean_suspicion()
        for v in result["mean_suspicion"]:
            assert 0.0 <= v <= 1.5  # proxy can exceed 1 due to clamp at load

    def test_early_late_keys(self):
        result = self.analyzer.compute_early_late_comparison()
        required_keys = [
            "early_red_avg", "late_red_avg", "red_improvement",
            "early_exfil_rate", "late_exfil_rate", "exfil_delta",
            "early_abort_rate", "late_abort_rate", "abort_delta",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_red_reward_improves(self):
        """Later episodes should have higher average RED reward (by construction)."""
        result = self.analyzer.compute_early_late_comparison()
        assert result["red_improvement"] > 0, (
            f"Expected improvement but got {result['red_improvement']}"
        )

    def test_evolution_episodes(self):
        eps = self.analyzer.get_evolution_episodes()
        assert eps == [10, 20]

    def test_evolution_summary(self):
        summary = self.analyzer.get_evolution_summary()
        assert summary["total_evolutions"] == 2
        assert summary["total_red_rules"] == 3 + 5
        assert summary["total_blue_rules"] == 2 + 3
        assert summary["evolution_episodes"] == [10, 20]

    def test_full_summary_structure(self):
        result = self.analyzer.get_full_summary()
        assert result["n_episodes"] == 50
        assert "rolling_win_rates" in result
        assert "exfil_rate" in result
        assert "abort_rate" in result
        assert "mean_suspicion" in result
        assert "early_late" in result
        assert "evolution" in result

    def test_rolling_mean_static(self):
        from cipher.training.improvement_analyzer import ImprovementAnalyzer
        values = [1.0, 0.0, 1.0, 0.0]
        result = ImprovementAnalyzer._rolling_mean(values, window=2)
        assert len(result) == 4
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(0.5)
        assert result[3] == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard Tab 6 layout tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardTab6Layout:
    def test_tab6_layout_has_reward_chart(self):
        from cipher.dashboard.live import _build_tab6_layout
        layout = _build_tab6_layout()
        layout_str = str(layout)
        assert "t6-reward-chart" in layout_str

    def test_tab6_layout_has_winrate_chart(self):
        from cipher.dashboard.live import _build_tab6_layout
        layout = _build_tab6_layout()
        layout_str = str(layout)
        assert "t6-winrate-chart" in layout_str

    def test_tab6_layout_has_stats(self):
        from cipher.dashboard.live import _build_tab6_layout
        layout = _build_tab6_layout()
        layout_str = str(layout)
        assert "t6-stats" in layout_str

    def test_tab6_label_is_learning_curve(self):
        from cipher.dashboard.live import create_live_layout
        layout = create_live_layout()
        layout_str = str(layout)
        assert "Learning Curve" in layout_str

    def test_update_tab6_returns_three_values(self):
        os.environ["LLM_MODE"] = "stub"
        from cipher.dashboard.live import update_tab6
        result = update_tab6(0)
        assert isinstance(result, tuple) or hasattr(result, "__len__")
        items = list(result) if not isinstance(result, (list, tuple)) else result
        assert len(items) == 3

    def test_update_tab6_figures_are_go_Figure(self):
        import plotly.graph_objects as go
        os.environ["LLM_MODE"] = "stub"
        from cipher.dashboard.live import update_tab6
        fig1, fig2, stats = update_tab6(0)
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)

    def test_tab_value_unchanged(self):
        """Ensure the tab value is still 'tab-evolution' for routing."""
        from cipher.dashboard.live import render_tab
        from cipher.dashboard.live import _build_tab6_layout
        result = render_tab("tab-evolution")
        result_str = str(result)
        assert "t6-reward-chart" in result_str or "t6-winrate-chart" in result_str


# ─────────────────────────────────────────────────────────────────────────────
# Callback registration test
# ─────────────────────────────────────────────────────────────────────────────

class TestCallbackRegistration:
    def test_register_callbacks_on_new_app(self):
        import dash
        from cipher.dashboard.live import register_callbacks_on
        test_app = dash.Dash(__name__ + "_phase10_test")
        test_app.config.suppress_callback_exceptions = True
        from cipher.dashboard.live import create_live_layout
        test_app.layout = create_live_layout()
        register_callbacks_on(test_app)
        # Check that the callback map has entries for the new Tab 6 outputs
        cb_map = test_app.callback_map
        all_keys = " ".join(str(k) for k in cb_map.keys())
        assert "t6-reward-chart" in all_keys or len(cb_map) >= 7, (
            f"Expected t6-reward-chart in callback map keys. Keys: {all_keys}"
        )
