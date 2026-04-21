"""
tests/test_phase13.py
Phase 13 - Live Training Dashboard tests.
No Selenium. No browser. Tests cover config, file I/O,
data helpers, moving average, and app structure.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("LLM_MODE", "stub")


class TestPhase13Config:
    def test_dashboard_live_port_exists(self):
        from cipher.utils.config import config

        assert hasattr(config, "dashboard_live_port")
        assert isinstance(config.dashboard_live_port, int)

    def test_dashboard_live_port_default(self):
        from cipher.utils.config import config

        assert config.dashboard_live_port == 8051

    def test_update_interval_exists(self):
        from cipher.utils.config import config

        assert hasattr(config, "dashboard_live_update_interval")
        assert config.dashboard_live_update_interval == 2000

    def test_no_port_conflict_with_phase12(self):
        from cipher.utils.config import config

        assert config.dashboard_live_port != 8050


class TestTrainingFileWrites:
    def test_append_event_creates_file(self, tmp_path, monkeypatch):
        import cipher.training.loop as tl

        monkeypatch.setattr(tl, "EVENTS_FILE", tmp_path / "ev.jsonl")
        tl._append_training_event({"episode": 1, "event_type": "episode_start"})
        assert (tmp_path / "ev.jsonl").exists()

    def test_append_event_valid_jsonl(self, tmp_path, monkeypatch):
        import cipher.training.loop as tl

        monkeypatch.setattr(tl, "EVENTS_FILE", tmp_path / "ev.jsonl")
        for idx in range(3):
            tl._append_training_event({"episode": idx + 1, "event_type": "trap_fired"})
        lines = (tmp_path / "ev.jsonl").read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["episode"] == 1

    def test_append_event_never_raises(self, monkeypatch):
        import cipher.training.loop as tl

        monkeypatch.setattr(tl, "EVENTS_FILE", Path("/bad/path/ev.jsonl"))
        tl._append_training_event({"episode": 1})

    def test_write_state_creates_file(self, tmp_path, monkeypatch):
        import cipher.training.loop as tl

        monkeypatch.setattr(tl, "STATE_FILE", tmp_path / "state.json")
        tl._write_training_state({"status": "running", "current_episode": 1})
        assert (tmp_path / "state.json").exists()

    def test_write_state_valid_json(self, tmp_path, monkeypatch):
        import cipher.training.loop as tl

        monkeypatch.setattr(tl, "STATE_FILE", tmp_path / "state.json")
        tl._write_training_state({"status": "running", "current_episode": 5})
        data = json.loads((tmp_path / "state.json").read_text())
        assert data["status"] == "running"
        assert data["current_episode"] == 5

    def test_write_state_atomic_no_tmp_left(self, tmp_path, monkeypatch):
        import cipher.training.loop as tl

        monkeypatch.setattr(tl, "STATE_FILE", tmp_path / "state.json")
        tl._write_training_state({"status": "idle"})
        assert not (tmp_path / "state.tmp").exists()

    def test_write_state_never_raises(self, monkeypatch):
        import cipher.training.loop as tl

        monkeypatch.setattr(tl, "STATE_FILE", Path("/bad/path/state.json"))
        tl._write_training_state({"status": "running"})


class TestDataLoaders:
    def test_load_csv_none_when_missing(self, tmp_path, monkeypatch):
        import cipher.dashboard.live as live

        monkeypatch.chdir(tmp_path)
        assert live._load_rewards_csv() is None

    def test_load_csv_returns_df(self, tmp_path, monkeypatch):
        import pandas as pd
        import cipher.dashboard.live as live

        monkeypatch.chdir(tmp_path)
        pd.DataFrame(
            [
                {
                    "episode": 1,
                    "red_total": -0.3,
                    "blue_total": 0.5,
                    "terminal_reason": "aborted",
                    "steps": 10,
                    "fleet_verdict": "contested",
                    "fleet_judgment": "test",
                    "oversight_flags": "none",
                }
            ]
        ).to_csv(tmp_path / "rewards_log.csv", index=False)
        frame = live._load_rewards_csv()
        assert frame is not None and len(frame) == 1

    def test_load_events_empty_when_missing(self, tmp_path, monkeypatch):
        import cipher.dashboard.live as live

        monkeypatch.chdir(tmp_path)
        assert live._load_training_events() == []

    def test_load_events_parses_jsonl(self, tmp_path, monkeypatch):
        import cipher.dashboard.live as live

        monkeypatch.chdir(tmp_path)
        lines = [json.dumps({"episode": idx + 1, "event_type": "trap_fired"}) for idx in range(3)]
        (tmp_path / "training_events.jsonl").write_text("\n".join(lines) + "\n")
        result = live._load_training_events()
        assert len(result) == 3

    def test_load_events_skips_bad_lines(self, tmp_path, monkeypatch):
        import cipher.dashboard.live as live

        monkeypatch.chdir(tmp_path)
        (tmp_path / "training_events.jsonl").write_text(
            '{"episode": 1, "event_type": "trap_fired"}\n'
            "bad line\n"
            '{"episode": 2, "event_type": "episode_end"}\n'
        )
        result = live._load_training_events()
        assert len(result) == 2

    def test_load_state_idle_when_missing(self, tmp_path, monkeypatch):
        import cipher.dashboard.live as live

        monkeypatch.chdir(tmp_path)
        assert live._load_training_state()["status"] == "idle"

    def test_load_state_parses_json(self, tmp_path, monkeypatch):
        import cipher.dashboard.live as live

        monkeypatch.chdir(tmp_path)
        (tmp_path / "training_state.json").write_text(
            json.dumps({"status": "running", "current_episode": 42})
        )
        result = live._load_training_state()
        assert result["status"] == "running"
        assert result["current_episode"] == 42


class TestMovingAverage:
    def test_empty_input(self):
        from cipher.dashboard.live import _moving_average

        assert _moving_average([]) == []

    def test_single_value(self):
        from cipher.dashboard.live import _moving_average

        assert _moving_average([0.5])[0] == pytest.approx(0.5)

    def test_window_10_last_value(self):
        from cipher.dashboard.live import _moving_average

        vals = list(range(1, 21))
        result = _moving_average(vals, window=10)
        assert result[-1] == pytest.approx(15.5)

    def test_first_value_unchanged(self):
        from cipher.dashboard.live import _moving_average

        result = _moving_average([0.3, 0.5, 0.7], window=10)
        assert result[0] == pytest.approx(0.3)

    def test_same_length_as_input(self):
        from cipher.dashboard.live import _moving_average

        vals = [1.0] * 15
        assert len(_moving_average(vals, window=5)) == 15


class TestDashboardApp:
    def test_live_app_importable(self):
        from cipher.dashboard.live import app

        assert app is not None

    def test_live_app_has_layout(self):
        from cipher.dashboard.live import app

        assert app.layout is not None

    def test_interval_in_layout(self):
        from cipher.dashboard.live import app

        assert "interval-component" in str(app.layout)

    def test_get_live_dashboard(self):
        from cipher.dashboard import get_live_dashboard

        assert get_live_dashboard() is not None

    def test_phase12_still_works(self):
        from cipher.dashboard.app import CipherDashboard

        assert CipherDashboard is not None

    def test_callbacks_registered(self):
        from cipher.dashboard.live import app

        assert len(app.callback_map) >= 5
