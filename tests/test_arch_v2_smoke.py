"""End-to-end smoke test for the v2 commander+subagent architecture.

Runs a short stub-mode episode with CIPHER_AGENT_ARCH=v2 and asserts:
* the runner does not crash
* the trace JSON contains commander metadata
* the canonical 4 RED specialists end up alive in the registry
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture
def force_v2(monkeypatch):
    monkeypatch.setenv("CIPHER_AGENT_ARCH", "v2")
    monkeypatch.setenv("LLM_MODE", "stub")
    yield


def test_v2_episode_runs_and_writes_trace(force_v2, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Re-import config so it picks up the env vars in this test process.
    import importlib
    import cipher.utils.config as cfgmod
    importlib.reload(cfgmod)

    from cipher.training._episode_runner import run_episode

    result = run_episode(
        episode_number=1,
        max_steps=15,
        verbose=False,
        save_trace=True,
        cfg=cfgmod.config,
    )
    # run_episode returns a tuple (red_total, blue_total)
    red_total, blue_total = result
    assert isinstance(red_total, float)
    assert isinstance(blue_total, float)

    traces_dir = tmp_path / "episode_traces"
    files = list(traces_dir.glob("*.json"))
    assert files, "expected at least one episode trace"

    with files[0].open(encoding="utf-8") as f:
        trace = json.load(f)
    assert "commanders" in trace
    assert trace["commanders"].get("arch") == "v2"
    red_meta = trace["commanders"].get("red_commander", {})
    blue_meta = trace["commanders"].get("blue_commander", {})
    assert red_meta.get("agent_id") == "red_commander_01"
    assert blue_meta.get("agent_id") == "blue_commander_01"

    # The default roster must have spawned the canonical RED specialists.
    final_roles = {r.get("role") for r in red_meta.get("final_roster", [])}
    assert {"planner", "analyst", "operative", "exfiltrator"}.issubset(final_roles)
