import os
import sys
import pytest
from cipher.environment.scenario import ScenarioGenerator
from cipher.training._episode_runner import run_episode
from cipher.utils.config import config

os.environ["LLM_MODE"] = "stub"
gen = ScenarioGenerator()
scenario, graph = gen.generate(1)
result = run_episode(scenario, graph, config, max_steps=30, verbose=False)
state = result["state"]
print("BLUE INTEGRITY REPORTS:", getattr(state, "blue_integrity_reports", []))
