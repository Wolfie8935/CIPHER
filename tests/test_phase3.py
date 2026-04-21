"""
CIPHER Phase 3 Test Suite — LLM Integration Tests.

Tests the LLM integration layer WITHOUT making actual API calls.
All tests run in stub mode (LLM_MODE=stub) by default.

Coverage:
- LLM mode toggle (stub vs live)
- LLM client JSON parsing and fallback
- Base agent observation-to-prompt serialization
- Base agent action parsing from JSON responses
- All 8 agents produce valid actions in stub mode
- Prompt history management and bounds
- Context reset clears prompt history
- Agent reasoning strings are populated
- Team-action validation (RED can't use BLUE actions etc.)
"""
from __future__ import annotations

import json
import os
import pytest

# Force stub mode for all tests
os.environ["LLM_MODE"] = "stub"

from cipher.agents.base_agent import Action, ActionType, BaseAgent, RED_ACTIONS, BLUE_ACTIONS
from cipher.agents.red.planner import RedPlanner
from cipher.agents.red.analyst import RedAnalyst
from cipher.agents.red.operative import RedOperative
from cipher.agents.red.exfiltrator import RedExfiltrator
from cipher.agents.blue.surveillance import BlueSurveillance
from cipher.agents.blue.threat_hunter import BlueThreatHunter
from cipher.agents.blue.deception_architect import BlueDeceptionArchitect
from cipher.agents.blue.forensics import BlueForensics
from cipher.environment.graph import generate_enterprise_graph, get_entry_points, get_high_value_target
from cipher.environment.observation import generate_red_observation, generate_blue_observation
from cipher.environment.state import EpisodeState
from cipher.utils.config import config
from cipher.utils.llm_mode import is_live_mode


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def graph():
    """Generate a test graph with fixed seed."""
    return generate_enterprise_graph(n_nodes=50, honeypot_density=0.15, seed=42)


@pytest.fixture
def state(graph):
    """Create a test episode state."""
    entry_points = get_entry_points(graph)
    start = entry_points[0] if entry_points else 0
    return EpisodeState(
        graph=graph,
        red_current_node=start,
        red_visited_nodes=[start],
    )


@pytest.fixture
def red_obs(state):
    """Generate a RED observation from the test state."""
    return generate_red_observation(state)


@pytest.fixture
def blue_obs(state):
    """Generate a BLUE observation from the test state."""
    return generate_blue_observation(state)


# ── LLM Mode Toggle Tests ────────────────────────────────────────


class TestLLMMode:
    """Test the LLM mode toggle."""

    def test_stub_mode_is_default(self):
        """LLM_MODE=stub should be the default."""
        assert not is_live_mode()

    def test_stub_mode_explicit(self):
        """Setting LLM_MODE=stub explicitly should return False."""
        os.environ["LLM_MODE"] = "stub"
        assert not is_live_mode()

    def test_live_mode_toggle(self):
        """Setting LLM_MODE=live should return True."""
        original = os.environ.get("LLM_MODE", "stub")
        try:
            os.environ["LLM_MODE"] = "live"
            assert is_live_mode()
        finally:
            os.environ["LLM_MODE"] = original

    def test_case_insensitive(self):
        """Mode toggle should be case insensitive."""
        original = os.environ.get("LLM_MODE", "stub")
        try:
            os.environ["LLM_MODE"] = "LIVE"
            assert is_live_mode()
            os.environ["LLM_MODE"] = "STUB"
            assert not is_live_mode()
        finally:
            os.environ["LLM_MODE"] = original


# ── JSON Parsing Tests ────────────────────────────────────────────


class TestJSONParsing:
    """Test the LLM client's JSON handling."""

    def test_strip_json_fences(self):
        """Should strip markdown code fences from JSON."""
        from cipher.utils.llm_client import LLMClient

        raw = '```json\n{"action_type": "wait"}\n```'
        result = LLMClient._strip_json_fences(raw)
        parsed = json.loads(result)
        assert parsed["action_type"] == "wait"

    def test_strip_plain_fences(self):
        """Should strip plain ``` fences."""
        from cipher.utils.llm_client import LLMClient

        raw = '```\n{"action_type": "move", "target_node": 5}\n```'
        result = LLMClient._strip_json_fences(raw)
        parsed = json.loads(result)
        assert parsed["action_type"] == "move"
        assert parsed["target_node"] == 5

    def test_no_fences_passthrough(self):
        """JSON without fences should pass through unchanged."""
        from cipher.utils.llm_client import LLMClient

        raw = '{"action_type": "wait", "reasoning": "test"}'
        result = LLMClient._strip_json_fences(raw)
        assert result == raw

    def test_fallback_action_is_valid_json(self):
        """Fallback action should be valid JSON with all required fields."""
        from cipher.utils.llm_client import LLMClient

        fallback = LLMClient._fallback_action()
        parsed = json.loads(fallback)
        assert parsed["action_type"] == "wait"
        assert "target_node" in parsed
        assert "reasoning" in parsed


# ── Action Parsing Tests ──────────────────────────────────────────


class TestActionParsing:
    """Test BaseAgent._parse_action_from_response."""

    def test_valid_red_action(self, red_obs):
        """Valid RED JSON should parse correctly."""
        agent = RedPlanner("test_planner", config)
        agent.observe(red_obs)

        response = json.dumps({
            "action_type": "move",
            "target_node": 5,
            "target_file": None,
            "reasoning": "Moving to next zone."
        })
        action = agent._parse_action_from_response(response)
        assert action.action_type == ActionType.MOVE
        assert action.target_node == 5
        assert action.reasoning == "Moving to next zone."

    def test_valid_blue_action(self, blue_obs):
        """Valid BLUE JSON should parse correctly."""
        agent = BlueSurveillance("test_surv", config)
        agent.observe(blue_obs)

        response = json.dumps({
            "action_type": "investigate_node",
            "target_node": 10,
            "reasoning": "Anomaly detected."
        })
        action = agent._parse_action_from_response(response)
        assert action.action_type == ActionType.INVESTIGATE_NODE
        assert action.target_node == 10

    def test_invalid_json_falls_back(self, red_obs):
        """Invalid JSON should produce WAIT for RED."""
        agent = RedPlanner("test_planner", config)
        agent.observe(red_obs)

        action = agent._parse_action_from_response("this is not json at all")
        assert action.action_type == ActionType.WAIT
        assert "parse failure" in action.reasoning.lower()

    def test_blue_invalid_json_falls_back(self, blue_obs):
        """Invalid JSON should produce STAND_DOWN for BLUE."""
        agent = BlueSurveillance("test_surv", config)
        agent.observe(blue_obs)

        action = agent._parse_action_from_response("{malformed")
        assert action.action_type == ActionType.STAND_DOWN

    def test_unknown_action_type_falls_back(self, red_obs):
        """Unknown action_type should fall back to team default."""
        agent = RedPlanner("test_planner", config)
        agent.observe(red_obs)

        response = json.dumps({
            "action_type": "fly_to_moon",
            "target_node": None,
            "reasoning": "Invalid action."
        })
        action = agent._parse_action_from_response(response)
        assert action.action_type == ActionType.WAIT

    def test_cross_team_action_rejected(self, red_obs):
        """RED agent trying BLUE action should fall back."""
        agent = RedPlanner("test_planner", config)
        agent.observe(red_obs)

        response = json.dumps({
            "action_type": "investigate_node",
            "target_node": 5,
            "reasoning": "Trying BLUE action."
        })
        action = agent._parse_action_from_response(response)
        assert action.action_type == ActionType.WAIT

    def test_target_node_coercion(self, red_obs):
        """String target_node should be coerced to int."""
        agent = RedPlanner("test_planner", config)
        agent.observe(red_obs)

        response = json.dumps({
            "action_type": "move",
            "target_node": "7",
            "reasoning": "Moving."
        })
        action = agent._parse_action_from_response(response)
        assert action.target_node == 7

    def test_null_target_node_is_none(self, red_obs):
        """null target_node should become Python None."""
        agent = RedPlanner("test_planner", config)
        agent.observe(red_obs)

        response = json.dumps({
            "action_type": "wait",
            "target_node": None,
            "reasoning": "Waiting."
        })
        action = agent._parse_action_from_response(response)
        assert action.target_node is None


# ── Observation Serialization Tests ───────────────────────────────


class TestObservationSerialization:
    """Test observation-to-prompt serialization."""

    def test_red_observation_to_text(self, red_obs):
        """RED observation should serialize to readable text."""
        agent = RedPlanner("test_planner", config)
        agent.observe(red_obs)

        text = agent._observation_to_prompt_text()
        assert "STEP" in text
        assert "RED TEAM OBSERVATION" in text
        assert "Current node" in text
        assert "Suspicion" in text

    def test_blue_observation_to_text(self, blue_obs):
        """BLUE observation should serialize to readable text."""
        agent = BlueSurveillance("test_surv", config)
        agent.observe(blue_obs)

        text = agent._observation_to_prompt_text()
        assert "STEP" in text
        assert "BLUE TEAM OBSERVATION" in text
        assert "Detection confidence" in text

    def test_no_observation_text(self):
        """No observation should produce safe text."""
        agent = RedPlanner("test_planner", config)
        text = agent._observation_to_prompt_text()
        assert "No observation" in text


# ── Agent Stub Behavior Tests ─────────────────────────────────────


class TestRedAgentStubs:
    """Test all RED agents produce valid actions in stub mode."""

    def test_red_planner_acts(self, red_obs):
        agent = RedPlanner("red_planner_01", config)
        agent.observe(red_obs)
        action = agent.act()
        assert isinstance(action, Action)
        assert action.action_type in RED_ACTIONS
        assert action.agent_id == "red_planner_01"
        assert len(action.reasoning) > 0

    def test_red_analyst_acts(self, red_obs):
        agent = RedAnalyst("red_analyst_01", config)
        agent.observe(red_obs)
        action = agent.act()
        assert isinstance(action, Action)
        assert action.action_type in RED_ACTIONS
        assert len(action.reasoning) > 0

    def test_red_operative_acts(self, red_obs):
        agent = RedOperative("red_operative_01", config)
        agent.observe(red_obs)
        action = agent.act()
        assert isinstance(action, Action)
        assert action.action_type in RED_ACTIONS
        assert len(action.reasoning) > 0

    def test_red_exfiltrator_acts(self, red_obs):
        agent = RedExfiltrator("red_exfiltrator_01", config)
        agent.observe(red_obs)
        action = agent.act()
        assert isinstance(action, Action)
        assert action.action_type in RED_ACTIONS
        assert len(action.reasoning) > 0


class TestBlueAgentStubs:
    """Test all BLUE agents produce valid actions in stub mode."""

    def test_blue_surveillance_acts(self, blue_obs):
        agent = BlueSurveillance("blue_surveillance_01", config)
        agent.observe(blue_obs)
        action = agent.act()
        assert isinstance(action, Action)
        assert action.action_type in BLUE_ACTIONS
        assert len(action.reasoning) > 0

    def test_blue_threat_hunter_acts(self, blue_obs):
        agent = BlueThreatHunter("blue_threat_hunter_01", config)
        agent.observe(blue_obs)
        action = agent.act()
        assert isinstance(action, Action)
        assert action.action_type in BLUE_ACTIONS
        assert len(action.reasoning) > 0

    def test_blue_deception_architect_acts(self, blue_obs):
        agent = BlueDeceptionArchitect("blue_deception_architect_01", config)
        agent.observe(blue_obs)
        action = agent.act()
        assert isinstance(action, Action)
        assert action.action_type in BLUE_ACTIONS
        assert len(action.reasoning) > 0

    def test_blue_forensics_acts(self, blue_obs):
        agent = BlueForensics("blue_forensics_01", config)
        agent.observe(blue_obs)
        action = agent.act()
        assert isinstance(action, Action)
        assert action.action_type in BLUE_ACTIONS
        assert len(action.reasoning) > 0


# ── Prompt History Tests ──────────────────────────────────────────


class TestPromptHistory:
    """Test prompt history management."""

    def test_history_updates(self, red_obs):
        """Prompt history should grow with _update_prompt_history."""
        agent = RedPlanner("test_planner", config)
        agent._update_prompt_history("user msg", "assistant msg")
        assert len(agent.prompt_history) == 2
        assert agent.prompt_history[0]["role"] == "user"
        assert agent.prompt_history[1]["role"] == "assistant"

    def test_history_bounded(self, red_obs):
        """Prompt history should not exceed MAX_PROMPT_HISTORY * 2."""
        from cipher.agents.base_agent import MAX_PROMPT_HISTORY
        agent = RedPlanner("test_planner", config)

        for i in range(MAX_PROMPT_HISTORY + 10):
            agent._update_prompt_history(f"user {i}", f"assistant {i}")

        assert len(agent.prompt_history) <= MAX_PROMPT_HISTORY * 2

    def test_reset_clears_history(self, red_obs):
        """Reset should clear prompt_history and action_history."""
        agent = RedPlanner("test_planner", config)
        agent.observe(red_obs)
        agent.act()  # produces an action
        agent._update_prompt_history("user", "assistant")

        assert len(agent.action_history) > 0
        assert len(agent.prompt_history) > 0

        agent.reset()
        assert len(agent.action_history) == 0
        assert len(agent.prompt_history) == 0
        assert agent.step_count == 0


# ── Message Building Tests ────────────────────────────────────────


class TestMessageBuilding:
    """Test _build_messages constructs proper message lists."""

    def test_messages_have_system_prompt(self, red_obs):
        """Messages should start with a system prompt."""
        agent = RedPlanner("test_planner", config)
        agent.observe(red_obs)

        messages = agent._build_messages()
        assert len(messages) >= 2  # system + user
        assert messages[0]["role"] == "system"
        assert len(messages[0]["content"]) > 50  # Non-trivial prompt

    def test_messages_end_with_user(self, red_obs):
        """Messages should end with the current observation as user."""
        agent = RedPlanner("test_planner", config)
        agent.observe(red_obs)

        messages = agent._build_messages()
        assert messages[-1]["role"] == "user"
        assert "RED TEAM OBSERVATION" in messages[-1]["content"]

    def test_blue_messages_have_observation(self, blue_obs):
        """BLUE messages should contain BLUE observation."""
        agent = BlueSurveillance("test_surv", config)
        agent.observe(blue_obs)

        messages = agent._build_messages()
        assert messages[-1]["role"] == "user"
        assert "BLUE TEAM OBSERVATION" in messages[-1]["content"]


# ── Model Key Tests ───────────────────────────────────────────────


class TestModelKeys:
    """Test that each agent has a valid _model_env_key."""

    def test_all_agents_have_model_keys(self):
        """Every agent should have a non-empty _model_env_key."""
        agents = [
            RedPlanner("t", config),
            RedAnalyst("t", config),
            RedOperative("t", config),
            RedExfiltrator("t", config),
            BlueSurveillance("t", config),
            BlueThreatHunter("t", config),
            BlueDeceptionArchitect("t", config),
            BlueForensics("t", config),
        ]
        for agent in agents:
            assert agent._model_env_key, f"{agent.__class__.__name__} has no _model_env_key"
            # Verify the key exists in config
            assert hasattr(config, agent._model_env_key), \
                f"{agent._model_env_key} not found in CipherConfig"


# ── Multi-Step Integration Tests ──────────────────────────────────


class TestMultiStepIntegration:
    """Test agents across multiple observation-act cycles."""

    def test_red_multi_step(self, state):
        """RED agents should handle multiple observe-act cycles."""
        agent = RedPlanner("red_planner_01", config)

        for step in range(1, 6):
            state.step = step
            obs = generate_red_observation(state)
            agent.observe(obs)
            action = agent.act()
            assert isinstance(action, Action)
            assert action.action_type in RED_ACTIONS

        assert agent.step_count == 5
        assert len(agent.action_history) == 5

    def test_blue_multi_step(self, state):
        """BLUE agents should handle multiple observe-act cycles."""
        agent = BlueSurveillance("blue_surv_01", config)

        for step in range(1, 6):
            state.step = step
            obs = generate_blue_observation(state)
            agent.observe(obs)
            action = agent.act()
            assert isinstance(action, Action)
            assert action.action_type in BLUE_ACTIONS

        assert agent.step_count == 5

    def test_full_episode_stub_mode(self, state, graph):
        """Full episode with all 8 agents should complete without error."""
        red_agents = [
            RedPlanner("red_planner_01", config),
            RedAnalyst("red_analyst_01", config),
            RedOperative("red_operative_01", config),
            RedExfiltrator("red_exfiltrator_01", config),
        ]
        blue_agents = [
            BlueSurveillance("blue_surveillance_01", config),
            BlueThreatHunter("blue_threat_hunter_01", config),
            BlueDeceptionArchitect("blue_deception_architect_01", config),
            BlueForensics("blue_forensics_01", config),
        ]

        for step in range(1, 6):
            state.step = step
            red_obs = generate_red_observation(state)
            blue_obs = generate_blue_observation(state)

            for agent in red_agents:
                agent.observe(red_obs)
                action = agent.act()
                assert action.action_type in RED_ACTIONS
                assert len(action.reasoning) > 0

            for agent in blue_agents:
                agent.observe(blue_obs)
                action = agent.act()
                assert action.action_type in BLUE_ACTIONS
                assert len(action.reasoning) > 0


# ── Action Validation Sets ────────────────────────────────────────


class TestActionSets:
    """Test RED_ACTIONS and BLUE_ACTIONS are correct and disjoint."""

    def test_red_actions_complete(self):
        """RED_ACTIONS should contain all red-specific actions."""
        assert ActionType.MOVE in RED_ACTIONS
        assert ActionType.EXFILTRATE in RED_ACTIONS
        assert ActionType.WRITE_DEAD_DROP in RED_ACTIONS
        assert ActionType.READ_DEAD_DROP in RED_ACTIONS
        assert ActionType.WAIT in RED_ACTIONS
        assert ActionType.ABORT in RED_ACTIONS

    def test_blue_actions_complete(self):
        """BLUE_ACTIONS should contain all blue-specific actions."""
        assert ActionType.INVESTIGATE_NODE in BLUE_ACTIONS
        assert ActionType.TRIGGER_ALERT in BLUE_ACTIONS
        assert ActionType.ANALYZE_ANOMALY in BLUE_ACTIONS
        assert ActionType.RECONSTRUCT_PATH in BLUE_ACTIONS
        assert ActionType.STAND_DOWN in BLUE_ACTIONS

    def test_sets_disjoint(self):
        """RED and BLUE action sets should have no overlap."""
        overlap = RED_ACTIONS & BLUE_ACTIONS
        assert len(overlap) == 0, f"Overlapping actions: {overlap}"
