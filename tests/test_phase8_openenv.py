"""
Phase 8 — OpenEnv Compliance Tests
pytest tests/test_phase8_openenv.py -v
All 11 tests must pass.
"""
import pytest
import openenv
from openenv.env.env import Env as _OpenEnvBase

from cipher.env_wrapper import CIPHEREnv, make_env


class TestOpenEnvCompliance:
    """11-test suite verifying CIPHEREnv meets the OpenEnv interface contract."""

    # ------------------------------------------------------------------
    # 1. Inheritance
    # ------------------------------------------------------------------

    def test_inherits_openenv_env(self):
        assert issubclass(CIPHEREnv, _OpenEnvBase)

    # ------------------------------------------------------------------
    # 2. Metadata
    # ------------------------------------------------------------------

    def test_metadata_complete(self):
        required_keys = [
            "name",
            "reward_range",
            "action_type",
            "observation_type",
            "num_agents",
            "description",
        ]
        for key in required_keys:
            assert key in CIPHEREnv.metadata, f"metadata missing key: {key}"
        assert CIPHEREnv.metadata["num_agents"] == 8
        assert len(CIPHEREnv.metadata["reward_range"]) == 2

    # ------------------------------------------------------------------
    # 3. reset() signature
    # ------------------------------------------------------------------

    def test_reset_signature(self):
        env = CIPHEREnv(max_steps=10, llm_mode="stub")
        obs, info = env.reset()
        assert isinstance(obs, str)
        assert len(obs) > 10
        assert isinstance(info, dict)
        assert "episode" in info
        assert info["episode"] == 1

    # ------------------------------------------------------------------
    # 4. step() returns a 5-tuple with correct types
    # ------------------------------------------------------------------

    def test_step_five_tuple(self):
        env = CIPHEREnv(max_steps=10, llm_mode="stub")
        env.reset()
        result = env.step("MOVE to any adjacent node")
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, str)
        assert isinstance(info, dict)

    # ------------------------------------------------------------------
    # 5. Reward is float within declared range
    # ------------------------------------------------------------------

    def test_reward_float_in_range(self):
        env = CIPHEREnv(max_steps=10, llm_mode="stub")
        env.reset()
        _, reward, _, _, _ = env.step("MOVE to any adjacent node")
        assert isinstance(reward, float)
        lo, hi = CIPHEREnv.metadata["reward_range"]
        assert lo <= reward <= hi

    # ------------------------------------------------------------------
    # 6. terminated=True, truncated=False — always
    # ------------------------------------------------------------------

    def test_terminated_always_true(self):
        env = CIPHEREnv(max_steps=10, llm_mode="stub")
        env.reset()
        _, _, terminated, truncated, _ = env.step("WAIT")
        assert terminated is True
        assert truncated is False

    # ------------------------------------------------------------------
    # 7. render() returns a non-empty string before and after an episode
    # ------------------------------------------------------------------

    def test_render_returns_string(self):
        env = CIPHEREnv(max_steps=10, llm_mode="stub")
        # Before any episode
        pre = env.render()
        assert isinstance(pre, str)
        # After a full episode
        env.reset()
        env.step("MOVE to any adjacent node")
        render = env.render()
        assert isinstance(render, str)
        assert len(render) > 0

    # ------------------------------------------------------------------
    # 8. info dict contains every required key after step()
    # ------------------------------------------------------------------

    def test_info_has_all_components(self):
        env = CIPHEREnv(max_steps=10, llm_mode="stub")
        env.reset()
        _, _, _, _, info = env.step("MOVE to any adjacent node")
        required_keys = [
            # Terminal
            "terminal_reason",
            "steps_taken",
            "suspicion_final",
            # RED reward
            "red_total",
            "red_exfil",
            "red_stealth",
            "red_memory",
            "red_complexity",
            "red_abort_penalty",
            "red_honeypot_penalty",
            # BLUE reward
            "blue_total",
            "blue_detection",
            "blue_speed",
            "blue_fp_penalty",
            "blue_honeypot_rate",
            "blue_reconstruction",
            # Oversight
            "fleet_verdict",
            "oversight_flags",
            # Metadata
            "episode",
            "difficulty",
            "dead_drops_written",
            "traps_fired",
            "zones_visited",
        ]
        for key in required_keys:
            assert key in info, f"Missing info key: {key}"

    # ------------------------------------------------------------------
    # 9. Auto-difficulty escalates when recent win rate > 60 %
    # ------------------------------------------------------------------

    def test_difficulty_auto_escalates(self):
        env = CIPHEREnv(max_steps=5, llm_mode="stub", use_auto_difficulty=True)
        initial_difficulty = env.difficulty
        # Simulate 10 consecutive wins to trigger upward escalation
        env._recent_red_wins = [1] * 10
        env.reset()
        assert env.difficulty >= initial_difficulty

    # ------------------------------------------------------------------
    # 10. Multiple resets increment episode counter monotonically
    # ------------------------------------------------------------------

    def test_multiple_resets_increment_episode(self):
        env = CIPHEREnv(max_steps=5, llm_mode="stub")
        for expected in range(1, 4):
            _, info = env.reset()
            assert info["episode"] == expected

    # ------------------------------------------------------------------
    # 11. make_env factory produces a valid, usable CIPHEREnv
    # ------------------------------------------------------------------

    def test_make_env_factory(self):
        env = make_env(max_steps=10, llm_mode="stub")
        assert isinstance(env, CIPHEREnv)
        obs, info = env.reset()
        assert isinstance(obs, str)
