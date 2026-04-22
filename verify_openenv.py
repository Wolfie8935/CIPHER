"""
CIPHER OpenEnv Compliance Verification Script.

Run from the project root:
    python verify_openenv.py

All 7 checks must pass to confirm OpenEnv compliance.
"""
import openenv
from openenv.env.env import Env as _OpenEnvBase

from cipher.env_wrapper import CIPHEREnv, make_env

print("=== CIPHER OpenEnv Compliance Verification ===\n")

# ── Check 1: Inheritance ─────────────────────────────────────────────────────
assert issubclass(CIPHEREnv, _OpenEnvBase), "CIPHEREnv must inherit from openenv.Env"
print("✓ Inherits openenv.Env")

# ── Check 2: Metadata fields ─────────────────────────────────────────────────
assert "name" in CIPHEREnv.metadata, "metadata missing 'name'"
assert "reward_range" in CIPHEREnv.metadata, "metadata missing 'reward_range'"
assert "action_type" in CIPHEREnv.metadata, "metadata missing 'action_type'"
print("✓ Metadata fields present")

# ── Check 3: reset() ─────────────────────────────────────────────────────────
env = CIPHEREnv(max_steps=15, llm_mode="stub")
obs, info = env.reset()
assert isinstance(obs, str) and len(obs) > 10, f"obs must be non-empty str, got: {obs!r}"
assert isinstance(info, dict) and "episode" in info, f"info must be dict with 'episode' key"
print(f"✓ reset() → obs ({len(obs)} chars), info ({len(info)} keys)")

# ── Check 4: step() ──────────────────────────────────────────────────────────
obs2, reward, terminated, truncated, info2 = env.step(
    "MOVE to the nearest auth_gateway node to begin zone traversal"
)
assert isinstance(reward, float), f"reward must be float, got: {type(reward)}"
assert terminated is True, "terminated must be True"
assert isinstance(info2, dict) and "terminal_reason" in info2, "info2 missing 'terminal_reason'"
print(f"✓ step() → reward={reward:+.3f}, terminal={info2['terminal_reason']}")

# ── Check 5: render() ────────────────────────────────────────────────────────
render = env.render()
assert isinstance(render, str), f"render() must return str, got: {type(render)}"
print(f"✓ render() → '{render[:60]}...'")

# ── Check 6: Second reset increments episode ──────────────────────────────────
_, i2 = env.reset()
assert i2["episode"] == 2, f"Expected episode 2, got: {i2['episode']}"
print(f"✓ Second reset() → episode {i2['episode']}")

# ── Check 7: Reward range in metadata ────────────────────────────────────────
print(f"✓ Reward range: {CIPHEREnv.metadata['reward_range']}")

print("\n=== ALL CHECKS PASSED — OpenEnv compliant ===")
