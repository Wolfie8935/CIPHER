"""
cipher/utils/llm_mode.py

Controls whether agents use real LLM calls or fall back to stub (random) behavior.

Modes:
  stub   — Random/heuristic actions. No API calls. Instant. Default.
  live   — All 8 agents use NVIDIA NIM API.
  hybrid — RED Planner uses local fine-tuned LoRA. Other 7 agents use NVIDIA NIM.
"""
from __future__ import annotations
import os


def get_llm_mode() -> str:
    """Returns the current LLM mode string: 'stub', 'live', or 'hybrid'."""
    env_mode = os.getenv("LLM_MODE", "").strip().lower()
    if env_mode:
        return env_mode
    try:
        from cipher.utils.config import config

        cfg_mode = str(getattr(config, "llm_mode", "stub")).strip().lower()
        return cfg_mode or "stub"
    except Exception:
        return "stub"


def is_live_mode() -> bool:
    """
    Returns True if agents should use the NVIDIA NIM API.
    True for both 'live' and 'hybrid' modes — in hybrid, non-specialist
    agents still use NIM.
    """
    return get_llm_mode() in ("live", "hybrid")


def is_hybrid_mode() -> bool:
    """Returns True if LLM_MODE=hybrid — specialist agents use local LoRA."""
    return get_llm_mode() == "hybrid"


def is_stub_mode() -> bool:
    """Returns True if LLM_MODE=stub (default)."""
    return get_llm_mode() == "stub"


def require_live_mode() -> None:
    """Raises if not in live or hybrid mode."""
    if not is_live_mode():
        raise RuntimeError(
            "LLM_MODE is not 'live' or 'hybrid'. "
            "Set LLM_MODE=live or LLM_MODE=hybrid to enable real API calls."
        )

