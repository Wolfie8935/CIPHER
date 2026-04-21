"""
cipher/utils/llm_mode.py

Controls whether agents use real LLM calls or fall back to stub (random) behavior.
Set LLM_MODE=stub in .env to use stubs (no API calls, no cost, instant).
Set LLM_MODE=live to use real NVIDIA API calls.
Default: stub (safe default, must explicitly opt into live mode).
"""
from __future__ import annotations
import os


def is_live_mode() -> bool:
    """Returns True if LLM_MODE=live in .env. False (stub mode) by default."""
    return os.getenv("LLM_MODE", "stub").lower() == "live"


def require_live_mode() -> None:
    """Raises if not in live mode. Call this at the top of LLM-dependent paths."""
    if not is_live_mode():
        raise RuntimeError(
            "LLM_MODE is not 'live'. Set LLM_MODE=live in .env to enable real API calls."
        )
