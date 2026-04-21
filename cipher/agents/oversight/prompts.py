"""
cipher/agents/oversight/prompts.py

Prompt constants for OversightAuditor.
Currently the system prompt lives in auditor.py directly.
This file is a placeholder for Phase 8 prompt iteration.
"""

# Re-export from auditor for convenience
from cipher.agents.oversight.auditor import OversightAuditor

OVERSIGHT_SYSTEM_PROMPT = OversightAuditor.SYSTEM_PROMPT

__all__ = ["OVERSIGHT_SYSTEM_PROMPT"]
