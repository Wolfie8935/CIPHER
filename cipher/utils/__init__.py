"""
CIPHER utilities package.

Contains shared infrastructure: configuration loader, logger, and LLM client.
These are the only modules that interact with external systems (env vars, APIs).

This package does NOT contain domain logic — only infrastructure.
"""
from __future__ import annotations
