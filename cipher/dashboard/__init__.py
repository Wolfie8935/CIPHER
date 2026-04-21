"""
CIPHER dashboard package.

Contains the Plotly Dash web dashboard for live training visualization,
episode replay, and dead drop inspection.

This is STUBBED in Phase 1 — the full implementation is Phase 12–13.
"""
from __future__ import annotations

from cipher.dashboard.app import CipherDashboard


def get_live_dashboard():
    """Lazy import to avoid Dash startup at import time."""
    from cipher.dashboard.live import app as live_app

    return live_app


__all__ = ["CipherDashboard", "get_live_dashboard"]
