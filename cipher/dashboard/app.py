"""
CIPHER Dashboard — Plotly Dash web application.

Provides live training visualization, episode replay, dead drop inspection,
and oversight feed display.

STUBBED in Phase 1 — the full implementation is Phase 12–13.
This file defines the app skeleton so imports work and the module is loadable.

Owns: dashboard layout, callbacks, and visualization rendering.
Does NOT own: data generation, training logic, or episode execution.
"""
from __future__ import annotations

from cipher.utils.config import config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class CipherDashboard:
    """
    Stubbed dashboard for CIPHER.

    In Phase 1, this class exists only to satisfy import requirements.
    Phase 12–13 will build the full Plotly Dash application with:
    - Tab 1: Dual reward curves
    - Tab 2: Dead drop inspector
    - Tab 3: Deception map
    - Tab 4: Oversight feed
    - Tab 5: Episode replay
    - Tab 6: Scenario difficulty curve
    """

    def __init__(self) -> None:
        self.port: int = config.dashboard_port
        self.update_interval: int = config.dashboard_live_update_interval
        logger.debug(
            f"CipherDashboard initialized (stub) — port={self.port}"
        )

    def run(self) -> None:
        """
        Launch the dashboard server.

        In Phase 1, prints a message indicating the dashboard is not yet
        implemented.
        """
        logger.info(
            f"Dashboard stub — full implementation in Phase 12–13. "
            f"Would serve on port {self.port}."
        )
