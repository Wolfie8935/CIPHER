"""
Centralized logger for CIPHER.

Provides colorized, structured terminal output using the `rich` library.
RED team actions log in red. BLUE team actions log in blue.
System events log in white. Rewards log in yellow. Errors log in bright red.

Owns: log formatting, color coding, and logger factory.
Does NOT own: any domain logic or file I/O beyond log output.
"""
from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ── Custom theme for CIPHER log output ────────────────────────
_CIPHER_THEME = Theme(
    {
        "red_team": "bold red",
        "blue_team": "bold blue",
        "system": "bold white",
        "reward": "bold yellow",
        "error": "bold bright_red",
        "memento": "bold magenta",
        "dead_drop": "bold cyan",
    }
)

# ── Shared console instance ──────────────────────────────────
console = Console(theme=_CIPHER_THEME, stderr=True)


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger with rich colorized output.

    Args:
        name: Logger name (e.g. 'cipher.agents.red.planner').
        level: Logging level. Defaults to DEBUG.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        handler.setLevel(level)
        fmt = logging.Formatter("%(name)s — %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger


def log_red(logger: logging.Logger, message: str) -> None:
    """Log a message styled as RED team output."""
    logger.info(f"[red_team]🔴 {message}[/red_team]")


def log_blue(logger: logging.Logger, message: str) -> None:
    """Log a message styled as BLUE team output."""
    logger.info(f"[blue_team]🔵 {message}[/blue_team]")


def log_system(logger: logging.Logger, message: str) -> None:
    """Log a system-level event."""
    logger.info(f"[system]⚙️  {message}[/system]")


def log_reward(logger: logging.Logger, message: str) -> None:
    """Log a reward computation event."""
    logger.info(f"[reward]💰 {message}[/reward]")


def log_memento(logger: logging.Logger, message: str) -> None:
    """Log a MEMENTO layer event (context resets, dead drops)."""
    logger.info(f"[memento]🧠 {message}[/memento]")


def log_error(logger: logging.Logger, message: str) -> None:
    """Log an error with bright red styling."""
    logger.error(f"[error]❌ {message}[/error]")
