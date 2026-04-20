"""
Base agent class and action vocabulary for CIPHER.

Defines the abstract BaseAgent that all 8 agents (4 RED, 4 BLUE) inherit from,
the Action model representing a single agent action, and the ActionType enum
enumerating all valid actions.

Owns: agent interface contract, action schema, prompt loading.
Does NOT own: specific agent behavior (that's each subclass), environment state,
or reward computation.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Union

from pydantic import BaseModel, Field

from cipher.environment.observation import BlueObservation, RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class ActionType(str, Enum):
    """All valid action types in CIPHER, for both RED and BLUE teams."""

    # ── RED actions ──────────────────────────────────────────────
    MOVE = "move"
    READ_FILE = "read_file"
    EXFILTRATE = "exfiltrate"
    WRITE_DEAD_DROP = "write_dead_drop"
    READ_DEAD_DROP = "read_dead_drop"
    PLANT_TRAP = "plant_trap"   # Stub — Phase 5
    WAIT = "wait"
    ABORT = "abort"

    # ── BLUE actions ─────────────────────────────────────────────
    INVESTIGATE_NODE = "investigate_node"
    PLACE_HONEYPOT = "place_honeypot"   # Stub — Phase 5
    TRIGGER_ALERT = "trigger_alert"
    ANALYZE_ANOMALY = "analyze_anomaly"
    RECONSTRUCT_PATH = "reconstruct_path"
    DEPLOY_TRAP = "deploy_trap"   # Stub — Phase 5
    STAND_DOWN = "stand_down"


class Action(BaseModel):
    """
    A single action taken by any CIPHER agent.

    All agent decisions are represented as Action objects. The training loop
    dispatches actions to the environment based on action_type.
    """

    agent_id: str
    action_type: ActionType
    target_node: int | None = None
    target_file: str | None = None
    dead_drop_content: dict[str, Any] | None = None
    reasoning: str = ""
    step: int = 0


# ── Prompt templates directory ────────────────────────────────
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


class BaseAgent(ABC):
    """
    Abstract base class for all CIPHER agents.

    Every agent — RED or BLUE — implements this interface.
    The training loop only ever calls observe(), act(), and reset().
    Everything else is agent-internal.
    """

    def __init__(
        self,
        agent_id: str,
        team: str,
        role: str,
        config: CipherConfig,
    ) -> None:
        """
        Initialize a CIPHER agent.

        Args:
            agent_id: Unique identifier (e.g. 'red_planner_01').
            team: Team affiliation ('red' | 'blue' | 'oversight').
            role: Agent role name (e.g. 'planner', 'analyst').
            config: The global CipherConfig instance.
        """
        self.agent_id = agent_id
        self.team = team
        self.role = role
        self.config = config
        self.action_history: list[Action] = []
        self.prompt_history: list[dict[str, str]] = []
        self.step_count: int = 0
        self._system_prompt: str = self._load_system_prompt()
        self._current_observation: RedObservation | BlueObservation | None = None

        logger.debug(f"Agent initialized: {agent_id} (team={team}, role={role})")

    @abstractmethod
    def observe(self, observation: RedObservation | BlueObservation) -> None:
        """
        Process the latest observation. Store it for use in act().

        Args:
            observation: The team-specific observation for this step.
        """

    @abstractmethod
    def act(self) -> Action:
        """
        Decide on and return an action based on the current observation.

        Returns:
            An Action object representing the agent's decision.
        """

    def reset(self) -> None:
        """
        Hard context reset. Clears prompt_history and action_history.

        Called by the MEMENTO layer when a context reset occurs.
        After reset, the agent has NO memory of prior steps.
        The only information available is: current observation + dead drop contents.
        """
        self.prompt_history = []
        self.action_history = []
        self.step_count = 0
        self._current_observation = None
        logger.debug(f"Agent {self.agent_id} context reset")

    def _load_system_prompt(self) -> str:
        """
        Load the prompt template from agents/prompts/<team>_<role>.txt.

        Returns:
            The system prompt text, or a fallback message if file not found.
        """
        # Map role names to prompt file names
        prompt_filename = f"{self.team}_{self.role}.txt"
        prompt_path = _PROMPTS_DIR / prompt_filename

        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")

        logger.warning(
            f"Prompt file not found: {prompt_path}. "
            f"Using fallback prompt for {self.agent_id}."
        )
        return (
            f"You are {self.agent_id}, a {self.role} on the {self.team.upper()} team "
            f"in the CIPHER adversarial intelligence simulation."
        )
