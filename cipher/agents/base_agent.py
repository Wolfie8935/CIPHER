"""
Base agent class and action vocabulary for CIPHER.

Defines the abstract BaseAgent that all 8 agents (4 RED, 4 BLUE) inherit from,
the Action model representing a single agent action, and the ActionType enum
enumerating all valid actions.

Phase 3: Adds LLM integration layer — observation-to-prompt serialization,
LLM completion calls, response parsing, and mode toggle (stub vs live).

Owns: agent interface contract, action schema, prompt loading, LLM integration.
Does NOT own: specific agent behavior (that's each subclass), environment state,
or reward computation.
"""
from __future__ import annotations

import json
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
    WAIT = "wait"
    ABORT = "abort"

    # ── RED trap actions (Phase 5) ───────────────────────────────
    PLANT_FALSE_TRAIL = "plant_false_trail"
    PLANT_TEMPORAL_DECOY = "plant_temporal_decoy"
    PLANT_HONEYPOT_POISON = "plant_honeypot_poison"
    WRITE_CORRUPTED_DROP = "write_corrupted_drop"

    # Legacy alias — maps to PLANT_FALSE_TRAIL in parsing
    PLANT_TRAP = "plant_trap"

    # ── BLUE actions ─────────────────────────────────────────────
    INVESTIGATE_NODE = "investigate_node"
    TRIGGER_ALERT = "trigger_alert"
    ANALYZE_ANOMALY = "analyze_anomaly"
    RECONSTRUCT_PATH = "reconstruct_path"
    STAND_DOWN = "stand_down"

    # ── BLUE trap actions (Phase 5) ──────────────────────────────
    PLACE_HONEYPOT = "place_honeypot"
    PLANT_BREADCRUMB = "plant_breadcrumb"
    TRIGGER_FALSE_ESCALATION = "trigger_false_escalation"
    TAMPER_DEAD_DROP = "tamper_dead_drop"

    # Legacy alias
    DEPLOY_TRAP = "deploy_trap"
    DEPLOY_BREADCRUMB = "deploy_breadcrumb"
    DEPLOY_FALSE_ESCALATION = "deploy_false_escalation"


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

    # Phase 5: trap-specific fields
    trap_type: str | None = None  # TrapType value string
    trap_params: dict[str, Any] | None = None
    trap_payload: dict[str, Any] | None = None


# ── Prompt templates directory ────────────────────────────────
_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


# ── Valid action sets per team ────────────────────────────────
RED_ACTIONS = {
    ActionType.MOVE,
    ActionType.READ_FILE,
    ActionType.EXFILTRATE,
    ActionType.WRITE_DEAD_DROP,
    ActionType.READ_DEAD_DROP,
    ActionType.WAIT,
    ActionType.ABORT,
    # Phase 5 trap actions
    ActionType.PLANT_FALSE_TRAIL,
    ActionType.PLANT_TEMPORAL_DECOY,
    ActionType.PLANT_HONEYPOT_POISON,
    ActionType.WRITE_CORRUPTED_DROP,
    ActionType.PLANT_TRAP,  # Legacy alias
}

BLUE_ACTIONS = {
    ActionType.INVESTIGATE_NODE,
    ActionType.TRIGGER_ALERT,
    ActionType.ANALYZE_ANOMALY,
    ActionType.RECONSTRUCT_PATH,
    ActionType.STAND_DOWN,
    # Phase 5 trap actions
    ActionType.PLACE_HONEYPOT,
    ActionType.PLANT_BREADCRUMB,
    ActionType.TRIGGER_FALSE_ESCALATION,
    ActionType.TAMPER_DEAD_DROP,
    ActionType.DEPLOY_TRAP,  # Legacy alias
    ActionType.DEPLOY_BREADCRUMB,  # Legacy alias
    ActionType.DEPLOY_FALSE_ESCALATION,  # Legacy alias
}

# Max history entries to keep in prompt (controls token budget)
MAX_PROMPT_HISTORY = 10


class BaseAgent(ABC):
    """
    Abstract base class for all CIPHER agents.

    Every agent — RED or BLUE — implements this interface.
    The training loop only ever calls observe(), act(), and reset().
    Everything else is agent-internal.

    Phase 3 additions:
    - _model_env_key: maps to the .env key for this agent's model
    - _observation_to_prompt_text: serializes observations for LLM
    - _build_messages: constructs the full message list for the LLM
    - _parse_action_from_response: parses JSON response into Action
    - act_live: LLM-backed decision making
    - _stub_act: random fallback for stub mode (must be overridden)
    """

    # Subclasses MUST override this with the config attribute name
    # e.g. "nvidia_model_red_planner"
    _model_env_key: str = ""

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
        self._last_reasoning: str = ""
        self._trap_budget_remaining: int = (
            config.env_trap_budget_red if team == "red" else config.env_trap_budget_blue
        )

        logger.debug(f"Agent initialized: {agent_id} (team={team}, role={role})")

    @abstractmethod
    def observe(self, observation: RedObservation | BlueObservation) -> None:
        """
        Process the latest observation. Store it for use in act().

        Args:
            observation: The team-specific observation for this step.
        """

    def act(self) -> Action:
        """
        Decide on and return an action based on the current observation.

        Mode routing:
          stub   → _stub_act()  (random/heuristic, no API calls)
          live   → _act_live()  (NVIDIA NIM for all 8 agents)
          hybrid → _act_lora()  for RED Planner specialist
                   _act_live()  for all other 7 agents (NVIDIA NIM)

        Returns:
            An Action object representing the agent's decision.
        """
        from cipher.utils.llm_mode import is_live_mode, is_hybrid_mode

        self.step_count += 1

        if is_hybrid_mode() and self._is_hybrid_specialist():
            action = self._act_lora()
        elif is_live_mode():
            action = self._act_live()
        else:
            action = self._stub_act()

        self.action_history.append(action)
        return action

    def _is_hybrid_specialist(self) -> bool:
        """
        Returns True if this agent is the fine-tuned specialist in hybrid mode.
        Currently: RED Planner only (trained with GRPO on CIPHER episodes).
        """
        return self.team == "red" and self.role == "planner"

    def _act_live(self) -> Action:
        """
        LLM-backed action selection via NVIDIA NIM API.
        Constructs prompt, calls LLM, parses response.
        Used by all 8 agents in live mode; used by 7 non-specialist agents in hybrid mode.
        """
        from cipher.utils.llm_client import get_llm_client

        messages = self._build_messages()
        client = get_llm_client()

        response_text = client.complete(
            model_env_key=self._model_env_key,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            expect_json=True,
            team=self.team,
        )

        action = self._parse_action_from_response(response_text)
        self._last_reasoning = action.reasoning

        self._update_prompt_history(
            user_content=self._observation_to_prompt_text(),
            assistant_content=response_text,
        )

        return action

    def _act_lora(self) -> Action:
        """
        Fine-tuned LoRA specialist inference for RED Planner in hybrid mode.

        Loads the trained Llama-3.2-1B LoRA adapter (cipher-red-planner) on
        first call, then reuses for all subsequent steps — no re-loading overhead.
        Falls back to _act_live() (NVIDIA NIM) on any loading error.
        """
        import os

        adapter_path = os.getenv(
            "RED_PLANNER_LORA_PATH",
            os.path.join("red trained", "cipher-red-planner"),
        )

        try:
            from cipher.utils.lora_client import LoRAClient
            messages = self._build_messages()
            client = LoRAClient()
            response_text = client.complete(
                messages=messages,
                adapter_path=adapter_path,
                max_new_tokens=256,
                temperature=0.7,
            )
            logger.info(f"[LoRA] RED Planner response: {response_text[:80]}...")
        except Exception as exc:
            logger.warning(
                f"[LoRA] RED Planner LoRA failed ({exc}). Falling back to NVIDIA NIM."
            )
            return self._act_live()

        action = self._parse_action_from_response(response_text)
        self._last_reasoning = action.reasoning

        self._update_prompt_history(
            user_content=self._observation_to_prompt_text(),
            assistant_content=response_text,
        )

        return action

    @abstractmethod
    def _stub_act(self) -> Action:
        """
        Random/heuristic fallback for stub mode (no API cost).

        Each subclass provides its own stub logic matching Phase 1 behavior.
        """

    def _build_messages(self) -> list[dict[str, str]]:
        """
        Construct the full OpenAI-format message list for the LLM.

        Structure:
        1. System prompt (role-specific, from prompts/*.txt)
        2. Conversation history (last N exchanges)
        3. Current observation as user message

        Subclasses can override to inject role-specific context.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
        ]

        # Add conversation history (bounded by MAX_PROMPT_HISTORY)
        history_slice = self.prompt_history[-MAX_PROMPT_HISTORY:]
        messages.extend(history_slice)

        # Add current observation as user message
        obs_text = self._observation_to_prompt_text()
        messages.append({"role": "user", "content": obs_text})

        return messages

    def _observation_to_prompt_text(self) -> str:
        """
        Serialize the current observation into natural language for the LLM.

        RED and BLUE have different observation structures.
        This method handles both via duck typing.
        """
        obs = self._current_observation
        if obs is None:
            return "No observation available. Choose WAIT."

        if isinstance(obs, RedObservation):
            return self._red_observation_to_text(obs)
        elif isinstance(obs, BlueObservation):
            return self._blue_observation_to_text(obs)
        else:
            return f"Unknown observation type: {type(obs)}"

    def _red_observation_to_text(self, obs: RedObservation) -> str:
        """Format a RedObservation into prompt text."""
        lines = [
            f"STEP {obs.step} — RED TEAM OBSERVATION",
            f"Current node: {obs.current_node}",
            f"Current zone: {obs.current_zone}",
            f"Current hostname: {obs.current_hostname}",
            f"Current node type: {obs.current_node_type.value if hasattr(obs.current_node_type, 'value') else obs.current_node_type}",
            f"Suspicion level: {obs.estimated_suspicion:.2f}",
            f"Privilege level: {obs.current_privilege_level}",
        ]

        # Services
        if obs.current_services:
            lines.append(f"Services: {', '.join(obs.current_services)}")

        # Adjacent nodes (parallel lists: ids, types, hostnames, protocols)
        if obs.adjacent_nodes:
            lines.append("Adjacent nodes:")
            for i, node_id in enumerate(obs.adjacent_nodes):
                node_type = obs.adjacent_node_types[i].value if i < len(obs.adjacent_node_types) and hasattr(obs.adjacent_node_types[i], 'value') else "unknown"
                hostname = obs.adjacent_hostnames[i] if i < len(obs.adjacent_hostnames) else "unknown"
                protocol = obs.edge_protocols[i] if i < len(obs.edge_protocols) else "ssh"
                req_priv = obs.required_privileges[i] if i < len(obs.required_privileges) else 0
                lines.append(
                    f"  node {node_id} (type={node_type}, hostname={hostname}, "
                    f"protocol={protocol}, req_priv={req_priv})"
                )

        # Zone boundary warning
        if obs.zone_boundary_ahead:
            lines.append("⚠ ZONE BOUNDARY AHEAD — next zone requires higher privileges")

        # Files at current node
        if obs.files_at_current_node:
            lines.append(f"Files here: {', '.join(obs.files_at_current_node)}")

        # Dead drops
        if obs.dead_drops_available:
            lines.append(f"Dead drops available: {len(obs.dead_drops_available)}")
        else:
            lines.append("Dead drops available: NONE")

        # Context reset
        if obs.context_reset_this_step:
            lines.append("*** CONTEXT RESET THIS STEP — You have no memory of prior actions ***")

        # Credentials
        if obs.credentials_held:
            lines.append(f"Credentials held: {', '.join(obs.credentials_held)}")

        return "\n".join(lines)

    def _blue_observation_to_text(self, obs: BlueObservation) -> str:
        """Format a BlueObservation into prompt text."""
        lines = [
            f"STEP {obs.step} — BLUE TEAM OBSERVATION",
            f"Detection confidence: {obs.current_detection_confidence:.2f}",
            f"Investigation budget remaining: {obs.investigation_budget_remaining}",
            f"Total alerts issued: {obs.total_alerts_issued}",
        ]

        # Anomaly feed
        if obs.anomaly_feed:
            lines.append(f"Anomaly events ({len(obs.anomaly_feed)}):")
            for anomaly in obs.anomaly_feed[-5:]:  # Show last 5
                lines.append(
                    f"  [{anomaly.event_type}] node {anomaly.node_id} "
                    f"(zone {anomaly.zone}, severity={anomaly.severity:.2f}, "
                    f"step={anomaly.step})"
                )
        else:
            lines.append("Anomaly feed: No anomalies detected this step")

        # Zone alert levels
        if obs.zone_alert_levels:
            alert_str = ", ".join(f"zone {z}={v:.2f}" for z, v in sorted(obs.zone_alert_levels.items()))
            lines.append(f"Zone alert levels: {alert_str}")

        # Honeypots
        if obs.active_honeypots:
            lines.append(f"Active honeypot nodes: {obs.active_honeypots}")
        if obs.honeypots_triggered_this_step:
            lines.append(
                f"HONEYPOTS TRIGGERED THIS STEP: {obs.honeypots_triggered_this_step}"
            )

        # Honeypot health
        if obs.honeypot_health:
            lines.append(f"Honeypot health: {obs.honeypot_health}")

        # Network anomaly rate
        lines.append(f"Network anomaly rate: {obs.network_anomaly_rate:.2f}")

        return "\n".join(lines)

    def _parse_action_from_response(self, response_text: str) -> Action:
        """
        Parse the LLM's JSON response into a validated Action.

        Handles:
        - Valid JSON with matching action_type
        - JSON with unknown action_type (fallback to WAIT/STAND_DOWN)
        - Invalid JSON (fallback to WAIT/STAND_DOWN)
        - Missing fields (filled with defaults)
        """
        fallback_type = ActionType.WAIT if self.team == "red" else ActionType.STAND_DOWN

        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```"):
            # Accept markdown-wrapped JSON even when the model ignores instruction.
            lines = cleaned_text.splitlines()
            cleaned_text = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            ).strip()

        try:
            data = json.loads(cleaned_text)
        except json.JSONDecodeError:
            logger.warning(
                f"{self.agent_id}: Failed to parse LLM response as JSON. "
                f"Falling back to {fallback_type.value}."
            )
            return Action(
                agent_id=self.agent_id,
                action_type=fallback_type,
                reasoning="JSON parse failure — falling back to safe action.",
            )

        # Extract action_type
        raw_action = data.get("action_type", fallback_type.value)
        if raw_action is None:
            raw_action = fallback_type.value
        elif not isinstance(raw_action, str):
            raw_action = str(raw_action)
        try:
            action_type = ActionType(raw_action.lower())
        except ValueError:
            logger.warning(
                f"{self.agent_id}: Unknown action_type '{raw_action}'. "
                f"Falling back to {fallback_type.value}."
            )
            action_type = fallback_type

        # Normalize legacy aliases.
        if action_type == ActionType.PLANT_TRAP:
            action_type = ActionType.PLANT_FALSE_TRAIL
        if action_type == ActionType.DEPLOY_TRAP:
            action_type = ActionType.PLACE_HONEYPOT
        if action_type == ActionType.DEPLOY_BREADCRUMB:
            action_type = ActionType.PLANT_BREADCRUMB
        if action_type == ActionType.DEPLOY_FALSE_ESCALATION:
            action_type = ActionType.TRIGGER_FALSE_ESCALATION

        # Validate team-action consistency
        valid_set = RED_ACTIONS if self.team == "red" else BLUE_ACTIONS
        if action_type not in valid_set:
            logger.warning(
                f"{self.agent_id}: Action '{action_type.value}' not valid for "
                f"{self.team} team. Falling back to {fallback_type.value}."
            )
            action_type = fallback_type

        # Extract optional fields
        target_node = data.get("target_node")
        if target_node is not None:
            try:
                target_node = int(target_node)
            except (ValueError, TypeError):
                target_node = None

        target_file = data.get("target_file")
        reasoning = data.get("reasoning", "")
        trap_params = data.get("trap_params") or data.get("trap_payload") or {}

        # EXFIL semantic guardrail: require a plausible file name target.
        if action_type == ActionType.EXFILTRATE:
            invalid_exfil_target = False
            if target_file is None:
                invalid_exfil_target = True
            elif not isinstance(target_file, str):
                invalid_exfil_target = True
            else:
                target_file = target_file.strip()
                if not target_file:
                    invalid_exfil_target = True
                # Common malformed patterns: numeric IDs or node-like aliases.
                if target_file.isdigit() or target_file.lower().startswith("node_"):
                    invalid_exfil_target = True
            if invalid_exfil_target:
                logger.warning(
                    f"{self.agent_id}: Invalid EXFILTRATE semantics "
                    f"(target_file={target_file!r}). Falling back to {fallback_type.value}."
                )
                action_type = fallback_type
                target_file = None
                if not reasoning:
                    reasoning = (
                        "Invalid EXFILTRATE target_file semantics — "
                        "falling back to safe action."
                    )
            # Normalize EXFIL action shape: file-based target only.
            target_node = None

        if not self._is_trap_action_allowed(action_type, target_node):
            action_type = fallback_type

        if self._is_trap_action(action_type):
            self._trap_budget_remaining = max(0, self._trap_budget_remaining - 1)
        return Action(
            agent_id=self.agent_id,
            action_type=action_type,
            target_node=target_node,
            target_file=target_file,
            reasoning=str(reasoning),
            trap_params=trap_params if isinstance(trap_params, dict) else {},
            trap_payload=trap_params if isinstance(trap_params, dict) else {},
        )

    def _is_trap_action(self, action_type: ActionType) -> bool:
        trap_actions = {
            ActionType.PLANT_FALSE_TRAIL,
            ActionType.PLANT_TEMPORAL_DECOY,
            ActionType.PLANT_HONEYPOT_POISON,
            ActionType.WRITE_CORRUPTED_DROP,
            ActionType.PLACE_HONEYPOT,
            ActionType.PLANT_BREADCRUMB,
            ActionType.TRIGGER_FALSE_ESCALATION,
            ActionType.TAMPER_DEAD_DROP,
        }
        return action_type in trap_actions

    def _is_trap_action_allowed(
        self,
        action_type: ActionType,
        target_node: int | None,
    ) -> bool:
        if not self._is_trap_action(action_type):
            return True

        if self._trap_budget_remaining <= 0:
            return False

        if action_type == ActionType.PLANT_FALSE_TRAIL:
            n_moves = sum(1 for a in self.action_history if a.action_type == ActionType.MOVE)
            return n_moves >= 3

        if action_type == ActionType.PLACE_HONEYPOT:
            if self.team != "blue" or target_node is None:
                return False
            obs = self._current_observation
            if obs is None or not isinstance(obs, BlueObservation):
                return False
            observed_nodes = {a.node_id for a in obs.anomaly_feed if a.node_id is not None}
            return target_node in observed_nodes

        if action_type == ActionType.TAMPER_DEAD_DROP:
            obs = self._current_observation
            discovered = getattr(obs, "blue_discovered_drop_paths", None)
            if discovered is None:
                discovered = getattr(self, "_blue_discovered_drop_paths", [])
            return bool(discovered)

        return True

    def _update_prompt_history(
        self,
        user_content: str,
        assistant_content: str,
    ) -> None:
        """
        Append the latest exchange to prompt_history, bounded by MAX_PROMPT_HISTORY.
        """
        self.prompt_history.append({"role": "user", "content": user_content})
        self.prompt_history.append({"role": "assistant", "content": assistant_content})

        # Trim to keep within budget (entries = 2 * MAX_PROMPT_HISTORY)
        max_entries = MAX_PROMPT_HISTORY * 2
        if len(self.prompt_history) > max_entries:
            self.prompt_history = self.prompt_history[-max_entries:]

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
        self._last_reasoning = ""
        self._trap_budget_remaining = (
            self.config.env_trap_budget_red if self.team == "red" else self.config.env_trap_budget_blue
        )
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

    @property
    def last_reasoning(self) -> str:
        """Returns the reasoning string from the last action taken."""
        return self._last_reasoning
