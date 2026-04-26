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
import re
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

    # LoRA specialist outputs — treated as recon/escalation for RED
    ESCALATE_PRIVILEGES = "escalate_privileges"
    SCAN = "scan"

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

    # ── EMERGENT (self-reliant novel actions) ─────────────────────
    EMERGENT = "emergent"  # Agent-proposed action outside the predefined vocabulary

    # ── Commander meta-actions (v2 architecture) ──────────────────
    # These are NEVER dispatched to the environment. They are consumed by the
    # commander's SubagentRegistry to manage the dynamic subagent roster.
    SPAWN_SUBAGENT = "spawn_subagent"
    DELEGATE_TASK = "delegate_task"
    DISMISS_SUBAGENT = "dismiss_subagent"


# Set of meta-actions the registry consumes (do not pass to env dispatcher).
META_ACTIONS = frozenset({
    "spawn_subagent",
    "delegate_task",
    "dismiss_subagent",
})


class EmergentAction(BaseModel):
    """An agent-proposed action outside the predefined ActionType vocabulary."""
    intent: str = ""          # e.g. "spoof_credentials", "network_quarantine"
    target_node: int | None = None
    target_file: str | None = None
    reasoning: str = ""       # Why the agent chose this novel action
    expected_effect: str = "" # What the agent thinks will happen


class SubagentSpec(BaseModel):
    """
    Declarative spec for a subagent. The commander emits this (inside an
    Action with action_type=SPAWN_SUBAGENT) and the SubagentRegistry uses it
    to instantiate a Subagent worker for the team.
    """

    role_name: str                       # must match a registered SubagentRoleProfile
    team: str                            # 'red' | 'blue'
    task_brief: str = ""                 # natural-language directive from commander
    lifespan_steps: int = 5              # auto-dismiss after this many steps
    allowed_actions: list[str] | None = None  # optional whitelist override
    parent_id: str = ""                  # commander id that spawned this subagent
    subagent_id: str = ""                # filled in by registry on spawn
    use_llm: bool = False                # if True, subagent calls LLM; else heuristic


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

    # Emergent action payload — populated when action_type == EMERGENT
    emergent_data: EmergentAction | None = None

    # ── Commander/subagent provenance (v2) ────────────────────────
    role: str | None = None          # role name (e.g. 'planner', 'commander', 'scout')
    spawned_by: str | None = None    # parent commander agent_id when emitted by a subagent
    subagent_spec: SubagentSpec | None = None  # populated when action_type == SPAWN_SUBAGENT
    target_subagent_id: str | None = None      # for DELEGATE_TASK / DISMISS_SUBAGENT


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
    # LoRA specialist recon actions
    ActionType.ESCALATE_PRIVILEGES,
    ActionType.SCAN,
    # Emergent (self-reliant) actions
    ActionType.EMERGENT,
    # Commander-only meta-actions (validated separately for role='commander')
    ActionType.SPAWN_SUBAGENT,
    ActionType.DELEGATE_TASK,
    ActionType.DISMISS_SUBAGENT,
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
    # Emergent (self-reliant) actions
    ActionType.EMERGENT,
    # Commander-only meta-actions (validated separately for role='commander')
    ActionType.SPAWN_SUBAGENT,
    ActionType.DELEGATE_TASK,
    ActionType.DISMISS_SUBAGENT,
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
    # e.g. "hf_model_red_planner"
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
          live   → _act_live()  (HuggingFace Inference API for all 8 agents)
          hybrid → _act_lora()  for RED Planner specialist
                   _act_live()  for all other 7 agents (HuggingFace API)

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

    # Maps (team, role) → (env_var, default_adapter_path) for LoRA specialists.
    # v2 adds the COMMANDER entries — when --hybrid, the trained commanders
    # take over the orchestration layer via these LoRAs.
    _LORA_PATH_MAP: dict = {
        ("red",  "commander"):     ("RED_COMMANDER_LORA_PATH",      os.path.join("red trained", "cipher-red-commander-v1")),
        ("blue", "commander"):     ("BLUE_COMMANDER_LORA_PATH",     os.path.join("blue trained", "cipher-blue-commander-v1")),
        ("red",  "planner"):       ("RED_PLANNER_LORA_PATH",        os.path.join("red trained", "cipher-red-planner-v1")),
        ("red",  "analyst"):       ("RED_ANALYST_LORA_PATH",        os.path.join("red trained", "cipher-red-analyst-v1")),
        ("blue", "surveillance"):  ("BLUE_SURVEILLANCE_LORA_PATH",  os.path.join("blue trained", "cipher-blue-surveillance-v1")),
        ("blue", "threat_hunter"): ("BLUE_THREAT_HUNTER_LORA_PATH", os.path.join("blue trained", "cipher-blue-threat-hunter-v1")),
    }

    def _is_hybrid_specialist(self) -> bool:
        """Returns True if this agent has a trained LoRA available in hybrid mode."""
        key = (self.team, self.role)
        if key not in self._LORA_PATH_MAP:
            return False
        env_key, default_path = self._LORA_PATH_MAP[key]
        adapter_path = os.getenv(env_key, default_path)
        return bool(os.path.exists(adapter_path))

    def _get_adaptive_temperature(self) -> float:
        """Change 6a: Return temperature based on recent losing streak.

        3+ consecutive losses → bump to 0.9 to encourage diverse outputs.
        After a win → back to baseline 0.7.
        """
        try:
            from cipher.training.episode_memory import count_consecutive_losses
            losses = count_consecutive_losses(team=self.team)
            if losses >= 3:
                return 0.9  # exploration pressure
        except ImportError:
            pass
        return 0.7

    def _get_exploration_directive(self) -> str:
        """Change 6b: Return an exploration directive when agent is in a losing streak."""
        try:
            from cipher.training.episode_memory import count_consecutive_losses
            losses = count_consecutive_losses(team=self.team)
            if losses >= 3:
                return (
                    f"\n\nIMPORTANT: Your recent strategy has failed {losses} times in a row. "
                    "You MUST try a fundamentally different approach this episode. "
                    "This includes trying emergent actions you haven't used before, "
                    "taking different paths, or using different timing strategies."
                )
        except ImportError:
            pass
        return ""

    def _act_live(self) -> Action:
        """
        LLM-backed action selection via HuggingFace Inference API.
        Constructs prompt, calls LLM, parses response.
        Used by all 8 agents in live mode; used by 7 non-specialist agents in hybrid mode.
        """
        from cipher.utils.llm_client import get_llm_client

        # Change 6: adaptive temperature + exploration directive on losing streaks
        temperature = self._get_adaptive_temperature()
        exploration_directive = self._get_exploration_directive()

        messages = self._build_messages()
        if exploration_directive:
            # Append directive to the last user message
            messages[-1]["content"] += exploration_directive

        client = get_llm_client()

        response_text = client.complete(
            model_env_key=self._model_env_key,
            messages=messages,
            max_tokens=512,
            temperature=temperature,
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
        Fine-tuned LoRA specialist inference for hybrid mode.

        Resolves the correct adapter path for whichever specialist this agent is,
        then calls the LoRA client. Falls back to HF API on any loading error.
        """
        key = (self.team, self.role)
        env_key, default_path = self._LORA_PATH_MAP.get(key, ("", ""))
        adapter_path = os.getenv(env_key, default_path) if env_key else default_path

        try:
            from cipher.utils.lora_client import LoRAClient
            messages = self._build_messages()
            client = LoRAClient()
            response_text = client.complete(
                messages=messages,
                adapter_path=adapter_path,
                max_new_tokens=80,
                temperature=0.7,
            )
            logger.info(f"[LoRA] {self.agent_id} response: {response_text[:80]}...")
        except Exception as exc:
            logger.warning(
                f"[LoRA] {self.agent_id} LoRA failed ({exc}). Falling back to HF API."
            )
            return self._act_live_hf_direct()

        action = self._parse_action_from_response(response_text)
        self._last_reasoning = action.reasoning

        self._update_prompt_history(
            user_content=self._observation_to_prompt_text(),
            assistant_content=response_text,
        )

        return action

    def _act_live_hf_direct(self) -> Action:
        """
        Call HuggingFace API directly, bypassing hybrid local routing.

        Used when the LoRA specialist is unavailable (e.g. torch not installed).
        Temporarily removes the RED Planner key from _LOCAL_KEYS_IN_HYBRID so
        _resolve() routes to the HF API instead of localhost.
        """
        from cipher.utils.llm_client import get_llm_client, _LOCAL_KEYS_IN_HYBRID

        messages = self._build_messages()
        client = get_llm_client()

        # Temporarily pull the planner key out of hybrid routing so it goes
        # to HF API instead of the unavailable local server.
        key = self._model_env_key
        was_in_hybrid = key in _LOCAL_KEYS_IN_HYBRID
        if was_in_hybrid:
            _LOCAL_KEYS_IN_HYBRID.discard(key)

        # Change 6: adaptive temperature on losing streaks
        temperature = self._get_adaptive_temperature()
        exploration_directive = self._get_exploration_directive()
        if exploration_directive:
            messages[-1]["content"] += exploration_directive

        try:
            response_text = client.complete(
                model_env_key=key,
                messages=messages,
                max_tokens=512,
                temperature=temperature,
                expect_json=True,
                team=self.team,
            )
        finally:
            # Always restore the routing table.
            if was_in_hybrid:
                _LOCAL_KEYS_IN_HYBRID.add(key)

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

    @staticmethod
    def _compress_history(history: list[dict[str, str]], keep_recent: int = 5) -> list[dict[str, str]]:
        """Token-Squeeze: compress old history into a summary + keep last N pairs.

        Reduces token usage by 30-50% on long episodes. Keeps the last
        `keep_recent` user/assistant pairs verbatim and summarises earlier ones
        into a single injected exchange that costs ~20 tokens instead of ~200.
        """
        pairs = keep_recent * 2  # each exchange = 1 user + 1 assistant msg
        if len(history) <= pairs:
            return history

        older = history[:-pairs]
        recent = history[-pairs:]

        step_summaries: list[str] = []
        for i in range(0, len(older), 2):
            user_msg = older[i].get("content", "") if i < len(older) else ""
            asst_msg = older[i + 1].get("content", "") if i + 1 < len(older) else ""

            step_match = re.search(r"STEP\s+(\d+)", user_msg)
            step_label = f"step {step_match.group(1)}" if step_match else f"step {i // 2 + 1}"

            try:
                action_data = json.loads(asst_msg[:512])
                action = action_data.get("action_type", "?")
                target = action_data.get("target_node")
                entry = f"{step_label}: {action}" + (f"→n{target}" if target is not None else "")
            except Exception:
                entry = f"{step_label}: (action)"

            step_summaries.append(entry)

        if step_summaries:
            summary = "COMPRESSED HISTORY: " + "; ".join(step_summaries)
            return [
                {"role": "user", "content": summary},
                {"role": "assistant", "content": "History acknowledged. Continuing."},
            ] + recent

        return recent

    def _build_messages(self) -> list[dict[str, str]]:
        """
        Construct the full OpenAI-format message list for the LLM.

        Structure:
        1. System prompt (role-specific, from prompts/*.txt) with episode memory prefix
        2. Compressed conversation history (last 5 full + summary of older)
        3. Current observation as user message
        """
        # Prepend recent episode history ONLY in hybrid/train mode so that live
        # episodes start fresh and don't inherit stale win/loss patterns.
        system_content = self._system_prompt
        if getattr(self.config, "llm_mode", "stub") == "hybrid":
            try:
                from cipher.training.episode_memory import get_recent_summary
                ep_summary = get_recent_summary(n=3)
                if ep_summary:
                    system_content = ep_summary + "\n\n" + system_content
            except ImportError:
                pass

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        # Compress history: keep last 5 exchanges verbatim, summarise older
        compressed = self._compress_history(self.prompt_history, keep_recent=5)
        messages.extend(compressed)

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
        node_type_val = (
            obs.current_node_type.value
            if hasattr(obs.current_node_type, "value")
            else str(obs.current_node_type)
        )
        at_hvt = node_type_val == "high_value_target"
        honeypot_nodes = set(getattr(obs, 'honeypot_nodes_nearby', []) or [])
        breadcrumb_nodes = set(getattr(obs, 'breadcrumb_nodes_nearby', []) or [])
        danger_nodes = honeypot_nodes | breadcrumb_nodes
        hvt_node_id = int(getattr(obs, 'hvt_node_id', -1))

        lines = [
            f"STEP {obs.step} — RED TEAM OBSERVATION",
            f"Current node: {obs.current_node} | Zone: {obs.current_zone} | Type: {node_type_val}",
            f"Hostname: {obs.current_hostname}",
            f"Suspicion level: {obs.estimated_suspicion:.2f} | Privilege: {obs.current_privilege_level}",
            f"BLUE detection confidence: {obs.blue_detection_confidence:.2f} (abort if this reaches 0.80+)",
        ]

        # Show HVT node ID prominently so RED can navigate directly
        if at_hvt:
            lines.append("EXFILTRATE ALLOWED: YES — you are AT the HIGH_VALUE_TARGET node. Use EXFILTRATE now.")
        elif hvt_node_id >= 0:
            lines.append(
                f"EXFILTRATE ALLOWED: NO — TARGET IS node n{hvt_node_id} (Zone 3). "
                f"You must MOVE to n{hvt_node_id} to exfiltrate. "
                f"{'It is adjacent — MOVE THERE NOW.' if hvt_node_id in (getattr(obs, 'adjacent_nodes', []) or []) else 'Navigate toward it now.'}"
            )
        else:
            lines.append("EXFILTRATE ALLOWED: NO — keep moving to Zone 3.")

        # Services
        if obs.current_services:
            lines.append(f"Services: {', '.join(obs.current_services)}")

        # Adjacent nodes with danger flags and HVT callouts
        if obs.adjacent_nodes:
            lines.append("Adjacent nodes (AVOID any marked DANGER):")
            hvt_adjacent = []
            for i, node_id in enumerate(obs.adjacent_nodes):
                node_type = (
                    obs.adjacent_node_types[i].value
                    if i < len(obs.adjacent_node_types) and hasattr(obs.adjacent_node_types[i], 'value')
                    else "unknown"
                )
                hostname = obs.adjacent_hostnames[i] if i < len(obs.adjacent_hostnames) else "unknown"
                protocol = obs.edge_protocols[i] if i < len(obs.edge_protocols) else "ssh"
                req_priv = obs.required_privileges[i] if i < len(obs.required_privileges) else 0
                flags = ""
                if node_id in honeypot_nodes:
                    flags += " ⚠HONEYPOT-AVOID"
                elif node_id in breadcrumb_nodes:
                    flags += " ⚠BREADCRUMB-AVOID"
                if node_type == "high_value_target":
                    flags += " ★HVT-TARGET-MOVE-HERE"
                    hvt_adjacent.append(node_id)
                lines.append(
                    f"  node {node_id} (type={node_type}, host={hostname}, "
                    f"proto={protocol}, req_priv={req_priv}){flags}"
                )
            # Urgent HVT callout
            if hvt_adjacent and not at_hvt:
                lines.append(
                    f"★★★ HVT IS ADJACENT: MOVE TO NODE {hvt_adjacent[0]} NOW — this is the exfiltration target! ★★★"
                )

        # Danger zone summary (honeypots + breadcrumbs)
        if danger_nodes:
            hp_list = sorted(honeypot_nodes)
            bc_list = sorted(breadcrumb_nodes)
            if hp_list:
                lines.append(f"⚠ HONEYPOT NODES (NEVER ENTER): {hp_list}")
            if bc_list:
                lines.append(f"⚠ BREADCRUMB TRAPS (AVOID PATH): {bc_list}")

        # Zone boundary
        if obs.zone_boundary_ahead:
            lines.append("→ ZONE BOUNDARY AHEAD — advancing here raises privileges and zone level")

        # Zone stall warning — uses getattr for safety on non-planner RED agents
        zone_stall = getattr(self, '_zone_stall_steps', 0)
        if zone_stall >= 6:
            lines.append(
                f"⚠⚠ CRITICAL ZONE STALL: {zone_stall} steps in Zone {obs.current_zone}. "
                f"You MUST advance to the next zone NOW. Staying here is mission failure."
            )

        # Change 2: danger map from shared RED coordination module
        try:
            from cipher.agents.red.coordination import (
                get_danger_nodes,
                get_danger_score,
            )
            danger_nodes = get_danger_nodes(threshold=0.4)
            if danger_nodes:
                # get_danger_nodes returns a set[int]; look up scores individually
                # so this works whether the upstream returns a set or a dict.
                if isinstance(danger_nodes, dict):
                    flagged = sorted(danger_nodes, key=danger_nodes.get, reverse=True)[:5]
                    scores = ", ".join(f"n{n}={danger_nodes[n]:.2f}" for n in flagged)
                else:
                    scored = [(n, get_danger_score(n)) for n in danger_nodes]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    scores = ", ".join(f"n{n}={s:.2f}" for n, s in scored[:5])
                lines.append(f"DANGER MAP (suspicion spikes observed by RED team): {scores}")
        except ImportError:
            pass

        # Change 2: available paths — sorted by ZONE DESC then suspicion ASC,
        # so the LLM sees the highest-zone advancement options FIRST and picks
        # them by default. Stalling in the same zone is the #1 cause of mission
        # failure in live runs.
        if obs.available_paths:
            sorted_paths = sorted(
                obs.available_paths,
                key=lambda p: (-int(p.get("zone", 0)), float(p.get("suspicion_delta", 0.0))),
            )
            lines.append(
                "PATH OPTIONS (PRIORITISE highest zone — advancing zones is the "
                "mission. Same-zone moves should only be picked if NO higher-zone "
                "option exists):"
            )
            for p in sorted_paths[:6]:
                node_id = p.get("node", "?")
                s_delta = p.get("suspicion_delta", 0.0)
                zone = p.get("zone", "?")
                host = p.get("hostname", "?")
                marker = " *** ZONE-ADVANCE" if zone != "?" and zone > obs.current_zone else ""
                # Mark the HVT node explicitly so RED navigates there directly
                if hvt_node_id >= 0 and node_id == hvt_node_id:
                    marker += " <<< THIS IS THE HVT — MOVE HERE TO EXFILTRATE"
                lines.append(
                    f"  -> n{node_id} (zone={zone}, host={host}, suspicion_cost={s_delta:+.3f}){marker}"
                )
            # If HVT is not in the top-6 adjacent paths, add a reminder
            if hvt_node_id >= 0 and not at_hvt:
                hvt_in_paths = any(p.get("node") == hvt_node_id for p in obs.available_paths)
                if not hvt_in_paths:
                    lines.append(
                        f"  [HVT node n{hvt_node_id} is not adjacent — keep moving toward Zone 3]"
                    )

        # Change 4b: contextual emergent hints when RED is stuck
        if obs.estimated_suspicion > 0.60 and zone_stall >= 3:
            lines.append(
                "STUCK DETECTED: High suspicion + zone stall. Standard actions are failing. "
                "Consider an EMERGENT action like 'spoof_credentials' or 'inject_noise' "
                "to change the situation immediately."
            )

        # Files at current node
        if obs.files_at_current_node:
            if at_hvt:
                exfil_done = set(getattr(obs, 'exfiltrated_files', []) or [])
                remaining = [f for f in obs.files_at_current_node if f not in exfil_done]
                if remaining:
                    lines.append(f"TARGET FILES HERE: {', '.join(remaining)} — EXFILTRATE THESE NOW")
                    lines.append(
                        f'EXACT JSON TO EMIT: '
                        f'{{"action_type":"exfiltrate","target_file":"{remaining[0]}",'
                        f'"reasoning":"At HVT — extracting target file."}}'
                    )
                if exfil_done:
                    lines.append(f"Already exfiltrated: {', '.join(exfil_done)}")
            else:
                lines.append(f"Files here: {', '.join(obs.files_at_current_node)} (not target files — skip, keep moving)")
        elif at_hvt:
            lines.append("No files here — the HVT may be a different node in Zone 3.")

        # Exfiltrated files summary
        exfil_done = getattr(obs, 'exfiltrated_files', [])
        if exfil_done:
            lines.append(f"Already exfiltrated this episode: {', '.join(exfil_done)}")

        # False escalation detection
        if getattr(obs, 'detection_spike_this_step', False):
            lines.append(
                "⚠ WARNING: BLUE detection confidence spiked suddenly — likely FALSE_ESCALATION trap. "
                "Do NOT abort on this signal alone. Verify the trend persists for 2+ steps."
            )

        # Dead drops
        if obs.dead_drops_available:
            lines.append(f"Dead drops available: {len(obs.dead_drops_available)} (read ONLY if context was reset)")
        else:
            lines.append("Dead drops available: NONE")

        # Context reset
        if obs.context_reset_this_step:
            lines.append("*** CONTEXT RESET THIS STEP — READ_DEAD_DROP immediately to recover mission state ***")

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

        # Coordinated sweep signal — when detection is high, direct all BLUE agents
        # to focus on the zone with the highest cumulative anomaly severity.
        if obs.current_detection_confidence > 0.60 and obs.anomaly_feed:
            zone_scores: dict[int, float] = {}
            for anomaly in obs.anomaly_feed:
                z = anomaly.zone
                zone_scores[z] = zone_scores.get(z, 0.0) + anomaly.severity
            if zone_scores:
                hot_zone = max(zone_scores, key=zone_scores.get)
                lines.append(
                    f"COORDINATED SWEEP ACTIVE: All BLUE agents focus on Zone {hot_zone}. "
                    f"Prioritize INVESTIGATE_NODE on Zone {hot_zone} nodes immediately."
                )

        # Change 2: recent traversal patterns — surface RED movement history for BLUE
        if obs.recent_traversal_patterns:
            lines.append("RECENT RED TRAVERSAL PATTERNS (last observed movement signatures):")
            for pattern in obs.recent_traversal_patterns[-4:]:
                from_node = pattern.get("from_node", "?")
                to_node = pattern.get("to_node", "?")
                zone = pattern.get("zone", "?")
                step_p = pattern.get("step", "?")
                lines.append(f"  Step {step_p}: {from_node} → {to_node} (zone {zone})")

        # Change 4b: contextual emergent hints when BLUE detection is stagnant
        if obs.current_detection_confidence < 0.40 and obs.step > 10:
            lines.append(
                "LOW DETECTION: Standard investigation hasn't found RED after 10+ steps. "
                "Consider EMERGENT actions like 'traffic_analysis' or 'shadow_agent' "
                "to break the stalemate."
            )

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
        # LoRA model outputs 'scan' — remap to ESCALATE_PRIVILEGES for RED,
        # or ANALYZE_ANOMALY for BLUE (the closest semantic equivalent).
        if action_type == ActionType.SCAN:
            action_type = ActionType.ESCALATE_PRIVILEGES if self.team == "red" else ActionType.ANALYZE_ANOMALY

        # Handle EMERGENT action — build EmergentAction payload from JSON fields
        # before the team-validation gate (EMERGENT is valid for both teams).
        emergent_payload: EmergentAction | None = None
        if action_type == ActionType.EMERGENT:
            intent = str(data.get("intent", data.get("action_intent", "unknown")))
            emergent_payload = EmergentAction(
                intent=intent,
                target_node=int(data["target_node"]) if data.get("target_node") is not None else None,
                target_file=str(data["target_file"]) if data.get("target_file") else None,
                reasoning=str(data.get("reasoning", ""))[:500],
                expected_effect=str(data.get("expected_effect", ""))[:300],
            )
            logger.info(
                f"{self.agent_id}: EMERGENT action proposed — intent='{intent}'"
            )

        # Validate team-action consistency (EMERGENT passes for both teams).
        # BLUE agents often emit 'wait' (LLMs use it as a generic no-op) — silently
        # remap to the blue equivalent 'stand_down' rather than logging a noisy warning.
        if self.team == "blue" and action_type == ActionType.WAIT:
            action_type = ActionType.STAND_DOWN

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

        # ── Commander meta-action payloads (v2) ──────────────────────
        subagent_spec_payload: SubagentSpec | None = None
        target_subagent_id_val: str | None = None
        if action_type in (
            ActionType.SPAWN_SUBAGENT,
            ActionType.DELEGATE_TASK,
            ActionType.DISMISS_SUBAGENT,
        ):
            # Only commanders may emit meta-actions; everyone else falls back.
            if getattr(self, "role", "") != "commander":
                logger.warning(
                    "%s: meta-action '%s' is commander-only — falling back.",
                    self.agent_id, action_type.value,
                )
                action_type = fallback_type
            else:
                if action_type == ActionType.SPAWN_SUBAGENT:
                    spec_data = data.get("subagent_spec") or data.get("spec") or {}
                    if isinstance(spec_data, dict) and spec_data.get("role_name"):
                        try:
                            subagent_spec_payload = SubagentSpec(
                                role_name=str(spec_data.get("role_name", "")),
                                team=str(spec_data.get("team", self.team)) or self.team,
                                task_brief=str(spec_data.get("task_brief", ""))[:500],
                                lifespan_steps=int(spec_data.get("lifespan_steps", 5)),
                                allowed_actions=spec_data.get("allowed_actions"),
                                use_llm=bool(spec_data.get("use_llm", False)),
                                parent_id=self.agent_id,
                            )
                        except Exception as exc:
                            logger.warning(
                                "%s: malformed SPAWN_SUBAGENT spec (%s) — falling back.",
                                self.agent_id, exc,
                            )
                            action_type = fallback_type
                    else:
                        logger.warning(
                            "%s: SPAWN_SUBAGENT without role_name — falling back.",
                            self.agent_id,
                        )
                        action_type = fallback_type
                elif action_type in (ActionType.DELEGATE_TASK, ActionType.DISMISS_SUBAGENT):
                    target_subagent_id_val = (
                        data.get("target_subagent_id")
                        or data.get("subagent_id")
                        or ""
                    )
                    if not target_subagent_id_val:
                        logger.warning(
                            "%s: %s missing target_subagent_id — falling back.",
                            self.agent_id, action_type.value,
                        )
                        action_type = fallback_type
                    else:
                        target_subagent_id_val = str(target_subagent_id_val)

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
                # AUTO-FILL: if RED is at the HVT and there are files available,
                # silently pick the first un-exfiltrated file. This lets the LLM
                # emit a bare {"action_type":"exfiltrate"} from the HVT and still
                # succeed instead of getting stuck waiting.
                obs = self._current_observation
                auto_filled = False
                if isinstance(obs, RedObservation):
                    files_here = list(getattr(obs, "files_at_current_node", []) or [])
                    already_exfil = set(getattr(obs, "exfiltrated_files", []) or [])
                    remaining = [f for f in files_here if f not in already_exfil]
                    if remaining:
                        target_file = remaining[0]
                        auto_filled = True
                        if not reasoning:
                            reasoning = f"Auto-filled target_file={target_file} at HVT."
                if not auto_filled:
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

        # EMERGENT and meta-actions bypass standard semantic guardrails and trap budget —
        # they are evaluated separately (EmergentEvaluator / SubagentRegistry).
        is_meta = action_type.value in META_ACTIONS
        rejected_action: str = ""
        if (
            action_type != ActionType.EMERGENT
            and not is_meta
            and not self._passes_semantic_guardrails(action_type)
        ):
            rejected_action = action_type.value
            logger.warning(
                f"{self.agent_id}: Semantic guardrail rejected action "
                f"'{rejected_action}'. Falling back to {fallback_type.value}."
            )
            action_type = fallback_type
            target_node = None
            target_file = None
            # Always overwrite reasoning so it matches the ACTUAL action taken
            reasoning = (
                f"Guardrail blocked '{rejected_action}' (pre-conditions not met) — "
                f"falling back to {fallback_type.value}."
            )

        # Special rescue: EXFILTRATE was rejected because RED is not at the HVT.
        # Instead of WAITing (which causes an infinite loop), redirect to MOVE
        # toward the HVT if it is adjacent, or any Zone-3 neighbour otherwise.
        if rejected_action == "exfiltrate" and self.team == "red":
            obs_r = self._current_observation
            if isinstance(obs_r, RedObservation):
                honeypots_r = set(getattr(obs_r, "honeypot_nodes_nearby", []) or [])
                hvt_node: int | None = None
                zone3_node: int | None = None
                for idx, adj_n in enumerate(obs_r.adjacent_nodes or []):
                    ntype = (
                        obs_r.adjacent_node_types[idx].value
                        if idx < len(obs_r.adjacent_node_types)
                        and hasattr(obs_r.adjacent_node_types[idx], "value")
                        else ""
                    )
                    if ntype == "high_value_target" and adj_n not in honeypots_r:
                        hvt_node = int(adj_n)
                        break
                    if adj_n not in honeypots_r and zone3_node is None:
                        for p in (getattr(obs_r, "available_paths", None) or []):
                            if int(p.get("node", -1)) == int(adj_n) and int(p.get("zone", 0)) >= 3:
                                zone3_node = int(adj_n)
                best_move = hvt_node if hvt_node is not None else zone3_node
                if best_move is not None:
                    action_type = ActionType.MOVE
                    target_node = best_move
                    target_file = None
                    reasoning = (
                        f"EXFIL-to-MOVE rescue: not at HVT — "
                        f"moving to n{best_move} to reach exfiltration point."
                    )

        if not is_meta and not self._is_trap_action_allowed(action_type, target_node):
            rejected_trap = action_type.value
            action_type = fallback_type
            target_node = None
            target_file = None
            # Critical: overwrite reasoning to match action — this is the source of
            # 'stand_down_with_trap_intent' mismatches in the mismatch detector.
            reasoning = (
                f"Trap '{rejected_trap}' not allowed (budget or pre-condition check) — "
                f"holding position ({fallback_type.value})."
            )

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
            emergent_data=emergent_payload,
            role=getattr(self, "role", None),
            subagent_spec=subagent_spec_payload,
            target_subagent_id=target_subagent_id_val,
        )

    def _passes_semantic_guardrails(self, action_type: ActionType) -> bool:
        """
        Central semantic safeguards to reduce unstable policy behavior.

        These checks prevent obviously invalid high-impact actions from being
        executed when the current observation does not support them.
        """
        obs = self._current_observation

        # ABORT authority is planner-only for RED team.
        if action_type == ActionType.ABORT and self.team == "red":
            if self.role != "planner":
                return False
            if isinstance(obs, RedObservation):
                blue_conf = float(getattr(obs, "blue_detection_confidence", 0.0))
                # Standard gate: critical suspicion + strong BLUE confirmation
                if obs.estimated_suspicion > 0.85 and blue_conf > 0.70:
                    return True
                # Early-warning gate: both metrics rising very fast
                history = getattr(self, '_suspicion_history', [])
                conf_history = getattr(self, '_blue_conf_history', [])
                if len(history) >= 3 and len(conf_history) >= 3:
                    recent_delta = history[-1] - history[-3]
                    blue_delta = conf_history[-1] - conf_history[-3]
                    if recent_delta > 0.20 and blue_delta > 0.18 and obs.estimated_suspicion > 0.78:
                        return True
                # Stuck gate: allow abort when genuinely trapped (4+ consecutive waits
                # with no progress). Prevents infinite guardrail-rejection loops.
                recent_waits = sum(
                    1 for a in self.action_history[-6:]
                    if a.action_type in (ActionType.WAIT, ActionType.STAND_DOWN)
                )
                if recent_waits >= 4:
                    return True
                return False
            return False

        # EXFILTRATE is only valid at HVT nodes, and only for planner/exfiltrator roles.
        if action_type == ActionType.EXFILTRATE and isinstance(obs, RedObservation):
            # Operative and Analyst should never exfiltrate — that's Planner/Exfiltrator's job
            if self.team == "red" and self.role in ("operative", "analyst"):
                return False
            node_type_val = (
                obs.current_node_type.value
                if hasattr(obs.current_node_type, "value")
                else str(obs.current_node_type)
            )
            return node_type_val == "high_value_target"

        # BLUE alerts should require reasonable evidence to reduce false positives.
        if action_type == ActionType.TRIGGER_ALERT and isinstance(obs, BlueObservation):
            # Lowered to 0.40 — with the new passive detection boost BLUE reaches
            # 40%+ quickly, matching the alert_judge subagent threshold so BLUE
            # commanders can trigger alerts at the same confidence level.
            confidence_ok = float(obs.current_detection_confidence) >= 0.40
            honeypot_ok = bool(obs.honeypots_triggered_this_step)
            return confidence_ok or honeypot_ok

        return True

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
            if not observed_nodes:
                # Anomaly feed is empty this step — allow placement while budget remains.
                # (Empty feed was causing stand_down-with-trap-intent mismatch loop.)
                return obs.investigation_budget_remaining > 0
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

    def _avoid_recent(self, pool: list, lookback: int = 3) -> list:
        """Return pool with recently-visited nodes removed to prevent cycling.

        Falls back to the full pool if deduplication would leave it empty.
        For planner callers, use lookback=6 to break 6-node cycles (25→28→29→25…).
        Pool elements may be ints or dicts with a 'node' key.
        """
        recent = {
            a.target_node for a in self.action_history[-lookback:]
            if a.action_type == ActionType.MOVE and a.target_node is not None
        }

        def _node_key(n) -> int:
            return int(n["node"]) if isinstance(n, dict) else int(n)

        fresh = [n for n in pool if _node_key(n) not in recent]
        return fresh if fresh else pool

    def _avoid_recent_long(self, pool: list[int]) -> list[int]:
        """Extended anti-loop version with 6-step lookback for planner."""
        return self._avoid_recent(pool, lookback=6)

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
