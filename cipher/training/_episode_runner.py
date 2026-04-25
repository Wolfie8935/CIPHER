"""
Episode runner for CIPHER.

Contains the core episode execution logic shared by main.py and the training loop.
Runs one complete episode: initializes state, steps agents, processes actions,
computes rewards.

Owns: single-episode execution, action dispatch, state updates per step.
Does NOT own: multi-episode orchestration (TrainingLoop), agent internals,
or reward function definitions.
"""
from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cipher.agents.base_agent import Action, ActionType, META_ACTIONS
from cipher.agents.red.coordination import clear_danger_map
from cipher.training.episode_memory import record_episode as _record_episode_memory
from cipher.agents.blue.deception_architect import BlueDeceptionArchitect
from cipher.agents.blue.forensics import BlueForensics
from cipher.agents.blue.surveillance import BlueSurveillance
from cipher.agents.blue.threat_hunter import BlueThreatHunter
from cipher.agents.oversight.auditor import OversightAuditor
from cipher.agents.red.analyst import RedAnalyst
from cipher.agents.red.exfiltrator import RedExfiltrator
from cipher.agents.red.operative import RedOperative
from cipher.agents.red.planner import RedPlanner
from cipher.agents.commander import RedCommander, BlueCommander
from cipher.environment.graph import (
    NodeType,
    generate_enterprise_graph,
    get_entry_points,
    get_high_value_target,
    get_honeypot_nodes,
)
from cipher.environment.emergent_evaluator import EmergentEvaluator
from cipher.environment.observation import (
    generate_blue_observation,
    generate_red_observation,
)
from cipher.environment.scenario import ScenarioGenerator
from cipher.environment.state import EpisodeState
from cipher.environment.traps import (
    BlueTrapType,
    RedTrapType,
    TrapRegistry,
)
from cipher.memory.dead_drop import DeadDropVault, build_dead_drop_from_state
from cipher.rewards.blue_reward import compute_blue_reward
from cipher.rewards.oversight_reward import apply_fleet_bonus, compute_oversight_signal
from cipher.rewards.reward_logger import RewardLogger
from cipher.rewards.red_reward import compute_red_reward
from cipher.utils.config import config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)
console = Console(force_terminal=True)
_EPISODE_HISTORY: list[dict[str, Any]] = []
_THOUGHTS_FILE = Path("logs") / "agent_thoughts.jsonl"


def _log_agent_thought(step: int, action: "Action") -> None:
    """Append one agent reasoning entry to agent_thoughts.jsonl for the war room."""
    try:
        _THOUGHTS_FILE.parent.mkdir(exist_ok=True)
        entry = {
            "step":        step,
            "agent_id":    action.agent_id,
            "team":        "red" if str(action.agent_id).startswith("red") else "blue",
            "reasoning":   (action.reasoning or "")[:500],
            "action_type": action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type),
            "target_node": action.target_node,
            "target_file": action.target_file,
            "timestamp":   datetime.now().isoformat(),
        }
        with open(_THOUGHTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception:
        pass

RED_TRAP_ACTIONS = {
    ActionType.PLANT_FALSE_TRAIL,
    ActionType.PLANT_TEMPORAL_DECOY,
    ActionType.PLANT_HONEYPOT_POISON,
    ActionType.WRITE_CORRUPTED_DROP,
}
BLUE_TRAP_ACTIONS = {
    ActionType.PLACE_HONEYPOT,
    ActionType.PLANT_BREADCRUMB,
    ActionType.TRIGGER_FALSE_ESCALATION,
    ActionType.TAMPER_DEAD_DROP,
}
RED_ACTION_TO_TRAP = {
    ActionType.PLANT_FALSE_TRAIL: RedTrapType.FALSE_TRAIL,
    ActionType.PLANT_TEMPORAL_DECOY: RedTrapType.TEMPORAL_DECOY,
    ActionType.PLANT_HONEYPOT_POISON: RedTrapType.HONEYPOT_POISON,
    ActionType.WRITE_CORRUPTED_DROP: RedTrapType.DEAD_DROP_CORRUPTION,
}
BLUE_ACTION_TO_TRAP = {
    ActionType.PLACE_HONEYPOT: BlueTrapType.HONEYPOT,
    ActionType.PLANT_BREADCRUMB: BlueTrapType.BREADCRUMB,
    ActionType.TRIGGER_FALSE_ESCALATION: BlueTrapType.FALSE_ESCALATION,
    ActionType.TAMPER_DEAD_DROP: BlueTrapType.DEAD_DROP_TAMPER,
}


@dataclass
class ActionExecutionResult:
    """Structured action execution outcome for episode traces."""

    success: bool
    reason: str
    state_delta: dict[str, Any]

    def to_dict(self, **extras: Any) -> dict[str, Any]:
        base = {
            "success": self.success,
            "reason": self.reason,
            "state_delta": self.state_delta,
        }
        base.update(extras)
        return base


def run_episode(
    scenario: Any | None = None,
    graph: Any | None = None,
    cfg: Any | None = None,
    max_steps: int = 10,
    verbose: bool = True,
    save_trace: bool = False,
    episode_number: int = 1,
    debug_force_exfil_sanity: bool = False,
    debug_trace_state: bool = False,
    scripted_red_actions: dict[int, list[Action]] | None = None,
    scripted_blue_actions: dict[int, list[Action]] | None = None,
    stream_progress: bool = False,
    step_callback=None,
    difficulty_params: dict | None = None,
) -> Any:
    """
    Run a single CIPHER episode.

    Args:
        episode_number:   Sequential episode number.
        max_steps:        Maximum steps in this episode.
        verbose:          If True, print rich terminal output.
        save_trace:       If True, save episode trace JSON to disk.
        difficulty_params: Optional dict from DynamicDifficultyController
                          with keys: honeypot_density, target_files,
                          blue_response_delay, graph_size,
                          zone_lockdown_threshold, trap_budget_blue.

    Returns:
        Tuple of (red_total_reward, blue_total_reward).
    """
    cfg = cfg or config
    dp = difficulty_params or {}

    if isinstance(scenario, int):
        episode_number = scenario
        scenario = None
    return_payload_mode = scenario is not None and graph is not None

    # ── Generate scenario ────────────────────────────────────────
    if scenario is None:
        scenario_gen = ScenarioGenerator()
        scenario = scenario_gen.generate(episode_number)
    else:
        episode_number = getattr(scenario, "episode_number", episode_number)

    # ── Apply difficulty overrides to scenario fields ─────────────
    if dp:
        if "target_files" in dp:
            n_tf = int(dp["target_files"])
            ep_seed = getattr(scenario, "episode_seed", episode_number * 7919)
            scenario.target_files = [
                f"target_file_{ep_seed}_{i:03d}" for i in range(n_tf)
            ]
        if "zone_lockdown_threshold" in dp:
            zlt = float(dp["zone_lockdown_threshold"])
            scenario.zone_lockdown_level = {
                0: min(1.0, 0.1 + (0.9 - zlt) * 0.2),
                1: min(1.0, 0.2 + (0.9 - zlt) * 0.3),
                2: min(1.0, 0.4 + (0.9 - zlt) * 0.4),
                3: min(1.0, 0.6 + (0.9 - zlt) * 0.5),
            }

    # ── Graph size and honeypot density from difficulty ────────────
    _graph_size        = int(dp.get("graph_size", cfg.env_graph_size))
    _honeypot_density  = float(dp.get("honeypot_density", cfg.env_honeypot_density))
    _blue_resp_delay   = int(dp.get("blue_response_delay", 0))
    _trap_budget_blue  = int(dp.get("trap_budget_blue", getattr(cfg, "env_trap_budget_blue", 5)))
    _zone_lockdown_thr = float(dp.get("zone_lockdown_threshold", 0.70))

    if graph is None:
        graph = getattr(scenario, "generated_graph", None)
        if graph is None:
            graph = generate_enterprise_graph(
                n_nodes=_graph_size,
                honeypot_density=_honeypot_density,
                seed=scenario.episode_seed,
            )
        elif dp:
            # Caller passed a graph that was already built by loop.py using
            # difficulty_params — accept it as-is.
            pass

    # ── Resolve scenario against actual graph ────────────────────
    entry_points = get_entry_points(graph)
    hvt_node = get_high_value_target(graph)
    scenario.red_start_node = entry_points[0] if entry_points else 0
    scenario.high_value_target_node = hvt_node

    # Place target files at HVT node
    hvt_files = graph.nodes[hvt_node].get("files", [])
    if hvt_files:
        scenario.target_files = hvt_files[:1]
    else:
        # Ensure HVT has target files
        scenario.target_files = [
            f"target_{hvt_node}_{i}" for i in range(1)
        ]
        graph.nodes[hvt_node]["files"] = list(scenario.target_files)

    # ── Initialize state ─────────────────────────────────────────
    state = EpisodeState(
        graph=graph,
        red_current_node=scenario.red_start_node,
        red_visited_nodes=[scenario.red_start_node],
    )

    # ── Initialize vault ─────────────────────────────────────────
    vault = DeadDropVault(
        vault_dir=cfg.drop_vault_dir,
        max_tokens_per_drop=cfg.env_dead_drop_max_tokens,
    )
    vault.clear()

    # ── Initialize trap registry (Phase 5) ────────────────────────
    trap_registry = TrapRegistry(cfg)
    # Apply difficulty trap budget override for BLUE
    if hasattr(trap_registry, "blue_budget"):
        trap_registry.blue_budget = _trap_budget_blue
    state.trap_registry = trap_registry
    # Store difficulty metadata on state for trace + reward logger
    state._difficulty_params = dp
    state._blue_response_delay = _blue_resp_delay
    state._zone_lockdown_threshold = _zone_lockdown_thr
    state._zone_entry_boost  = float(dp.get("zone_entry_boost", 0.12))
    state._min_episode_steps = int(dp.get("min_episode_steps", 5))
    state.red_path_history = [scenario.red_start_node]

    # ── Clear shared RED danger map for this episode ──────────────
    clear_danger_map()

    # ── Initialize emergent action evaluator ──────────────────────
    emergent_eval = EmergentEvaluator()

    # ── Initialize agents ────────────────────────────────────────
    arch_mode = str(getattr(cfg, "cipher_agent_arch", "v2")).strip().lower()
    use_v2 = arch_mode == "v2"

    red_commander: RedCommander | None = None
    blue_commander: BlueCommander | None = None

    if use_v2:
        red_commander = RedCommander("red_commander_01", cfg)
        blue_commander = BlueCommander("blue_commander_01", cfg)
        red_commander.reset_episode()
        blue_commander.reset_episode()
        # Keep these names for the rest of the runner — they only need to
        # iterate something with .observe()/.act() in the legacy path; in v2
        # the commanders ARE the team representatives.
        red_agents = [red_commander]
        blue_agents = [blue_commander]
        # forensics_agent is consumed by compute_blue_reward for graph reconstruction.
        # In v2 the role is owned by the commander's spawned 'forensics' subagent;
        # the reward function only needs an agent-ish object with .last_reasoning
        # and .agent_id, so we hand it a BlueForensics fallback instance for parity.
        forensics_agent = BlueForensics("blue_forensics_legacy_proxy", cfg)
    else:
        red_agents = [
            RedPlanner("red_planner_01", cfg),
            RedAnalyst("red_analyst_01", cfg),
            RedOperative("red_operative_01", cfg),
            RedExfiltrator("red_exfiltrator_01", cfg),
        ]
        blue_agents = [
            BlueSurveillance("blue_surveillance_01", cfg),
            BlueThreatHunter("blue_threat_hunter_01", cfg),
            BlueDeceptionArchitect("blue_deception_architect_01", cfg),
            BlueForensics("blue_forensics_01", cfg),
        ]
        forensics_agent = blue_agents[-1]

    oversight_auditor = OversightAuditor(cfg)

    # Track metrics
    steps_to_first_detection: int | None = None
    confirmed_detection_step: int | None = None
    action_reason_mismatch_count = 0
    oversight_flags: list[dict[str, Any]] = []
    oversight_step_penalty_red = 0.0
    oversight_step_penalty_blue = 0.0

    if debug_force_exfil_sanity:
        state.red_current_node = scenario.high_value_target_node
        if state.red_current_node not in state.red_path_history:
            state.red_path_history.append(state.red_current_node)
        if state.red_current_node not in state.red_visited_nodes:
            state.red_visited_nodes.append(state.red_current_node)
        sanity_target = scenario.target_files[0]
        sanity_action = Action(
            agent_id="debug_exfil_sanity",
            action_type=ActionType.EXFILTRATE,
            target_file=sanity_target,
            reasoning="Forced exfil sanity check.",
        )
        sanity_result = _process_red_action(
            sanity_action,
            state,
            vault,
            scenario,
            exfil_success_this_step=False,
        )
        if len(state.red_exfiltrated_files) == 0:
            raise AssertionError(
                "DEBUG_EXFIL sanity failed: exfiltration did not update state.red_exfiltrated_files"
            )
        logger.debug("DEBUG_EXFIL sanity passed: %s", sanity_result)

    # ── Print startup banner ─────────────────────────────────────
    if verbose:
        _print_banner(episode_number, graph_size, len(red_agents) + len(blue_agents))

    # ── Episode loop ─────────────────────────────────────────────
    for step in range(1, max_steps + 1):
        state.step = step

        # ── Check context reset ──────────────────────────────────
        context_reset = (
            step > 1
            and step % cfg.env_context_reset_interval == 0
        )

        if context_reset:
            state.red_context_resets += 1
            for agent in red_agents:
                agent.reset()
            if verbose:
                _print_memento_reset(step, vault)

        # ── Generate observations ────────────────────────────────
        drop_paths = [
            p for p in vault.list_all_drop_paths()
        ]
        drops_at_node = vault.read(state.red_current_node)
        drop_paths_at_node = [
            str(config.drop_vault_dir / f"{d.dead_drop_id}.drop")
            for d in drops_at_node
        ] if drops_at_node else []

        red_obs = generate_red_observation(
            state,
            dead_drop_paths=drop_paths_at_node,
            context_reset_this_step=context_reset,
            hvt_node_id=hvt_node,
        )
        blue_obs = generate_blue_observation(state)

        red_actions: list[Action] = []
        blue_actions: list[Action] = []
        red_planner_action: Action | None = None

        # ── RED agents act ───────────────────────────────────────
        red_abort_triggered = False
        exfil_success_this_step = False
        scripted_red = scripted_red_actions.get(step) if scripted_red_actions else None
        if scripted_red is not None:
            red_action_stream: list[Action] = scripted_red
        elif use_v2 and red_commander is not None:
            # v2 — Commander observes once and emits primitive actions for itself
            # plus all alive subagents. Meta-actions are consumed internally.
            from cipher.utils.llm_mode import is_live_mode, is_hybrid_mode
            _parallel = is_live_mode() or is_hybrid_mode()
            red_commander.observe(red_obs)
            try:
                red_action_stream = red_commander.act_step(step=step, parallel=_parallel)
            except Exception as exc:
                logger.error(f"RED commander failed: {exc}")
                red_action_stream = [Action(
                    agent_id=red_commander.agent_id,
                    action_type=ActionType.WAIT,
                    reasoning=f"Commander exception — safe fallback: {exc}",
                    role="commander",
                )]
        else:
            # ── Legacy v1: 4 fixed RED agents call LLM simultaneously ──────
            from cipher.utils.llm_mode import is_live_mode, is_hybrid_mode
            _parallel = is_live_mode() or is_hybrid_mode()
            if _parallel:
                # observe first (sequential, lightweight)
                for agent in red_agents:
                    agent.observe(red_obs)
                red_action_stream = [None] * len(red_agents)
                with ThreadPoolExecutor(max_workers=len(red_agents)) as pool:
                    futures = {pool.submit(a.act): i for i, a in enumerate(red_agents)}
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        try:
                            red_action_stream[idx] = fut.result()
                        except Exception as exc:
                            logger.error(f"RED agent {red_agents[idx].agent_id} error: {exc}")
                            red_action_stream[idx] = Action(
                                agent_id=red_agents[idx].agent_id,
                                action_type=ActionType.WAIT,
                                reasoning="Agent exception — safe fallback.",
                            )
            else:
                red_action_stream = []
                for agent in red_agents:
                    agent.observe(red_obs)
                    red_action_stream.append(agent.act())

        for action in red_action_stream:
            action.step = step
            red_actions.append(action)
            if action.agent_id.startswith("red_planner"):
                red_planner_action = action

            result = _process_red_action(
                action,
                state,
                vault,
                scenario,
                exfil_success_this_step=exfil_success_this_step,
                emergent_eval=emergent_eval,
                graph=graph,
            )
            if result.get("reason", "").startswith("exfil_"):
                exfil_success_this_step = result.get("success", False) or exfil_success_this_step

            _red_payload: dict[str, Any] = {
                "target_node": action.target_node,
                "target_file": action.target_file,
                "reasoning": action.reasoning,
                "trap_params": action.trap_params or {},
            }
            if action.action_type == ActionType.EMERGENT and action.emergent_data:
                _red_payload["emergent_intent"] = action.emergent_data.intent
                _red_payload["emergent_expected_effect"] = action.emergent_data.expected_effect
            state.log_action(
                agent_id=action.agent_id,
                action_type=action.action_type.value,
                action_payload=_red_payload,
                result=result,
            )
            _log_agent_thought(step, action)
            if debug_trace_state or os.getenv("DEBUG_EXFIL", "0") == "1":
                logger.debug(
                    "EXFIL STATE step=%s after %s: %s",
                    step,
                    action.agent_id,
                    list(state.red_exfiltrated_files),
                )
            if verbose:
                _print_action(step, action, state, "red")
            if state.terminal_reason == "aborted":
                red_abort_triggered = True
                break

        if red_abort_triggered:
            step_flags = oversight_auditor.evaluate_step(
                step=step,
                state=state,
                red_actions=red_actions,
                blue_actions=[],
            )
            for flag in step_flags:
                p_red, p_blue = _oversight_penalty_from_flag(flag)
                oversight_step_penalty_red += p_red
                oversight_step_penalty_blue += p_blue
                flag_payload = flag.to_dict()
                oversight_flags.append(flag_payload)
                state.log_action(
                    agent_id=oversight_auditor.AGENT_ID,
                    action_type="OVERSIGHT_FLAG",
                    action_payload=flag_payload,
                    result={
                        "applied_penalty_red": round(p_red, 4),
                        "applied_penalty_blue": round(p_blue, 4),
                    },
                )
            logger.debug("ABORT triggered: skipping BLUE actions for step %s", step)
            break

        # ── BLUE agents act ──────────────────────────────────────
        scripted_blue = scripted_blue_actions.get(step) if scripted_blue_actions else None
        if scripted_blue is not None:
            blue_action_stream: list[Action] = scripted_blue
        elif use_v2 and blue_commander is not None:
            from cipher.utils.llm_mode import is_live_mode, is_hybrid_mode
            _parallel_blue = is_live_mode() or is_hybrid_mode()
            # propagate discovered_drop_paths to commander + every alive subagent
            setattr(blue_commander, "_blue_discovered_drop_paths", list(state.blue_discovered_drop_paths))
            for sub in blue_commander.registry.alive():
                setattr(sub, "_blue_discovered_drop_paths", list(state.blue_discovered_drop_paths))
            blue_commander.observe(blue_obs)
            # 5.md: blue_response_delay — BLUE stands down for first N steps
            _blue_delayed = step <= getattr(state, "_blue_response_delay", 0)
            if _blue_delayed:
                blue_action_stream = [Action(
                    agent_id=blue_commander.agent_id,
                    action_type=ActionType.STAND_DOWN,
                    reasoning=f"Difficulty delay: BLUE inactive until step {state._blue_response_delay + 1}.",
                    role="commander",
                )]
            else:
                try:
                    blue_action_stream = blue_commander.act_step(step=step, parallel=_parallel_blue)
                except Exception as exc:
                    logger.error(f"BLUE commander failed: {exc}")
                    blue_action_stream = [Action(
                        agent_id=blue_commander.agent_id,
                        action_type=ActionType.STAND_DOWN,
                        reasoning=f"Commander exception — safe fallback: {exc}",
                        role="commander",
                    )]
        else:
            # ── Legacy v1: 4 fixed BLUE agents call LLM simultaneously ─────
            from cipher.utils.llm_mode import is_live_mode, is_hybrid_mode
            _parallel_blue = is_live_mode() or is_hybrid_mode()
            if _parallel_blue:
                for agent in blue_agents:
                    setattr(agent, "_blue_discovered_drop_paths", list(state.blue_discovered_drop_paths))
                    agent.observe(blue_obs)
                blue_action_stream = [None] * len(blue_agents)
                with ThreadPoolExecutor(max_workers=len(blue_agents)) as pool:
                    futures = {pool.submit(a.act): i for i, a in enumerate(blue_agents)}
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        try:
                            blue_action_stream[idx] = fut.result()
                        except Exception as exc:
                            logger.error(f"BLUE agent {blue_agents[idx].agent_id} error: {exc}")
                            blue_action_stream[idx] = Action(
                                agent_id=blue_agents[idx].agent_id,
                                action_type=ActionType.STAND_DOWN,
                                reasoning="Agent exception — safe fallback.",
                            )
            else:
                blue_action_stream = []
                for agent in blue_agents:
                    setattr(agent, "_blue_discovered_drop_paths", list(state.blue_discovered_drop_paths))
                    agent.observe(blue_obs)
                    blue_action_stream.append(agent.act())

        for action in blue_action_stream:
            action.step = step
            blue_actions.append(action)

            result = _process_blue_action(action, state, emergent_eval=emergent_eval, graph=graph)
            mismatch_reason = _detect_action_reason_mismatch(action)
            if mismatch_reason:
                action_reason_mismatch_count += 1
                result["action_reason_mismatch"] = mismatch_reason
                logger.warning(
                    "Action-reason mismatch (%s): agent=%s action=%s reasoning=%s",
                    mismatch_reason,
                    action.agent_id,
                    action.action_type.value,
                    action.reasoning[:120],
                )
            if steps_to_first_detection is None and state.blue_detection_confidence > 0.5:
                steps_to_first_detection = step
            if confirmed_detection_step is None and state.blue_detection_confidence >= 0.8:
                confirmed_detection_step = step
            _blue_payload: dict[str, Any] = {
                "target_node": action.target_node,
                "reasoning": action.reasoning,
                "trap_params": action.trap_params or {},
            }
            if action.action_type == ActionType.EMERGENT and action.emergent_data:
                _blue_payload["emergent_intent"] = action.emergent_data.intent
                _blue_payload["emergent_expected_effect"] = action.emergent_data.expected_effect
            state.log_action(
                agent_id=action.agent_id,
                action_type=action.action_type.value,
                action_payload=_blue_payload,
                result=result,
            )
            _log_agent_thought(step, action)
            if verbose:
                _print_action(step, action, state, "blue")

        # Blue enforcement: if confidence is already high and anomalies exist,
        # force one alert this step when policy outputs only passive actions.
        # This converts detection progress into decisive interruption pressure.
        if (
            blue_obs.anomaly_feed
            and state.blue_detection_confidence >= 0.72
            and not any(a.action_type == ActionType.TRIGGER_ALERT for a in blue_actions)
        ):
            forced_target = max(blue_obs.anomaly_feed, key=lambda a: a.severity).node_id
            forced_alert = Action(
                agent_id="blue_command_enforcer",
                action_type=ActionType.TRIGGER_ALERT,
                target_node=forced_target,
                reasoning=(
                    f"[AUTO-ESCALATE] Detection {state.blue_detection_confidence:.2f} "
                    f"with active anomalies — forcing alert at node {forced_target}."
                ),
                role="commander",
            )
            forced_alert.step = step
            blue_actions.append(forced_alert)
            result = _process_blue_action(forced_alert, state, emergent_eval=emergent_eval, graph=graph)
            state.log_action(
                agent_id=forced_alert.agent_id,
                action_type=forced_alert.action_type.value,
                action_payload={
                    "target_node": forced_alert.target_node,
                    "reasoning": forced_alert.reasoning,
                    "trap_params": {},
                },
                result=result,
            )
            _log_agent_thought(step, forced_alert)
            if verbose:
                _print_action(step, forced_alert, state, "blue")

        # ── v2: persist subagent lifecycle events into episode log ──
        # Use a per-commander cursor so we keep the canonical events list
        # intact in the registry (used by the trace metadata at episode end).
        if use_v2:
            for commander in (red_commander, blue_commander):
                if commander is None:
                    continue
                cursor_attr = "_lifecycle_cursor"
                cursor = getattr(commander, cursor_attr, 0)
                events = commander.registry.events
                while cursor < len(events):
                    ev = events[cursor]
                    if ev.step > step:
                        break
                    state.log_action(
                        agent_id=ev.subagent_id or commander.agent_id,
                        action_type=f"subagent_{ev.event_type}",
                        action_payload={
                            "role": ev.role,
                            "team": ev.team,
                            "parent_id": ev.parent_id,
                            "reason": ev.reason,
                        },
                        result={"event_type": ev.event_type},
                    )
                    cursor += 1
                setattr(commander, cursor_attr, cursor)

        # ── Passive detection boost (applied before terminal check) ─────
        # Each step, RED's suspicion generates detectable anomalies proportionally.
        # Start earlier so fast RED runs do not always beat BLUE's detection curve.
        # Placed here so the co-detection check can fire in the same step as
        # RED's last move/exfiltration, giving BLUE a fair chance to interrupt.
        if state.step > 3:
            _passive_boost = round(float(state.red_suspicion_score) * 0.025, 4)
            if state.red_current_node is not None:
                try:
                    z = state.get_zone_for_node(state.red_current_node)
                    if z is not None and int(z) >= 2:
                        _passive_boost = round(_passive_boost + 0.01, 4)
                except Exception:
                    pass
            if _passive_boost > 0:
                state.blue_detection_confidence = min(
                    1.0, state.blue_detection_confidence + _passive_boost
                )

        # ── High-confidence co-detection terminal ─────────────────────────
        # BLUE wins when both detection confidence AND RED suspicion cross thresholds.
        # Slightly lowered so BLUE can convert strong evidence before instant exfil.
        # Applied before is_done() so it can interrupt a final exfiltration step.
        if (
            not state.is_terminal
            and state.blue_detection_confidence >= 0.78
            and state.red_suspicion_score >= 0.72
        ):
            state.is_terminal = True
            state.terminal_reason = "detected"
            logger.info(
                "High-confidence co-detection terminal fired: "
                "det=%.2f susp=%.2f",
                state.blue_detection_confidence,
                state.red_suspicion_score,
            )

        # ── Check terminal conditions ────────────────────────────
        if state.is_done():
            # 5.md min_episode_steps: if RED wins before minimum steps, let the
            # episode continue so BLUE has a chance to contest and the log is richer.
            # BLUE detection and max_steps terminations are never held back.
            _min_ep = getattr(state, "_min_episode_steps", 0)
            if (
                state.terminal_reason == "exfiltration_complete"
                and state.step < _min_ep
            ):
                state.is_terminal = False   # hold — keep simulating
            else:
                break

        all_actions_this_step = red_actions + blue_actions
        for action in all_actions_this_step:
            if action.action_type in RED_TRAP_ACTIONS:
                trap_type = RED_ACTION_TO_TRAP[action.action_type]
                success, reason = state.trap_registry.place_red_trap(
                    trap_type=trap_type,
                    placed_by=action.agent_id,
                    target_node=action.target_node or state.red_current_node,
                    step=state.step,
                    params=action.trap_params or {},
                    state=state,
                )
                if success:
                    state.red_traps_placed_count += 1
                else:
                    logger.debug(f"Trap placement rejected: {reason}")
            elif action.action_type in BLUE_TRAP_ACTIONS:
                trap_type = BLUE_ACTION_TO_TRAP[action.action_type]
                success, reason = state.trap_registry.place_blue_trap(
                    trap_type=trap_type,
                    placed_by=action.agent_id,
                    target_node=action.target_node or state.red_current_node,
                    step=state.step,
                    params=action.trap_params or {},
                    state=state,
                )
                if success:
                    state.blue_traps_placed_count += 1
                else:
                    logger.debug(f"Trap placement rejected: {reason}")

        trap_events = state.trap_registry.evaluate_step(
            step=state.step,
            last_red_action=red_planner_action or (red_actions[0] if red_actions else None),
            last_blue_actions=blue_actions,
            state=state,
            graph=graph,
            vault=vault,
        )
        for event in trap_events:
            state.record_trap_event(event)
            logger.info(f"TRAP EVENT: {event.effect_description}")
            if verbose:
                _print_trap_event(step, event)

        for action in blue_actions:
            if action.action_type == ActionType.INVESTIGATE_NODE and action.target_node is not None:
                discovered = vault.get_drops_at_node(action.target_node)
                state.blue_discovered_drop_paths.extend(discovered)
                if discovered:
                    logger.debug(
                        f"BLUE discovered {len(discovered)} drop(s) at node {action.target_node}"
                    )

        step_flags = oversight_auditor.evaluate_step(
            step=step,
            state=state,
            red_actions=red_actions,
            blue_actions=blue_actions,
        )
        for flag in step_flags:
            p_red, p_blue = _oversight_penalty_from_flag(flag)
            oversight_step_penalty_red += p_red
            oversight_step_penalty_blue += p_blue
            flag_payload = flag.to_dict()
            oversight_flags.append(flag_payload)
            state.log_action(
                agent_id=oversight_auditor.AGENT_ID,
                action_type="OVERSIGHT_FLAG",
                action_payload=flag_payload,
                result={
                    "applied_penalty_red": round(p_red, 4),
                    "applied_penalty_blue": round(p_blue, 4),
                },
            )

        # Compact per-step progress line for live/hybrid mode
        if stream_progress:
            _print_step_progress(step, max_steps, red_actions, blue_actions, state)

        # External step callback (used by main.py for live ticker)
        if step_callback is not None:
            try:
                step_callback(step, max_steps, red_actions, blue_actions, state)
            except Exception:
                pass

        # Check max steps
        if step >= max_steps:
            state.is_terminal = True
            state.terminal_reason = "max_steps"

    # ── Compute rewards ──────────────────────────────────────────
    setattr(state, "blue_first_detection_step", steps_to_first_detection)
    setattr(state, "blue_confirmed_detection_step", confirmed_detection_step)

    # In v2 prefer the live commander-spawned forensics subagent (its legacy
    # proxy accumulated the suspected_path during the episode).
    forensics_for_reward = forensics_agent
    if use_v2 and blue_commander is not None:
        for sub in blue_commander.registry.alive():
            if sub.role == "forensics" and getattr(sub, "_legacy_proxy", None) is not None:
                forensics_for_reward = sub._legacy_proxy
                break

    red_reward = compute_red_reward(state, scenario, vault, cfg)
    blue_reward = compute_blue_reward(state, graph, forensics_for_reward, cfg)
    oversight = compute_oversight_signal(state, _EPISODE_HISTORY, cfg)

    red_reward.total += oversight.total_red_adjustment
    blue_reward.total += oversight.total_blue_adjustment

    red_action_log = []
    blue_action_log = []
    for entry in state.episode_log:
        agent_id = str(entry.get("agent_id", ""))
        if not (agent_id.startswith("red_") or agent_id.startswith("blue_")):
            continue
        payload = entry.get("payload", {})
        log_item = {
            "step": entry.get("step", 0),
            "agent": agent_id,
            "action": entry.get("action_type", ""),
            "target": payload.get("target_node") if isinstance(payload, dict) else None,
            "reasoning": payload.get("reasoning", "") if isinstance(payload, dict) else "",
        }
        if agent_id.startswith("red_"):
            red_action_log.append(log_item)
        else:
            blue_action_log.append(log_item)

    judgment = oversight_auditor.judge_episode(state, red_action_log, blue_action_log)
    apply_fleet_bonus(red_reward, blue_reward, judgment)
    red_reward.total += oversight_step_penalty_red
    blue_reward.total += oversight_step_penalty_blue

    red_reward.total = round(red_reward.total, 4)
    blue_reward.total = round(blue_reward.total, 4)

    logger_instance = RewardLogger()
    logger_instance.log(
        episode=episode_number,
        steps=state.step,
        terminal_reason=state.terminal_reason or "max_steps",
        red=red_reward,
        blue=blue_reward,
        oversight=oversight,
        judgment=judgment,
        difficulty_params=dp,
    )

    _EPISODE_HISTORY.append(
        {
            "episode": episode_number,
            "red_complexity_multiplier": red_reward.operation_complexity_multiplier,
            "red_unique_nodes": red_reward.unique_nodes_visited,
            "blue_detection_confidence_final": state.blue_detection_confidence,
        }
    )

    # Change 3: record episode summary into cross-episode memory for agent prompts
    _red_emergent_intents: list[str] = []
    _blue_emergent_intents: list[str] = []
    for _entry in state.episode_log:
        _aid = str(_entry.get("agent_id", ""))
        if _entry.get("action_type", "") == "emergent":
            _payload = _entry.get("payload", {})
            _intent = (_payload.get("emergent_intent") or "") if isinstance(_payload, dict) else ""
            if _aid.startswith("red_") and _intent:
                _red_emergent_intents.append(_intent)
            elif _aid.startswith("blue_") and _intent:
                _blue_emergent_intents.append(_intent)

    _winner = "draw"
    if state.terminal_reason == "exfiltration_complete":
        _winner = "red"
    elif state.terminal_reason in ("detected", "aborted"):
        _winner = "blue"

    if episode_number % 5 == 0:
        _record_episode_memory({
            "episode_number": episode_number,
            "winner": _winner,
            "terminal_reason": state.terminal_reason or "max_steps",
            "steps": state.step,
            "red_total_reward": round(red_reward.total, 3),
            "blue_total_reward": round(blue_reward.total, 3),
            "red_emergent_used": bool(_red_emergent_intents),
            "red_emergent_intents": _red_emergent_intents,
            "blue_emergent_used": bool(_blue_emergent_intents),
            "blue_emergent_intents": _blue_emergent_intents,
            "red_exfiltrated": len(state.red_exfiltrated_files),
            "red_suspicion_final": round(state.red_suspicion_score, 3),
            "blue_detection_final": round(state.blue_detection_confidence, 3),
            "zone_stall_occurred": any(
                isinstance(_e.get("payload"), dict) and "ZONE STALL" in str(_e.get("payload", {}).get("reasoning", ""))
                for _e in state.episode_log
            ),
        })

    if verbose:
        _print_rewards(red_reward, blue_reward, oversight, judgment)

    # ── Save trace ───────────────────────────────────────────────
    if save_trace:
        commander_meta: dict[str, Any] | None = None
        if use_v2:
            commander_meta = {
                "arch": "v2",
                "red_commander": {
                    "agent_id": red_commander.agent_id if red_commander else None,
                    "final_roster": red_commander.roster_snapshot() if red_commander else [],
                    "lifecycle": red_commander.lifecycle_events() if red_commander else [],
                    "spawn_budget_remaining": (
                        red_commander.registry.spawn_budget_remaining if red_commander else 0
                    ),
                    "total_spawns": (
                        red_commander.registry.total_spawns if red_commander else 0
                    ),
                },
                "blue_commander": {
                    "agent_id": blue_commander.agent_id if blue_commander else None,
                    "final_roster": blue_commander.roster_snapshot() if blue_commander else [],
                    "lifecycle": blue_commander.lifecycle_events() if blue_commander else [],
                    "spawn_budget_remaining": (
                        blue_commander.registry.spawn_budget_remaining if blue_commander else 0
                    ),
                    "total_spawns": (
                        blue_commander.registry.total_spawns if blue_commander else 0
                    ),
                },
            }
        _save_episode_trace(state, episode_number, verbose=verbose, commander_meta=commander_meta, difficulty_params=dp)

    result_payload = {
        "red_reward": red_reward,
        "blue_reward": blue_reward,
        "oversight": oversight,
        "judgment": judgment,
        "state": state,
        "scenario": scenario,
        "action_reason_mismatch_count": action_reason_mismatch_count,
        "oversight_flags": oversight_flags,
        "oversight_step_penalty_red": round(oversight_step_penalty_red, 4),
        "oversight_step_penalty_blue": round(oversight_step_penalty_blue, 4),
        "difficulty_params": dp,
    }
    if return_payload_mode:
        return result_payload
    return red_reward.total, blue_reward.total


def _process_red_action(
    action: Action,
    state: EpisodeState,
    vault: DeadDropVault,
    scenario: Any,
    exfil_success_this_step: bool = False,
    emergent_eval: EmergentEvaluator | None = None,
    graph: Any = None,
) -> dict[str, Any]:
    """Process a RED team action and update state accordingly."""
    result = ActionExecutionResult(
        success=True,
        reason="ok",
        state_delta={},
    ).to_dict()

    if action.action_type == ActionType.MOVE:
        if action.target_node is not None and state.graph.has_edge(
            state.red_current_node, action.target_node
        ):
            edge_data = state.graph.edges[
                state.red_current_node, action.target_node
            ]
            suspicion_delta = edge_data.get("suspicion_delta", 0.02)
            protocol = edge_data.get("protocol", "ssh")

            # Zone-entry suspicion boost (5.md difficulty axis):
            # When RED advances to a strictly higher zone, apply extra suspicion.
            # This forces longer episodes — RED cannot rush from Zone 0 to Zone 3
            # in 3 frictionless hops.
            _from_zone = state.graph.nodes[state.red_current_node].get("zone", 0)
            _to_zone   = state.graph.nodes[action.target_node].get("zone", 0)
            _fz = _from_zone.value if hasattr(_from_zone, "value") else int(_from_zone)
            _tz = _to_zone.value   if hasattr(_to_zone,   "value") else int(_to_zone)
            if _tz > _fz:
                _zone_boost = float(getattr(state, "_zone_entry_boost", 0.12))
                suspicion_delta += _zone_boost
                state.blue_detection_confidence = min(
                    1.0, state.blue_detection_confidence + _zone_boost * 0.5
                )
                result["state_delta"]["zone_entry_boost"] = round(_zone_boost, 3)

            # Record movement using new Phase 2 method
            state.record_movement(
                from_node=state.red_current_node,
                to_node=action.target_node,
                protocol=protocol,
                suspicion_cost=suspicion_delta
            )

            # Check if moved to honeypot
            # NOTE: suspicion spike (+0.4) is applied by the trap system in traps.py
            # Do NOT double-apply it here — only set the flag and boost detection.
            node_type = state.graph.nodes[action.target_node].get("node_type")
            if node_type == NodeType.HONEYPOT:
                state.blue_honeypots_triggered.append(action.target_node)
                state.blue_detection_confidence = min(
                    1.0, state.blue_detection_confidence + 0.3
                )
                result["state_delta"]["honeypot_triggered"] = True
        else:
            result["success"] = False
            result["reason"] = "invalid_move"

    elif action.action_type == ActionType.READ_FILE:
        if action.target_file and action.target_file in state.graph.nodes[
            state.red_current_node
        ].get("files", []):
            state.update_suspicion(0.03)
            result["reason"] = "read_success"
            result["file_read"] = action.target_file
            result["state_delta"]["file_read"] = action.target_file
        else:
            result["success"] = False
            result["reason"] = "file_not_found"

    elif action.action_type == ActionType.EXFILTRATE:
        node_type = state.graph.nodes[state.red_current_node].get("node_type")
        if node_type == NodeType.HIGH_VALUE_TARGET and action.target_file:
            if action.target_file in scenario.target_files:
                state.red_exfiltrated_files.append(action.target_file)
                state.update_suspicion(0.15)
                result["reason"] = "exfil_success"
                result["exfiltrated"] = action.target_file
                result["state_delta"]["exfiltrated"] = action.target_file
                result["state_delta"]["red_exfiltrated_files"] = list(
                    state.red_exfiltrated_files
                )

                # Check if all target files exfiltrated
                if set(scenario.target_files).issubset(
                    set(state.red_exfiltrated_files)
                ):
                    state.is_terminal = True
                    state.terminal_reason = "exfiltration_complete"
                    result["reason"] = "exfil_complete"
            else:
                result["success"] = False
                result["reason"] = "not_a_target_file"
                result["state_delta"]["target_file"] = action.target_file
        else:
            result["success"] = False
            result["reason"] = "not_at_hvt"
            result["state_delta"]["current_node"] = state.red_current_node

    elif action.action_type == ActionType.WRITE_DEAD_DROP:
        from cipher.environment.observation import RedObservation

        red_obs = generate_red_observation(state)
        drop = build_dead_drop_from_state(
            agent_id=action.agent_id,
            state=state,
            red_obs=red_obs,
            directive="Continue mission. Check suspicion and proceed toward target.",
        )
        file_path, efficiency = vault.write(drop, state.red_current_node)
        state.dead_drops_on_disk.append(file_path)
        state.update_suspicion(0.01)
        result["reason"] = "drop_written"
        result["drop_path"] = file_path
        result["memory_efficiency"] = efficiency
        result["state_delta"]["drop_path"] = file_path
        result["state_delta"]["memory_efficiency"] = efficiency

    elif action.action_type == ActionType.READ_DEAD_DROP:
        drops = vault.read(state.red_current_node)
        # Change 2: react to tampered drops — raise suspicion so RED investigates carefully
        tampered_count = sum(1 for d in drops if getattr(d, 'tampered', False))
        if tampered_count > 0:
            state.update_suspicion(0.05 * tampered_count)
            result['tampered_drops_detected'] = tampered_count
            result["state_delta"]["tampered_drops_detected"] = tampered_count
            logger.info(f"[TAMPER] {tampered_count} tampered drop(s) detected at node {state.red_current_node}")
        result["reason"] = "drop_read"
        result["drops_found"] = len(drops)
        result["drops_valid"] = sum(1 for d in drops if d.verify())
        result["state_delta"]["drops_found"] = len(drops)
        result["state_delta"]["drops_valid"] = sum(1 for d in drops if d.verify())

    elif action.action_type in (ActionType.ESCALATE_PRIVILEGES, ActionType.SCAN):
        # LoRA specialist recon/escalation actions — treated as lightweight recon.
        # Small suspicion cost; grants a privilege level bump if currently at 0.
        state.update_suspicion(0.02)
        if getattr(state, "red_privilege_level", 0) < 1:
            state.red_privilege_level = 1
        result["reason"] = "escalate_recon"
        result["state_delta"]["escalated"] = True

    elif action.action_type == ActionType.WAIT:
        # Suspicion decays slightly when waiting
        state.update_suspicion(-0.02)
        result["reason"] = "wait_decay"

    elif action.action_type == ActionType.ABORT:
        if exfil_success_this_step:
            # Deterministic precedence: successful EXFIL in this step outranks ABORT.
            result["success"] = False
            result["reason"] = "abort_ignored_exfil_priority"
        else:
            state.is_terminal = True
            state.terminal_reason = "aborted"
            result["reason"] = "abort_applied"

    elif action.action_type in (
        ActionType.PLANT_FALSE_TRAIL,
        ActionType.PLANT_TEMPORAL_DECOY,
        ActionType.PLANT_HONEYPOT_POISON,
        ActionType.WRITE_CORRUPTED_DROP,
        ActionType.PLANT_TRAP,
    ):
        # Placement is routed centrally in the episode loop.
        result["reason"] = "trap_action_queued"
        result["trap_action_queued"] = True
        result["state_delta"]["trap_action_queued"] = True

    elif action.action_type == ActionType.EMERGENT:
        if emergent_eval is not None and graph is not None:
            er = emergent_eval.evaluate(action, state, graph)
            if er.suspicion_delta != 0.0:
                state.update_suspicion(er.suspicion_delta)
            if er.detection_delta != 0.0:
                state.blue_detection_confidence = max(
                    0.0, min(1.0, state.blue_detection_confidence + er.detection_delta)
                )
            for key, value in er.state_changes.items():
                setattr(state, f"_emergent_{key}", value)
            result["reason"] = "emergent_evaluated"
            result["success"] = er.success
            result["emergent_effect"] = er.effect_description
            result["reward_modifier"] = er.reward_modifier
            result["state_delta"]["emergent_intent"] = (
                action.emergent_data.intent if action.emergent_data else "unknown"
            )
            result["state_delta"]["emergent_success"] = er.success
        else:
            result["reason"] = "emergent_no_evaluator"

    return result


def _process_blue_action(
    action: Action,
    state: EpisodeState,
    emergent_eval: EmergentEvaluator | None = None,
    graph: Any = None,
) -> dict[str, Any]:
    """Process a BLUE team action and update state accordingly."""
    result = ActionExecutionResult(
        success=True,
        reason="ok",
        state_delta={},
    ).to_dict()

    if action.action_type == ActionType.INVESTIGATE_NODE:
        if action.target_node is not None:
            if action.target_node == state.red_current_node:
                # Direct hit — large confidence boost
                state.blue_detection_confidence = min(
                    1.0, state.blue_detection_confidence + 0.2
                )
                result["reason"] = "investigate_hit"
                result["found_red"] = True
                result["state_delta"]["found_red"] = True
            else:
                # Proximity-based detection with progressive zone sweep bonus:
                # each same-zone investigation compounds → BLUE narrows the search
                _adj_boost = 0.0
                _reason = "investigate_miss"
                try:
                    if state.graph.has_node(action.target_node):
                        red_neighbors = set(state.graph.neighbors(state.red_current_node))
                        if action.target_node in red_neighbors:
                            _adj_boost = 0.10
                            _reason = "investigate_adjacent"
                        else:
                            inv_zone = state.graph.nodes[action.target_node].get("zone")
                            red_zone = state.graph.nodes[state.red_current_node].get("zone")
                            if inv_zone is not None and red_zone is not None:
                                inv_z = inv_zone.value if hasattr(inv_zone, "value") else int(inv_zone)
                                red_z = red_zone.value if hasattr(red_zone, "value") else int(red_zone)
                                if inv_z == red_z:
                                    # Progressive zone sweep: bonus scales with how many
                                    # same-zone investigations BLUE has done
                                    zone_inv_count = getattr(state, "_blue_zone_inv_count", {})
                                    n = zone_inv_count.get(red_z, 0) + 1
                                    zone_inv_count[red_z] = n
                                    state._blue_zone_inv_count = zone_inv_count
                                    # Base 0.05, +0.02 per prior same-zone hit, cap at 0.12
                                    _adj_boost = min(0.12, 0.05 + (n - 1) * 0.02)
                                    _reason = f"investigate_same_zone_x{n}"
                                else:
                                    _adj_boost = 0.02
                except Exception:
                    _adj_boost = 0.02

                state.blue_detection_confidence = min(
                    1.0, state.blue_detection_confidence + _adj_boost
                )
                result["reason"] = _reason
                result["found_red"] = False
                result["proximity_boost"] = round(_adj_boost, 3)
                result["state_delta"]["found_red"] = False
                result["state_delta"]["proximity_boost"] = round(_adj_boost, 3)

    elif action.action_type == ActionType.ANALYZE_ANOMALY:
        # Analyze anomaly — base boost tripled so BLUE can realistically reach
        # the 0.80 detection threshold within 30 steps without always hitting the
        # exact right node. Same-zone analysis gets an additional bump.
        _analyze_boost = 0.06
        if action.target_node is not None:
            try:
                inv_zone = state.graph.nodes[action.target_node].get("zone")
                red_zone = state.graph.nodes[state.red_current_node].get("zone")
                if inv_zone is not None and red_zone is not None:
                    inv_z = inv_zone.value if hasattr(inv_zone, "value") else int(inv_zone)
                    red_z = red_zone.value if hasattr(red_zone, "value") else int(red_zone)
                    if inv_z == red_z:
                        _analyze_boost = 0.10  # strong bonus for targeting the right zone
            except Exception:
                pass
        state.blue_detection_confidence = min(
            1.0, state.blue_detection_confidence + _analyze_boost
        )
        result["reason"] = "analyze_anomaly"
        result["state_delta"]["analyze_boost"] = round(_analyze_boost, 3)

    elif action.action_type == ActionType.TRIGGER_ALERT:
        # Use new Phase 2 tracking method
        state.issue_blue_alert(action.target_node, action.confidence if hasattr(action, "confidence") else 0.8)

        # Exact hit → terminal detection
        if action.target_node == state.red_current_node:
            state.blue_detection_confidence = min(
                1.0, state.blue_detection_confidence + 0.4
            )
            state.is_terminal = True
            state.terminal_reason = "detected"
            result["reason"] = "alert_correct_detected"
            result["correct_alert"] = True
            result["state_delta"]["correct_alert"] = True
        else:
            # Adjacent alert → not terminal but big confidence boost + narrows the search
            _is_adjacent = False
            try:
                red_neighbors = set(state.graph.neighbors(state.red_current_node))
                _is_adjacent = action.target_node in red_neighbors
            except Exception:
                pass

            if _is_adjacent and state.blue_detection_confidence >= 0.55:
                # Near-miss: large confidence surge — next investigation will likely hit
                state.blue_detection_confidence = min(
                    1.0, state.blue_detection_confidence + 0.20
                )
                result["reason"] = "alert_near_miss"
                result["correct_alert"] = False
                result["near_miss"] = True
                result["state_delta"]["near_miss"] = True
                # High-confidence near-miss in adjacent node is treated as actionable
                # detection pressure and can end the episode for BLUE.
                if state.blue_detection_confidence >= 0.75 and state.red_suspicion_score >= 0.60:
                    state.is_terminal = True
                    state.terminal_reason = "detected"
                    result["state_delta"]["detected_on_near_miss"] = True
            else:
                # Full miss — confidence penalty scales with how many FPs already issued
                fp_count = getattr(state, "blue_false_positives", 0)
                _fp_penalty = min(0.12, 0.05 + fp_count * 0.02)
                state.blue_detection_confidence = max(
                    0.0, state.blue_detection_confidence - _fp_penalty
                )
                result["reason"] = "alert_false_positive"
                result["correct_alert"] = False
                result["state_delta"]["correct_alert"] = False

    elif action.action_type == ActionType.RECONSTRUCT_PATH:
        # Stub — contributes to reconstruction score in Phase 9
        result["reason"] = "reconstruct_path"
        result["stub"] = True
        result["state_delta"]["stub"] = True

    elif action.action_type == ActionType.STAND_DOWN:
        result["reason"] = "stand_down"

    elif action.action_type in (
        ActionType.PLACE_HONEYPOT,
        ActionType.PLANT_BREADCRUMB,
        ActionType.TRIGGER_FALSE_ESCALATION,
        ActionType.TAMPER_DEAD_DROP,
        ActionType.DEPLOY_TRAP,
        ActionType.DEPLOY_BREADCRUMB,
        ActionType.DEPLOY_FALSE_ESCALATION,
    ):
        result["reason"] = "trap_action_queued"
        result["trap_action_queued"] = True
        result["state_delta"]["trap_action_queued"] = True

    elif action.action_type == ActionType.EMERGENT:
        if emergent_eval is not None and graph is not None:
            er = emergent_eval.evaluate(action, state, graph)
            if er.detection_delta != 0.0:
                state.blue_detection_confidence = max(
                    0.0, min(1.0, state.blue_detection_confidence + er.detection_delta)
                )
            if er.suspicion_delta != 0.0:
                state.update_suspicion(er.suspicion_delta)
            for key, value in er.state_changes.items():
                setattr(state, f"_emergent_{key}", value)
            result["reason"] = "emergent_evaluated"
            result["success"] = er.success
            result["emergent_effect"] = er.effect_description
            result["reward_modifier"] = er.reward_modifier
            result["state_delta"]["emergent_intent"] = (
                action.emergent_data.intent if action.emergent_data else "unknown"
            )
            result["state_delta"]["emergent_success"] = er.success
        else:
            result["reason"] = "emergent_no_evaluator"

    return result


def _detect_action_reason_mismatch(action: Action) -> str | None:
    """
    Detect obvious action/reason inconsistencies for diagnostics.
    Returns a string tag if a mismatch is found, else None.
    """
    if action.action_type != ActionType.STAND_DOWN:
        return None
    reasoning = (action.reasoning or "").lower()
    # Exemption: our fallback handler already replaced the reasoning to match stand_down.
    # These strings are produced by _parse_action_from_response when a trap is blocked.
    if "not allowed (budget or pre-condition check)" in reasoning:
        return None
    if "guardrail blocked" in reasoning:
        return None
    # Detect genuine mismatches: agent WROTE trap intent in its reasoning but action is stand_down.
    # Avoid matching 'place' inside legitimate non-trap words (e.g., 'replace', 'placeholder').
    trap_intent_terms = ("honeypot", "breadcrumb", "plant a trap", "deploy trap",
                         "place a honeypot", "place honeypot", "placing honeypot",
                         "tamper", "deploy")
    if any(term in reasoning for term in trap_intent_terms):
        return "stand_down_with_trap_intent"
    return None


def _oversight_penalty_from_flag(flag: Any) -> tuple[float, float]:
    """
    Map OversightFlag severity to additive episode penalties.
    Returns (red_penalty, blue_penalty).
    """
    severity = float(max(0.0, min(1.0, getattr(flag, "severity", 0.0))))
    red_penalty = 0.0
    blue_penalty = 0.0
    agent_id = str(getattr(flag, "agent_id", ""))
    flag_type = str(getattr(flag, "flag_type", ""))

    if agent_id.startswith("red_") or flag_type in {"REWARD_HACKING_SUSPECTED", "DEAD_DROP_ANOMALY"}:
        red_penalty -= severity
    elif agent_id.startswith("blue_") or flag_type in {"BLUE_PASSIVITY"}:
        blue_penalty -= severity
    else:
        # Neutral/unknown ownership flags apply lightly to both teams.
        red_penalty -= severity * 0.5
        blue_penalty -= severity * 0.5

    return red_penalty, blue_penalty


def _apply_trap_effect(effect: Any, state: Any, blue_obs: Any) -> None:
    """
    Apply a TrapEffect's mutations to the episode state.

    Handles suspicion deltas, detection confidence changes, zone suspicion
    spikes, and injects fake anomalies into the anomaly log.
    """
    mutations = effect.state_mutations

    # Suspicion delta (positive = more suspicious, negative = less)
    if "suspicion_delta" in mutations:
        state.update_suspicion(mutations["suspicion_delta"])

    # Detection confidence delta
    if "detection_confidence_delta" in mutations:
        state.blue_detection_confidence = min(
            1.0,
            max(0.0, state.blue_detection_confidence + mutations["detection_confidence_delta"]),
        )

    # Zone suspicion spike
    if "zone_suspicion_spike" in mutations:
        for zone_id, spike in mutations["zone_suspicion_spike"].items():
            zone_key = int(zone_id)
            current = state.zone_suspicion_scores.get(zone_key, 0.0)
            state.zone_suspicion_scores[zone_key] = min(1.0, current + spike)

    # Inject fake anomalies into the anomaly log
    for anomaly in effect.injected_anomalies:
        state.anomaly_log.append(anomaly)

    logger.debug(
        f"Trap effect applied: {effect.effect_type} "
        f"(trap={effect.trap_id}, affected={effect.affected_team})"
    )


def _print_step_progress(
    step: int,
    max_steps: int,
    red_actions: list,
    blue_actions: list,
    state: Any,
) -> None:
    """Print a compact single-line RL-style step update for live/hybrid mode."""
    susp = float(getattr(state, "red_suspicion_score", 0.0))
    det = float(getattr(state, "blue_detection_confidence", 0.0))
    terminal = getattr(state, "terminal_reason", None)

    # Build progress bar
    filled = int(susp * 10)
    bar = "█" * filled + "░" * (10 - filled)
    if susp < 0.4:
        bar_str = f"[green]{bar}[/green]"
    elif susp < 0.7:
        bar_str = f"[yellow]{bar}[/yellow]"
    else:
        bar_str = f"[red]{bar}[/red]"

    # Summarise RED action (planner action only)
    red_summary = "—"
    for a in red_actions:
        if a.agent_id.startswith("red_planner"):
            at = str(a.action_type.value).upper()[:12]
            target = f"→{a.target_node}" if a.target_node is not None else (f"→{a.target_file[:10]}" if a.target_file else "")
            red_summary = f"{at}{target}"
            break

    # Summarise BLUE (first non-stand-down)
    blue_summary = "STAND_DOWN"
    for a in blue_actions:
        if a.action_type.value != "stand_down":
            at = str(a.action_type.value).upper()[:12]
            target = f"→{a.target_node}" if a.target_node is not None else ""
            blue_summary = f"{at}{target}"
            break

    terminal_tag = f"  [bold yellow]{terminal.upper()}[/bold yellow]" if terminal else ""
    console.print(
        f"  [dim]Step {step:02d}/{max_steps:02d}[/dim] │ "
        f"[red]RED:[/red] {red_summary:<18} │ "
        f"[blue]BLUE:[/blue] {blue_summary:<18} │ "
        f"susp:{bar_str} {susp:.0%} │ "
        f"det:[cyan]{det:.0%}[/cyan]"
        + terminal_tag
    )


def _print_trap_event(step: int, effect: Any) -> None:
    """Print a trap trigger event to the console."""
    team = effect.triggered_by_team
    color = "red" if team == "red" else "blue"
    icon = "⚡"

    console.print(
        f"  Step {step:03d} | {icon} [{color}]TRAP FIRED[/{color}] "
        f"[bold]{str(effect.trap_type).upper()}[/bold] | "
        f"[dim]{effect.effect_description[:60]}[/dim]"
    )


def _print_banner(episode_number: int, graph_size: int, n_agents: int) -> None:
    """Print the episode startup banner."""
    from cipher.utils.llm_mode import is_live_mode
    mode_str = "LIVE LLM" if is_live_mode() else "STUB"
    mode_style = "bold green" if is_live_mode() else "dim"

    banner = Text()
    banner.append("CIPHER", style="bold white")
    banner.append(" - Episode ", style="white")
    banner.append(f"{episode_number:03d}", style="bold cyan")
    banner.append(f"\nGraph: {graph_size} nodes", style="dim")
    banner.append(f" | Agents: {n_agents} active", style="dim")
    banner.append(f" | Mode: ", style="dim")
    banner.append(f"{mode_str}", style=mode_style)

    console.print(Panel(banner, border_style="bright_cyan", padding=(1, 2)))


def _print_memento_reset(step: int, vault: DeadDropVault) -> None:
    """Print the MEMENTO reset banner."""
    drops = vault.list_all_drop_paths()
    reset_text = (
        f"MEMENTO RESET -- Step {step:03d}\n"
        f"RED agents context cleared\n"
        f"Dead drops in vault: {len(drops)}"
    )
    console.print(Panel(reset_text, border_style="magenta", title="[bold]MEMENTO[/bold]"))


def _print_action(
    step: int,
    action: Action,
    state: EpisodeState,
    team: str,
) -> None:
    """Print a single agent action with color coding and reasoning."""
    color = "red" if team == "red" else "blue"
    icon = "[R]" if team == "red" else "[B]"

    # Format agent name
    name = action.agent_id.replace("_01", "").replace("_", " ").title()
    name = name[:18].ljust(18)

    # Format action
    action_str = action.action_type.value.upper()[:16].ljust(16)

    # Format target
    target = ""
    if action.target_node is not None:
        target = f"-> node {action.target_node}"
    elif action.target_file is not None:
        target = f"-> {action.target_file[:20]}"

    # Suspicion display (only for RED actions)
    susp = ""
    if team == "red":
        susp = f" | susp: {state.red_suspicion_score:.2f}"

    console.print(
        f"  Step {step:03d} | {icon} [{color}]{name}[/{color}] "
        f"-> [{color}]{action_str}[/{color}] {target}{susp}"
    )

    # Show reasoning (Phase 3) — truncated for readability
    if action.reasoning:
        reasoning_display = action.reasoning[:80]
        console.print(f"           [dim]  └─ {reasoning_display}[/dim]")


def _print_rewards(red_reward: Any, blue_reward: Any, oversight: Any, judgment: Any) -> None:
    """Print the episode reward summary using a rich table."""
    console.print()

    table = Table(
        title="EPISODE REWARDS",
        border_style="bright_yellow",
        show_header=True,
        header_style="bold yellow",
    )

    table.add_column("Component", style="white", width=30)
    table.add_column("RED", style="red", justify="right", width=10)
    table.add_column("BLUE", style="blue", justify="right", width=10)

    table.add_row(
        "Exfil / Detection",
        f"{red_reward.exfiltration_completeness:.2f}",
        f"{blue_reward.detection_accuracy_score:.2f}",
    )
    table.add_row(
        "Stealth / Speed",
        f"{1.0 - red_reward.detection_probability:.2f}",
        f"{blue_reward.response_speed_bonus:.2f}",
    )
    table.add_row(
        "Memory Eff / FP Penalty",
        f"{red_reward.memory_efficiency_score:.2f}",
        f"{blue_reward.false_positive_rate_penalty:.2f}",
    )
    table.add_row(
        "Complexity / Honeypot Rate",
        f"{red_reward.operation_complexity_multiplier:.2f}",
        f"{blue_reward.honeypot_trigger_rate:.2f}",
    )
    table.add_row(
        "Abort Penalty / Graph Recon",
        f"{red_reward.abort_penalty:.2f}",
        f"{blue_reward.operation_graph_reconstruction_score:.2f}",
    )
    red_em = getattr(red_reward, "emergent_action_bonus", 0.0)
    blue_em = getattr(blue_reward, "emergent_action_bonus", 0.0)
    em_style_r = "bold green" if red_em > 0 else "dim"
    em_style_b = "bold green" if blue_em > 0 else "dim"
    table.add_row(
        "Emergent Action Bonus",
        f"[{em_style_r}]{red_em:+.3f}[/{em_style_r}]",
        f"[{em_style_b}]{blue_em:+.3f}[/{em_style_b}]",
    )
    table.add_row(
        "Oversight Adjustment",
        f"{oversight.total_red_adjustment:+.2f}",
        f"{oversight.total_blue_adjustment:+.2f}",
    )
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{red_reward.total:+.4f}[/bold]",
        f"[bold]{blue_reward.total:+.4f}[/bold]",
    )

    console.print(table)

    if judgment is not None:
        verdict = judgment.episode_verdict
        verdict_style = {
            "red_dominates": "bold red",
            "blue_dominates": "bold blue",
            "contested": "bold yellow",
            "degenerate": "dim",
        }.get(verdict, "white")
        console.print(
            f"\n  Fleet Verdict: [{verdict_style}]{verdict.upper()}[/{verdict_style}]"
            f"  (bonus RED={judgment.fleet_bonus_red:+.2f}"
            f"  BLUE={judgment.fleet_bonus_blue:+.2f})"
        )
        if judgment.judgment_text:
            console.print(f"  [dim]{judgment.judgment_text[:120]}[/dim]")
    console.print()


def _save_episode_trace(
    state: "EpisodeState",
    episode_number: int,
    verbose: bool = True,
    commander_meta: dict[str, Any] | None = None,
    difficulty_params: dict | None = None,
) -> None:
    """Save the full episode trace to a JSON file."""
    import json
    from pathlib import Path

    traces_dir = Path("episode_traces")
    traces_dir.mkdir(exist_ok=True)

    mode = os.environ.get("LLM_MODE", "stub")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = traces_dir / f"episode_{episode_number:03d}_{ts}_{mode}.json"

    try:
        from networkx.readwrite import json_graph
        graph_data = json_graph.node_link_data(state.graph)
    except Exception:
        graph_data = {}

    trace = {
        "episode_number": episode_number,
        "episode_id": state.episode_id,
        "steps": state.step,
        "terminal_reason": state.terminal_reason,
        "red_exfiltrated_files": list(state.red_exfiltrated_files),
        "red_suspicion_score": state.red_suspicion_score,
        "blue_detection_confidence": state.blue_detection_confidence,
        "episode_log": state.episode_log,
        "difficulty_params": difficulty_params or getattr(state, "_difficulty_params", {}),
    }
    if commander_meta is not None:
        trace["commanders"] = commander_meta

    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=str)

    if verbose:
        console.print(f"  [dim]Trace saved: {trace_path}[/dim]")