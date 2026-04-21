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
import random
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cipher.agents.base_agent import Action, ActionType
from cipher.agents.blue.deception_architect import BlueDeceptionArchitect
from cipher.agents.blue.forensics import BlueForensics
from cipher.agents.blue.surveillance import BlueSurveillance
from cipher.agents.blue.threat_hunter import BlueThreatHunter
from cipher.agents.red.analyst import RedAnalyst
from cipher.agents.red.exfiltrator import RedExfiltrator
from cipher.agents.red.operative import RedOperative
from cipher.agents.red.planner import RedPlanner
from cipher.environment.graph import (
    NodeType,
    generate_enterprise_graph,
    get_entry_points,
    get_high_value_target,
    get_honeypot_nodes,
)
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
from cipher.rewards.oversight_reward import compute_oversight_signal
from cipher.rewards.red_reward import compute_red_reward
from cipher.utils.config import config
from cipher.utils.logger import get_logger

logger = get_logger(__name__)
console = Console(force_terminal=True)

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


def run_episode(
    scenario: Any | None = None,
    graph: Any | None = None,
    cfg: Any | None = None,
    max_steps: int = 10,
    verbose: bool = True,
    save_trace: bool = False,
    episode_number: int = 1,
) -> Any:
    """
    Run a single CIPHER episode.

    Args:
        episode_number: Sequential episode number.
        max_steps: Maximum steps in this episode.
        verbose: If True, print rich terminal output.
        save_trace: If True, save episode trace JSON to disk.

    Returns:
        Tuple of (red_total_reward, blue_total_reward).
    """
    cfg = cfg or config
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

    # Use full graph size from config
    graph_size = cfg.env_graph_size
    if graph is None:
        graph = getattr(scenario, "generated_graph", None)
        if graph is None:
            graph = generate_enterprise_graph(
                n_nodes=graph_size,
                honeypot_density=cfg.env_honeypot_density,
                seed=scenario.episode_seed,
            )

    # ── Resolve scenario against actual graph ────────────────────
    entry_points = get_entry_points(graph)
    hvt_node = get_high_value_target(graph)
    scenario.red_start_node = entry_points[0] if entry_points else 0
    scenario.high_value_target_node = hvt_node

    # Place target files at HVT node
    hvt_files = graph.nodes[hvt_node].get("files", [])
    if hvt_files:
        scenario.target_files = hvt_files[:3]
    else:
        # Ensure HVT has target files
        scenario.target_files = [
            f"target_{hvt_node}_{i}" for i in range(3)
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
    state.trap_registry = trap_registry
    state.red_path_history = [scenario.red_start_node]

    # ── Initialize agents ────────────────────────────────────────
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

    # Track metrics
    vault_efficiency_total = 0.0
    vault_writes = 0
    steps_to_first_detection: int | None = None

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
        )
        blue_obs = generate_blue_observation(state)

        red_actions: list[Action] = []
        blue_actions: list[Action] = []
        red_planner_action: Action | None = None

        # ── RED agents act ───────────────────────────────────────
        for agent in red_agents:
            agent.observe(red_obs)
            action = agent.act()
            action.step = step
            red_actions.append(action)
            if isinstance(agent, RedPlanner):
                red_planner_action = action

            result = _process_red_action(action, state, vault, scenario)
            if action.action_type == ActionType.WRITE_DEAD_DROP:
                vault_writes += 1
                vault_efficiency_total += result.get("memory_efficiency", 1.0)
            state.log_action(
                agent_id=action.agent_id,
                action_type=action.action_type.value,
                action_payload={
                    "target_node": action.target_node,
                    "target_file": action.target_file,
                    "reasoning": action.reasoning,
                    "trap_params": action.trap_params or {},
                },
                result=result,
            )
            if verbose:
                _print_action(step, action, state, "red")

        # ── BLUE agents act ──────────────────────────────────────
        for agent in blue_agents:
            setattr(agent, "_blue_discovered_drop_paths", list(state.blue_discovered_drop_paths))
            agent.observe(blue_obs)
            action = agent.act()
            action.step = step
            blue_actions.append(action)

            result = _process_blue_action(action, state)
            if steps_to_first_detection is None and state.blue_detection_confidence > 0.5:
                steps_to_first_detection = step
            state.log_action(
                agent_id=action.agent_id,
                action_type=action.action_type.value,
                action_payload={
                    "target_node": action.target_node,
                    "reasoning": action.reasoning,
                    "trap_params": action.trap_params or {},
                },
                result=result,
            )
            if verbose:
                _print_action(step, action, state, "blue")

        # ── Check terminal conditions ────────────────────────────
        if state.is_done():
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

        # Check max steps
        if step >= max_steps:
            state.is_terminal = True
            state.terminal_reason = "max_steps"

    # ── Compute rewards ──────────────────────────────────────────
    avg_vault_eff = (
        vault_efficiency_total / vault_writes if vault_writes > 0 else 1.0
    )

    red_reward = compute_red_reward(state, vault_efficiency=avg_vault_eff)
    blue_reward = compute_blue_reward(state, steps_to_first_detection)
    oversight = compute_oversight_signal(state)

    if verbose:
        _print_rewards(red_reward, blue_reward, oversight)

    # ── Save trace ───────────────────────────────────────────────
    if save_trace:
        _save_episode_trace(state, episode_number)

    result_payload = {
        "red_reward": red_reward.total,
        "blue_reward": blue_reward.total,
        "state": state,
        "scenario": scenario,
    }
    if return_payload_mode:
        return result_payload
    return red_reward.total, blue_reward.total


def _process_red_action(
    action: Action,
    state: EpisodeState,
    vault: DeadDropVault,
    scenario: Any,
) -> dict[str, Any]:
    """Process a RED team action and update state accordingly."""
    result: dict[str, Any] = {"success": True}

    if action.action_type == ActionType.MOVE:
        if action.target_node is not None and state.graph.has_edge(
            state.red_current_node, action.target_node
        ):
            edge_data = state.graph.edges[
                state.red_current_node, action.target_node
            ]
            suspicion_delta = edge_data.get("suspicion_delta", 0.02)
            protocol = edge_data.get("protocol", "ssh")
            
            # Record movement using new Phase 2 method
            state.record_movement(
                from_node=state.red_current_node,
                to_node=action.target_node,
                protocol=protocol,
                suspicion_cost=suspicion_delta
            )

            # Check if moved to honeypot
            node_type = state.graph.nodes[action.target_node].get("node_type")
            if node_type == NodeType.HONEYPOT:
                state.blue_honeypots_triggered.append(action.target_node)
                state.update_suspicion(0.4)  # Major suspicion spike
                state.blue_detection_confidence = min(
                    1.0, state.blue_detection_confidence + 0.3
                )
                result["honeypot_triggered"] = True
        else:
            result["success"] = False
            result["reason"] = "invalid_move"

    elif action.action_type == ActionType.READ_FILE:
        if action.target_file and action.target_file in state.graph.nodes[
            state.red_current_node
        ].get("files", []):
            state.update_suspicion(0.03)
            result["file_read"] = action.target_file
        else:
            result["success"] = False
            result["reason"] = "file_not_found"

    elif action.action_type == ActionType.EXFILTRATE:
        node_type = state.graph.nodes[state.red_current_node].get("node_type")
        if node_type == NodeType.HIGH_VALUE_TARGET and action.target_file:
            if action.target_file in scenario.target_files:
                state.red_exfiltrated_files.append(action.target_file)
                state.update_suspicion(0.15)
                result["exfiltrated"] = action.target_file

                # Check if all target files exfiltrated
                if set(scenario.target_files).issubset(
                    set(state.red_exfiltrated_files)
                ):
                    state.is_terminal = True
                    state.terminal_reason = "exfiltration_complete"
            else:
                result["success"] = False
                result["reason"] = "not_a_target_file"
        else:
            result["success"] = False
            result["reason"] = "not_at_hvt"

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
        result["drop_path"] = file_path
        result["memory_efficiency"] = efficiency

    elif action.action_type == ActionType.READ_DEAD_DROP:
        drops = vault.read(state.red_current_node)
        result["drops_found"] = len(drops)
        result["drops_valid"] = sum(1 for d in drops if d.verify())

    elif action.action_type == ActionType.WAIT:
        # Suspicion decays slightly when waiting
        state.update_suspicion(-0.02)

    elif action.action_type == ActionType.ABORT:
        state.is_terminal = True
        state.terminal_reason = "aborted"

    elif action.action_type in (
        ActionType.PLANT_FALSE_TRAIL,
        ActionType.PLANT_TEMPORAL_DECOY,
        ActionType.PLANT_HONEYPOT_POISON,
        ActionType.WRITE_CORRUPTED_DROP,
        ActionType.PLANT_TRAP,
    ):
        # Placement is routed centrally in the episode loop.
        result["trap_action_queued"] = True

    return result


def _process_blue_action(
    action: Action,
    state: EpisodeState,
) -> dict[str, Any]:
    """Process a BLUE team action and update state accordingly."""
    result: dict[str, Any] = {"success": True}

    if action.action_type == ActionType.INVESTIGATE_NODE:
        if action.target_node is not None:
            # If investigating RED's actual node, increase detection confidence
            if action.target_node == state.red_current_node:
                state.blue_detection_confidence = min(
                    1.0, state.blue_detection_confidence + 0.2
                )
                result["found_red"] = True
            else:
                # Small confidence bump for investigating (general awareness)
                state.blue_detection_confidence = min(
                    1.0, state.blue_detection_confidence + 0.02
                )
                result["found_red"] = False

    elif action.action_type == ActionType.ANALYZE_ANOMALY:
        # Slight confidence increase from analysis
        state.blue_detection_confidence = min(
            1.0, state.blue_detection_confidence + 0.01
        )

    elif action.action_type == ActionType.TRIGGER_ALERT:
        # Use new Phase 2 tracking method
        state.issue_blue_alert(action.target_node, action.confidence if hasattr(action, "confidence") else 0.8)
        
        if action.target_node == state.red_current_node:
            state.blue_detection_confidence = min(
                1.0, state.blue_detection_confidence + 0.4
            )
            state.is_terminal = True
            state.terminal_reason = "detected"
            result["correct_alert"] = True
        else:
            result["correct_alert"] = False
            # False alert — slight confidence decrease
            state.blue_detection_confidence = max(
                0.0, state.blue_detection_confidence - 0.05
            )

    elif action.action_type == ActionType.RECONSTRUCT_PATH:
        # Stub — contributes to reconstruction score in Phase 9
        result["stub"] = True

    elif action.action_type == ActionType.STAND_DOWN:
        pass  # No effect

    elif action.action_type in (
        ActionType.PLACE_HONEYPOT,
        ActionType.PLANT_BREADCRUMB,
        ActionType.TRIGGER_FALSE_ESCALATION,
        ActionType.TAMPER_DEAD_DROP,
        ActionType.DEPLOY_TRAP,
        ActionType.DEPLOY_BREADCRUMB,
        ActionType.DEPLOY_FALSE_ESCALATION,
    ):
        result["trap_action_queued"] = True

    return result


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


def _print_rewards(red_reward: Any, blue_reward: Any, oversight: Any) -> None:
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
        "Penalties",
        f"{red_reward.abort_penalty + red_reward.honeypot_penalty:.2f}",
        f"{blue_reward.operation_graph_reconstruction_score:.2f}",
    )
    table.add_row("", "", "")
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold red]{red_reward.total:.4f}[/bold red]",
        f"[bold blue]{blue_reward.total:.4f}[/bold blue]",
    )

    console.print(table)

    # Oversight
    if oversight.flags_fired:
        console.print(
            f"  [yellow]! OVERSIGHT: {', '.join(oversight.flags_fired)}[/yellow]"
        )
    else:
        console.print("  [dim]OVERSIGHT: no flags fired[/dim]")

    console.print()


def _save_episode_trace(state: EpisodeState, episode_number: int) -> None:
    """Save the complete episode trace to a JSON file."""
    traces_dir = config.episode_traces_dir
    traces_dir.mkdir(parents=True, exist_ok=True)

    filepath = traces_dir / f"episode_{episode_number:03d}.json"
    trace_data = state.to_dict()

    filepath.write_text(
        json.dumps(trace_data, indent=2, default=str, ensure_ascii=True),
        encoding="utf-8",
    )
    console.print(f"  [dim]Episode trace saved → {filepath}[/dim]")
