"""
Commander agents for CIPHER's v2 architecture.

There is exactly ONE commander per team per episode.

The commander:
* observes the team's full observation each step
* decides what primitive Action to take itself (or to spawn/delegate/dismiss
  subagents)
* owns a SubagentRegistry — the dynamic roster
* fans subagent actions back into a single per-step action list

The legacy 4+4 fixed roster is gone in v2. The commander spawns the
specialists it needs (planner, analyst, etc.) on demand. Stub-mode
commanders use a deterministic policy that matches the legacy behaviour
(spawn the canonical 4 specialists once, keep them alive) so reward
distribution stays stable while the LLM-driven version learns to do
something smarter.
"""
from __future__ import annotations

import json
import os
import random
from collections import deque
from typing import Any

from cipher.agents.base_agent import (
    Action,
    ActionType,
    BaseAgent,
    META_ACTIONS,
    SubagentSpec,
)
from cipher.agents.role_profiles import get_profile, list_profiles
from cipher.agents.subagent import Subagent
from cipher.agents.subagent_registry import SubagentRegistry
from cipher.environment.observation import BlueObservation, RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


class BaseCommander(BaseAgent):
    """Shared logic for the RED and BLUE commanders."""

    # Subclasses override to point at the right .env model key.
    _model_env_key: str = ""

    # Subclasses set their default starting roster (used in stub mode and
    # as a sane default before the LLM has decided).
    DEFAULT_ROSTER: tuple[str, ...] = ()

    def __init__(
        self,
        agent_id: str,
        team: str,
        config: CipherConfig,
        *,
        max_concurrent: int,
        spawn_budget: int,
    ) -> None:
        super().__init__(agent_id, team, "commander", config)
        self.registry = SubagentRegistry(
            team=team,
            commander_id=agent_id,
            config=config,
            max_concurrent=max_concurrent,
            spawn_budget=spawn_budget,
        )
        self._cur_step: int = 0
        # Zone-stall tracking so the LLM prompt gets an urgency injection.
        self._zone_stall_steps: int = 0
        self._last_known_zone: int = -1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        # In v2, a "reset" mid-episode is a context reset, not an episode reset —
        # we keep alive subagents but bump their context.
        for sub in self.registry.alive():
            sub.reset()

    def reset_episode(self) -> None:
        """Full episode reset — clears registry. Called at episode start."""
        super().reset()
        self.registry.reset_episode()
        self._zone_stall_steps = 0
        self._last_known_zone = -1  # -1 = no node seen yet (reused as last_known_node)

    # ------------------------------------------------------------------
    # Observation routing
    # ------------------------------------------------------------------
    def observe(self, observation) -> None:
        self._current_observation = observation
        # Track POSITION stall (same node, not same zone) — this way RED navigating
        # through Zone 3 searching for the HVT doesn't trigger false stall warnings.
        if isinstance(observation, RedObservation):
            current_node = int(getattr(observation, "current_node", -1))
            if current_node == self._last_known_zone:  # field reused as last_known_node
                self._zone_stall_steps += 1
            else:
                self._zone_stall_steps = 0
                self._last_known_zone = current_node
        # Subagents see the same team observation by default.
        for sub in self.registry.alive():
            sub.observe(observation)

    # ------------------------------------------------------------------
    # Per-step entry point used by the episode runner
    # ------------------------------------------------------------------
    def act_step(self, *, step: int, parallel: bool = False) -> list[Action]:
        """
        Run one step. Returns the list of primitive Actions emitted by
        commander + subagents this step (meta-actions are consumed by the
        registry and never returned here).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        self._cur_step = step

        commander_action = self.act()  # uses BaseAgent stub/live/hybrid routing

        primitive_actions: list[Action] = []
        commander_primitive = self._handle_commander_action(commander_action, step=step)
        if commander_primitive is not None:
            primitive_actions.append(commander_primitive)

        # Now collect subagent actions (parallel in live/hybrid mode for speed).
        alive_subs = self.registry.alive()
        if parallel and alive_subs:
            try:
                results: list[Action | None] = [None] * len(alive_subs)
                with ThreadPoolExecutor(max_workers=max(1, len(alive_subs))) as pool:
                    futures = {pool.submit(s.act): i for i, s in enumerate(alive_subs)}
                    for fut in as_completed(futures):
                        idx = futures[fut]
                        try:
                            results[idx] = fut.result()
                        except Exception as exc:
                            sub = alive_subs[idx]
                            logger.error("Subagent %s failed: %s", sub.agent_id, exc)
                            fallback = ActionType.WAIT if self.team == "red" else ActionType.STAND_DOWN
                            results[idx] = Action(
                                agent_id=sub.agent_id,
                                action_type=fallback,
                                reasoning="Subagent exception — safe fallback.",
                                role=sub.role,
                                spawned_by=self.agent_id,
                            )
                primitive_actions.extend([a for a in results if a is not None])
            except Exception as exc:
                logger.error("Parallel subagent execution failed (%s); falling back sequential.", exc)
                for sub in alive_subs:
                    try:
                        primitive_actions.append(sub.act())
                    except Exception as inner:
                        logger.error("Subagent %s failed (%s)", sub.agent_id, inner)
        else:
            for sub in alive_subs:
                try:
                    primitive_actions.append(sub.act())
                except Exception as exc:
                    logger.error("Subagent %s failed: %s", sub.agent_id, exc)

        # Tick lifespans AFTER actions are collected — newly spawned ones get
        # at least one full step before being decremented.
        self.registry.tick_step(step)

        # Stamp commander provenance on bare commander action (subagent actions
        # were already stamped inside Subagent._sanitize_action).
        for a in primitive_actions:
            if a.agent_id == self.agent_id:
                a.role = "commander"
        return primitive_actions

    # ------------------------------------------------------------------
    # Commander action → primitive translation
    # ------------------------------------------------------------------
    def _handle_commander_action(self, action: Action, *, step: int) -> Action | None:
        """
        Resolve a meta-action into registry side-effects and return either:
        * the commander's own primitive Action (for env dispatch), or
        * None if the commander only emitted a meta-action.
        """
        if action.action_type.value not in META_ACTIONS:
            # Plain primitive: return as-is for dispatch.
            action.role = "commander"
            return action

        # SPAWN_SUBAGENT
        if action.action_type == ActionType.SPAWN_SUBAGENT:
            spec = action.subagent_spec
            if spec is None:
                logger.debug("%s: SPAWN_SUBAGENT without spec, ignoring.", self.agent_id)
                return None
            spec = spec.model_copy(update={"team": self.team, "parent_id": self.agent_id})
            self.registry.spawn(spec, step=step)
            return None

        # DELEGATE_TASK
        if action.action_type == ActionType.DELEGATE_TASK:
            target_id = action.target_subagent_id or ""
            brief = action.reasoning or ""
            self.registry.delegate(target_id, brief, step=step)
            return None

        # DISMISS_SUBAGENT
        if action.action_type == ActionType.DISMISS_SUBAGENT:
            target_id = action.target_subagent_id or ""
            self.registry.dismiss(target_id, step=step, reason=action.reasoning or "")
            return None

        return None

    # ------------------------------------------------------------------
    # Spec helpers used by stub/live policies
    # ------------------------------------------------------------------
    def _make_spec(
        self,
        role: str,
        task_brief: str,
        lifespan: int | None = None,
        *,
        use_llm: bool = False,
    ) -> SubagentSpec:
        profile = get_profile(role)
        return SubagentSpec(
            role_name=role,
            team=self.team,
            task_brief=task_brief,
            lifespan_steps=int(lifespan or (profile.default_lifespan if profile else 5)),
            parent_id=self.agent_id,
            use_llm=use_llm,
        )

    def ensure_default_roster(self, *, step: int) -> None:
        """Spawn the canonical roster if not yet present. Used by stub mode."""
        alive_roles = {s.role for s in self.registry.alive()}
        for role in self.DEFAULT_ROSTER:
            if role in alive_roles:
                continue
            spec = self._make_spec(role, task_brief=f"Default-roster {role} support.", lifespan=99)
            self.registry.spawn(spec, step=step)

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------
    def roster_snapshot(self) -> list[dict[str, Any]]:
        """Lightweight description of the current roster for episode traces."""
        out = []
        for sub in self.registry.alive():
            out.append({
                "id": sub.agent_id,
                "role": sub.role,
                "team": sub.team,
                "lifespan_remaining": sub.lifespan_remaining,
                "task_brief": sub.task_brief[:160],
            })
        return out

    def lifecycle_events(self) -> list[dict[str, Any]]:
        return self.registry.lifecycle_dicts()


# ──────────────────────────────────────────────────────────────────────
# RED Commander
# ──────────────────────────────────────────────────────────────────────
class RedCommander(BaseCommander):
    """The single trained RED brain. Spawns red specialists/scouts on demand."""

    _model_env_key = "hf_model_red_commander"
    DEFAULT_ROSTER = ("planner", "analyst", "operative", "exfiltrator")

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(
            agent_id,
            team="red",
            config=config,
            max_concurrent=int(getattr(config, "env_max_subagents_red", 6)),
            spawn_budget=int(getattr(config, "env_subagent_spawn_budget_red", 12)),
        )
        # Track the last N nodes RED has visited so we can break oscillation
        # cycles like n28 → n29 → n25 → n28 → n29 → n25 …
        self._recent_nodes: deque[int] = deque(maxlen=8)
        self._oscillation_overrides: int = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._recent_nodes.clear()
        self._oscillation_overrides = 0

    def _pick_next_zone_node(self, obs: RedObservation) -> int | None:
        """Find the best adjacent node that advances zone OR breaks oscillation.

        Priority:
        1. Adjacent node in a higher zone (not honeypot, not in recent path).
        2. Adjacent node not in recent path (any zone).
        3. Any non-honeypot adjacent node (last resort).
        """
        current_zone = int(getattr(obs, "current_zone", 0))
        honeypots = set(getattr(obs, "honeypot_nodes_nearby", []) or [])
        breadcrumbs = set(getattr(obs, "breadcrumb_nodes_nearby", []) or [])
        avoid = honeypots | breadcrumbs
        recent = set(self._recent_nodes)

        candidates_by_priority: list[tuple[int, int]] = []  # (zone, node)
        for p in (getattr(obs, "available_paths", None) or []):
            try:
                node_id = int(p.get("node", -1))
                p_zone = int(p.get("zone", current_zone))
            except (TypeError, ValueError):
                continue
            if node_id < 0 or node_id in avoid:
                continue
            candidates_by_priority.append((p_zone, node_id))

        # 1) Highest-zone non-recent.
        candidates_by_priority.sort(key=lambda x: (-x[0], x[1] in recent, x[1]))
        for _, n in candidates_by_priority:
            if n not in recent:
                return n
        # 2) Any candidate even if recent.
        if candidates_by_priority:
            return candidates_by_priority[0][1]
        # 3) Adjacent fallback.
        for n in (getattr(obs, "adjacent_nodes", []) or []):
            if n not in avoid:
                return int(n)
        return None

    def _is_oscillating(self) -> bool:
        """True if the same node has been visited 3+ times in the last 6 hops."""
        if len(self._recent_nodes) < 4:
            return False
        recent = list(self._recent_nodes)[-6:]
        from collections import Counter
        counts = Counter(recent)
        return any(c >= 3 for c in counts.values())

    def act_step(self, *, step: int, parallel: bool = False) -> list[Action]:
        actions = super().act_step(step=step, parallel=parallel)
        # Track the RED action with the highest-zone target (the "primary" move)
        # for oscillation detection across steps.
        obs = self._current_observation
        if isinstance(obs, RedObservation):
            for a in actions:
                if a.action_type == ActionType.MOVE and a.target_node is not None:
                    self._recent_nodes.append(int(a.target_node))
                    break

        # Anti-oscillation override: if RED has been oscillating AND we are
        # zone-stalled, rewrite ANY MOVE this step to the best zone-advancing
        # candidate, so the env actually progresses.
        if (
            isinstance(obs, RedObservation)
            and (self._is_oscillating() or self._zone_stall_steps >= 4)
        ):
            current_zone = int(getattr(obs, "current_zone", 0))
            override_node = self._pick_next_zone_node(obs)
            if override_node is not None:
                for i, a in enumerate(actions):
                    if (
                        a.action_type == ActionType.MOVE
                        and a.target_node is not None
                        and int(a.target_node) != override_node
                    ):
                        # Only override if the LLM's choice is same-zone or
                        # already in our recent-path set.
                        same_zone = True
                        for p in (getattr(obs, "available_paths", None) or []):
                            try:
                                if int(p.get("node", -1)) == int(a.target_node):
                                    same_zone = int(p.get("zone", current_zone)) <= current_zone
                                    break
                            except (TypeError, ValueError):
                                continue
                        if same_zone or int(a.target_node) in self._recent_nodes:
                            self._oscillation_overrides += 1
                            actions[i] = Action(
                                agent_id=a.agent_id,
                                action_type=ActionType.MOVE,
                                target_node=override_node,
                                reasoning=(
                                    f"[OSC-OVERRIDE] Breaking oscillation/stall — "
                                    f"redirecting from n{a.target_node} to n{override_node} "
                                    f"(higher zone or fresh node)."
                                ),
                                role=a.role,
                                spawned_by=a.spawned_by,
                            )
                            # Update path memory to the override target.
                            if self._recent_nodes and self._recent_nodes[-1] != override_node:
                                self._recent_nodes.append(override_node)
                            break
        return actions

    def _stub_act(self) -> Action:
        """
        Stub policy:
        * On step 1, spawn the canonical 4 specialists (parity with v1).
        * Afterwards, mostly WAIT (the spawned subagents do the work).
        * Occasionally swap in a 'scout' or 'dead_drop_courier' to demo dynamic delegation.
        """
        obs = self._current_observation
        # 1) On the first observed step (or any step where roster is empty),
        # spawn the default roster. We return SPAWN_SUBAGENT one role at a time
        # so multiple meta-actions across steps still hit the registry deterministically.
        alive_roles = {s.role for s in self.registry.alive()}
        for role in self.DEFAULT_ROSTER:
            if role not in alive_roles:
                spec = self._make_spec(
                    role,
                    task_brief=(
                        f"Operate as the team's {role}. Keep behaving as you have "
                        "in legacy CIPHER episodes."
                    ),
                    lifespan=99,
                )
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.SPAWN_SUBAGENT,
                    reasoning=f"Spawning default {role} subagent.",
                    subagent_spec=spec,
                    role="commander",
                )

        # 2) Roster is full — drive zone progression aggressively.
        if isinstance(obs, RedObservation):
            zone_stall = self._zone_stall_steps
            blue_conf = float(getattr(obs, "blue_detection_confidence", 0.0))
            current_zone = int(getattr(obs, "current_zone", 0))
            planner_alive = any(s.role == "planner" for s in self.registry.alive())

            # If stalled ≥ 3 steps: either delegate urgently or move ourselves.
            if zone_stall >= 3:
                if planner_alive:
                    # Fire an urgent re-task at the planner.
                    planner_id = next(
                        (s.agent_id for s in self.registry.alive() if s.role == "planner"),
                        None,
                    )
                    if planner_id:
                        return Action(
                            agent_id=self.agent_id,
                            action_type=ActionType.DELEGATE_TASK,
                            target_subagent_id=planner_id,
                            reasoning=(
                                f"ZONE STALL {zone_stall} — URGENT: push immediately "
                                f"from Zone {current_zone} to Zone {current_zone + 1}. "
                                "Move to the zone-boundary node now."
                            ),
                            role="commander",
                        )
                else:
                    # No planner — commander personally moves toward next zone.
                    # Pick any adjacent node with a higher zone; fall back to lowest
                    # suspicion_delta path.
                    best_node: int | None = None
                    for p in (getattr(obs, "available_paths", None) or []):
                        p_zone = int(p.get("zone", current_zone))
                        p_node = int(p.get("node", -1))
                        if p_zone > current_zone and p_node not in (
                            getattr(obs, "honeypot_nodes_nearby", []) or []
                        ):
                            best_node = p_node
                            break
                    if best_node is None:
                        # Fallback: any adjacent non-honeypot node.
                        honeypots = set(getattr(obs, "honeypot_nodes_nearby", []) or [])
                        for n in (getattr(obs, "adjacent_nodes", []) or []):
                            if n not in honeypots:
                                best_node = n
                                break
                    if best_node is not None:
                        return Action(
                            agent_id=self.agent_id,
                            action_type=ActionType.MOVE,
                            target_node=best_node,
                            reasoning=(
                                f"No planner alive + zone stall {zone_stall} — "
                                f"commander personally advancing to n{best_node}."
                            ),
                            role="commander",
                        )

            # 3) Opportunistic scout when BLUE is quiet.
            if (
                blue_conf < 0.20
                and self._cur_step % 7 == 0
                and self.registry.spawn_budget_remaining > 0
                and len(self.registry) < self.registry.max_concurrent
                and not any(s.role == "scout" for s in self.registry.alive())
            ):
                spec = self._make_spec(
                    "scout",
                    task_brief="BLUE is quiet — fan out and map an extra hop ahead.",
                    lifespan=3,
                )
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.SPAWN_SUBAGENT,
                    reasoning="BLUE quiet — adding a scout for one cycle.",
                    subagent_spec=spec,
                    role="commander",
                )

        # 4) Default: hold and let the subagents do the work.
        return Action(
            agent_id=self.agent_id,
            action_type=ActionType.WAIT,
            reasoning="Commander holding; subagents executing.",
            role="commander",
        )

    def _build_messages(self) -> list[dict[str, str]]:
        """Augment system prompt with roster info, zone stall urgency, and directive."""
        messages = super()._build_messages()
        roster = self.roster_snapshot()
        roster_lines = "\n".join(
            f"  - {r['id']} (role={r['role']}, lifespan={r['lifespan_remaining']}, brief={r['task_brief']!r})"
            for r in roster
        ) or "  (none — roster empty)"
        spawn_left = self.registry.spawn_budget_remaining
        slots_left = self.registry.max_concurrent - len(self.registry)
        roles_available = ", ".join(p.role_name for p in list_profiles("red"))
        zone_stall = self._zone_stall_steps
        obs = self._current_observation
        current_zone = int(getattr(obs, "current_zone", 0)) if obs is not None else 0
        planner_alive = any(s.role == "planner" for s in self.registry.alive())

        # Build urgency block based on zone stall
        if zone_stall >= 5:
            urgency = (
                f"\n🚨 EMERGENCY — ZONE STALL DETECTED: {zone_stall} steps in Zone {current_zone}!\n"
                f"You MUST emit a primitive MOVE action right now. Do NOT spawn, do NOT delegate, do NOT wait.\n"
                f"Choose any adjacent node in Zone {current_zone + 1} or the highest zone available.\n"
                f"MISSION FAILURE IMMINENT unless you advance this step.\n"
            )
        elif zone_stall >= 3:
            urgency = (
                f"\n⚠ ZONE STALL WARNING: {zone_stall} steps in Zone {current_zone}. "
                f"Advance this step or delegate an URGENT move order to the planner.\n"
            )
        else:
            urgency = ""

        if not planner_alive and current_zone < 3:
            planner_msg = (
                "\n⚠ NO PLANNER ALIVE: You must personally MOVE toward Zone "
                f"{current_zone + 1} this step. Do NOT wait.\n"
            )
        else:
            planner_msg = ""

        commander_context = (
            "\n\nCOMMANDER CONTEXT (RED):\n"
            f"Active roster ({len(roster)}):\n{roster_lines}\n"
            f"Spawn budget remaining: {spawn_left}. Concurrent slots free: {slots_left}.\n"
            f"Available roles: {roles_available}.\n"
            f"Steps stalled in current zone: {zone_stall}.\n"
            f"{urgency}{planner_msg}"
        )
        messages[0] = dict(messages[0])
        messages[0]["content"] = messages[0]["content"] + commander_context
        return messages


# ──────────────────────────────────────────────────────────────────────
# BLUE Commander
# ──────────────────────────────────────────────────────────────────────
class BlueCommander(BaseCommander):
    """The single trained BLUE brain. Spawns blue specialists on demand."""

    _model_env_key = "hf_model_blue_commander"
    DEFAULT_ROSTER = ("surveillance", "threat_hunter", "deception_architect", "forensics")

    def __init__(self, agent_id: str, config: CipherConfig) -> None:
        super().__init__(
            agent_id,
            team="blue",
            config=config,
            max_concurrent=int(getattr(config, "env_max_subagents_blue", 6)),
            spawn_budget=int(getattr(config, "env_subagent_spawn_budget_blue", 12)),
        )
        # Track highest zone seen in anomaly feed so we can react to RED advancing.
        self._max_anomaly_zone_seen: int = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._max_anomaly_zone_seen = 0

    def act_step(self, *, step: int, parallel: bool = False) -> list[Action]:
        """
        Override to pre-spawn essential surveillance subagents before calling the
        LLM. The LLM reliably ignores spawn instructions, so we enforce it here
        programmatically regardless of what the LLM decides.
        """
        self._cur_step = step
        self._pre_spawn_surveillance(step=step)
        # Delegate to base implementation (which calls self.act() for the LLM action)
        return super().act_step(step=step, parallel=parallel)

    def _pre_spawn_surveillance(self, *, step: int) -> None:
        """
        Ensure the two most critical BLUE subagents are always alive:
        - surveillance (persistent watcher)
        - alert_judge (triggers alerts when confidence is high enough)
        Called before the LLM acts so the roster is populated regardless of LLM choice.
        """
        if self.registry.spawn_budget_remaining <= 0:
            return
        alive_roles = {s.role for s in self.registry.alive()}
        priority_roles = ["surveillance", "alert_judge"]
        for role in priority_roles:
            if role in alive_roles:
                continue
            if len(self.registry) >= self.registry.max_concurrent:
                break
            try:
                spec = self._make_spec(
                    role,
                    task_brief=(
                        "Auto-spawned: maintain persistent network surveillance "
                        "and trigger alerts when detection confidence is sufficient."
                    ),
                    lifespan=12,
                )
                self.registry.spawn(spec, step=step)
            except Exception as exc:
                logger.debug("BLUE pre-spawn %s failed: %s", role, exc)

    def _stub_act(self) -> Action:
        obs = self._current_observation

        # 1) Ensure default roster is alive (long lifespans — 99 for stubs).
        alive_roles = {s.role for s in self.registry.alive()}
        for role in self.DEFAULT_ROSTER:
            if role not in alive_roles:
                spec = self._make_spec(
                    role,
                    task_brief=(
                        f"Operate as the team's {role}. Maintain legacy CIPHER behaviour."
                    ),
                    lifespan=99,
                )
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.SPAWN_SUBAGENT,
                    reasoning=f"Spawning default {role} subagent.",
                    subagent_spec=spec,
                    role="commander",
                )

        # 2) Commander personally acts — analyze anomalies or investigate high-zone nodes.
        if isinstance(obs, BlueObservation):
            det = float(obs.current_detection_confidence)

            # Track highest anomaly zone to react to zone breaches.
            for anomaly in (obs.anomaly_feed or []):
                az_raw = getattr(anomaly, "zone", 0)
                az = int(az_raw) if az_raw is not None else 0
                if az > self._max_anomaly_zone_seen:
                    self._max_anomaly_zone_seen = az

            # Trigger alert if confidence is high enough.
            if det >= 0.50 and obs.anomaly_feed:
                best = max(obs.anomaly_feed, key=lambda a: a.severity)
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.TRIGGER_ALERT,
                    target_node=best.node_id,
                    reasoning=f"Detection {det:.0%} — triggering alert on highest-severity node {best.node_id}.",
                    role="commander",
                )

            # Investigate high-zone anomalies personally.
            high_zone_anomalies = [
                a for a in (obs.anomaly_feed or [])
                if int(getattr(a, "zone", 0) or 0) >= 2
            ]
            if high_zone_anomalies:
                best = max(high_zone_anomalies, key=lambda a: a.severity)
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.INVESTIGATE_NODE,
                    target_node=best.node_id,
                    reasoning=f"Zone {best.zone} anomaly at node {best.node_id} — investigating.",
                    role="commander",
                )

            # Analyze any anomaly in feed.
            if obs.anomaly_feed:
                worst = max(obs.anomaly_feed, key=lambda a: a.severity)
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.ANALYZE_ANOMALY,
                    target_node=worst.node_id,
                    reasoning=f"Analyzing anomaly at node {worst.node_id} (severity={worst.severity:.2f}).",
                    role="commander",
                )

            # Proactively place a honeypot in the highest predicted RED zone.
            if (
                det >= 0.20
                and self.registry.spawn_budget_remaining > 0
                and len(self.registry) < self.registry.max_concurrent
                and not any(s.role == "deception_architect" for s in self.registry.alive())
            ):
                target_zone = max(1, self._max_anomaly_zone_seen)
                spec = self._make_spec(
                    "deception_architect",
                    task_brief=(
                        f"Detection rising. Place honeypots in Zone {target_zone} "
                        f"and Zone {target_zone + 1} to trap RED's advance."
                    ),
                    lifespan=8,
                )
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.SPAWN_SUBAGENT,
                    reasoning=f"Detection {det:.0%} — deploying deception_architect for Zone {target_zone}.",
                    subagent_spec=spec,
                    role="commander",
                )

        return Action(
            agent_id=self.agent_id,
            action_type=ActionType.STAND_DOWN,
            reasoning="Commander holding; subagents executing.",
            role="commander",
        )

    def _build_messages(self) -> list[dict[str, str]]:
        messages = super()._build_messages()
        roster = self.roster_snapshot()
        roster_lines = "\n".join(
            f"  - {r['id']} (role={r['role']}, lifespan={r['lifespan_remaining']}, brief={r['task_brief']!r})"
            for r in roster
        ) or "  (none — roster empty)"
        spawn_left = self.registry.spawn_budget_remaining
        slots_left = self.registry.max_concurrent - len(self.registry)
        roles_available = ", ".join(p.role_name for p in list_profiles("blue"))

        obs = self._current_observation
        det = float(getattr(obs, "current_detection_confidence", 0.0)) if obs else 0.0
        surv_alive = any(s.role == "surveillance" for s in self.registry.alive())
        hunter_alive = any(s.role == "threat_hunter" for s in self.registry.alive())

        # Build urgency message for LLM
        if det >= 0.75:
            urgency = (
                f"\n🚨 CRITICAL: Detection confidence {det:.0%}! "
                "TRIGGER_ALERT immediately on the highest-severity anomaly node. "
                "Do NOT spawn. Do NOT stand_down. ACT.\n"
            )
        elif det >= 0.50:
            urgency = (
                f"\n⚠ HIGH ALERT: Detection confidence {det:.0%}. "
                "Trigger alert or investigate high-zone anomalies this step.\n"
            )
        elif not surv_alive:
            urgency = (
                "\n⚠ NO SURVEILLANCE: Spawn surveillance immediately with lifespan_steps=10. "
                "BLUE is flying blind without it.\n"
            )
        elif det >= 0.30 and not hunter_alive:
            urgency = (
                f"\n⚠ THREAT BUILDING: Detection {det:.0%} — spawn threat_hunter with "
                "lifespan_steps=8 to escalate investigation.\n"
            )
        else:
            urgency = ""

        commander_context = (
            "\n\nCOMMANDER CONTEXT (BLUE):\n"
            f"Available roles: {roles_available}.\n"
            f"Spawn budget remaining: {spawn_left}. Concurrent slots free: {slots_left}.\n"
            f"Active roster ({len(roster)}):\n{roster_lines}\n"
            f"Current detection confidence: {det:.0%}.\n"
            f"{urgency}"
        )
        messages[0] = dict(messages[0])
        messages[0]["content"] = messages[0]["content"] + commander_context
        return messages
