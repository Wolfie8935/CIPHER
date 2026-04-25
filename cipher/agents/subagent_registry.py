"""
SubagentRegistry — per-commander state holder for dynamically spawned subagents.

Responsibilities
----------------
* Spawn / dismiss subagents based on Commander meta-actions.
* Enforce per-team caps (max concurrent + spawn budget per episode).
* Tick lifespans every step and auto-dismiss expired subagents.
* Provide stable, unique agent_ids (e.g. red_scout_03).
* Track lifecycle events for the dashboard / oversight layer.

This module is intentionally engine-agnostic: it doesn't know how the
commander chooses what to spawn, only how to maintain the roster.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from cipher.agents.role_profiles import get_profile
from cipher.agents.subagent import Subagent, build_subagent
from cipher.agents.base_agent import SubagentSpec
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LifecycleEvent:
    """A single spawn/dismiss/expire event for the dashboard + oversight."""
    step: int
    event_type: str  # 'spawn' | 'dismiss' | 'expire' | 'delegate' | 'reject'
    subagent_id: str
    role: str
    team: str
    reason: str = ""
    parent_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "event_type": self.event_type,
            "subagent_id": self.subagent_id,
            "role": self.role,
            "team": self.team,
            "reason": self.reason,
            "parent_id": self.parent_id,
        }


class SubagentRegistry:
    """One per Commander. Tracks alive subagents and their budgets."""

    def __init__(
        self,
        team: str,
        commander_id: str,
        config: CipherConfig,
        max_concurrent: int,
        spawn_budget: int,
    ) -> None:
        self.team = team
        self.commander_id = commander_id
        self.config = config
        self.max_concurrent = int(max_concurrent)
        self.spawn_budget_remaining = int(spawn_budget)
        self.total_spawns = 0
        self._alive: dict[str, Subagent] = {}
        self._role_counts: dict[str, int] = {}
        self.events: list[LifecycleEvent] = []
        self._spawn_history: list[tuple[int, str]] = []  # (step, role)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def alive(self) -> list[Subagent]:
        """Return alive subagents in deterministic insertion order."""
        return list(self._alive.values())

    def alive_ids(self) -> list[str]:
        return list(self._alive.keys())

    def __len__(self) -> int:
        return len(self._alive)

    def __contains__(self, subagent_id: str) -> bool:
        return subagent_id in self._alive

    def get(self, subagent_id: str) -> Subagent | None:
        return self._alive.get(subagent_id)

    def spawn(self, spec: SubagentSpec, *, step: int) -> Subagent | None:
        """
        Try to spawn a new subagent. Returns the Subagent on success, None on
        rejection (cap reached, budget exhausted, unknown role, etc.).
        """
        if self.spawn_budget_remaining <= 0:
            self._record(step, "reject", spec.role_name, reason="spawn_budget_exhausted")
            return None
        if len(self._alive) >= self.max_concurrent:
            self._record(step, "reject", spec.role_name, reason="max_concurrent_reached")
            return None
        profile = get_profile(spec.role_name)
        if profile is None:
            self._record(step, "reject", spec.role_name, reason="unknown_role")
            return None
        if profile.team != self.team:
            self._record(step, "reject", spec.role_name, reason="wrong_team")
            return None

        # Assign deterministic unique id
        self._role_counts[spec.role_name] = self._role_counts.get(spec.role_name, 0) + 1
        subagent_id = f"{self.team}_{spec.role_name}_{self._role_counts[spec.role_name]:02d}"

        # Clamp lifespan: enforce a floor of half the profile's default_lifespan
        # so the LLM can't accidentally specify lifespan=1 and kill a surveillance
        # agent in a single tick. Short-lived roles (scout, alert_judge) keep a
        # lower floor so tests and intentional short spawns still work.
        raw_lifespan = int(spec.lifespan_steps or profile.default_lifespan)
        min_lifespan = max(2, profile.default_lifespan // 2)
        clamped_lifespan = max(raw_lifespan, min_lifespan)
        spec = spec.model_copy(update={
            "team": self.team,
            "parent_id": spec.parent_id or self.commander_id,
            "subagent_id": subagent_id,
            "lifespan_steps": clamped_lifespan,
        })

        subagent = build_subagent(spec, self.config, agent_id=subagent_id)
        if subagent is None:
            self._record(step, "reject", spec.role_name, reason="build_failed")
            return None

        subagent._spawned_at_step = step
        self._alive[subagent_id] = subagent
        self.spawn_budget_remaining -= 1
        self.total_spawns += 1
        self._spawn_history.append((step, spec.role_name))
        self._record(
            step,
            "spawn",
            spec.role_name,
            subagent_id=subagent_id,
            reason=spec.task_brief[:120],
        )
        logger.debug(
            "[%s] spawned %s (lifespan=%d, brief=%r)",
            self.commander_id, subagent_id, spec.lifespan_steps, spec.task_brief[:60],
        )
        return subagent

    def delegate(self, subagent_id: str, task_brief: str, *, step: int,
                 lifespan_steps: int | None = None) -> bool:
        """Refresh an existing subagent with a new task brief."""
        sub = self._alive.get(subagent_id)
        if sub is None:
            self._record(step, "reject", "?", subagent_id=subagent_id, reason="delegate_unknown_id")
            return False
        sub.assign_task(task_brief, lifespan_steps)
        self._record(step, "delegate", sub.role, subagent_id=subagent_id, reason=task_brief[:120])
        return True

    def dismiss(self, subagent_id: str, *, step: int, reason: str = "commander_dismiss") -> bool:
        sub = self._alive.pop(subagent_id, None)
        if sub is None:
            return False
        self._record(step, "dismiss", sub.role, subagent_id=subagent_id, reason=reason)
        return True

    def tick_step(self, step: int) -> None:
        """
        Decrement lifespans and auto-dismiss anyone whose timer hit zero.
        Call this AFTER subagents have acted for the step.
        """
        expired: list[str] = []
        for sub_id, sub in self._alive.items():
            sub.tick_lifespan()
            if not sub.is_alive():
                expired.append(sub_id)
        for sub_id in expired:
            sub = self._alive.pop(sub_id, None)
            if sub is not None:
                self._record(step, "expire", sub.role, subagent_id=sub_id, reason="lifespan_zero")

    def reset_episode(self) -> None:
        """Wipe all alive subagents and reset budget — call at episode start."""
        self._alive.clear()
        self._role_counts.clear()
        self.events.clear()
        self._spawn_history.clear()
        self.total_spawns = 0
        # Budget intentionally NOT restored here; caller provides fresh budget on init.

    def detect_thrash(self, step: int, window: int = 2) -> list[str]:
        """
        Detect spawn/dismiss thrash: same role spawned twice within `window` steps.
        Returns the list of thrashing role names.
        """
        thrash: list[str] = []
        recent = [(s, r) for s, r in self._spawn_history if step - s <= window]
        seen: dict[str, int] = {}
        for _, role in recent:
            seen[role] = seen.get(role, 0) + 1
        for role, n in seen.items():
            if n >= 3:
                thrash.append(role)
        return thrash

    def lifecycle_dicts(self) -> list[dict[str, Any]]:
        """Return all events as plain dicts for trace serialization."""
        return [e.to_dict() for e in self.events]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record(
        self,
        step: int,
        event_type: str,
        role: str,
        *,
        subagent_id: str = "",
        reason: str = "",
    ) -> None:
        self.events.append(
            LifecycleEvent(
                step=step,
                event_type=event_type,
                subagent_id=subagent_id,
                role=role,
                team=self.team,
                reason=reason,
                parent_id=self.commander_id,
            )
        )
