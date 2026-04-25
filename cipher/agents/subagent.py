"""
Generic Subagent worker for CIPHER's v2 commander+subagent architecture.

A Subagent is constructed from a SubagentRoleProfile + a SubagentSpec
(task brief from the commander). It implements the same `BaseAgent`
contract — observe() / act() / reset() — so the rest of the system
(episode runner, reward functions, oversight) doesn't need to know
whether it is talking to a hand-coded RedPlanner or a dynamically
spawned 'scout'.

Behaviour by mode
-----------------
* stub mode: if the role has a legacy_class_path (e.g. RedPlanner),
  we proxy observe/_stub_act to the legacy class so we preserve every
  heuristic. Otherwise we fall back to a tiny generic policy that
  picks a sensible primitive from `allowed_actions` based on the
  observation.
* live / hybrid mode: by default subagents do NOT call the LLM
  (commander is the brain). When the commander spawns a subagent
  with `use_llm=True`, the subagent calls the LLM with its role
  prompt + the task brief from the commander.
"""
from __future__ import annotations

import importlib
import random
from typing import Any

from cipher.agents.base_agent import (
    Action,
    ActionType,
    BaseAgent,
    BLUE_ACTIONS,
    META_ACTIONS,
    RED_ACTIONS,
    SubagentSpec,
)
from cipher.agents.role_profiles import SubagentRoleProfile, get_profile
from cipher.environment.observation import BlueObservation, RedObservation
from cipher.utils.config import CipherConfig
from cipher.utils.logger import get_logger

logger = get_logger(__name__)


# Resolve role → .env LoRA mapping for hybrid specialists at runtime.
def _build_lora_map() -> dict[tuple[str, str], tuple[str, str]]:
    out: dict[tuple[str, str], tuple[str, str]] = {}
    from cipher.agents.role_profiles import list_profiles
    for p in list_profiles():
        if p.lora_env_key and p.lora_default_path:
            out[(p.team, p.role_name)] = (p.lora_env_key, p.lora_default_path)
    return out


class Subagent(BaseAgent):
    """
    Dynamic worker spawned by a Commander. Inherits BaseAgent so it gets
    LLM plumbing, prompt building, parsing, and history compression for free.
    """

    # No LoRA by default; populated dynamically based on role profile.
    _model_env_key: str = ""

    def __init__(
        self,
        agent_id: str,
        spec: SubagentSpec,
        profile: SubagentRoleProfile,
        config: CipherConfig,
    ) -> None:
        # The team/role come from the profile (canonical) — spec may override.
        super().__init__(agent_id, spec.team or profile.team, profile.role_name, config)
        self._profile = profile
        self._spec = spec
        self._task_brief: str = spec.task_brief or ""
        self._spawned_at_step: int | None = None
        self._lifespan_remaining: int = max(1, int(spec.lifespan_steps or profile.default_lifespan))
        self._use_llm: bool = bool(spec.use_llm)

        # Resolve model env key from the profile (so commander/LoRA routing still
        # works for legacy specialists like 'planner', 'analyst', etc.).
        self._model_env_key = self._profile_model_env_key(profile)

        # If the profile points to a legacy class, instantiate it as a
        # behaviour proxy. We only use its observe()/_stub_act() — its
        # action_history etc. is owned by THIS Subagent.
        self._legacy_proxy: BaseAgent | None = self._build_legacy_proxy(profile, config)

        # Compute allowed actions (whitelist).
        if profile.allowed_actions:
            self._allowed_actions: set[ActionType] = {
                ActionType(a) for a in profile.allowed_actions if _is_known_action(a)
            }
        else:
            base = RED_ACTIONS if self.team == "red" else BLUE_ACTIONS
            self._allowed_actions = {a for a in base if a.value not in META_ACTIONS}

    # ------------------------------------------------------------------
    # BaseAgent contract
    # ------------------------------------------------------------------
    def observe(self, observation) -> None:
        self._current_observation = observation
        if self._legacy_proxy is not None:
            try:
                self._legacy_proxy.observe(observation)
            except Exception:
                pass

    def reset(self) -> None:
        super().reset()
        if self._legacy_proxy is not None:
            try:
                self._legacy_proxy.reset()
            except Exception:
                pass

    def _stub_act(self) -> Action:
        action = self._stub_act_inner()
        return self._sanitize_action(action)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _stub_act_inner(self) -> Action:
        # Preferred path: legacy class heuristic (preserves all hand-coded behaviour).
        if self._legacy_proxy is not None:
            try:
                proxy_action = self._legacy_proxy._stub_act()
                proxy_action.agent_id = self.agent_id
                return proxy_action
            except Exception as exc:
                logger.warning(
                    "Legacy proxy %s failed (%s); falling back to generic policy.",
                    self._profile.legacy_class_path,
                    exc,
                )

        return self._generic_policy()

    def _generic_policy(self) -> Action:
        """
        Tiny fallback policy used when no legacy class is mapped (i.e. for
        emergent roles like 'scout', 'dead_drop_courier').
        """
        obs = self._current_observation
        team = self.team
        role = self.role

        if obs is None:
            fallback = ActionType.WAIT if team == "red" else ActionType.STAND_DOWN
            return Action(
                agent_id=self.agent_id,
                action_type=fallback,
                reasoning="No observation yet — waiting.",
            )

        # Role-specific tiny heuristics
        if role == "scout" and isinstance(obs, RedObservation) and obs.adjacent_nodes:
            honeypots = set(getattr(obs, "honeypot_nodes_nearby", []) or [])
            breadcrumbs = set(getattr(obs, "breadcrumb_nodes_nearby", []) or [])
            safe = [n for n in obs.adjacent_nodes if n not in honeypots and n not in breadcrumbs]
            target = random.choice(safe if safe else obs.adjacent_nodes)
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.MOVE,
                target_node=target,
                reasoning=f"Scout exploring adjacent node {target}.",
            )

        if role == "dead_drop_courier" and isinstance(obs, RedObservation):
            if obs.context_reset_this_step and obs.dead_drops_available:
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.READ_DEAD_DROP,
                    reasoning="Courier: context reset — recovering mission state from dead drop.",
                )
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.WRITE_DEAD_DROP,
                reasoning="Courier: persisting current state to a dead drop.",
            )

        if role == "anomaly_triager" and isinstance(obs, BlueObservation):
            if obs.anomaly_feed:
                worst = max(obs.anomaly_feed, key=lambda a: a.severity)
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.ANALYZE_ANOMALY,
                    target_node=worst.node_id,
                    reasoning=f"Triager: deepest-severity anomaly is at node {worst.node_id}.",
                )
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.STAND_DOWN,
                reasoning="Triager: anomaly feed empty.",
            )

        if role == "alert_judge" and isinstance(obs, BlueObservation):
            # Lower threshold to 0.40 so BLUE alerts faster.
            confident = float(obs.current_detection_confidence) >= 0.40
            if confident and obs.anomaly_feed:
                target = max(obs.anomaly_feed, key=lambda a: a.severity).node_id
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.TRIGGER_ALERT,
                    target_node=target,
                    reasoning=f"Alert judge: detection={obs.current_detection_confidence:.2f} — alerting on node {target}.",
                )
            # Even without anomaly feed, investigate if confidence is meaningful.
            if float(obs.current_detection_confidence) >= 0.40 and obs.zone_alert_levels:
                hottest_zone = max(obs.zone_alert_levels, key=obs.zone_alert_levels.get)
                return Action(
                    agent_id=self.agent_id,
                    action_type=ActionType.INVESTIGATE_NODE,
                    target_node=int(hottest_zone) * 10 + 1,  # best-guess zone entry node
                    reasoning=f"Alert judge: confidence {obs.current_detection_confidence:.2f} — investigating hottest zone {hottest_zone}.",
                )
            return Action(
                agent_id=self.agent_id,
                action_type=ActionType.STAND_DOWN,
                reasoning="Alert judge: evidence below threshold — holding alert.",
            )

        # Generic fallback: pick the safest no-op of the right team.
        fallback = ActionType.WAIT if team == "red" else ActionType.STAND_DOWN
        return Action(
            agent_id=self.agent_id,
            action_type=fallback,
            reasoning=f"{role}: no high-value primitive available — holding.",
        )

    def _sanitize_action(self, action: Action) -> Action:
        """
        Enforce the role's allowed_actions whitelist and stamp provenance.
        Meta-actions are NEVER allowed for subagents.
        """
        # Meta-actions are commander-only.
        if action.action_type.value in META_ACTIONS:
            fallback = ActionType.WAIT if self.team == "red" else ActionType.STAND_DOWN
            action = Action(
                agent_id=self.agent_id,
                action_type=fallback,
                reasoning=f"{self.role}: subagents may not emit meta-actions — falling back.",
            )

        # Whitelist check (skip for emergent — always allowed).
        if (
            action.action_type != ActionType.EMERGENT
            and self._allowed_actions
            and action.action_type not in self._allowed_actions
        ):
            fallback = ActionType.WAIT if self.team == "red" else ActionType.STAND_DOWN
            logger.debug(
                "Subagent %s tried disallowed action %s — falling back.",
                self.agent_id,
                action.action_type.value,
            )
            action = Action(
                agent_id=self.agent_id,
                action_type=fallback,
                reasoning=f"{self.role}: action not in whitelist — falling back.",
            )

        # Stamp provenance
        action.agent_id = self.agent_id
        action.role = self.role
        action.spawned_by = self._spec.parent_id or None
        return action

    def _build_messages(self) -> list[dict[str, str]]:
        """
        Append the commander's task brief to the system prompt so the LLM
        sees what the subagent is being asked to do.
        """
        messages = super()._build_messages()
        if self._task_brief:
            messages[0] = dict(messages[0])
            messages[0]["content"] = (
                messages[0]["content"]
                + "\n\nCOMMANDER TASK BRIEF:\n"
                + self._task_brief
                + f"\n\n(You are a short-lived '{self.role}' subagent — focus only on the brief above.)"
            )
        return messages

    # ------------------------------------------------------------------
    # Lifespan helpers used by SubagentRegistry
    # ------------------------------------------------------------------
    @property
    def lifespan_remaining(self) -> int:
        return self._lifespan_remaining

    def tick_lifespan(self) -> None:
        self._lifespan_remaining = max(0, self._lifespan_remaining - 1)

    def is_alive(self) -> bool:
        return self._lifespan_remaining > 0

    def assign_task(self, task_brief: str, lifespan_steps: int | None = None) -> None:
        """Refresh this subagent with a new task brief and optional new lifespan."""
        self._task_brief = task_brief
        self._spec = self._spec.model_copy(update={"task_brief": task_brief})
        if lifespan_steps is not None:
            self._lifespan_remaining = max(1, int(lifespan_steps))

    @property
    def task_brief(self) -> str:
        return self._task_brief

    @property
    def profile(self) -> SubagentRoleProfile:
        return self._profile

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _profile_model_env_key(profile: SubagentRoleProfile) -> str:
        """Pick a sensible HF model env key for the role."""
        team = profile.team
        role = profile.role_name
        # Match legacy keys when possible.
        legacy_map = {
            ("red", "planner"): "hf_model_red_planner",
            ("red", "analyst"): "hf_model_red_analyst",
            ("red", "operative"): "hf_model_red_operative",
            ("red", "exfiltrator"): "hf_model_red_exfil",
            ("blue", "surveillance"): "hf_model_blue_surv",
            ("blue", "threat_hunter"): "hf_model_blue_hunter",
            ("blue", "deception_architect"): "hf_model_blue_deceiver",
            ("blue", "forensics"): "hf_model_blue_forensics",
        }
        if (team, role) in legacy_map:
            return legacy_map[(team, role)]
        # Emergent roles share the commander's model.
        return "hf_model_red_commander" if team == "red" else "hf_model_blue_commander"

    @staticmethod
    def _build_legacy_proxy(
        profile: SubagentRoleProfile, config: CipherConfig
    ) -> BaseAgent | None:
        if not profile.legacy_class_path:
            return None
        try:
            module_path, class_name = profile.legacy_class_path.split(":")
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            # Use a deterministic id so legacy code that introspects agent_id still works.
            proxy_id = f"{profile.team}_{profile.role_name}_proxy"
            return cls(proxy_id, config)
        except Exception as exc:
            logger.warning(
                "Failed to build legacy proxy for role %s (%s): %s",
                profile.role_name,
                profile.legacy_class_path,
                exc,
            )
            return None


def _is_known_action(name: str) -> bool:
    try:
        ActionType(name)
        return True
    except ValueError:
        return False


def build_subagent(
    spec: SubagentSpec,
    config: CipherConfig,
    *,
    agent_id: str | None = None,
) -> Subagent | None:
    """
    Factory: instantiate a Subagent from a SubagentSpec, looking up its profile.

    Returns None if the role is not registered.
    """
    profile = get_profile(spec.role_name)
    if profile is None:
        logger.warning("Subagent role '%s' is not registered — refusing spawn.", spec.role_name)
        return None
    if profile.team != spec.team:
        logger.warning(
            "Subagent role '%s' belongs to team '%s' but spec asked for '%s' — refusing spawn.",
            spec.role_name, profile.team, spec.team,
        )
        return None
    final_id = agent_id or spec.subagent_id
    if not final_id:
        # Caller should normally pass the registry-assigned id; guard anyway.
        final_id = f"{spec.team}_{spec.role_name}_anon"
    return Subagent(final_id, spec, profile, config)
