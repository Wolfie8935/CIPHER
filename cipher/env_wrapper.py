"""
OpenEnv-compliant wrapper for the CIPHER adversarial multi-agent environment.
The agent being trained via this wrapper is the RED PLANNER.
One env.step() = one complete CIPHER episode.
Observation = RED Planner's text observation (network state, mission briefing).
Action      = RED Planner's first-step action string (text).
Reward      = red_reward.total at episode end (float, range approx -1.0 to 2.0).
The remaining 7 agents (RED Analyst, Operative, Exfiltrator, 4x BLUE) operate
via their own policies (stub or live LLM) for the rest of the episode.
"""
import openenv
import os
from pathlib import Path
from typing import Optional, Any

from openenv.env.env import Env as _OpenEnvBase

from cipher.agents.base_agent import Action, ActionType
from cipher.environment.scenario import ScenarioGenerator
from cipher.environment.graph import generate_enterprise_graph
from cipher.environment.state import EpisodeState
from cipher.environment.observation import generate_red_observation
from cipher.training._episode_runner import run_episode
from cipher.utils.config import config as cipher_config


def _parse_text_action(text: str, agent_id: str = "red_planner_01") -> Action:
    """
    Convert a free-text action string into an Action object for the RED Planner.

    The text is inspected for keywords to determine the best-fitting ActionType.
    If no keywords match, defaults to WAIT.
    """
    low = text.lower()

    # Extract a potential numeric node target from the text
    import re
    node_match = re.search(r'\bnode[_ ]?(\d+)\b', low)
    target_node: Optional[int] = int(node_match.group(1)) if node_match else None

    # Extract a potential file name from the text
    file_match = re.search(r'\b(target_\S+|file_\S+|\S+\.(?:txt|dat|csv|log))\b', low)
    target_file: Optional[str] = file_match.group(1) if file_match else None

    # Keyword priority matching
    if "exfiltrat" in low:
        action_type = ActionType.EXFILTRATE
    elif "abort" in low:
        action_type = ActionType.ABORT
    elif "wait" in low or "hold" in low:
        action_type = ActionType.WAIT
    elif "read" in low and "drop" in low:
        action_type = ActionType.READ_DEAD_DROP
    elif "write" in low and "drop" in low:
        action_type = ActionType.WRITE_DEAD_DROP
    elif "read" in low and "file" in low:
        action_type = ActionType.READ_FILE
    elif "move" in low or "travers" in low or "navigate" in low or "advance" in low:
        action_type = ActionType.MOVE
    else:
        action_type = ActionType.WAIT

    return Action(
        agent_id=agent_id,
        action_type=action_type,
        target_node=target_node,
        target_file=target_file,
        reasoning=text[:500],  # Store the original text as reasoning
    )


class CIPHEREnv(_OpenEnvBase):
    """
    OpenEnv-compliant wrapper for the CIPHER adversarial multi-agent environment.

    Trained agent: RED Planner.
    One env.step() call executes a complete CIPHER episode internally.
    The other 7 agents run via their own stub or live-LLM policies.
    """

    metadata = {
        "name": "CIPHER-v1",
        "version": "1.0.0",
        "reward_range": (-1.0, 2.5),
        "action_type": "text",
        "observation_type": "text",
        "num_agents": 8,
        "description": (
            "Adversarial multi-agent network infiltration. "
            "RED team attempts to exfiltrate a classified file from a 50-node "
            "enterprise network. BLUE team attempts to detect and neutralize. "
            "Trained agent is RED Planner. One step = one full episode."
        ),
    }

    def __init__(
        self,
        max_steps: int = 20,
        difficulty: float = 0.30,
        llm_mode: str = "stub",
        use_auto_difficulty: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="CIPHER-v1")
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.llm_mode = llm_mode
        self.use_auto_difficulty = use_auto_difficulty
        self._fixed_seed = seed

        os.environ["LLM_MODE"] = llm_mode

        self._gen = ScenarioGenerator()

        # Episode tracking
        self._episode_count: int = 0
        self._scenario: Any = None
        self._graph: Any = None
        self._result: Optional[dict] = None
        self._recent_red_wins: list[int] = []

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """Reset the environment and return the initial observation and info."""

        # 1. Increment episode counter
        self._episode_count += 1

        # 2. Auto-difficulty adjustment based on last 10 episodes
        if self.use_auto_difficulty and len(self._recent_red_wins) >= 10:
            win_rate = sum(self._recent_red_wins[-10:]) / 10
            if win_rate > 0.6:
                self.difficulty = min(1.0, self.difficulty + 0.05)
            elif win_rate < 0.3:
                self.difficulty = max(0.10, self.difficulty - 0.02)

        # 3. Resolve seed
        use_seed = self._fixed_seed or seed

        # 4. Generate scenario — inject difficulty into the generator's curve
        self._gen._difficulty_curve = self.difficulty
        scenario = self._gen.generate(self._episode_count)
        self._scenario = scenario
        self._graph = scenario.generated_graph

        # 5. Create initial EpisodeState for observation generation
        initial_state = EpisodeState(graph=self._graph)
        # Resolve entry point (same logic as run_episode)
        from cipher.environment.graph import get_entry_points
        entry_points = get_entry_points(self._graph)
        initial_state.red_current_node = entry_points[0] if entry_points else 0

        # 6. Generate RED Planner's opening observation
        obs_obj = generate_red_observation(state=initial_state)

        # 7. Clear last result
        self._result = None

        # 8. Build info dict
        entry_node = initial_state.red_current_node
        info = {
            "episode": self._episode_count,
            "difficulty": self.difficulty,
            "seed": use_seed,
            "graph_nodes": self._graph.number_of_nodes() if self._graph is not None else 0,
            "graph_edges": self._graph.number_of_edges() if self._graph is not None else 0,
            "hvt_zone": 3,
            "entry_node": entry_node,
            "max_steps": self.max_steps,
            "llm_mode": self.llm_mode,
            "current_difficulty": self.difficulty,
        }

        # Serialize observation to str
        obs_text = self._obs_to_str(obs_obj)

        return obs_text, info

    def step(
        self, action: str
    ) -> tuple[str, float, bool, bool, dict]:
        """
        Execute one full CIPHER episode using *action* as the RED Planner's
        first-step directive. Always returns terminated=True, truncated=False.
        """
        if self._scenario is None or self._graph is None:
            raise RuntimeError("Call reset() before step().")

        # Convert the text action into a scripted Action for step 1
        planner_action = _parse_text_action(action, agent_id="red_planner_01")
        scripted_red = {1: [planner_action]}

        # 2. Run a complete episode
        result = run_episode(
            scenario=self._scenario,
            graph=self._graph,
            cfg=cipher_config,
            max_steps=self.max_steps,
            verbose=False,
            scripted_red_actions=scripted_red,
        )

        # 3. Store result
        self._result = result

        # 4. Extract RED reward total
        red_reward_total = float(result["red_reward"].total)

        # 5. Track win/loss history (positive reward = win for RED)
        self._recent_red_wins.append(1 if red_reward_total > 0 else 0)
        if len(self._recent_red_wins) > 50:
            self._recent_red_wins.pop(0)

        # 6. Format terminal observation string
        obs = self._format_terminal_observation(result)

        # 7-9. Standard returns
        reward = red_reward_total
        terminated = True
        truncated = False

        # 10. Build comprehensive info dict
        state_obj = result.get("state")
        rr = result["red_reward"]
        br = result["blue_reward"]

        terminal_reason = "max_steps"
        steps_taken = self.max_steps
        suspicion_final = 0.0
        if state_obj is not None:
            terminal_reason = str(getattr(state_obj, "terminal_reason", None) or "max_steps")
            steps_taken = int(getattr(state_obj, "step", self.max_steps))
            suspicion_final = float(getattr(state_obj, "blue_detection_confidence", 0.0))

        # Zones visited by RED
        zones_visited = 0
        if state_obj is not None:
            try:
                zones_seen = set()
                for n in getattr(state_obj, "red_path_history", []):
                    if isinstance(n, int) and n in state_obj.graph.nodes:
                        z = state_obj.graph.nodes[n].get("zone")
                        if z is not None:
                            zones_seen.add(z.value if hasattr(z, "value") else int(z))
                zones_visited = len(zones_seen)
            except Exception:
                zones_visited = 0

        # Traps fired
        traps_fired = 0
        if state_obj is not None:
            try:
                trap_reg = getattr(state_obj, "trap_registry", None)
                if trap_reg is not None and hasattr(trap_reg, "get_trap_summary"):
                    summary = trap_reg.get_trap_summary() or {}
                    traps_fired = int(summary.get("traps_fired", 0))
            except Exception:
                traps_fired = 0

        # Resolve AuditorJudgment object (not a dict)
        _judgment_obj = result.get("judgment")
        _fleet_verdict = (
            getattr(_judgment_obj, "episode_verdict", "contested")
            if _judgment_obj is not None
            else "contested"
        )

        # oversight_flags is a list; convert to comma-separated string (or "none")
        _oversight_raw = result.get("oversight_flags", [])
        if isinstance(_oversight_raw, list):
            _oversight_flags = ", ".join(str(f) for f in _oversight_raw) if _oversight_raw else "none"
        else:
            _oversight_flags = str(_oversight_raw) if _oversight_raw else "none"

        info: dict[str, Any] = {
            # Terminal
            "terminal_reason": terminal_reason,
            "steps_taken": steps_taken,
            "suspicion_final": suspicion_final,
            # RED reward components
            "red_total": float(rr.total),
            "red_exfil": float(rr.exfiltration_completeness),
            "red_stealth": float(1.0 - rr.detection_probability),
            "red_memory": float(rr.memory_efficiency_score),
            "red_complexity": float(rr.operation_complexity_multiplier),
            "red_abort_penalty": float(rr.abort_penalty),
            "red_honeypot_penalty": float(rr.honeypot_trigger_penalty),
            # BLUE reward components
            "blue_total": float(br.total),
            "blue_detection": float(br.detection_accuracy_score),
            "blue_speed": float(br.response_speed_bonus),
            "blue_fp_penalty": float(br.false_positive_rate_penalty),
            "blue_honeypot_rate": float(br.honeypot_trigger_rate),
            "blue_reconstruction": float(br.operation_graph_reconstruction_score),
            # Oversight
            "fleet_verdict": _fleet_verdict,
            "oversight_flags": _oversight_flags,
            # Metadata
            "episode": self._episode_count,
            "difficulty": self.difficulty,
            "dead_drops_written": int(getattr(rr, "drops_written", 0)),
            "traps_fired": traps_fired,
            "zones_visited": zones_visited,
        }

        return obs, reward, terminated, truncated, info

    def render(self) -> str:
        """Return a human-readable summary of the last episode."""
        if self._result is None:
            return "No episode run yet. Call reset() then step()."

        r = self._result
        state_obj = r.get("state")

        terminal_str = "?"
        steps_str = "?"
        if state_obj is not None:
            terminal_str = str(getattr(state_obj, "terminal_reason", "?") or "?")
            steps_str = str(getattr(state_obj, "step", "?"))

        _j = r.get("judgment")
        _verdict = (
            getattr(_j, "episode_verdict", "contested")
            if _j is not None
            else "contested"
        )
        return (
            f"CIPHER Episode {self._episode_count} | "
            f"Difficulty: {self.difficulty:.2f} | "
            f"Terminal: {terminal_str} | "
            f"Steps: {steps_str} | "
            f"RED: {r['red_reward'].total:+.3f} | "
            f"BLUE: {r['blue_reward'].total:+.3f} | "
            f"Verdict: {_verdict}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_terminal_observation(self, result: dict) -> str:
        """Build the text observation returned after a completed episode."""
        state_obj = result.get("state")
        reason = str(getattr(state_obj, "terminal_reason", None) or "max_steps")
        red_reward = float(result["red_reward"].total)
        exfil = float(result["red_reward"].exfiltration_completeness)
        susp = float(getattr(state_obj, "blue_detection_confidence", 0.0) if state_obj else 0.0)
        steps = int(getattr(state_obj, "step", self.max_steps) if state_obj else self.max_steps)

        outcome_map = {
            "exfil_success": "MISSION SUCCESS — classified file exfiltrated.",
            "exfiltration_complete": "MISSION SUCCESS — classified file exfiltrated.",
            "exfil_complete": "MISSION SUCCESS — classified file exfiltrated.",
            "detected": "MISSION FAILED — RED team detected by BLUE.",
            "aborted": "MISSION ABORTED — suspicion too high to continue.",
        }
        outcome = outcome_map.get(reason, f"MISSION INCOMPLETE — {reason}.")

        return (
            f"{outcome} Exfiltration: {exfil:.0%}. Final suspicion: {susp:.3f}. "
            f"Steps taken: {steps}/{self.max_steps}. RED reward: {red_reward:+.3f}."
        )

    @staticmethod
    def _obs_to_str(obs_obj: Any) -> str:
        """Serialize a RedObservation (or any object) to a plain string."""
        if obs_obj is None:
            return "No observation available."
        if isinstance(obs_obj, str):
            return obs_obj
        # Use __str__ if defined meaningfully, else build a compact summary
        parts = []
        for attr in (
            "step", "current_node", "current_zone", "current_hostname",
            "current_node_type", "estimated_suspicion", "adjacent_nodes",
            "files_at_current_node", "dead_drops_available",
            "zone_boundary_ahead", "credentials_held",
        ):
            val = getattr(obs_obj, attr, None)
            if val is None:
                continue
            if hasattr(val, "value"):
                val = val.value
            parts.append(f"{attr}: {val}")
        return "\n".join(parts) if parts else str(obs_obj)


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

def make_env(max_steps: int = 20, llm_mode: str = "stub") -> CIPHEREnv:
    """Factory function for creating a CIPHER environment instance."""
    return CIPHEREnv(max_steps=max_steps, llm_mode=llm_mode)
