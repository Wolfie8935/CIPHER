"""
cipher/environment/difficulty.py

Dynamic Difficulty Controller for CIPHER — Task 5 (5.md).

Replaces the simple +0.05/-0.02 win-rate escalation with a multi-axis
adaptive system that scales 6 independent environment parameters based
on rolling agent performance patterns, keeping both teams in the
learning sweet spot.

6 axes:
  honeypot_density     0.05 – 0.30
  target_files         1 – 4
  blue_response_delay  3 – 0 steps
  graph_size           40 – 60 nodes
  zone_lockdown_thresh 0.90 – 0.50  (suspicion at which zone locks down)
  trap_budget_blue     3 – 8

Used by:
  cipher/training/loop.py            — instantiate, record, compute
  cipher/training/_episode_runner.py — apply difficulty_params to env
"""
from __future__ import annotations

from typing import Any


class DynamicDifficultyController:
    """
    Multi-axis adaptive difficulty controller.

    Tracks the last `window_size` episode results and adjusts a scalar
    difficulty level [0.10, 1.0] based on RED win rate, then maps that
    scalar to 6 per-axis environment parameters.
    """

    # How much each axis responds to a difficulty delta.
    # Weights closer to 1.0 = axis changes faster with difficulty.
    AXIS_WEIGHTS = {
        "honeypot_density":    0.30,
        "target_files":        0.30,
        "blue_response_delay": 0.50,
        "graph_size":          0.50,
        "zone_lockdown":       0.30,
        "trap_budget":         0.50,
    }

    def __init__(
        self,
        window_size: int = 10,
        initial_difficulty: float = 0.30,
        max_steps: int = 20,
    ) -> None:
        self.window_size = window_size
        self.current_difficulty: float = float(initial_difficulty)
        self.max_steps: int = max_steps
        self.history: list[dict[str, Any]] = []

    # ── Recording ──────────────────────────────────────────────────

    def record_episode(self, result: dict[str, Any]) -> None:
        """
        Record the outcome of one episode.

        Expected keys (all optional — falls back to safe defaults):
          red_reward      float — RED's total reward
          blue_reward     float — BLUE's total reward
          steps           int   — steps taken this episode
          max_steps       int   — max allowed steps (for fraction calculation)
          max_zone        int   — highest RED zone reached (0-3)
          traps_triggered int   — number of trap hits
          emergent_count  int   — emergent actions used
        """
        red_r = float(result.get("red_reward", 0.0))
        steps_used = int(result.get("steps", self.max_steps))
        max_steps  = int(result.get("max_steps", self.max_steps))
        entry = {
            "red_won":          red_r > 0.0,
            "red_reward":       red_r,
            "steps_used":       steps_used,
            "steps_fraction":   steps_used / max(1, max_steps),
            "zones_reached":    int(result.get("max_zone", 0)),
            "traps_triggered":  int(result.get("traps_triggered", 0)),
            "emergent_actions": int(result.get("emergent_count", 0)),
        }
        self.history.append(entry)
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

    # ── Computation ─────────────────────────────────────────────────

    MIN_DIFFICULTY: float = 0.30   # hard floor — never drop below this
    MAX_DIFFICULTY: float = 0.90

    RED_WIN_DELTA:  float = +0.15  # RED wins  → increase difficulty by 0.15
    BLUE_WIN_DELTA: float = -0.05  # BLUE wins → decrease difficulty by 0.05

    def compute_next_difficulty(self) -> dict[str, Any]:
        """
        Compute difficulty parameters for the next episode.

        Rules (applied in all modes — stub, live, hybrid, training):
          - RED wins  → difficulty += 0.15
          - BLUE wins → difficulty -= 0.05
          - difficulty never drops below 0.30 (min floor)
          - difficulty never rises above 0.90 (max cap)
          - No history yet → return current defaults unchanged
        """
        if not self.history:
            return self._default_params()

        n = len(self.history)
        win_rate = sum(1 for h in self.history if h["red_won"]) / n
        last = self.history[-1]

        delta = self.RED_WIN_DELTA if last["red_won"] else self.BLUE_WIN_DELTA

        new_d = self.current_difficulty + delta
        self.current_difficulty = max(self.MIN_DIFFICULTY, min(self.MAX_DIFFICULTY, new_d))
        return self._params_from_difficulty(self.current_difficulty, win_rate)

    def _params_from_difficulty(
        self, d: float, win_rate: float = 0.5
    ) -> dict[str, Any]:
        return {
            # 6 original axes
            "honeypot_density":        round(0.05 + d * 0.25, 4),   # 5%–30%
            "target_files":            max(1, int(1 + d * 3)),        # 1–4
            "blue_response_delay":     max(0, int(3 - d * 3)),        # 3–0 steps
            "graph_size":              int(50 + d * 10),               # 50–60 nodes
            "zone_lockdown_threshold": round(0.90 - d * 0.40, 3),    # 0.90–0.50
            "trap_budget_blue":        int(3 + d * 5),                 # 3–8
            # 2 new axes for episode quality
            # Suspicion penalty RED receives on entering each new (higher) zone.
            # At low difficulty RED can rush; at high difficulty zones resist entry.
            "zone_entry_boost":        round(0.10 + d * 0.15, 3),    # 0.10–0.25 susp per zone hop
            # Minimum steps before the episode is allowed to end in a RED win.
            # Forces at least partial gameplay even when RED is dominant.
            "min_episode_steps":       max(5, int(self.max_steps * (0.25 + d * 0.25))),
            # 5 steps at d=0 → 10 steps at d=1 (for max_steps=20)
            # Metadata
            "difficulty":              round(d, 4),
            "win_rate_window":         round(win_rate, 4),
        }

    def _default_params(self) -> dict[str, Any]:
        return self._params_from_difficulty(self.current_difficulty, 0.5)

    # ── Helpers ─────────────────────────────────────────────────────

    def to_scenario_overrides(self) -> dict[str, Any]:
        """
        Return a dict of params that can be applied directly to a
        Scenario / graph generation call.  Excludes metadata keys.
        """
        params = self.compute_next_difficulty()
        return {k: v for k, v in params.items()
                if k not in ("difficulty", "win_rate_window")}

    def get_difficulty_report(self) -> dict[str, Any]:
        """
        Human-readable snapshot of current difficulty state.
        Used by the dashboard difficulty panel.
        """
        params = self._params_from_difficulty(self.current_difficulty)
        return {
            "current_difficulty":    self.current_difficulty,
            "episodes_in_window":    len(self.history),
            "window_size":           self.window_size,
            "recent_red_win_rate":   (
                sum(1 for h in self.history if h["red_won"]) / len(self.history)
                if self.history else 0.5
            ),
            "axes": {
                "honeypot_density":        params["honeypot_density"],
                "target_files":            params["target_files"],
                "blue_response_delay":     params["blue_response_delay"],
                "graph_size":              params["graph_size"],
                "zone_lockdown_threshold": params["zone_lockdown_threshold"],
                "trap_budget_blue":        params["trap_budget_blue"],
            },
        }
