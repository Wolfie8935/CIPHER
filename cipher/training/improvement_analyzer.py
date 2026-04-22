"""
CIPHER Phase 10 — Reward Improvement Metrics Analyzer.

Computes all improvement metrics from training history files:
- rewards_log.csv
- prompt_evolution_log.jsonl
- training_events.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class ImprovementAnalyzer:
    """
    Computes all improvement metrics from CIPHER training history.

    Reads from:
    - rewards_log.csv            : episode rewards, terminal reasons
    - prompt_evolution_log.jsonl : evolution events with episode numbers
    - training_events.jsonl      : episode-start events with difficulty
    """

    REWARDS_LOG = Path("rewards_log.csv")
    EVOLUTION_LOG = Path("prompt_evolution_log.jsonl")
    EVENTS_LOG = Path("training_events.jsonl")

    def __init__(self) -> None:
        self._df: Optional[pd.DataFrame] = None
        self._evolutions: Optional[list[dict]] = None
        self._events: Optional[list[dict]] = None

    # ------------------------------------------------------------------ #
    # Lazy-loaded data properties                                          #
    # ------------------------------------------------------------------ #

    @property
    def df(self) -> Optional[pd.DataFrame]:
        if self._df is None:
            self._df = self._load_rewards()
        return self._df

    @property
    def evolutions(self) -> list[dict]:
        if self._evolutions is None:
            self._evolutions = self._load_evolutions()
        return self._evolutions

    @property
    def events(self) -> list[dict]:
        if self._events is None:
            self._events = self._load_events()
        return self._events

    # ------------------------------------------------------------------ #
    # Data loaders                                                         #
    # ------------------------------------------------------------------ #

    def _load_rewards(self) -> Optional[pd.DataFrame]:
        if not self.REWARDS_LOG.exists():
            return None
        try:
            df = pd.read_csv(self.REWARDS_LOG)
            if df.empty:
                return None
            if "episode" in df.columns:
                df["episode"] = pd.to_numeric(df["episode"], errors="coerce")
                df = df.dropna(subset=["episode"])
                df["episode"] = df["episode"].astype(int)
            for col in (
                "red_total", "blue_total", "red_exfil", "red_stealth",
                "red_complexity", "red_abort_penalty", "red_honeypot_penalty",
                "blue_detection", "blue_speed", "blue_fp_penalty",
                "blue_honeypot_rate", "blue_graph_reconstruction",
                "red_complexity_multiplier",
            ):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df if not df.empty else None
        except Exception:
            return None

    def _load_evolutions(self) -> list[dict]:
        if not self.EVOLUTION_LOG.exists():
            return []
        try:
            rows: list[dict] = []
            for line in self.EVOLUTION_LOG.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return rows
        except Exception:
            return []

    def _load_events(self) -> list[dict]:
        if not self.EVENTS_LOG.exists():
            return []
        try:
            rows: list[dict] = []
            for line in self.EVENTS_LOG.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return rows
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    # Public metric methods                                                #
    # ------------------------------------------------------------------ #

    def get_evolution_episodes(self) -> list[int]:
        """Return list of episode numbers where prompt evolution occurred."""
        return [int(e.get("episode", 0)) for e in self.evolutions]

    def compute_rolling_win_rates(self, window: int = 10) -> dict:
        """
        Compute rolling RED and BLUE win rates over a sliding window.

        Returns:
            {episodes, red_win_rate, blue_win_rate} — all list[float]
        """
        df = self.df
        if df is None or df.empty:
            return {"episodes": [], "red_win_rate": [], "blue_win_rate": []}

        episodes = df["episode"].tolist()
        red_wins = (df["red_total"] > 0).astype(float).tolist()
        blue_wins = (df["blue_total"] > 0).astype(float).tolist()

        return {
            "episodes": episodes,
            "red_win_rate": self._rolling_mean(red_wins, window),
            "blue_win_rate": self._rolling_mean(blue_wins, window),
        }

    def compute_exfil_rate(self, window: int = 10) -> dict:
        """
        Compute exfiltration rate (fraction with red_exfil > 0) per rolling window.
        """
        df = self.df
        if df is None or df.empty:
            return {"episodes": [], "exfil_rate": []}

        episodes = df["episode"].tolist()
        if "red_exfil" in df.columns:
            exfils = (df["red_exfil"] > 0).astype(float).tolist()
        else:
            exfils = [0.0] * len(episodes)

        return {
            "episodes": episodes,
            "exfil_rate": self._rolling_mean(exfils, window),
        }

    def compute_abort_rate(self, window: int = 10) -> dict:
        """
        Compute abort rate (fraction of aborted episodes) per rolling window.
        """
        df = self.df
        if df is None or df.empty:
            return {"episodes": [], "abort_rate": []}

        episodes = df["episode"].tolist()
        if "terminal_reason" in df.columns:
            aborts = (df["terminal_reason"] == "aborted").astype(float).tolist()
        else:
            aborts = [0.0] * len(episodes)

        return {
            "episodes": episodes,
            "abort_rate": self._rolling_mean(aborts, window),
        }

    def compute_mean_suspicion(self, window: int = 10) -> dict:
        """
        Compute mean episode suspicion (proxy: 1 - red_stealth) per rolling window.
        """
        df = self.df
        if df is None or df.empty:
            return {"episodes": [], "mean_suspicion": []}

        episodes = df["episode"].tolist()
        if "red_stealth" in df.columns:
            suspicion = (1.0 - df["red_stealth"].fillna(0.5)).tolist()
        else:
            suspicion = [0.5] * len(episodes)

        return {
            "episodes": episodes,
            "mean_suspicion": self._rolling_mean(suspicion, window),
        }

    def compute_early_late_comparison(self, pct: float = 0.10) -> dict:
        """
        Compare early (first pct%) vs late (last pct%) episode performance.

        Args:
            pct: Fraction of total episodes used for each window (default 10%).

        Returns dict with early/late averages and deltas for:
        - red reward, exfil rate, abort rate
        """
        df = self.df
        if df is None or len(df) < 2:
            return {
                "early_red_avg": 0.0, "late_red_avg": 0.0, "red_improvement": 0.0,
                "early_blue_avg": 0.0, "late_blue_avg": 0.0, "blue_improvement": 0.0,
                "early_exfil_rate": 0.0, "late_exfil_rate": 0.0, "exfil_delta": 0.0,
                "early_abort_rate": 0.0, "late_abort_rate": 0.0, "abort_delta": 0.0,
                "n_early": 0, "n_late": 0,
            }

        n = len(df)
        n_window = max(1, int(n * pct))
        early = df.head(n_window)
        late = df.tail(n_window)

        early_red = float(early["red_total"].mean())
        late_red = float(late["red_total"].mean())

        if "blue_total" in df.columns:
            early_blue = float(early["blue_total"].mean())
            late_blue = float(late["blue_total"].mean())
        else:
            early_blue = late_blue = 0.0

        if "red_exfil" in df.columns:
            early_exfil = float((early["red_exfil"] > 0).mean())
            late_exfil = float((late["red_exfil"] > 0).mean())
        else:
            early_exfil = late_exfil = 0.0

        if "terminal_reason" in df.columns:
            early_abort = float((early["terminal_reason"] == "aborted").mean())
            late_abort = float((late["terminal_reason"] == "aborted").mean())
        else:
            early_abort = late_abort = 0.0

        return {
            "early_red_avg": round(early_red, 3),
            "late_red_avg": round(late_red, 3),
            "red_improvement": round(late_red - early_red, 3),
            "early_blue_avg": round(early_blue, 3),
            "late_blue_avg": round(late_blue, 3),
            "blue_improvement": round(late_blue - early_blue, 3),
            "early_exfil_rate": round(early_exfil, 3),
            "late_exfil_rate": round(late_exfil, 3),
            "exfil_delta": round(late_exfil - early_exfil, 3),
            "early_abort_rate": round(early_abort, 3),
            "late_abort_rate": round(late_abort, 3),
            "abort_delta": round(late_abort - early_abort, 3),
            "n_early": n_window,
            "n_late": n_window,
        }

    def compute_difficulty_reward_correlation(self) -> float:
        """
        Compute Pearson correlation between episode difficulty and RED reward.

        Expected negative: harder episodes yield lower RED reward.
        Returns NaN if insufficient data.
        """
        df = self.df
        if df is None or len(df) < 5:
            return float("nan")

        difficulty_map: dict[int, float] = {}
        for event in self.events:
            if event.get("event_type") == "episode_start":
                detail = str(event.get("detail", ""))
                try:
                    diff_str = detail.split("difficulty=")[1].split(")")[0]
                    ep = int(event.get("episode", 0))
                    difficulty_map[ep] = float(diff_str)
                except Exception:
                    pass

        if len(difficulty_map) < 3:
            return float("nan")

        episodes = df["episode"].tolist()
        difficulties = [difficulty_map.get(int(ep)) for ep in episodes]
        rewards = df["red_total"].tolist()

        pairs = [(d, r) for d, r in zip(difficulties, rewards) if d is not None]
        if len(pairs) < 3:
            return float("nan")

        d_arr = np.array([p[0] for p in pairs], dtype=float)
        r_arr = np.array([p[1] for p in pairs], dtype=float)

        try:
            return float(np.corrcoef(d_arr, r_arr)[0, 1])
        except Exception:
            return float("nan")

    def get_evolution_summary(self) -> dict:
        """
        Return aggregate evolution statistics.

        Returns:
            {total_evolutions, total_red_rules, total_blue_rules, evolution_episodes}
        """
        evols = self.evolutions
        return {
            "total_evolutions": len(evols),
            "total_red_rules": sum(e.get("red_rules_count", 0) for e in evols),
            "total_blue_rules": sum(e.get("blue_rules_count", 0) for e in evols),
            "evolution_episodes": [int(e.get("episode", 0)) for e in evols],
        }

    def get_full_summary(self) -> dict:
        """
        Compute and return all improvement metrics in a single dict.
        """
        df = self.df
        n_episodes = len(df) if df is not None else 0

        return {
            "n_episodes": n_episodes,
            "rolling_win_rates": self.compute_rolling_win_rates(),
            "exfil_rate": self.compute_exfil_rate(),
            "abort_rate": self.compute_abort_rate(),
            "mean_suspicion": self.compute_mean_suspicion(),
            "early_late": self.compute_early_late_comparison(),
            "difficulty_corr": self.compute_difficulty_reward_correlation(),
            "evolution": self.get_evolution_summary(),
        }

    # ------------------------------------------------------------------ #
    # Static helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rolling_mean(values: list[float], window: int) -> list[float]:
        """Rolling mean with expanding window for the first window-1 elements."""
        if not values:
            return []
        arr = np.array(values, dtype=float)
        result = np.empty_like(arr)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            result[i] = float(np.nanmean(arr[start: i + 1]))
        return result.tolist()
