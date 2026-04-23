"""
cipher/rewards/reward_logger.py

Logs reward components to CSV and SQLite for training curve visualization.
Each episode appends one row. SQLite is the primary source (thread-safe);
CSV is kept for backwards compatibility.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from cipher.rewards.blue_reward import BlueRewardComponents
from cipher.rewards.oversight_reward import OversightSignal
from cipher.rewards.red_reward import RedRewardComponents


class RewardLogger:
    LOG_FILE = Path("rewards_log.csv")

    COLUMNS = [
        "episode",
        "timestamp",
        "steps",
        "terminal_reason",
        "red_total",
        "red_exfil",
        "red_stealth",
        "red_memory",
        "red_complexity",
        "red_abort_penalty",
        "red_honeypot_penalty",
        "blue_total",
        "blue_detection",
        "blue_speed",
        "blue_fp_penalty",
        "blue_honeypot_rate",
        "blue_graph_reconstruction",
        "oversight_red_adj",
        "oversight_blue_adj",
        "oversight_flags",
        "red_unique_nodes",
        "red_drops_written",
        "red_traps_placed",
        "red_context_resets",
        "red_complexity_multiplier",
        "fleet_verdict",
        "fleet_judgment",
    ]

    def __init__(self) -> None:
        self._run_id = os.environ.get("CIPHER_RUN_ID", "default")
        self._llm_mode = os.environ.get("LLM_MODE", "stub")
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.LOG_FILE.exists():
            with open(self.LOG_FILE, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()

    def log(
        self,
        episode: int,
        steps: int,
        terminal_reason: str,
        red: RedRewardComponents,
        blue: BlueRewardComponents,
        oversight: OversightSignal,
        judgment: Optional["AuditorJudgment"] = None,
    ) -> None:
        ts = datetime.now().isoformat()
        row = {
            "episode": episode,
            "timestamp": ts,
            "steps": steps,
            "terminal_reason": terminal_reason,
            "red_total": round(red.total, 4),
            "red_exfil": round(red.exfiltration_completeness, 4),
            "red_stealth": round(1 - red.detection_probability, 4),
            "red_memory": round(red.memory_efficiency_score, 4),
            "red_complexity": round(red.operation_complexity_multiplier, 4),
            "red_abort_penalty": round(red.abort_penalty, 4),
            "red_honeypot_penalty": round(red.honeypot_trigger_penalty, 4),
            "blue_total": round(blue.total, 4),
            "blue_detection": round(blue.detection_accuracy_score, 4),
            "blue_speed": round(blue.response_speed_bonus, 4),
            "blue_fp_penalty": round(blue.false_positive_rate_penalty, 4),
            "blue_honeypot_rate": round(blue.honeypot_trigger_rate, 4),
            "blue_graph_reconstruction": round(blue.operation_graph_reconstruction_score, 4),
            "oversight_red_adj": round(oversight.total_red_adjustment, 4),
            "oversight_blue_adj": round(oversight.total_blue_adjustment, 4),
            "oversight_flags": "|".join(oversight.flags_fired) or "none",
            "red_unique_nodes": red.unique_nodes_visited,
            "red_drops_written": red.drops_written,
            "red_traps_placed": red.traps_placed,
            "red_context_resets": red.context_resets,
            "red_complexity_multiplier": round(red.operation_complexity_multiplier, 4),
            "fleet_verdict": judgment.episode_verdict if judgment else "none",
            "fleet_judgment": (judgment.judgment_text[:120] if judgment else "none"),
        }

        # ── CSV (backwards compat) ───────────────────────────────
        with open(self.LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(row)

        # ── SQLite (primary, thread-safe) ────────────────────────
        try:
            from cipher.utils.telemetry_db import get_db
            db_row = dict(row)
            db_row["run_id"] = self._run_id
            db_row["llm_mode"] = self._llm_mode
            get_db().write_episode(db_row)
        except Exception:
            pass
