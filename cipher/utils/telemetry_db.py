"""
cipher/utils/telemetry_db.py  —  B2: Thread-safe SQLite telemetry store.

Primary episode data source for the dashboard.
rewards_log.csv kept for backwards compatibility.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("telemetry.db")

_COLUMNS = [
    "run_id", "llm_mode", "episode", "timestamp", "steps", "terminal_reason",
    "red_total", "blue_total",
    "red_exfil", "red_stealth", "red_memory", "red_complexity",
    "red_abort_penalty", "red_honeypot_penalty", "red_emergent_bonus",
    "blue_detection", "blue_speed", "blue_fp_penalty",
    "blue_honeypot_rate", "blue_graph_reconstruction", "blue_emergent_bonus",
    "oversight_red_adj", "oversight_blue_adj", "oversight_flags",
    "red_unique_nodes", "red_drops_written", "red_traps_placed",
    "red_context_resets", "red_complexity_multiplier",
    "fleet_verdict", "fleet_judgment",
]

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, llm_mode TEXT, episode INTEGER, timestamp TEXT,
    steps INTEGER, terminal_reason TEXT,
    red_total REAL, blue_total REAL,
    red_exfil REAL, red_stealth REAL, red_memory REAL, red_complexity REAL,
    red_abort_penalty REAL, red_honeypot_penalty REAL, red_emergent_bonus REAL,
    blue_detection REAL, blue_speed REAL, blue_fp_penalty REAL,
    blue_honeypot_rate REAL, blue_graph_reconstruction REAL, blue_emergent_bonus REAL,
    oversight_red_adj REAL, oversight_blue_adj REAL, oversight_flags TEXT,
    red_unique_nodes INTEGER, red_drops_written INTEGER, red_traps_placed INTEGER,
    red_context_resets INTEGER, red_complexity_multiplier REAL,
    fleet_verdict TEXT, fleet_judgment TEXT
)
"""


class TelemetryDB:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self._db_path), check_same_thread=False)

    def _init_db(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.execute(_CREATE_SQL)
            conn.commit()
            conn.close()

    def write_episode(self, row: Dict[str, Any]) -> None:
        cols = [c for c in _COLUMNS if c in row]
        sql = f"INSERT INTO episodes ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
        with self._lock:
            conn = self._conn()
            conn.execute(sql, [row[c] for c in cols])
            conn.commit()
            conn.close()

    def get_last_n_episodes(self, n: int, run_id: Optional[str] = None) -> List[Dict]:
        with self._lock:
            conn = self._conn()
            if run_id:
                cur = conn.execute(
                    "SELECT * FROM episodes WHERE run_id=? ORDER BY id DESC LIMIT ?",
                    (run_id, n),
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM episodes ORDER BY id DESC LIMIT ?", (n,)
                )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.close()
        return list(reversed(rows))

    def get_all_episodes(self, run_id: Optional[str] = None) -> List[Dict]:
        with self._lock:
            conn = self._conn()
            if run_id:
                cur = conn.execute(
                    "SELECT * FROM episodes WHERE run_id=? ORDER BY id", (run_id,)
                )
            else:
                cur = conn.execute("SELECT * FROM episodes ORDER BY id")
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.close()
        return rows

    def get_distinct_runs(self) -> List[Dict]:
        """Return each unique (run_id, llm_mode, min_timestamp) sorted newest first."""
        with self._lock:
            conn = self._conn()
            cur = conn.execute(
                "SELECT run_id, llm_mode, MIN(timestamp) as started, COUNT(*) as eps "
                "FROM episodes GROUP BY run_id ORDER BY started DESC"
            )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            conn.close()
        return rows


_db: Optional[TelemetryDB] = None


def get_db() -> TelemetryDB:
    global _db
    if _db is None:
        _db = TelemetryDB()
    return _db
