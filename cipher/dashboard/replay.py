"""
Phase 12 replay utilities.

This module keeps replay-specific parsing and export logic separated from the
Dash app wiring so the visualization layer is easier to reason about.
"""

import json
from pathlib import Path
from datetime import datetime

import plotly.io as pio


def infer_runtime_mode(episode_data: dict | None = None) -> str:
    """Return a human-readable runtime mode label without hardcoded STUB fallback."""
    data = episode_data or {}
    for key in ("llm_mode", "mode", "runtime_mode"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().upper()

    from cipher.utils.config import config
    from cipher.utils.llm_mode import get_llm_mode

    mode = get_llm_mode().strip().lower()
    backend = str(config.llm_backend).strip().lower()

    # Explicit mode always wins (main.py sets LLM_MODE in os.environ at runtime).
    if mode:
        return mode.upper()
    if backend in {"hf", "openai", "huggingface"}:
        return f"LIVE ({backend.upper()})"

    return "UNKNOWN"


def extract_trap_steps(data: dict) -> list[int]:
    steps: list[int] = []
    for evt in data.get("trap_events_log", data.get("traps_triggered_log", [])):
        if not isinstance(evt, dict):
            continue
        st = evt.get("step")
        if isinstance(st, int):
            steps.append(st)
    if steps:
        return sorted(set(steps))

    # Fallback from action log.
    for entry in data.get("episode_log", []):
        action_type = str(entry.get("action_type", "")).lower()
        if "trap" in action_type or "honeypot" in action_type:
            st = entry.get("step", entry.get("t"))
            if isinstance(st, int):
                steps.append(st)
    return sorted(set(steps))


def extract_honeypot_trigger_steps(data: dict) -> list[int]:
    steps: list[int] = []
    for evt in data.get("blue_anomaly_history", []):
        if not isinstance(evt, dict):
            continue
        if str(evt.get("event_type", "")).upper() == "HONEYPOT_TRIGGERED":
            st = evt.get("step")
            if isinstance(st, int):
                steps.append(st)
    if steps:
        return sorted(set(steps))

    last = data.get("last_honeypot_trigger_step")
    if isinstance(last, int):
        return [last]
    return []


def extract_forensics_path(data: dict) -> list[int]:
    """Best-effort BLUE Forensics reconstructed path."""
    for key in (
        "forensics_reconstructed_path",
        "blue_reconstructed_path",
        "operation_graph_reconstructed_path",
    ):
        v = data.get(key)
        if isinstance(v, list) and v:
            out = [int(x) for x in v if isinstance(x, (int, float, str)) and str(x).isdigit()]
            if out:
                return out

    path: list[int] = []
    for entry in data.get("episode_log", []):
        if str(entry.get("agent_id", "")).startswith("blue_forensics"):
            payload = entry.get("payload", {})
            if isinstance(payload, dict):
                n = payload.get("target_node")
                if isinstance(n, (int, float, str)) and str(n).isdigit():
                    path.append(int(n))
    if path:
        return path

    nodes = data.get("blue_investigated_nodes", [])
    if isinstance(nodes, list):
        return [int(n) for n in nodes if isinstance(n, (int, float, str)) and str(n).isdigit()]
    return []


def operation_complexity_score(data: dict) -> float:
    """Derived complexity score based on path depth, traps, and reset survival."""
    uniq_nodes = len(set(data.get("red_visited_nodes", []) or data.get("red_path_history", [])))
    resets = int(data.get("red_context_resets", 0) or 0)
    red_traps = int(data.get("red_traps_placed_count", 0) or len(data.get("red_traps_placed", [])))
    trap_events = len(data.get("trap_events_log", []) or data.get("traps_triggered_log", []))

    raw = 1.0 + 0.02 * uniq_nodes + 0.20 * resets + 0.15 * red_traps + 0.05 * trap_events
    return round(min(3.0, max(1.0, raw)), 2)


def _safe_name(path: str) -> str:
    stem = Path(path).stem if path else "episode"
    return "".join(ch for ch in stem if ch.isalnum() or ch in ("-", "_")).strip("_") or "episode"


def export_replay_html(
    *,
    trace_path: str,
    network_fig,
    timeline_fig,
    out_dir: Path | None = None,
) -> Path:
    """Export current replay view to a standalone HTML file."""
    target_dir = out_dir or Path("dashboard_exports")
    target_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = target_dir / f"replay_{_safe_name(trace_path)}_{ts}.html"

    net_html = pio.to_html(network_fig, include_plotlyjs="cdn", full_html=False)
    tl_html = pio.to_html(timeline_fig, include_plotlyjs=False, full_html=False)
    title = f"CIPHER Replay Export — {Path(trace_path).name if trace_path else 'episode'}"
    html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <style>
      body {{ background: #0a0e1a; color: #e2e8f0; font-family: Inter, Arial, sans-serif; margin: 18px; }}
      h2 {{ margin: 0 0 12px 0; }}
      .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
      .card {{ background: #111827; border: 1px solid #2a3550; border-radius: 8px; padding: 8px; }}
    </style>
  </head>
  <body>
    <h2>{title}</h2>
    <div class="grid">
      <div class="card">{tl_html}</div>
      <div class="card">{net_html}</div>
    </div>
  </body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path

