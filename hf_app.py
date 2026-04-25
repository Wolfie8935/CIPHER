"""
hf_app.py — Hugging Face Spaces entrypoint for CIPHER Dashboard.

This wraps cipher/dashboard/app.py for HF Spaces (port 7860).
Optionally downloads demo traces from a HF Dataset on startup.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Ensure src is on path ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Optional: pull demo traces from HF Dataset on cold start ─────────────────
def _fetch_demo_traces() -> None:
    """Download a few demo episode traces from HF if traces/ is empty."""
    traces_dir = Path("episode_traces")
    traces_dir.mkdir(exist_ok=True)

    if list(traces_dir.glob("*.json")):
        return  # Already have traces

    from cipher.utils.config import config

    repo_id = str(config.hf_traces_repo)
    try:
        from huggingface_hub import hf_hub_download, list_repo_files, HfApi
        api = HfApi()
        files = list(api.list_repo_files(repo_id, repo_type="dataset"))
        trace_files = [f for f in files if f.startswith("traces/") and f.endswith(".json")]
        for remote_path in trace_files[:5]:  # Limit to 5 demo traces
            local = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                local_dir=".",
            )
            # Move to episode_traces/
            src = Path(local)
            dst = traces_dir / src.name
            if not dst.exists():
                src.rename(dst)
        print(f"[hf_app] Fetched {len(trace_files[:5])} demo traces from {repo_id}")
    except Exception as e:
        print(f"[hf_app] Could not fetch demo traces: {e}")


_fetch_demo_traces()

# ── Import the Dash app ───────────────────────────────────────────────────────
from cipher.dashboard.app import app, server  # noqa: F401

# ── Configure for HF Spaces ──────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
