"""
hf_app.py — HuggingFace Spaces entrypoint for CIPHER.

Serves the React War Room dashboard (built to dashboard-react/dist/) via the
Flask API server on port 7860. No Flask vs Dash ambiguity — judges see the
React dashboard.

Set HF_TOKEN and API_BASE_URL as HF Space secrets (not hardcoded here).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "dashboard-react"))

# ── Optional: pull demo traces from HF Dataset on cold start ─────────────────
def _fetch_demo_traces() -> None:
    traces_dir = Path("episode_traces")
    traces_dir.mkdir(exist_ok=True)
    if list(traces_dir.glob("*.json")):
        return
    repo_id = os.getenv("HF_TRACES_REPO", "wolfie8935/cipher-traces")
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        files = list(api.list_repo_files(repo_id, repo_type="dataset"))
        trace_files = [f for f in files if f.startswith("traces/") and f.endswith(".json")]
        for remote_path in trace_files[:5]:
            local = hf_hub_download(
                repo_id=repo_id, filename=remote_path,
                repo_type="dataset", local_dir=".",
            )
            src = Path(local)
            dst = traces_dir / src.name
            if not dst.exists():
                src.rename(dst)
        print(f"[hf_app] Fetched {len(trace_files[:5])} demo traces from {repo_id}")
    except Exception as e:
        print(f"[hf_app] Could not fetch demo traces: {e}")


_fetch_demo_traces()

# ── Import Flask app from api_server ─────────────────────────────────────────
from api_server import app  # noqa: F401 — gunicorn targets this

# ── Direct run (python hf_app.py) ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
