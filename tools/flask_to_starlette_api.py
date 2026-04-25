"""One-off: convert dashboard-react/api_server_flask_backup.py to Starlette api_server.py"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "dashboard-react" / "api_server_flask_backup.py").read_text(encoding="utf-8")

# Drop Flask header through CORS and React routes (keep from "# ── Helpers")
start = src.index("# ── Helpers")
body = src[start:]

# Remove @app.route lines; Flask `request` global → Starlette `request` arg (same name)
body = re.sub(r"@app\.route\([^\n]+\)\n", "", body)
body = body.replace("request.args.get", "request.query_params.get")
body = body.replace("request.args", "request.query_params")

# jsonify(...) -> _j(...) (including multi-line returns)
body = body.replace("jsonify(", "_j(")

# Manual tuple returns in episode_summary (after jsonify→_j)
body = body.replace(
    """return (
            _j(
                {
                    "error": "invalid_name",
                    "message": "name must be a basename only (no path components).",
                    "received": name,
                    "traces_dir": str(traces_dir),
                }
            ),
            400,
        )""",
    """return _j(
                {
                    "error": "invalid_name",
                    "message": "name must be a basename only (no path components).",
                    "received": name,
                    "traces_dir": str(traces_dir),
                },
                status_code=400,
            )""",
)
body = body.replace(
    """return (
            _j(
                {
                    "error": "trace_file_not_found",
                    "message": f"No trace file at episode_traces/{name}.",
                    "resolved_path": str(path),
                    "requested_name": name,
                    "traces_dir": str(traces_dir),
                    "traces_dir_exists": traces_dir.is_dir(),
                }
            ),
            404,
        )""",
    """return _j(
                {
                    "error": "trace_file_not_found",
                    "message": f"No trace file at episode_traces/{name}.",
                    "resolved_path": str(path),
                    "requested_name": name,
                    "traces_dir": str(traces_dir),
                    "traces_dir_exists": traces_dir.is_dir(),
                },
                status_code=404,
            )""",
)
body = body.replace(
    'return _j({"error": f"read failed: {exc}"}), 500',
    'return _j({"error": f"read failed: {exc}"}, status_code=500)',
)
body = body.replace("return _j({}), 404", "return _j({}, status_code=404)")

# live_logs limit: Flask type=int → parse int from query string
body = body.replace(
    "raw_limit = request.query_params.get(\"limit\", default=200, type=int)",
    """_raw_lim = request.query_params.get("limit")
    try:
        raw_limit = int(_raw_lim) if _raw_lim not in (None, "") else 200
    except ValueError:
        raw_limit = 200""",
)

# get_episode(filename) -> async def with request
body = body.replace(
    "def get_episode(filename):",
    "async def get_episode(request):\n    filename = request.path_params[\"filename\"]",
)

# episode_summary and others need request param
route_names_need_request = [
    "live_steps",
    "thoughts",
    "agent_status",
    "architecture_doc",
    "network_graph",
    "list_episodes",
    "episode_summary",
    "rewards",
    "commanders",
    "health",
    "training_state",
    "training_events",
    "dead_drops",
    "battle_log",
    "live_logs",
    "rl_stats",
    "history",
    "analytics",
    "lore_reports",
]
for name in route_names_need_request:
    body = re.sub(rf"^def {name}\(\):", f"async def {name}(request):", body, flags=re.MULTILINE)

# _qp_get mimic for Starlette (inject at top of body after helpers)
header = '''"""
CIPHER War Room — ASGI API (Starlette)
Serves React `dist/` and `/api/*` on port 5001 (local) or 7860 (HF).
Run: uvicorn ... or python api_server.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

_HERE = Path(__file__).resolve().parent
DIST_DIR = _HERE / "dist"
ROOT = _HERE.parent


def _j(data, status_code: int = 200) -> JSONResponse:
    return JSONResponse(data, status_code=status_code)


_BUILD_HTML = """<html><body style="font-family:monospace;background:#0b1120;color:#e8f0fe;padding:40px">
<h2 style="color:#ff5252">CIPHER War Room — build needed</h2>
<p>Run: <code>cd dashboard-react &amp;&amp; npm run build</code></p>
<p>Or dev: <code>npm run dev</code> →
<a href="http://localhost:5173" style="color:#00e5ff">localhost:5173</a></p>
</body></html>"""


async def _serve_index(_request):
    idx = DIST_DIR / "index.html"
    if idx.is_file():
        return FileResponse(idx)
    return HTMLResponse(_BUILD_HTML, status_code=200)


async def _serve_vite_svg(_request):
    p = DIST_DIR / "vite.svg"
    if p.is_file():
        return FileResponse(p)
    return Response(status_code=404)


'''

# Fix architecture doc string Flask -> Starlette
body = body.replace("Flask (`api_server.py`)", "Starlette (`api_server.py`)")

# Build routes table at end — replace old entry point in body
old_main = """# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    print(f"[cipher-api] War Room API listening on http://localhost:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
"""

routes_block = """
# ── ASGI app ───────────────────────────────────────────────────────────

_routes = [
    Route("/api/live-steps", live_steps, methods=["GET"]),
    Route("/api/thoughts", thoughts, methods=["GET"]),
    Route("/api/agent-status", agent_status, methods=["GET"]),
    Route("/api/architecture-doc", architecture_doc, methods=["GET"]),
    Route("/api/network-graph", network_graph, methods=["GET"]),
    Route("/api/episodes", list_episodes, methods=["GET"]),
    Route("/api/episode-summary", episode_summary, methods=["GET"]),
    Route("/api/episode/{filename:path}", get_episode, methods=["GET"]),
    Route("/api/rewards", rewards, methods=["GET"]),
    Route("/api/commanders", commanders, methods=["GET"]),
    Route("/api/health", health, methods=["GET"]),
    Route("/api/training-state", training_state, methods=["GET"]),
    Route("/api/training-events", training_events, methods=["GET"]),
    Route("/api/dead-drops", dead_drops, methods=["GET"]),
    Route("/api/battle-log", battle_log, methods=["GET"]),
    Route("/api/live-logs", live_logs, methods=["GET"]),
    Route("/api/rl-stats", rl_stats, methods=["GET"]),
    Route("/api/history", history, methods=["GET"]),
    Route("/api/analytics", analytics, methods=["GET"]),
    Route("/api/lore", lore_reports, methods=["GET"]),
    Route("/", _serve_index, methods=["GET"]),
    Route("/vite.svg", _serve_vite_svg, methods=["GET"]),
]
_assets = DIST_DIR / "assets"
if _assets.is_dir():
    _routes.insert(-2, Mount("/assets", StaticFiles(directory=str(_assets)), name="assets"))

app = Starlette(routes=_routes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    print(f"[cipher-api] War Room ASGI on http://localhost:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
"""

if old_main not in body:
    raise SystemExit("old main block not found")
body = body.replace(old_main, routes_block)

# Ensure all _j used — any remaining jsonify
if "jsonify" in body:
    bad = [l for l in body.splitlines() if "jsonify" in l][:5]
    raise SystemExit("jsonify still in body:\n" + "\n".join(bad))

out = header + body
out_path = ROOT / "dashboard-react" / "api_server.py"
out_path.write_text(out, encoding="utf-8")
print("Wrote", out_path, "lines", len(out.splitlines()))
