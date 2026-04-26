"""
CIPHER War Room — Flask API Server
Serves live episode data to the React war room dashboard on port 5001.
Run: python dashboard-react/api_server.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

import subprocess
import threading
from datetime import datetime
from flask import Flask, jsonify, send_from_directory, redirect, Response, request
from flask_cors import CORS

_HERE    = Path(__file__).resolve().parent
DIST_DIR = _HERE / 'dist'
ROOT     = _HERE.parent          # project root

app  = Flask(__name__)
CORS(app)

# ── Background episode state ─────────────────────────────────────────
_training_lock    = threading.Lock()
_training_running = False


def _fetch_hf_live_file(filename: str) -> bytes | None:
    """Download a live data file from HF Dataset (fallback when local file is empty)."""
    repo_id = os.getenv("HF_TRACES_REPO", "wolfie8935/cipher-traces")
    token   = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        return None
    try:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(
            repo_id=repo_id,
            filename=f"live/{filename}",
            repo_type="dataset",
            token=token,
            force_download=True,       # always fetch freshest copy
        )
        return Path(local).read_bytes()
    except Exception:
        return None


def _run_bg_episode() -> None:
    """Run one CIPHER episode in background — triggered by /api/control start."""
    global _training_running
    ts_path = ROOT / "training_state.json"
    try:
        sys.path.insert(0, str(ROOT))
        from cipher.training._episode_runner import run_episode

        run_id = f"hf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Clear stale live data
        (ROOT / "live_steps.jsonl").write_text("", encoding="utf-8")
        (ROOT / "logs").mkdir(exist_ok=True)
        (ROOT / "logs" / "agent_thoughts.jsonl").write_text("", encoding="utf-8")

        ts = {
            "status": "running",
            "run_id": run_id,
            "current_episode": 1,
            "total_episodes": 1,
            "llm_mode": os.getenv("LLM_MODE", "stub"),
            "started_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_steps": 0,
        }
        ts_path.write_text(json.dumps(ts, indent=2), encoding="utf-8")
        print(f"[START] Background episode starting (run_id={run_id})", flush=True)

        # Build a simple step callback that writes live_steps.jsonl
        def _step_cb(step, max_steps, red_actions, blue_actions, state):
            try:
                from datetime import datetime as _dt
                ra = red_actions[0] if red_actions else None
                red_info = ""
                if ra:
                    atype = ra.action_type.value if hasattr(ra.action_type, "value") else str(ra.action_type)
                    node = f" → n{ra.target_node}" if ra.target_node is not None else ""
                    red_info = f"{atype}{node}"
                blue_counts: dict = {}
                for a in blue_actions:
                    if a:
                        k = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
                        blue_counts[k] = blue_counts.get(k, 0) + 1
                blue_info = " ".join(f"{k}×{v}" for k, v in blue_counts.items())
                zone_raw = state.graph.nodes[state.red_current_node].get("zone", 0)
                zone_val = getattr(zone_raw, "value", zone_raw) if zone_raw is not None else 0
                zone_names = {0: "Perimeter", 1: "General", 2: "Sensitive", 3: "Critical/HVT"}
                zone_label = zone_names.get(int(zone_val or 0), "Unknown")
                susp = round(float(getattr(state, "red_suspicion_score", 0.0)), 3)
                det  = round(float(getattr(state, "blue_detection_confidence", 0.0)), 3)
                exfil = list(getattr(state, "red_exfiltrated_files", []))
                _all_agents = []
                for a in (red_actions or []) + (blue_actions or []):
                    if not a:
                        continue
                    atype = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
                    _all_agents.append({
                        "agent_id": a.agent_id,
                        "team": "red" if str(a.agent_id).startswith("red") else "blue",
                        "action_type": atype,
                        "target_node": a.target_node,
                    })
                step_data = {
                    "run_id": run_id, "episode": 1, "step": step, "max_steps": max_steps,
                    "red_action": red_info or "waiting", "blue_actions": blue_info or "—",
                    "suspicion": susp, "detection": det, "zone": zone_label,
                    "exfil_count": len(exfil), "exfil_files": exfil[-3:],
                    "timestamp": _dt.now().isoformat(), "all_agents": _all_agents,
                }
                with open(ROOT / "live_steps.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(step_data) + "\n")
                # Update training_state step counter
                ts["total_steps"] = step
                ts["last_updated"] = _dt.now().isoformat()
                ts_path.write_text(json.dumps(ts, indent=2), encoding="utf-8")
                # Write agent_status
                agents_detail = {}
                for a in (red_actions or []) + (blue_actions or []):
                    if a:
                        atype = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
                        agents_detail[a.agent_id] = {
                            "action": atype, "node": a.target_node,
                            "team": "red" if str(a.agent_id).startswith("red") else "blue",
                        }
                (ROOT / "logs" / "agent_status.json").write_text(
                    json.dumps({
                        "run_id": run_id, "episode": 1, "step": step,
                        "suspicion": susp, "detection": det, "zone": zone_label,
                        "agents": agents_detail, "timestamp": _dt.now().isoformat(),
                    }, indent=2),
                    encoding="utf-8",
                )
                print(f"[STEP] {step}/{max_steps} | ep=1 | sus={susp:.2f} | det={det:.2f}", flush=True)
                # Push to HF Hub in background (non-blocking)
                if step % 2 == 0:
                    try:
                        from cipher.utils.hf_uploader import push_live_data
                        threading.Thread(target=push_live_data, daemon=True).start()
                    except Exception:
                        pass
            except Exception as _cb_err:
                print(f"[WARN] step_cb error: {_cb_err}", flush=True)

        run_episode(
            episode_number=1,
            max_steps=int(os.getenv("HF_EPISODE_MAX_STEPS", "20")),
            verbose=True,
            save_trace=True,
            step_callback=_step_cb,
        )

        ts["status"] = "complete"
        ts["last_updated"] = datetime.now().isoformat()
        ts_path.write_text(json.dumps(ts, indent=2), encoding="utf-8")
        print("[END] Background episode complete", flush=True)

        # Push final data to HF Hub
        try:
            from cipher.utils.hf_uploader import push_live_data
            push_live_data(root_dir=ROOT)
        except Exception:
            pass

    except Exception as exc:
        print(f"[ERROR] Background episode failed: {exc}", flush=True)
        try:
            state = json.loads(ts_path.read_text(encoding="utf-8")) if ts_path.exists() else {}
            state["status"] = "error"
            state["error"] = str(exc)
            ts_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass
    finally:
        with _training_lock:
            _training_running = False


# ── Serve React app from dist/ ───────────────────────────────────────
def _serve_react(subpath='index.html'):
    from flask import send_file as _sf
    target = DIST_DIR / subpath
    if not target.exists() or not target.is_file():
        target = DIST_DIR / 'index.html'
    if target.exists():
        return _sf(str(target))
    return Response(
        '<html><body style="font-family:monospace;background:#0b1120;color:#e8f0fe;padding:40px">'
        '<h2 style="color:#ff5252">CIPHER War Room — build needed</h2>'
        '<p>Run: <code>cd dashboard-react &amp;&amp; npm run build</code></p>'
        '<p>Or dev: <code>npm run dev</code> → '
        '<a href="http://localhost:5173" style="color:#00e5ff">localhost:5173</a></p>'
        '</body></html>',
        mimetype='text/html', status=200
    )

@app.route('/')
def index():
    return _serve_react('index.html')

@app.route('/assets/<path:filename>')
def assets(filename):
    return _serve_react(f'assets/{filename}')

@app.route('/vite.svg')
def vite_svg():
    return _serve_react('vite.svg')


# ── Helpers ─────────────────────────────────────────────────────────

def _read_rewards() -> list[dict]:
    """Read rewards_log.csv from local disk; fall back to HF Dataset live/ folder."""
    import io as _io
    path = ROOT / "rewards_log.csv"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    return rows
        except Exception:
            pass
    raw = _fetch_hf_live_file("rewards_log.csv")
    if raw:
        try:
            return list(csv.DictReader(_io.StringIO(raw.decode("utf-8", errors="ignore"))))
        except Exception:
            pass
    return []


def _read_jsonl(path: Path, last_n: int = 50) -> list:
    if not path.exists():
        return []
    try:
        raw  = path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        lines = [l for l in raw.split("\n") if l.strip()]
        parsed = []
        for ln in lines[-last_n:]:
            try:
                parsed.append(json.loads(ln))
            except Exception:
                pass
        return parsed
    except Exception:
        return []


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── Routes ──────────────────────────────────────────────────────────

@app.route("/api/live-steps")
def live_steps():
    data = _read_jsonl(ROOT / "live_steps.jsonl", last_n=50)
    if not data:
        raw = _fetch_hf_live_file("live_steps.jsonl")
        if raw:
            lines = raw.decode("utf-8", errors="ignore").strip().split("\n")
            for ln in lines[-50:]:
                try:
                    data.append(json.loads(ln))
                except Exception:
                    pass
    return jsonify(data)


@app.route("/api/thoughts")
def thoughts():
    data = _read_jsonl(ROOT / "logs" / "agent_thoughts.jsonl", last_n=24)
    return jsonify(data)


@app.route("/api/agent-status")
def agent_status():
    data = _read_json(ROOT / "logs" / "agent_status.json")
    if not data:
        raw = _fetch_hf_live_file("agent_status.json")
        if raw:
            try:
                data = json.loads(raw.decode("utf-8", errors="ignore"))
            except Exception:
                pass
    return jsonify(data)


@app.route("/api/architecture-doc")
def architecture_doc():
    """Return project-root `changed.md` for the Architecture panel (JSON: {markdown: ...})."""
    path = ROOT / "changed.md"
    if not path.is_file():
        return jsonify(
            {
                "markdown": (
                    "*`changed.md` is not present at the project root next to the CIPHER environment.*\n\n"
                    "Add `changed.md` to document stack: React (Vite) → proxy → "
                    "Flask (`api_server.py`) → CIPHER Python env + `logs/`."
                ),
            }
        )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - IO edge cases
        return jsonify(
            {
                "markdown": f"*Could not read `changed.md`:* {exc}",
            }
        )
    return jsonify({"markdown": text})


@app.route("/api/network-graph")
def network_graph():
    # 1. Try pre-serialised live graph
    live_path = ROOT / "logs" / "network_graph.json"
    if live_path.exists():
        data = _read_json(live_path)
        if data.get("nodes"):
            return jsonify(data)

    # 2. Fall back to latest episode trace
    traces = sorted(glob.glob(str(ROOT / "episode_traces" / "*.json")))
    if traces:
        try:
            trace = json.loads(Path(traces[-1]).read_text(encoding="utf-8"))
            g     = trace.get("graph", {})
            # networkx node_link_data format → {nodes, edges}
            if "nodes" in g and "links" in g:
                edges = [{"source": l["source"], "target": l["target"]} for l in g.get("links", [])]
                nodes = []
                for n in g.get("nodes", []):
                    zone_raw  = n.get("zone", 0)
                    zone_val  = zone_raw.get("_value_", zone_raw) if isinstance(zone_raw, dict) else int(zone_raw or 0)
                    type_raw  = n.get("type", n.get("node_type", "server"))
                    type_val  = type_raw.get("_value_", str(type_raw)) if isinstance(type_raw, dict) else str(type_raw)
                    nodes.append({
                        "id":         n.get("id", n.get("node_id")),
                        "hostname":   n.get("hostname", f"node_{n.get('id',0)}"),
                        "zone":       zone_val,
                        "type":       type_val,
                        "files":      n.get("files", []),
                        "services":   n.get("services", []),
                        "is_honeypot":n.get("is_honeypot", False),
                        "is_entry":   n.get("is_entry", False),
                        "is_hvt":     n.get("is_hvt", False),
                    })
                return jsonify({"nodes": nodes, "edges": edges})
        except Exception:
            pass

    return jsonify({"nodes": [], "edges": []})


def _discover_episode_traces() -> list[Path]:
    """Use only project-root episode_traces/*.json files."""
    traces_dir = ROOT / "episode_traces"
    if not traces_dir.exists():
        return []
    traces = [p for p in traces_dir.glob("*.json") if p.is_file()]
    traces.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return traces


@app.route("/api/episodes")
def list_episodes():
    traces = _discover_episode_traces()
    episodes = []
    for p in traces:
        winner = "UNKNOWN"
        terminal_reason = ""
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            terminal_reason = str(data.get("terminal_reason", "") or "")
            term = terminal_reason.lower()
            if term in ("exfil_success", "exfiltration_complete", "exfil_complete"):
                winner = "RED"
            elif term == "aborted":
                winner = "DRAW"
            elif term:
                winner = "BLUE"
        except Exception:
            pass
        episodes.append({
            "name": p.name,
            "winner": winner,
            "terminal_reason": terminal_reason or "unknown",
        })
    return jsonify(episodes)


def _winner_from_terminal(terminal_reason: str) -> str:
    tr = str(terminal_reason or "").lower().replace("-", "_")
    if tr in ("exfil_success", "exfiltration_complete", "exfil_complete"):
        return "RED"
    if tr == "aborted":
        return "DRAW"
    if tr and tr != "unknown":
        return "BLUE"
    return "UNKNOWN"


def _match_rewards_row_for_trace(ep_num: int, steps: int, terminal: str) -> dict | None:
    """Find a `rewards_log.csv` row best matching a saved episode trace."""
    path = ROOT / "rewards_log.csv"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    if not rows:
        return None

    def _fi(v) -> int:
        try:
            return int(float(v))
        except Exception:
            return 0

    def _norm_t(t) -> str:
        return str(t or "").lower().replace("-", "_").strip()

    t_trace = _norm_t(terminal)
    exfils = {"exfil_success", "exfiltration_complete", "exfil_complete"}

    def row_score(r: dict) -> tuple:
        r_term = _norm_t(r.get("terminal_reason"))
        pr = 0
        if t_trace and r_term == t_trace:
            pr = 2
        elif t_trace in exfils and r_term in exfils:
            pr = 1
        # file order: later rows win on tie
        idx = rows.index(r) if r in rows else 0
        return (pr, idx)

    by_ep_step: list[dict] = [r for r in rows if _fi(r.get("episode")) == ep_num and _fi(r.get("steps")) == steps]
    pool = by_ep_step if by_ep_step else [r for r in rows if _fi(r.get("episode")) == ep_num]
    if not pool:
        return None
    return max(pool, key=row_score)


@app.route("/api/episode-summary")
def episode_summary():
    """
    One episode: trace metadata from `episode_traces/<name>.json` plus
    `red_total` / `blue_total` from `rewards_log.csv` when a row matches
    (episode, steps) and, when possible, terminal reason.
    """
    name = (request.args.get("name") or "").strip()
    traces_dir = ROOT / "episode_traces"
    if not name or ".." in name or "/" in name or "\\" in name:
        return (
            jsonify(
                {
                    "error": "invalid_name",
                    "message": "name must be a basename only (no path components).",
                    "received": name,
                    "traces_dir": str(traces_dir),
                }
            ),
            400,
        )
    # Align with `/api/episodes` (filenames with `.json`); allow stem-only for clients that omit the suffix
    if not name.lower().endswith(".json"):
        name = f"{name}.json"
    path = traces_dir / name
    if not path.is_file():
        return (
            jsonify(
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
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": f"read failed: {exc}"}), 500

    ep_num = int(data.get("episode_number") or data.get("episode") or 0)
    steps = int(data.get("steps") or 0)
    terminal = str(data.get("terminal_reason") or "")
    winner = _winner_from_terminal(terminal)

    row = _match_rewards_row_for_trace(ep_num, steps, terminal) if ep_num else None
    def _rfloat(k: str) -> float | None:
        if not row:
            return None
        v = row.get(k)
        try:
            return float(v)
        except Exception:
            return None

    return jsonify(
        {
            "name": name,
            "episode": ep_num,
            "steps": steps,
            "terminal_reason": terminal or "unknown",
            "winner": winner,
            "red_total": _rfloat("red_total"),
            "blue_total": _rfloat("blue_total"),
            "rewards_matched": row is not None,
        }
    )


_CANONICAL_GRAPH_CACHE = None

def _get_canonical_graph():
    global _CANONICAL_GRAPH_CACHE
    if _CANONICAL_GRAPH_CACHE is not None:
        return _CANONICAL_GRAPH_CACHE
    try:
        sys.path.insert(0, str(ROOT))
        from cipher.environment.graph import generate_enterprise_graph
        from cipher.utils.config import config
        g = generate_enterprise_graph(
            n_nodes=config.env_graph_size,
            honeypot_density=config.env_honeypot_density,
            seed=7961,
        )
        nodes = []
        for nid in g.nodes():
            n = g.nodes[nid]
            zone_raw = n.get('zone', 0)
            zone_val = getattr(zone_raw, 'value', zone_raw)
            try:
                zone_val = int(zone_val)
            except Exception:
                zone_val = 0
            type_raw = n.get('type', n.get('node_type', 'server'))
            type_val = str(getattr(type_raw, 'value', type_raw))
            nodes.append({
                'id':          int(nid),
                'hostname':    n.get('hostname', f'node_{nid}'),
                'zone':        zone_val,
                'type':        type_val,
                'is_honeypot': bool(n.get('is_honeypot', False)),
                'is_entry':    bool(n.get('is_entry', False)),
                'is_hvt':      bool(n.get('is_hvt', False)),
                'files':       list(n.get('files', [])),
                'services':    list(n.get('services', [])),
            })
        edges = [{'source': int(s), 'target': int(t)} for s, t in g.edges()]
        _CANONICAL_GRAPH_CACHE = {'nodes': nodes, 'edges': edges}
    except Exception:
        _CANONICAL_GRAPH_CACHE = {'nodes': [], 'edges': []}
    return _CANONICAL_GRAPH_CACHE


@app.route("/api/episode/<filename>")
def get_episode(filename):
    candidates = [p for p in _discover_episode_traces() if p.name == filename]
    if not candidates:
        candidates = [ROOT / "episode_traces" / filename]

    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Inject canonical graph when episode file has no graph data
            if not data.get('graph', {}).get('nodes'):
                data['graph'] = _get_canonical_graph()
            return jsonify(data)
        except Exception:
            continue

    return jsonify({}), 404


@app.route("/api/rewards")
def rewards():
    return jsonify(_read_rewards())


@app.route("/api/commanders")
def commanders():
    """
    Return the commander/subagent roster from the most recent episode trace.
    Returns:
      {
        "arch": "v2" | "v1",
        "red": { "agent_id", "final_roster": [...], "lifecycle": [...], "spawn_budget_remaining", "total_spawns" },
        "blue": { ... }
      }
    Falls back to {"arch":"v1"} when no v2 trace exists yet.
    """
    traces = _discover_episode_traces()
    for path in traces:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = data.get("commanders") or {}
        if meta:
            return jsonify({
                "arch": meta.get("arch", "v2"),
                "red": meta.get("red_commander", {}),
                "blue": meta.get("blue_commander", {}),
                "source_trace": path.name,
            })
    return jsonify({"arch": "v1", "red": {}, "blue": {}})


@app.route("/api/health")
def health():
    traces_repo = os.getenv("HF_TRACES_REPO", "wolfie8935/cipher-traces")
    space_url = os.getenv(
        "HF_SPACE_URL",
        "https://huggingface.co/spaces/wolfie8935/cipher-openenv",
    )
    push = (os.getenv("CIPHER_PUSH_TRACES_HF") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )
    ts = _read_json(ROOT / "training_state.json")
    return jsonify(
        {
            "status": "ok",
            "root": str(ROOT),
            "hf_space_url": space_url,
            "hf_traces_dataset_url": f"https://huggingface.co/datasets/{traces_repo}",
            "hf_traces_repo": traces_repo,
            "cipher_push_traces_hf": push,
            "training_status": ts.get("status", "idle"),
            "current_episode": ts.get("current_episode"),
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.route("/api/control", methods=["POST"])
def control():
    """Start / Step / Reset control endpoint for the HF Space UI and local War Room."""
    global _training_running
    data = request.get_json(silent=True) or {}
    action = str(data.get("action", "")).strip().lower()
    ts_path = ROOT / "training_state.json"

    if action == "start":
        with _training_lock:
            already = _training_running
            if not already:
                _training_running = True
                threading.Thread(target=_run_bg_episode, daemon=True).start()
        if already:
            return jsonify({"ok": False, "action": "start", "message": "Episode already running."})
        print("[START] Control: background episode launched", flush=True)
        return jsonify({"ok": True, "action": "start", "message": "Episode started — watch the dashboard update live."})

    elif action == "reset":
        with _training_lock:
            _training_running = False
        state = _read_json(ts_path) or {}
        state.update({
            "status": "idle",
            "control_action": "reset",
            "control_timestamp": datetime.now().isoformat(),
            "current_episode": 0,
            "total_steps": 0,
        })
        ts_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        for p in [ROOT / "live_steps.jsonl", ROOT / "logs" / "agent_thoughts.jsonl"]:
            if p.exists():
                p.write_text("", encoding="utf-8")
        print("[RESET] Control: environment reset", flush=True)
        return jsonify({"ok": True, "action": "reset", "message": "Environment reset. Click ▶ START to run an episode."})

    elif action == "step":
        state = _read_json(ts_path) or {}
        state.update({
            "control_action": "step",
            "control_timestamp": datetime.now().isoformat(),
        })
        ts_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print("[STEP] Control: step action received", flush=True)
        return jsonify({"ok": True, "action": "step", "message": "Step requested."})

    return jsonify({"ok": False, "error": f"Unknown action: {action!r}. Use start|step|reset."}), 400


@app.route("/api/training-state")
def training_state():
    return jsonify(_read_json(ROOT / "training_state.json"))


@app.route("/api/training-events")
def training_events():
    return jsonify(_read_jsonl(ROOT / "training_events.jsonl", last_n=200))


@app.route("/api/dead-drops")
def dead_drops():
    events = _read_jsonl(ROOT / "training_events.jsonl", last_n=500)
    drops = [e for e in events if e.get("event_type") in (
        "dead_drop_written", "dead_drop_read", "dead_drop_tampered",
        "write_dead_drop", "read_dead_drop", "tamper_dead_drop",
    )]
    # Also pull drop events from live_steps
    for step in _read_jsonl(ROOT / "live_steps.jsonl", last_n=200):
        for d in step.get("drop_events", []):
            d["step"] = step.get("step")
            d["episode"] = step.get("episode")
            d["timestamp"] = step.get("timestamp")
            drops.append(d)
    drops.sort(key=lambda x: (x.get("episode", 0), x.get("step", 0)))
    return jsonify(drops[-100:])


@app.route("/api/battle-log")
def battle_log():
    """
    Rich merged event stream combining:
    - agent_thoughts.jsonl  (full reasoning per agent per step)
    - live_steps.jsonl      (step summaries + trap/drop events + key events)
    - training_events.jsonl (trap fired, drop written events)
    Returns up to 150 events newest-first.
    """
    thoughts = _read_jsonl(ROOT / "logs" / "agent_thoughts.jsonl", last_n=200)
    steps    = _read_jsonl(ROOT / "live_steps.jsonl", last_n=100)
    tevents  = _read_jsonl(ROOT / "training_events.jsonl", last_n=200)

    # Build a step -> summary lookup from live_steps
    step_map = {}
    for s in steps:
        key = (s.get("episode", 0), s.get("step", 0))
        step_map[key] = s

    events = []
    seen = set()

    # ── From agent_thoughts: one event per agent action ──────────
    for t in thoughts:
        eid = f"thought-{t.get('agent_id')}-{t.get('step')}"
        if eid in seen:
            continue
        seen.add(eid)
        step_num = t.get("step", 0)
        ep_num   = t.get("episode", 0)
        skey     = (ep_num, step_num)
        summary  = step_map.get(skey, {})
        atype    = t.get("action_type", "")
        team     = t.get("team", "")
        agent_id = t.get("agent_id", "")
        role     = agent_id.replace("red_", "").replace("blue_", "").replace("_01", "").replace("_", " ")
        is_critical = (
            "exfil" in atype.lower()
            or atype in ("trigger_alert", "abort", "place_honeypot")
        )
        target_node = t.get("target_node")
        target_file = t.get("target_file")
        detail_parts = [atype.replace("_", " ")]
        if target_node is not None:
            detail_parts.append(f"→ n{target_node}")
        if target_file:
            detail_parts.append(f"[{target_file}]")
        events.append({
            "id":          eid,
            "type":        "action",
            "step":        step_num,
            "episode":     ep_num,
            "timestamp":   t.get("timestamp", ""),
            "team":        team,
            "agent_id":    agent_id,
            "role":        role,
            "action_type": atype,
            "target_node": target_node,
            "target_file": target_file,
            "reasoning":   t.get("reasoning", ""),
            "detail":      " ".join(detail_parts),
            "critical":    is_critical,
            "suspicion":   summary.get("suspicion"),
            "detection":   summary.get("detection"),
            "zone":        summary.get("zone"),
        })

    # ── From live_steps: key events, trap events, drop events ────
    for s in steps:
        step_num = s.get("step", 0)
        ep_num   = s.get("episode", 0)
        ts       = s.get("timestamp", "")

        # Key event (zone advance, high suspicion, exfil)
        if s.get("key_event"):
            eid = f"key-{ep_num}-{step_num}"
            if eid not in seen:
                seen.add(eid)
                events.append({
                    "id": eid, "type": "system", "step": step_num,
                    "episode": ep_num, "timestamp": ts, "team": "system",
                    "agent_id": "system", "role": "system",
                    "action_type": "key_event",
                    "detail": s["key_event"],
                    "critical": True,
                    "suspicion": s.get("suspicion"),
                    "detection": s.get("detection"),
                    "zone": s.get("zone"),
                    "reasoning": "",
                })

        # Trap events embedded in step
        for i, te in enumerate(s.get("trap_events", [])):
            eid = f"trap-step-{ep_num}-{step_num}-{i}"
            if eid not in seen:
                seen.add(eid)
                events.append({
                    "id": eid, "type": "trap", "step": step_num,
                    "episode": ep_num, "timestamp": ts,
                    "team": te.get("team", ""),
                    "agent_id": te.get("agent_id", ""),
                    "role": "trap",
                    "action_type": te.get("action_type", ""),
                    "target_node": te.get("node"),
                    "detail": f"{te.get('action_type','trap').replace('_',' ')} @ n{te.get('node','?')}",
                    "critical": False,
                    "suspicion": s.get("suspicion"),
                    "detection": s.get("detection"),
                    "zone": s.get("zone"),
                    "reasoning": "",
                })

        # Dead drop events embedded in step
        for i, de in enumerate(s.get("drop_events", [])):
            eid = f"drop-step-{ep_num}-{step_num}-{i}"
            if eid not in seen:
                seen.add(eid)
                events.append({
                    "id": eid, "type": "dead_drop", "step": step_num,
                    "episode": ep_num, "timestamp": ts,
                    "team": "red",
                    "agent_id": de.get("agent_id", ""),
                    "role": "drop",
                    "action_type": de.get("action_type", ""),
                    "target_node": de.get("node"),
                    "detail": f"{de.get('action_type','drop').replace('_',' ')} @ n{de.get('node','?')}: {de.get('content_preview','')}",
                    "critical": "tamper" in str(de.get("action_type", "")),
                    "suspicion": s.get("suspicion"),
                    "detection": s.get("detection"),
                    "zone": s.get("zone"),
                    "reasoning": de.get("content_preview", ""),
                })

        # Exfil event
        if s.get("exfil_count", 0) > 0 and s.get("exfil_files"):
            eid = f"exfil-{ep_num}-{step_num}"
            if eid not in seen:
                seen.add(eid)
                files_str = ", ".join(str(f) for f in s["exfil_files"])
                events.append({
                    "id": eid, "type": "exfil", "step": step_num,
                    "episode": ep_num, "timestamp": ts, "team": "red",
                    "agent_id": "red_exfiltrator_01", "role": "exfiltrator",
                    "action_type": "exfiltrate",
                    "detail": f"💀 EXFILTRATED: {files_str}",
                    "critical": True,
                    "suspicion": s.get("suspicion"),
                    "detection": s.get("detection"),
                    "zone": s.get("zone"),
                    "reasoning": "",
                })

    # ── From training_events.jsonl ────────────────────────────────
    for i, te in enumerate(tevents):
        etype = te.get("event_type", "")
        step_num = te.get("step", 0)
        ep_num   = te.get("episode", 0)
        eid = f"tevent-{ep_num}-{step_num}-{i}"
        if eid in seen:
            continue
        seen.add(eid)
        is_trap = "trap" in etype
        is_drop = "drop" in etype
        if not (is_trap or is_drop or etype in ("dead_drop_written", "dead_drop_tampered")):
            continue
        events.append({
            "id": eid,
            "type": "trap" if is_trap else "dead_drop",
            "step": step_num,
            "episode": ep_num,
            "timestamp": te.get("timestamp", ""),
            "team": te.get("team", ""),
            "agent_id": "",
            "role": te.get("trap_type", "event"),
            "action_type": etype,
            "target_node": te.get("node"),
            "detail": te.get("detail", etype.replace("_", " ")),
            "critical": "tamper" in etype or "fired" in etype,
            "suspicion": None,
            "detection": None,
            "zone": None,
            "reasoning": "",
        })

    # Sort newest-first and return last 150
    events.sort(key=lambda e: (e.get("episode", 0), e.get("step", 0), e.get("timestamp", "")))
    return jsonify(events[-150:])


def _live_logs_sort_key(entry: dict) -> tuple:
    """Stable chronological ordering for merged judge log lines."""
    type_order = {"step": 0, "thought": 1, "training": 2}
    return (
        int(entry.get("episode") or 0),
        int(entry.get("step") or 0),
        str(entry.get("timestamp") or ""),
        type_order.get(str(entry.get("type") or ""), 9),
        str(entry.get("agent_id") or ""),
        str(entry.get("id") or ""),
    )


@app.route("/api/live-logs")
def live_logs():
    """
    Judge/demo-friendly merged tail: live_steps + agent_thoughts (+ trap/drop training_events).
    Sorted ascending by (episode, step, timestamp); returns the last *limit* lines.
    Query: limit=50..500 (default 200).
    """
    raw_limit = request.args.get("limit", default=200, type=int)
    limit = 200 if raw_limit is None else max(50, min(500, int(raw_limit)))

    steps = _read_jsonl(ROOT / "live_steps.jsonl", last_n=120)
    thoughts = _read_jsonl(ROOT / "logs" / "agent_thoughts.jsonl", last_n=500)
    tevents = _read_jsonl(ROOT / "training_events.jsonl", last_n=300)

    entries: list[dict] = []

    for s in steps:
        ep = int(s.get("episode") or 0)
        st = int(s.get("step") or 0)
        ts = str(s.get("timestamp") or "")
        parts = []
        rid = s.get("run_id")
        if rid:
            parts.append(f"run={rid}")
        ra = s.get("red_action")
        if ra:
            parts.append(f"RED {ra}")
        ba = s.get("blue_actions")
        if ba:
            parts.append(f"BLUE {ba}")
        zn = s.get("zone")
        if zn:
            parts.append(f"zone={zn}")
        if s.get("suspicion") is not None:
            parts.append(f"sus={s.get('suspicion')}")
        if s.get("detection") is not None:
            parts.append(f"det={s.get('detection')}")
        if int(s.get("exfil_count") or 0) > 0:
            ex = s.get("exfil_files") or []
            parts.append(f"EXFIL×{s.get('exfil_count')} {','.join(str(x) for x in ex[:3])}")
        if s.get("key_event"):
            parts.append(f"KEY:{s.get('key_event')}")
        msg = " │ ".join(str(p) for p in parts if p) or "(empty step)"
        raw_step = json.dumps(s, ensure_ascii=False)
        entries.append({
            "id": f"step-{ep}-{st}",
            "type": "step",
            "side": "SYSTEM",
            "severity": "info",
            "episode": ep,
            "step": st,
            "timestamp": ts,
            "message": msg,
            "snippet": (raw_step[:420] + "…") if len(raw_step) > 420 else raw_step,
        })

        for i, te in enumerate(s.get("trap_events") or []):
            raw_te = json.dumps(te, ensure_ascii=False)
            entries.append({
                "id": f"trap-{ep}-{st}-{i}",
                "type": "training",
                "side": "SYSTEM",
                "severity": "warn",
                "episode": ep,
                "step": st,
                "timestamp": ts,
                "message": f"TRAP {te.get('action_type','')} team={te.get('team','')} @n{te.get('node','?')}",
                "snippet": (raw_te[:320] + "…") if len(raw_te) > 320 else raw_te,
            })
        for i, de in enumerate(s.get("drop_events") or []):
            raw_de = json.dumps(de, ensure_ascii=False)
            entries.append({
                "id": f"drop-{ep}-{st}-{i}",
                "type": "training",
                "side": "RED",
                "severity": "info",
                "episode": ep,
                "step": st,
                "timestamp": ts,
                "message": f"DROP {de.get('action_type','')} @n{de.get('node','?')}",
                "snippet": (raw_de[:320] + "…") if len(raw_de) > 320 else raw_de,
            })

    for t in thoughts:
        team = str(t.get("team") or "").lower()
        if team == "red":
            side = "RED"
        elif team == "blue":
            side = "BLUE"
        else:
            side = "SYSTEM"
        ep = int(t.get("episode") or 0)
        st = int(t.get("step") or 0)
        ts = str(t.get("timestamp") or "")
        aid = str(t.get("agent_id") or "")
        atype = str(t.get("action_type") or "")
        tn = t.get("target_node")
        tf = t.get("target_file")
        tail = []
        if tn is not None:
            tail.append(f"→n{tn}")
        if tf:
            tail.append(str(tf))
        tail_s = " ".join(tail)
        reason = str(t.get("reasoning") or "")
        reason_short = reason if len(reason) <= 240 else reason[:237] + "…"
        msg = f"{aid} {atype}{(' ' + tail_s) if tail_s else ''} — {reason_short}"
        sev = "critical" if ("exfil" in atype.lower() or atype in ("trigger_alert", "abort")) else "info"
        raw_th = json.dumps(t, ensure_ascii=False)
        entries.append({
            "id": f"thought-{aid}-{st}-{ts}",
            "type": "thought",
            "side": side,
            "severity": sev,
            "episode": ep,
            "step": st,
            "timestamp": ts,
            "agent_id": aid,
            "message": msg,
            "snippet": (raw_th[:400] + "…") if len(raw_th) > 400 else raw_th,
        })

    for i, te in enumerate(tevents):
        etype = str(te.get("event_type") or "")
        if not any(k in etype for k in ("trap", "drop", "dead_drop")):
            continue
        ep = int(te.get("episode") or 0)
        st = int(te.get("step") or 0)
        ts = str(te.get("timestamp") or "")
        tm = str(te.get("team") or "").lower()
        tside = "RED" if tm == "red" else "BLUE" if tm == "blue" else "SYSTEM"
        raw_ev = json.dumps(te, ensure_ascii=False)
        entries.append({
            "id": f"tevent-{ep}-{st}-{i}-{etype}",
            "type": "training",
            "side": tside,
            "severity": "warn" if "fired" in etype or "tamper" in etype else "info",
            "episode": ep,
            "step": st,
            "timestamp": ts,
            "message": str(te.get("detail") or etype.replace("_", " ")),
            "snippet": (raw_ev[:360] + "…") if len(raw_ev) > 360 else raw_ev,
        })

    entries.sort(key=_live_logs_sort_key)
    tail = entries[-limit:]
    return jsonify({"lines": tail, "count": len(tail), "limit": limit})


@app.route("/api/rl-stats")
def rl_stats():
    """
    Compute training metrics from rewards_log.csv:
    - per-episode reward series (last 50)
    - win_rate (rolling 10-ep)
    - RED/BLUE component averages
    - difficulty curve (if training_state.json has it)
    """
    rows = _read_rewards()
    if not rows:
        return jsonify({
            "episodes": [], "red_totals": [], "blue_totals": [],
            "win_rate_red": 0, "win_rate_blue": 0, "total_episodes": 0,
            "red_avg": 0, "blue_avg": 0,
            "best_red": 0, "best_blue": 0,
            "terminal_counts": {},
            "component_avgs": {},
            "episode_table": [],
            "evo_events": [],
            "difficulty_vs_rewards": [],
        })

    # Take last 50 unique episodes by row order
    recent = rows[-50:]

    def _f(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    ep_nums    = [int(r.get("episode", 0)) for r in recent]
    red_totals = [_f(r.get("red_total", 0)) for r in recent]
    blue_totals= [_f(r.get("blue_total", 0)) for r in recent]
    terminals  = [r.get("terminal_reason", "max_steps") for r in recent]

    terminal_counts = {}
    for t in terminals:
        terminal_counts[t] = terminal_counts.get(t, 0) + 1

    red_wins  = sum(1 for t in terminals if t in ("exfiltration_complete", "exfil_success", "exfil_complete"))
    blue_wins = sum(1 for t in terminals if t in ("detected",))
    total     = max(1, len(recent))

    # Rolling 10-ep win rate
    last10_terminals = terminals[-10:]
    red_win_rate_10  = sum(1 for t in last10_terminals if t in ("exfiltration_complete", "exfil_success", "exfil_complete")) / max(1, len(last10_terminals))
    blue_win_rate_10 = sum(1 for t in last10_terminals if t == "detected") / max(1, len(last10_terminals))

    # Component averages
    comp_cols = [
        "red_exfil", "red_stealth", "red_complexity", "red_memory",
        "blue_detection", "blue_speed", "blue_fp_penalty", "blue_honeypot_rate",
        "blue_graph_reconstruction",
    ]
    component_avgs = {}
    for col in comp_cols:
        vals = [_f(r.get(col, 0)) for r in recent if r.get(col) is not None]
        component_avgs[col] = round(sum(vals) / max(1, len(vals)), 4) if vals else 0.0

    # Recent episodes table (last 20) for display
    episode_table = []
    for r in recent[-20:]:
        episode_table.append({
            "episode":      r.get("episode"),
            "steps":        r.get("steps"),
            "terminal":     r.get("terminal_reason"),
            "red_total":    _f(r.get("red_total")),
            "blue_total":   _f(r.get("blue_total")),
            "verdict":      r.get("fleet_verdict", "?"),
            "flags":        r.get("oversight_flags", "none"),
            "red_exfil":    _f(r.get("red_exfil")),
            "red_stealth":  _f(r.get("red_stealth")),
            "red_complexity": _f(r.get("red_complexity")),
            "blue_detection": _f(r.get("blue_detection")),
            "blue_fp_penalty": _f(r.get("blue_fp_penalty")),
            "judgment":     r.get("fleet_judgment", "")[:80],
        })

    # Prompt evolution events
    evo_events = _read_jsonl(ROOT / "prompt_evolution_log.jsonl", last_n=20)

    # Last N log rows: RED/BLUE totals vs difficulty (complexity mult or step pressure)
    chart_slice = rows[-100:]
    step_vals = [int(_f(r.get("steps", 0))) for r in chart_slice]
    max_steps_denom = max(30, max(step_vals, default=30))
    difficulty_vs_rewards = []
    base = len(rows) - len(chart_slice)
    for i, r in enumerate(chart_slice):
        st = int(_f(r.get("steps", 0)))
        mult = _f(r.get("red_complexity_multiplier", 0))
        if mult <= 0:
            mult = _f(r.get("red_complexity", 0))
        diff_val = float(mult) if mult > 0 else min(2.0, st / float(max_steps_denom))
        difficulty_vs_rewards.append({
            "run": base + i + 1,
            "episode": int(_f(r.get("episode", 0))),
            "steps": st,
            "red": round(_f(r.get("red_total")), 4),
            "blue": round(_f(r.get("blue_total")), 4),
            "difficulty": round(diff_val, 4),
            "steps_norm": round(st / float(max_steps_denom), 4),
        })

    # Full convergence data — all episodes (downsampled for large runs)
    all_rows = rows  # full dataset
    if len(all_rows) > 500:
        # Downsample evenly to keep max 500 points
        step = len(all_rows) // 500
        all_rows = all_rows[::step]
    all_red = [_f(r.get("red_total", 0)) for r in all_rows]
    all_blue = [_f(r.get("blue_total", 0)) for r in all_rows]
    all_eps = [int(_f(r.get("episode", 0))) for r in all_rows]

    return jsonify({
        "episodes":          ep_nums,
        "red_totals":        red_totals,
        "blue_totals":       blue_totals,
        "all_red_totals":    all_red,
        "all_blue_totals":   all_blue,
        "all_episodes":      all_eps,
        "win_rate_red":      round(red_wins / total, 3),
        "win_rate_blue":     round(blue_wins / total, 3),
        "win_rate_red_10":   round(red_win_rate_10, 3),
        "win_rate_blue_10":  round(blue_win_rate_10, 3),
        "total_episodes":    len(rows),
        "red_avg":           round(sum(red_totals) / total, 4),
        "blue_avg":          round(sum(blue_totals) / total, 4),
        "best_red":          round(max(red_totals, default=0), 4),
        "best_blue":         round(max(blue_totals, default=0), 4),
        "terminal_counts":   terminal_counts,
        "component_avgs":    component_avgs,
        "episode_table":     episode_table,
        "evo_events":        evo_events,
        "difficulty_vs_rewards": difficulty_vs_rewards,
    })


@app.route("/api/history")
def history():
    """Cross-run episode history for React parity with Dash History tab."""
    rows = _read_rewards()

    def _f(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    normalized = []
    for r in rows:
        normalized.append({
            "episode": int(_f(r.get("episode"), 0)),
            "run_id": str(r.get("run_id", "") or ""),
            "steps": int(_f(r.get("steps"), 0)),
            "terminal_reason": str(r.get("terminal_reason", "unknown") or "unknown"),
            "fleet_verdict": str(r.get("fleet_verdict", "") or ""),
            "red_total": _f(r.get("red_total")),
            "blue_total": _f(r.get("blue_total")),
            "red_exfil": _f(r.get("red_exfil")),
            "red_stealth": _f(r.get("red_stealth")),
            "red_complexity": _f(r.get("red_complexity")),
            "red_memory": _f(r.get("red_memory")),
            "blue_detection": _f(r.get("blue_detection")),
            "blue_speed": _f(r.get("blue_speed")),
            "blue_graph_reconstruction": _f(r.get("blue_graph_reconstruction")),
            "blue_fp_penalty": _f(r.get("blue_fp_penalty")),
            "blue_honeypot_rate": _f(r.get("blue_honeypot_rate")),
        })
    return jsonify(normalized)


@app.route("/api/analytics")
def analytics():
    """
    Unified analytics payload for React:
    - Elo trend (derived from terminal outcomes)
    - Reward curves
    - Terminal reason distribution
    - Trap hot nodes (from training_events)
    """
    reward_rows = _read_rewards()

    def _f(v, default=0.0):
        try:
            return float(v)
        except Exception:
            return default

    episodes = []
    red_rewards = []
    blue_rewards = []
    terminal_counts = {}
    red_elo = []
    blue_elo = []
    r_rating = 1500.0
    b_rating = 1500.0
    k = 18.0

    for idx, row in enumerate(reward_rows, start=1):
        ep = int(_f(row.get("episode"), idx))
        term = str(row.get("terminal_reason", "unknown") or "unknown").lower()
        r_win = 1.0 if term in ("exfil_success", "exfiltration_complete", "exfil_complete") else 0.0
        b_win = 1.0 if term in ("detected", "timeout", "failure", "max_steps") else 0.0
        if term == "aborted":
            r_win = 0.5
            b_win = 0.5
        # fallback draw-ish when unknown
        if r_win == 0.0 and b_win == 0.0:
            r_win = 0.5
            b_win = 0.5

        exp_r = 1.0 / (1.0 + (10 ** ((b_rating - r_rating) / 400.0)))
        exp_b = 1.0 - exp_r
        r_rating += k * (r_win - exp_r)
        b_rating += k * (b_win - exp_b)

        episodes.append(ep)
        red_rewards.append(_f(row.get("red_total")))
        blue_rewards.append(_f(row.get("blue_total")))
        red_elo.append(round(r_rating, 2))
        blue_elo.append(round(b_rating, 2))
        terminal_counts[term] = terminal_counts.get(term, 0) + 1

    # Trap node heat from training events and live steps
    trap_heat = {}
    for e in _read_jsonl(ROOT / "training_events.jsonl", last_n=1500):
        et = str(e.get("event_type", "")).lower()
        if "trap" not in et and "honeypot" not in et:
            continue
        node = e.get("node")
        if node is None:
            continue
        n = str(node)
        trap_heat[n] = trap_heat.get(n, 0) + 1
    for s in _read_jsonl(ROOT / "live_steps.jsonl", last_n=400):
        for te in (s.get("trap_events", []) or []):
            node = te.get("node")
            if node is None:
                continue
            n = str(node)
            trap_heat[n] = trap_heat.get(n, 0) + 1

    top_traps = sorted(
        [{"node": n, "count": c} for n, c in trap_heat.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:20]

    return jsonify({
        "episodes": episodes,
        "red_rewards": red_rewards,
        "blue_rewards": blue_rewards,
        "red_elo": red_elo,
        "blue_elo": blue_elo,
        "terminal_counts": terminal_counts,
        "trap_hot_nodes": top_traps,
    })


@app.route("/api/lore")
def lore_reports():
    """Expose storyteller reports for React Lore panel."""
    try:
        sys.path.insert(0, str(ROOT))
        from cipher.utils.storyteller import load_reports

        reports = load_reports() or []
        out = []
        for r in reports:
            out.append({
                "episode": r.get("episode"),
                "filename": r.get("filename", ""),
                "text": r.get("text", ""),
            })
        return jsonify(out)
    except Exception:
        return jsonify([])


@app.route("/api/forensics")
def forensics_latest():
    """Return the most recent forensics reconstruction from episode traces."""
    try:
        traces_dir = ROOT / "episode_traces"
        if not traces_dir.exists():
            return jsonify(None)
        files = sorted(traces_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files[:5]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                recon = data.get("forensics_reconstruction")
                if recon:
                    return jsonify(recon)
            except Exception:
                continue
        return jsonify(None)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    print(f"[cipher-api] War Room API listening on http://localhost:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
