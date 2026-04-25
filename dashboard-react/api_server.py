"""
CIPHER War Room — Flask API Server
Serves live episode data to the React war room dashboard on port 5001.
Run: python dashboard-react/api_server.py
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import subprocess
import threading
from flask import Flask, jsonify, send_from_directory, redirect, Response
from flask_cors import CORS

_HERE    = Path(__file__).resolve().parent
DIST_DIR = _HERE / 'dist'
ROOT     = _HERE.parent          # project root

app  = Flask(__name__)
CORS(app)


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
    return jsonify(data)


@app.route("/api/thoughts")
def thoughts():
    data = _read_jsonl(ROOT / "logs" / "agent_thoughts.jsonl", last_n=24)
    return jsonify(data)


@app.route("/api/agent-status")
def agent_status():
    return jsonify(_read_json(ROOT / "logs" / "agent_status.json"))


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


@app.route("/api/episodes")
def list_episodes():
    traces = sorted(glob.glob(str(ROOT / "episode_traces" / "*.json")), reverse=True)
    return jsonify([Path(t).name for t in traces])


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
    path = ROOT / "episode_traces" / filename
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Inject canonical graph when episode file has no graph data
            if not data.get('graph', {}).get('nodes'):
                data['graph'] = _get_canonical_graph()
            return jsonify(data)
        except Exception:
            pass
    return jsonify({}), 404


@app.route("/api/rewards")
def rewards():
    path = ROOT / "rewards_log.csv"
    if not path.exists():
        return jsonify([])
    try:
        with open(path, encoding="utf-8") as f:
            return jsonify(list(csv.DictReader(f)))
    except Exception:
        return jsonify([])


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "root": str(ROOT)})


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    print(f"[cipher-api] War Room API listening on http://localhost:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
