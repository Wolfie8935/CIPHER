# CIPHER system — what changed (architecture snapshot)

## War room (React + Vite)

- **`dashboard-react/`** — CIPHER War Room UI: live map, agent thoughts, analytics, and replay of `episode_traces/`.
- **Dev** — Vite on port `5173` with **`/api` → `http://localhost:5001`** (Flask). **Prod** — build to `dist/`; Flask serves static assets and APIs from the same origin.

## Flask API

- **`dashboard-react/api_server.py`** — reads project-root JSON/JSONL (`live_steps.jsonl`, `logs/agent_*.jsonl`, traces) and exposes REST endpoints for the dashboard.

## Python environment (CIPHER)

- **`main.py`**, **`cipher/`** — multi-agent cyber-ops RL environment: RED/BLUE commanders, subagents, dead drops, zones, and oversight. Training hooks write logs the war room consumes.

## Data flow (high level)

- Training/simulation steps → `live_steps.jsonl` / traces → **Flask** → **React** charts & map. Agent telemetry → `logs/`.
