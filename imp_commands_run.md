# Important Run Commands

Quick, copy-paste command reference to run the React War Room app, common `main.py` modes, and the Flask dashboard from project root.

## Run React War Room App Properly

Run these from project root (`C:/Users/amanc/Desktop/OPENENV FINAL`).

```bash
# 1) Install frontend deps (one-time or after package changes)
cd dashboard-react
npm install
```

```bash
# 2) Start War Room API backend (Flask, port 5001)
cd dashboard-react
python api_server.py
```

```bash
# 3) Start Vite frontend (default dev port 5173)
cd dashboard-react
npm run dev
```

```bash
# Optional: single-terminal sequence (PowerShell)
cd dashboard-react
npm install
Start-Process powershell -ArgumentList "-NoExit","-Command","python api_server.py"
npm run dev
```

Ports:
- React frontend: `http://localhost:5173`
- War Room API: `http://localhost:5001`

## Run main.py Combinations

Run these from project root.

```bash
# Baseline single stub episode
python main.py
```

```bash
# Multi-episode competition (stub mode)
python main.py --episodes 5 --steps 40
```

```bash
# Live mode with API-backed agents
python main.py --live --episodes 3 --steps 30
```

```bash
# Training mode (short)
python main.py --train --train-episodes 10 --steps 30
```

```bash
# Training mode where --episodes overrides --train-episodes in this repo
python main.py --train --episodes 20 --steps 35
```

```bash
# Demo/showcase mode in this repo
python main.py --demo --steps 30
```

```bash
# If your branch exposes showcase alias
python main.py --showcase --steps 30
```

```bash
# Evaluation suite (stub + hybrid comparison)
python main.py --eval 20 --steps 30
```

```bash
# Faster run without writing episode trace files
python main.py --live --episodes 2 --no-trace
```

```bash
# If your branch exposes no-dashboard toggle
python main.py --live --episodes 3 --no-dashboard
```

What each key flag does:
- `--live`: Use live LLM mode for agents.
- `--episodes N`: Number of episodes to run.
- `--steps N`: Max steps per episode.
- `--train`: Run training loop.
- `--train-episodes N`: Training episode count (unless `--episodes` is explicitly passed).
- `--showcase`: Showcase mode alias (branch-dependent; this repo uses `--demo`).
- `--eval N`: Run evaluation suite for `N` episodes per mode.
- `--no-dashboard`: Skip dashboard auto-launch (branch-dependent).
- `--no-trace`: Skip saving episode trace JSON files.

## Run Flask Dashboard App

From project root:

```bash
python -m cipher.dashboard.app
```

This dashboard is served on `http://localhost:8050` in this repo.

## Quick Troubleshooting

```bash
# ECONNREFUSED (frontend cannot reach API)
# Start/restart API server first, then frontend:
cd dashboard-react
python api_server.py
npm run dev
```

```bash
# Missing dependencies (Python + Node)
pip install -r requirements.txt
cd dashboard-react
npm install
```

```bash
# Port already in use (Windows): check and kill process on a port
netstat -ano | findstr :5001
taskkill /PID <PID> /F
```

```bash
# Also check common UI ports
netstat -ano | findstr :5173
netstat -ano | findstr :8050
```
