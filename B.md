# CIPHER — Teammate Work (B): Dashboard & Infrastructure

> **Owner**: Teammate (parallel developer)
> **Parallel track**: Fully independent of A.md — no shared file conflicts.
> **Goal**: Make the live dashboard production-ready, add telemetry, and improve network visualization.

---

## Track B1 — Fix Dashboard Crash on Filter Change ✅ DONE

**Problem**: Switching the "Agent Filter" dropdown in Tab 2 while a run is in progress causes a React key collision crash in Dash. The issue is `update_tab2` rebuilds the entire figure on every interval tick even when the filter hasn't changed.

**Files**: `cipher/dashboard/live.py`

### Tasks
- [x] Open `cipher/dashboard/live.py` → `update_tab2()` (around line 700)
- [x] Add `prevent_initial_call=True` to the `@app.callback` decorator if not already present
- [x] Add Dash `State("t2-filter-cache", "data")` to the callback so we track the last value
- [x] Cache the last-computed figure in a `dcc.Store(id="t2-filter-cache")` component
- [x] Only recompute the figure if `filter_val` has changed since last render (compare via Store)
- [x] Add the Store component to the Tab 2 layout section

**Implementation**: `_TAB2_CACHE` module-level dict + `dcc.Store(id="t2-filter-cache")` added to `_build_tab2_layout()`. Callback now takes `State("t2-filter-cache", "data")` and uses `callback_context.triggered` to skip recompute on interval ticks when filter unchanged.

---

## Track B2 — SQLite Telemetry (Replace CSV) ✅ DONE

**Problem**: `rewards_log.csv` is written by the episode runner and read by the dashboard simultaneously, causing file-lock errors on Windows. The retry loop (`_load_rewards_csv`) masks this but adds latency.

**Files**: `cipher/utils/reward_logger.py`, `cipher/dashboard/live.py`

### Tasks
- [x] Create `cipher/utils/telemetry_db.py`:
  - Thread-safe SQLite store with `threading.Lock`
  - `write_episode(...)`, `get_last_n_episodes(n)`, `get_all_episodes(run_id)`
  - Stores `run_id`, `llm_mode`, all reward columns
- [x] `cipher/utils/reward_logger.py` → `RewardLogger.log()` now dual-writes to SQLite AND CSV
- [x] `cipher/dashboard/live.py` → `_get_run_frame()` tries SQLite first; CSV is fallback
- [x] Retry loop kept for CSV fallback only; SQLite path never blocks

**Implementation**: `cipher/utils/telemetry_db.py` created (124 lines). `telemetry.db` stored in project root.

---

## Track B3 — Network Map Real-Time Delta Updates ✅ DONE

**Problem**: The `t3-map` (Tab 3 network visualization) redraws the entire 50-node graph every 2 seconds via Plotly. This causes a jarring full-redraw flash on every tick.

**Files**: `cipher/dashboard/live.py` (around the `update_tab3` function)

### Tasks
- [x] Find `update_tab3()` in `live.py`
- [x] Cache edge traces in `_GRAPH_CACHE["edge_traces"]` — built once on first call
- [x] On subsequent ticks: only rebuild node traces (RED position, trap counts, zone colors)
- [x] Show live RED position: parse last live_step for `→ nNN`, mark with red star
- [x] Color coding: zone colors for nodes, gold rings for trap-triggered nodes, star for RED
- [x] Show RED node label in tooltip with `← RED HERE`

**Implementation**: Edge traces cached after first build. Node traces rebuilt each tick (cheap). RED current node parsed from `live_steps.jsonl` and highlighted as a red star marker.

---

## Track B4 — Dashboard Header: Live Agent Status ✅ DONE

**Problem**: The dashboard header only shows "Episode N". Add a live per-agent status row showing which agents called LLM this step and their last action.

**Files**: `cipher/dashboard/live.py`, `main.py`

### Tasks
- [x] After each step completes in `main.py` step callback, write `logs/agent_status.json`
  ```json
  {
    "step": 7, "episode": 2, "zone": "Critical/HVT",
    "suspicion": 0.85, "detection": 0.18,
    "agents": {
      "red_planner_01": {"action": "move", "node": 23, "team": "red"},
      "blue_surveillance_01": {"action": "scan", "node": null, "team": "blue"}
    }
  }
  ```
- [x] Added `dcc.Interval(id="interval-fast", interval=1500)` (1.5s tick for status + logs)
- [x] `update_agent_status_bar()` reads `agent_status.json`, renders colored chips per agent
- [x] Status bar placed below header, above tabs — always visible
- [x] New **"Live Logs"** tab (`tab-logs`) shows step-by-step narrative feed, newest on top
- [x] `_write_agent_status()` helper in `main.py` writes status on every step

**Implementation**: Status bar updates every 1.5s via `interval-fast`. Per-agent colored chips show `team:action→node`. Live Logs tab shows monospace step feed grouped by episode.

---

## Track B5 — Add Episode History Tab ✅ DONE

**Problem**: There's no way to compare rewards across episodes during a multi-episode run.

**Files**: `cipher/dashboard/live.py`

### Tasks
- [x] Added **"History"** tab (`tab-history`) with:
  - Line chart: RED vs BLUE reward per episode, grouped by run (each run = separate trace)
  - Bar chart: terminal reasons distribution (exfil/detected/aborted/max_steps)
  - Table: last 30 episodes with run_id, mode, ep, steps, outcome, RED, BLUE, verdict
- [x] Data source: `telemetry_db.get_last_n_episodes(200)` (from B2)
- [x] Fully pre-rendered (all tab IDs always in DOM — no callback errors)
- [x] `th-rewards-chart`, `th-outcomes-chart`, `th-run-selector`, `th-table` IDs registered

**Implementation**: `_build_tab_history_layout()` + `update_tab_history()` added. Pulls all runs from SQLite, color-codes outcomes, shows run comparison.

---

## Bonus — API Cost Display ✅ DONE

Added `header-cost` span to the dashboard header showing estimated API spend:
- `stub` mode → `$0.00 (stub)`
- `live/hybrid` → `~$X.XXXX` based on `training_state.json → estimated_cost_usd`
- `main.py` now calculates cost on run completion and writes to `training_state.json`

---

## Bonus — Enhanced Learning Curve (Tab 6) ✅ DONE

Tab 6 now has **3 charts** instead of 2:
1. **Reward curves**: raw (dim) + 10-ep moving average (bright) + evolution vlines
2. **RED−BLUE gap chart**: bar per episode colored red/blue by sign, + gold moving avg
3. **Rolling win rate**: 10-ep window, 50% reference line

---

## Bonus — Trace Naming ✅ DONE

Episode traces now saved as `episode_{NNN}_{YYYYMMDD_HHMMSS}_{mode}.json` (e.g. `episode_001_20260423_183700_live.json`) for easy identification in the dashboard trace selector.

---

## Verification Checklist

```bash
# Syntax check
python -c "import main; print('OK')"
python -c "import cipher.dashboard.live; print('OK')"

# Dashboard loads without errors
python main.py --live
# Open http://127.0.0.1:8050

# Check no CSV file-lock errors in console
# Check Tab 2 filter doesn't crash
# Check network map doesn't flash
# Check agent status bar updates
# Check Live Logs tab shows steps
# Check History tab shows cross-run data
```

## Notes for Teammate

- Do NOT edit `cipher/training/_episode_runner.py` — that's being modified in A.md Track A4
- Do NOT edit `cipher/utils/llm_client.py` — that's A.md territory
- The `live.py` file is ~1880 lines — use `Ctrl+F` / grep to navigate
- All new Store components must be added to the `_build_layout()` function OR the pre-rendered tab layout
- Test with `python -m pytest tests/test_dashboard*.py -v` if dashboard tests exist
