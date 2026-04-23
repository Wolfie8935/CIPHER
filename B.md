# CIPHER — Teammate Work (B): Dashboard & Infrastructure

> **Owner**: Teammate (parallel developer)
> **Parallel track**: Fully independent of A.md — no shared file conflicts.
> **Goal**: Make the live dashboard production-ready, add telemetry, and improve network visualization.

---

## Track B1 — Fix Dashboard Crash on Filter Change

**Problem**: Switching the "Agent Filter" dropdown in Tab 2 while a run is in progress causes a React key collision crash in Dash. The issue is `update_tab2` rebuilds the entire figure on every interval tick even when the filter hasn't changed.

**Files**: `cipher/dashboard/live.py`

### Tasks
- [ ] Open `cipher/dashboard/live.py` → `update_tab2()` (around line 700)
- [ ] Add `prevent_initial_call=True` to the `@app.callback` decorator if not already present
- [ ] Add Dash `State("agent-filter", "value")` to the callback signature so we track the last value
- [ ] Cache the last-computed figure in a `dcc.Store(id="tab2-figure-cache")` component
- [ ] Only recompute the figure if `filter_val` has changed since last render (compare via Store)
- [ ] Add the Store component to the Tab 2 layout section

**Verify**: Switch filter rapidly 5 times while simulation is running — no crash

---

## Track B2 — SQLite Telemetry (Replace CSV)

**Problem**: `rewards_log.csv` is written by the episode runner and read by the dashboard simultaneously, causing file-lock errors on Windows. The retry loop (`_load_rewards_csv`) masks this but adds latency.

**Files**: `cipher/utils/reward_logger.py`, `cipher/dashboard/live.py`

### Tasks
- [ ] Create `cipher/utils/telemetry_db.py`:
  ```python
  # SQLite-backed thread-safe episode telemetry store
  # Table: episodes(id, episode, steps, terminal_reason,
  #                red_total, blue_total, oversight_total,
  #                timestamp REAL)
  ```
- [ ] Use `threading.Lock` + `sqlite3.connect(check_same_thread=False)` for thread safety
- [ ] Expose: `write_episode(...)`, `get_last_n_episodes(n)`, `get_all_episodes()`
- [ ] Open `cipher/utils/reward_logger.py` → modify `RewardLogger.log()` to write to SQLite **in addition to** CSV (keep CSV for backward compat for 1 sprint)
- [ ] Open `cipher/dashboard/live.py` → `_load_rewards_csv()` — add SQLite path as primary data source, CSV as fallback
- [ ] Remove the retry/sleep loop once SQLite is primary

**Verify**: Run `python main.py --live` — dashboard Tab 1 updates smoothly with no "retry" messages in logs

---

## Track B3 — Network Map Real-Time Delta Updates

**Problem**: The `t3-map` (Tab 3 network visualization) redraws the entire 50-node graph every 2 seconds via Plotly. This causes a jarring full-redraw flash on every tick.

**Files**: `cipher/dashboard/live.py` (around the `update_tab3` function)

### Tasks
- [ ] Find `update_tab3()` in `live.py`
- [ ] Switch from `px.scatter_graph` full rebuild to Plotly `extendData` partial updates:
  - Track `red_current_node` position across ticks
  - Only update the RED agent marker's x/y coords + color
- [ ] Add `dcc.Store(id="network-state-cache")` to hold the last full graph layout
- [ ] On first render: build full graph layout, save to Store
- [ ] On subsequent ticks: only patch changed nodes (RED position, detected nodes in red)
- [ ] Color coding: unvisited = grey, RED visited = orange, detected by BLUE = blue outline, honeypot = purple

**Verify**: Network map updates smoothly with no flash. Zone progression should be visible as RED moves.

---

## Track B4 — Dashboard Header: Live Agent Status

**Problem**: The dashboard header only shows "Episode N". Add a live per-agent status row showing which agents called LLM this step and their last action.

**Files**: `cipher/dashboard/live.py`

### Tasks
- [ ] Add `dcc.Store(id="agent-status-store")` to the app layout
- [ ] After each step completes in `_episode_runner.py`, write agent statuses to a sidecar JSON file: `logs/agent_status.json`
  ```json
  {
    "step": 7,
    "agents": {
      "red_planner_01": {"action": "move", "node": 23, "elapsed_ms": 1243},
      "blue_surveillance_01": {"action": "scan", "node": null, "elapsed_ms": 876}
    }
  }
  ```
- [ ] Add a new `dcc.Interval(id="status-interval", interval=1000)` (1s tick)
- [ ] Add callback `update_agent_status_bar()` that reads `agent_status.json` and renders a compact HTML row
- [ ] Place the status bar below the main header, above the tabs

**Verify**: After each LLM step, agent status row updates within ~1.5s showing current actions

---

## Track B5 — Add Episode History Tab (Tab 6 or new)

**Problem**: There's no way to compare rewards across episodes during a multi-episode run.

**Files**: `cipher/dashboard/live.py`

### Tasks
- [ ] Add Tab 6: "Episode History" with:
  - Line chart: RED reward vs BLUE reward per episode
  - Bar chart: terminal reasons (exfil / detected / max_steps / stalled)
  - Table: last 20 episodes with columns: ep, steps, outcome, red_total, blue_total
- [ ] Data source: `telemetry_db.get_last_n_episodes(20)` (from B2) or CSV fallback
- [ ] Add to tab pre-rendering so no callback ID errors

**Verify**: Run 3 episodes with `python main.py --episodes 3 --live` — Tab 6 shows all 3 episodes

---

## Verification Checklist

```bash
# Dashboard loads without errors
python main.py --live
# Open http://127.0.0.1:8050

# Check no CSV file-lock errors in console
# Check Tab 2 filter doesn't crash
# Check network map doesn't flash
# Check agent status bar updates

# Dash app should not print any "Callback error" to console
```

## Notes for Teammate

- Do NOT edit `cipher/training/_episode_runner.py` — that's being modified in A.md Track A4
- Do NOT edit `cipher/utils/llm_client.py` — that's A.md territory
- The `live.py` file is ~1400 lines — use `Ctrl+F` / grep to navigate
- All new Store components must be added to the `_build_layout()` function OR the pre-rendered tab layout
- Test with `python -m pytest tests/test_dashboard*.py -v` if dashboard tests exist
