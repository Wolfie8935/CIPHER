# CIPHER — PHASE 12 MASTER PROMPT
## For Your Claude Instance — Episode Replay Dashboard
### Read This Fully Before Writing Any Code

---

## WHAT PROJECT YOU ARE JOINING

You are building the visualization layer for **CIPHER** — an adversarial multi-agent reinforcement learning environment. Two teams of AI agents (RED infiltrators vs BLUE defenders) battle inside a simulated 50-node corporate network. Your job is to make their battles visible and compelling.

The person giving you this prompt has already built:
- A fully working environment (50-node zone-based enterprise network)
- 8 LLM-backed agents (4 RED, 4 BLUE) that make real strategic decisions
- A dead drop memory system (RED writes notes to itself when its memory resets)
- A complete reward function for both sides
- Episode traces saved to `episode_traces/` as JSON files after every run

**You are building Phase 12: the Episode Replay Dashboard.** This is the visual that judges will look at. It needs to tell the story of an episode — who went where, when BLUE started detecting RED, when dead drops were written, when traps fired — in a way that a judge understands in 30 seconds.

This is **30% of the judging score** (Storytelling). Build it like it matters.

---

## YOUR CONSTRAINTS

1. **Read the episode trace files first.** They are in `episode_traces/`. Run `python main.py` if none exist. The JSON structure is your data source — understand it before designing any UI.

2. **Use Plotly Dash.** It is already in `requirements.txt`. The dashboard file already exists as a stub at `cipher/dashboard/app.py`. Expand it — do not create a new file.

3. **Stub mode works perfectly.** You do NOT need the NVIDIA API or `LLM_MODE=live`. Run everything with `LLM_MODE=stub`. The episode traces from stub mode have the same JSON structure as live mode.

4. **Do not touch anything outside `cipher/dashboard/`.** The environment, agents, rewards, and training loop are complete and tested. Touch them and you will break 87 passing tests.

5. **The dashboard reads from disk.** It does not run episodes. It visualizes saved episode traces. This is the correct architecture — the training loop writes traces, the dashboard reads them.

---

## STEP 0 — READ THESE FILES BEFORE WRITING CODE

```bash
# Generate a fresh episode trace to work with
python main.py

# Read the trace structure
python -c "
import json
from pathlib import Path
traces = list(Path('episode_traces').glob('*.json'))
if traces:
    data = json.loads(traces[0].read_text())
    print('Top-level keys:', list(data.keys()))
    print('Episode log length:', len(data.get('episode_log', [])))
    print('First log entry:', data['episode_log'][0] if data.get('episode_log') else 'empty')
    print('Reward keys:', list(data.get('rewards', {}).keys()) if 'rewards' in data else 'no rewards key')
"
```

Also read:
- `cipher/environment/state.py` — the EpisodeState structure, `to_dict()` method
- `cipher/agents/base_agent.py` — the Action dataclass structure  
- `cipher/environment/graph.py` — the graph structure (zones, node types)
- `cipher/dashboard/app.py` — the existing stub to expand

---

## WHAT YOU ARE BUILDING

A Plotly Dash web application at `http://localhost:8050` with **5 panels** on a single page. No tabs — everything visible at once on a dark-themed dashboard.

---

### PANEL 1 — Network Graph (center, largest panel)

A visual of the 50-node enterprise network. This is the spatial anchor — everything else references this map.

**What it shows:**
- All 50 nodes as circles, colored by zone:
  - Zone 0 (Perimeter): gray
  - Zone 1 (General): blue  
  - Zone 2 (Sensitive): orange
  - Zone 3 (Critical): red
- Node labels showing the hostname (e.g., `corp-fs-02`)
- Edges shown as faint lines
- RED's path through the episode as a colored trail — each step a slightly brighter line, so the path direction is readable
- A pulsing marker showing RED's position at the currently selected step
- BLUE honeypot nodes marked with a ⚠ symbol (visible here since this is BLUE's view / ground truth)
- Nodes BLUE investigated highlighted with a blue ring
- Dead drop locations marked with a 📦 symbol

**Interactivity:**
- A step slider below this panel (step 1 → max step). Dragging it updates all panels to show the state at that step.
- Clicking a node shows its details in a sidebar tooltip: node type, zone, files, suspicion risk score.

**Implementation note:** Use `plotly.graph_objects.Figure` with `Scatter` traces on a fixed layout. Use NetworkX's `spring_layout` seeded with the episode seed for consistent positioning. Node positions should not change as the slider moves — only the overlay markers move.

---

### PANEL 2 — Suspicion & Detection Timeline (top right)

A dual-line chart showing the tension of the episode over time.

**What it shows:**
- X axis: step number (0 → max)
- Y axis: 0.0 → 1.0
- RED line: `red_suspicion_score` at each step — how hot RED is running
- BLUE line: `blue_detection_confidence` at each step — how close BLUE is to catching RED
- A horizontal dashed line at 0.8 labeled "Detection Threshold"
- Vertical dashed lines at context reset steps (when RED's memory wiped) labeled "MEMENTO"
- Small icon markers on the RED line wherever suspicion spiked sharply (Δ > 0.1) — these are the moments of danger
- The currently selected step highlighted with a vertical cursor line

**What this tells the story:** You can see RED running hot, pausing to let suspicion decay, then pushing again. You can see BLUE's confidence climbing as RED gets sloppy. The MEMENTO reset lines show the moments RED was most vulnerable.

---

### PANEL 3 — Episode Action Log (bottom right, scrollable)

A step-by-step log of what happened, styled like a terminal.

**What it shows:**
- Each step as a group of entries
- RED actions in red text: `[Step 07] [R] Red Planner → MOVE → node 42`
- BLUE actions in blue text: `[Step 07] [B] Blue Forensics → RECONSTRUCT_PATH`
- Reasoning lines in dim gray italic beneath each action (if reasoning field is non-empty)
- Dead drop events in yellow: `[Step 07] 📦 Dead drop written at node 42 (210 tokens, efficiency=1.00)`
- Trap events in purple: `[Step 07] ⚡ TRAP: FalseTrail fired — BLUE chasing node 31`
- Context reset events in white bold: `[Step 08] ══ MEMENTO RESET — RED memory cleared ══`

**Interactivity:** Clicking a step in this log moves the step slider and updates the network graph to that step.

---

### PANEL 4 — Dead Drop Inspector (top left)

A table showing every dead drop written during the episode.

**Columns:**
| Step | Written By | Node | Tokens | Efficiency | Integrity | Contents Preview |
|------|-----------|------|--------|------------|-----------|-----------------|

- **Integrity** column: ✅ Valid or ⚠ Tampered (red text)
- **Contents Preview**: first 60 chars of `continuation_directive`
- Clicking a row highlights that node on the network graph and jumps the step slider to that step

**Why judges care about this:** This is the MEMENTO mechanic made visible. Judges can literally see what RED wrote to itself, when, and whether BLUE tampered with it.

---

### PANEL 5 — Episode Summary (small panel, top center)

A compact summary card showing the final outcome.

**What it shows:**
```
EPISODE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Outcome:      RED ABORTED  (or DETECTED / EXFILTRATED / MAX_STEPS)
Steps:        47 / 200
RED Reward:   -0.29  (exfil=0.33, stealth=0.62, memory=1.00)
BLUE Reward:   0.53  (detection=1.00, speed=0.89, fp_penalty=-1.00)
Oversight:    DEAD_DROP_ANOMALY flagged
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Context Resets:  1
Dead Drops Written:  8
Traps Placed — RED: 0  BLUE: 0
Zone Crossings:  3
```

Outcome is color-coded:
- RED EXFILTRATED → red text (RED won)
- DETECTED → blue text (BLUE won)  
- ABORTED → gray text
- MAX_STEPS → yellow text

---

## FILE TO BUILD — `cipher/dashboard/app.py`

Full implementation. The stub already exists — expand it.

### App structure

```python
"""
cipher/dashboard/app.py

CIPHER Episode Replay Dashboard.
Reads saved episode traces from episode_traces/ and renders an
interactive 5-panel visualization.

Usage:
    python -m cipher.dashboard.app
    # Opens at http://localhost:8050

Does NOT run episodes. Only reads from disk.
Does NOT import from cipher.training or call any LLM.
Only imports from: cipher.utils.config, cipher.utils.logger,
                   cipher.environment.graph (for layout generation)
"""
```

### Key implementation decisions

**Episode selector:** A dropdown at the top of the page listing all JSON files in `episode_traces/`. Selecting one loads that episode. If no traces exist, show a clear message: "No episode traces found. Run `python main.py` to generate one."

**Graph layout:** Generate node positions using `nx.spring_layout(graph, seed=episode_seed)` where `episode_seed` comes from the episode trace JSON. This gives consistent, deterministic layouts per episode.

**Step slider:** `dcc.Slider` from 0 to `max_step`. Callback updates all panels simultaneously.

**Dark theme:** Use `plotly.graph_objects` with `template="plotly_dark"` on all charts. Background `#0d1117` (GitHub dark). RED accent `#ff4444`. BLUE accent `#4488ff`. Dead drop yellow `#ffcc00`. Trap purple `#aa44ff`.

**Loading state:** Show a spinner while episode data is loading. Episode traces can be several MB for long runs.

**No authentication, no server state.** Single user, local use. Keep it simple.

---

## HOW TO GENERATE TEST DATA

```bash
# Generate 3 episode traces to test with
python main.py
# Rename the trace:
# episode_traces/episode_001.json already there from Phase 1

# Generate more
LLM_MODE=stub python -c "
from cipher.training.loop import TrainingLoop
from cipher.utils.config import config
loop = TrainingLoop(config)
loop.run(n_episodes=3)
"
```

---

## VERIFICATION COMMANDS — Run After Building

```bash
# 1. Dashboard imports cleanly
python -c "from cipher.dashboard.app import CipherDashboard; print('import OK')"

# 2. Dashboard initializes without error
python -c "
from cipher.dashboard.app import CipherDashboard
from cipher.utils.config import config
dash = CipherDashboard(config)
print('Dashboard initialized OK')
print(f'Episode traces found: {dash.get_available_traces()}')
"

# 3. Episode trace loads and parses correctly
python -c "
from cipher.dashboard.app import CipherDashboard
from cipher.utils.config import config
from pathlib import Path
import json

# Make sure we have a trace
traces = list(Path('episode_traces').glob('*.json'))
assert len(traces) > 0, 'No traces found — run python main.py first'

dash = CipherDashboard(config)
data = dash.load_episode(str(traces[0]))
assert 'episode_log' in data, 'episode_log missing from trace'
assert 'rewards' in data or any('reward' in k for k in data.keys()), 'no reward data in trace'
print(f'Trace loaded: {len(data[\"episode_log\"])} log entries')
print('Episode trace loading: PASSED')
"

# 4. Graph figure generates for an episode
python -c "
from cipher.dashboard.app import CipherDashboard
from cipher.utils.config import config
from pathlib import Path
import json

dash = CipherDashboard(config)
traces = list(Path('episode_traces').glob('*.json'))
data = dash.load_episode(str(traces[0]))
fig = dash.build_network_figure(data, step=1)
assert fig is not None
assert hasattr(fig, 'data'), 'Figure has no data traces'
print(f'Network figure: {len(fig.data)} traces — PASSED')
"

# 5. Timeline figure generates
python -c "
from cipher.dashboard.app import CipherDashboard
from cipher.utils.config import config
from pathlib import Path

dash = CipherDashboard(config)
traces = list(Path('episode_traces').glob('*.json'))
data = dash.load_episode(str(traces[0]))
fig = dash.build_timeline_figure(data)
assert fig is not None
print('Timeline figure: PASSED')
"

# 6. Dashboard server starts
python -c "
import os; os.environ['LLM_MODE'] = 'stub'
from cipher.dashboard.app import CipherDashboard
from cipher.utils.config import config
dash = CipherDashboard(config)
# Just verify server object creates without error — don't actually start it
print('Dashboard server: PASSED')
"

# 7. Actually run it and check in browser
python -m cipher.dashboard.app
# Open http://localhost:8050
# Verify: dropdown shows episode files, network graph renders, slider works,
#         action log scrolls, dead drop table shows rows
```

**Phase 12 is complete when:**
- All 6 script checks pass
- `python -m cipher.dashboard.app` starts without error
- Opening `http://localhost:8050` shows the 5-panel dashboard
- Dragging the step slider moves the RED position marker on the network graph
- Dead drop rows are visible in the inspector table
- The timeline chart shows two lines (RED suspicion, BLUE confidence)

---

## WHAT GOOD LOOKS LIKE

When a judge opens this dashboard and loads an episode, they should be able to:
1. See the 50-node network and immediately understand the 4-zone structure
2. Watch RED's path light up as they drag the slider
3. See the MEMENTO reset moments as vertical lines on the timeline
4. Click a dead drop row and see where it was written on the map
5. See the episode outcome in the summary card

That is the story. Make it visible.

---

## ONE THING TO KEEP IN MIND

The dashboard will eventually show live training data (Phase 13 adds live-updating charts). Design Panel 2 (the timeline) so it can accept either a complete episode trace OR a growing list of per-episode rewards. This makes Phase 13 a small addition rather than a rewrite. Specifically: `build_timeline_figure()` should accept an optional `reward_history: list[dict]` parameter that, if provided, shows the training curve instead of the single-episode suspicion/confidence chart.

---

Do not start Phase 13 (live training dashboard). Stop and report when Phase 12 is complete.
