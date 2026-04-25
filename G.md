# G.md — Demo Excellence & Deployment Polish
## Your job: Make the hackathon demo unforgettable in the last 2 hours

> **Start after E.md and F.md are done** (or run in parallel on parts that don't depend on them).
> **Run in parallel** with teammates doing E.md and F.md.
>
> This task transforms a working system into a **jaw-dropping hackathon demo**.
> Judges have 5 minutes per project. G.md maximises impression per second.

---

## Context: What Judges See

Current demo flow:
1. `python main.py --live --episodes 3` → terminal output → hard to follow
2. `python -m cipher.dashboard.app` → dashboard → impressive but requires explanation

Desired demo flow:
1. **30-second terminal "cinematic" intro** showing CIPHER banner + what's happening
2. **Live dashboard auto-launches** in browser
3. Judges see Winning Metrics banner immediately
4. Analytics tab shows Elo + heatmap
5. **One-click "Judge Demo" runs 3 pre-configured showcase episodes**

---

## Change 1 — "Judge Demo" Mode

**File:** `main.py`

**What to do:**

Add `--demo` flag that runs 3 curated showcase episodes:
- Episode 1: RED wins fast (shows exfiltration)
- Episode 2: BLUE wins with honeypot trap (shows defensive mechanics)
- Episode 3: Contested (shows deception + dead drops)

```python
# Add to argument parser:
parser.add_argument("--demo", action="store_true", 
                    help="Run 3 curated showcase episodes for judge demo")

# After parsing:
if args.demo:
    print(DEMO_BANNER)  # ASCII art + explanation of what's about to happen
    # Run 3 episodes with fixed seeds that produce interesting outcomes
    _run_demo_mode()
```

**Demo banner** (put in a constant at the top of `main.py`):
```
╔══════════════════════════════════════════════════════╗
║  C I P H E R  — Judge Demo Mode                     ║
║  OpenEnv Hackathon | Multi-Agent Adversarial RL      ║
╠══════════════════════════════════════════════════════╣
║  3 episodes: Exfiltration · Detection · Contested    ║
║  8 LLM agents · 50-node network · Live inference     ║
╚══════════════════════════════════════════════════════╝
```

**Fixed seeds** — find seeds that produce the 3 outcome types from existing episode traces 
in `episode_traces/`. Pick:
- A seed where RED wins in < 8 steps (fast showcase)
- A seed where BLUE wins via honeypot trigger
- A seed where episode goes 15+ steps (back-and-forth tension)

---

## Change 2 — Auto-Launch Dashboard

**File:** `main.py`

**What to do:**

When `--demo` or `--live` is used, automatically open the dashboard in a browser after 
the first episode completes:

```python
import threading, webbrowser, time

def _auto_launch_dashboard():
    """Open dashboard in browser 3 seconds after training starts."""
    time.sleep(3)
    webbrowser.open("http://localhost:8050")

# In main(), after starting training:
if args.demo or args.live:
    t = threading.Thread(target=_auto_launch_dashboard, daemon=True)
    t.start()
    # Also start dashboard in background if not already running:
    _maybe_start_dashboard_process()
```

Add `_maybe_start_dashboard_process()` that checks if port 8050 is already in use and 
only starts the dashboard if not:

```python
import socket
def _maybe_start_dashboard_process():
    with socket.socket() as s:
        if s.connect_ex(("localhost", 8050)) != 0:
            import subprocess, sys
            subprocess.Popen(
                [sys.executable, "-m", "cipher.dashboard.app"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
```

---

## Change 3 — Improved Terminal Output Formatting

**File:** `main.py`

**What to do:**

The current step ticker is good but lacks color context. Add ANSI colors to the terminal 
output so judges can follow the action at a glance:

```python
# Color codes (already used in the project — check if colorama is available)
RED_ANSI  = "\033[91m"
BLUE_ANSI = "\033[94m"
GREEN_ANSI = "\033[92m"
GOLD_ANSI  = "\033[93m"
DIM_ANSI   = "\033[90m"
RESET_ANSI = "\033[0m"

# In _print_live_step(), wrap RED actions in RED_ANSI, BLUE in BLUE_ANSI:
red_action_str = f"{RED_ANSI}{red_action_str}{RESET_ANSI}"
blue_action_str = f"{BLUE_ANSI}{blue_action_str}{RESET_ANSI}"

# Highlight exfiltration events:
if "exfil" in line.lower():
    print(f"{GREEN_ANSI}{'█' * 20} EXFILTRATION SUCCESS {'█' * 20}{RESET_ANSI}")
```

Also add a **real-time timer** showing elapsed time vs estimated total:
```
Step 07/30   42.6s  [est: ~180s total]  Zone ◆ CRITICAL/HVT  RED: exfiltrate ✓
```

---

## Change 4 — Dashboard Quick-Start Guide Improvement

**File:** `cipher/dashboard/app.py`

**What to do:**

The current "How to read this dashboard" section is text-heavy. Replace with a visual 
legend that judges can scan in 10 seconds:

```
🔴 RED PATH ——→ route RED took (Zone 0→3)    🔵 BLUE CONFIDENCE: detection level
📦 DEAD DROP: encrypted memory node         ⚡ TRAP FIRED: deception activated
★ RED POSITION: current node                ◎ HONEYPOT: disguised as FILE_SERVER
```

Add a "QUICK START" badge that blinks/pulses on first load:
```python
# Use CSS animation via className or inline keyframes
html.Div("▶ CLICK PLAY TO START REPLAY", style={
    "animation": "pulse 2s infinite",
    ...
})
```

---

## Change 5 — Export "Shareable Report" Button

**File:** `cipher/dashboard/app.py`

**What to do:**

Add a "Share" button that exports the current episode as a self-contained HTML file 
with embedded charts (the existing `export_replay_html` function already does this — 
just wire it to a download button):

```python
# The function exists: cipher/dashboard/replay.py::export_replay_html()
# Wire it to a Dash Download component:

html.Button("📤 Export HTML Report", id="export-btn"),
dcc.Download(id="export-download"),

@app.callback(
    Output("export-download", "data"),
    Input("export-btn", "n_clicks"),
    State("episode-data-store", "data"),
    prevent_initial_call=True,
)
def export_report(n_clicks, data):
    html_str = export_replay_html(data, theme="dark")
    return dict(content=html_str, filename="cipher_episode_report.html")
```

---

## Checklist

- [ ] Add `--demo` flag with 3 curated showcase episodes and ASCII banner
- [ ] Add `_maybe_start_dashboard_process()` to start dashboard if not running
- [ ] Add ANSI colors to terminal output (RED for RED actions, BLUE for BLUE)
- [ ] Add real-time timer to step ticker
- [ ] Improve dashboard quick guide with visual legend
- [ ] Wire `export_replay_html` to a Download button in app.py
- [ ] Test full demo flow end-to-end: `python main.py --demo`

---

## Files You Will Touch

| File | Change |
|------|--------|
| `main.py` | Add `--demo` mode, `_auto_launch_dashboard`, ANSI colors, timer |
| `cipher/dashboard/app.py` | Visual legend, Export button, Download component |

---

## Demo Script (for the 5-minute judge presentation)

```
[0:00] python main.py --demo
       → ASCII banner appears
       → Dashboard opens automatically in browser
       → Episode 1 starts (RED wins fast — 7 steps)

[1:30] Switch to browser — show Winning Metrics banner
       → "7 Total Exfils | 18.3 steps avg TTD | +45% efficiency | HIGH confidence"
       → Click "Analytics ★" tab — show Elo chart rising for RED

[2:30] Click "Network Map" tab during episode 2
       → Show RED path as red trail through 4 zones
       → Show honeypot traps (blue dots) on graph

[3:30] Episode 2 ends (BLUE wins via honeypot)
       → Battle log shows: "BLUE ALERT CORRECT — RED agent located"
       → Elo chart: BLUE Elo spikes up

[4:00] Show Analytics tab heatmap
       → "These red nodes are Death Traps — BLUE places honeypots here every time"

[4:30] Show download: "Export HTML Report" button
       → "Judges, you can take this replay home"

[5:00] Q&A
```
