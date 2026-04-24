# C.md — Adversarial Theater: Visual Storytelling & HF Deployment
## Your job: Transform simulation logs into a premium "Show & Tell" experience

> **Run in parallel** with teammate doing D.md (Optimization & Analytics).
> This task focuses on the "WOW" factor for the hackathon judges.

---

## Context: Technical excellence is boring without a story
Judges want to see the "Battle" happen. Currently, we have a dashboard, but it's a tool for engineers. We need to create **Storytelling Assets** that explain *what* happened in a narrative and visual way.

---

## Change 1 — "CIPHER Cinema": Automated Video Highlight Generation

**Objective:** Generate a "Matrix-style" or "Terminal-style" video recap for every episode.

**What to do:**
1. Create `cipher/utils/video_gen.py`.
2. Use `MoviePy` or `OpenCV` to render a sequence of images/frames.
3. **The Script:**
   - Frame 1: Episode Title & Teams.
   - Frame 2-N: "CCTV" snapshots of the graph (captured from Plotly) + Scrolling "Hacker Console" text from the `episode_log`.
   - Final Frame: Scorecard & Result.
4. Add a CLI flag `--video` to `main.py` that triggers this after an episode ends.

**Implementation Hint:**
```python
def generate_episode_video(episode_data, output_path="highlights.mp4"):
    # 1. Capture snapshots of the network graph at key steps
    # 2. Overlay "RED: Moving to n5" / "BLUE: Investigating n5" text
    # 3. Add a glitch effect or "scanlines" overlay for hacker aesthetics
    # 4. Export using moviepy.editor
```

---

## Change 2 — "The Daily Breach": AI-Powered Narrative Post-Mortems

**Objective:** Use an LLM to write a dramatic news report or a "Red Team After Action Report" (AAR).

**File:** `cipher/training/loop.py` or a new `cipher/utils/storyteller.py`

**What to do:**
After an episode ends, feed the `episode_log` (compressed) to an LLM with this prompt:
> "You are a cyber-warfare correspondent. Write a 3-paragraph dramatic report of the following simulation. RED is 'The Gravity Collective', BLUE is 'Aegis Systems'. Highlight the turning point where [Trap Triggered] or [Exfiltration Successful]."

**Integration:**
- Save this as `episode_X_report.md`.
- Display it in a new "Lore" tab in `cipher/dashboard/app.py`.

---

## Change 3 — Hugging Face Hub & Spaces Deployment

**Objective:** Make the project "Live" on the web so judges can play with it.

**Hugging Face Hub:**
1. Create `cipher/utils/hf_uploader.py`.
2. Use `huggingface_hub` library to upload the `.zip` models from `A.md` to a repository (e.g., `wolfie8935/cipher-specialists`).
3. This allows anyone to run our project by just downloading the models via API.

**Hugging Face Spaces:**
1. Create a `app.py` wrapper in the root (or use the existing one).
2. Create a `Dockerfile` for HF Spaces to run the Dash dashboard.
3. Ensure it loads traces from a public `traces/` folder or a Dataset on HF.

---

## Change 4 — Aesthetic "Glitch" UI Overlays

**Objective:** Make the dashboard feel like a high-stakes cyber-warfare console.

**File:** `cipher/dashboard/assets/custom.css` (Create if missing)
**What to do:**
1. Add a **Glitch Animation** to the `mode-badge` when it's in `LIVE` mode.
2. Add **Scanlines** or a subtle **CRT Flicker** to the main graph container.
3. Use a high-contrast "Cyberpunk" color palette (Neons on Deep Blacks).

---

## Files You Will Touch (Summary)

| File | Change |
|------|--------|
| `cipher/utils/video_gen.py` | [NEW] Logic for rendering .mp4 highlights |
| `cipher/utils/storyteller.py` | [NEW] LLM prompt for narrative generation |
| `cipher/dashboard/app.py` | Add "Lore" tab and video playback support |
| `Dockerfile` | [NEW] For Hugging Face Spaces deployment |
| `cipher/utils/hf_uploader.py` | [NEW] Auto-upload models to HF Hub |

---

## Checklist
- [x] Install `moviepy`, `huggingface_hub`
- [x] Implement `Storyteller` logic and hook into `loop.py`
- [x] Create the "Lore" tab in Dash
- [x] Generate first highlight video from a `live` episode
- [ ] Deploy dashboard to HF Spaces
- [ ] Add glitch aesthetics to CSS
