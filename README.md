# Volleyball Timing Analyzer

A computer-vision pipeline that watches volleyball footage and measures
whether a hitter contacts the ball at the right moment in their jump —
not just "did they jump," but the actual timing offset in milliseconds
between the jump's apex and ball contact.

**[Live demo](https://volleyball-analysis-vincentp.streamlit.app/)**

## What it does

Four separable phases, each producing its own JSON + annotated overlay video:

1. **Ball tracking** (`ball/`) — a fine-tuned TrackNet-based model (vball-net)
   detects and tracks the ball, interpolates short gaps, marks the apex of
   each flight arc.
2. **Pose & jump detection** (`pose/`) — YOLO11-pose tracks every player;
   takeoff/landing is detected from ankle trajectory against a rolling
   ground-baseline, and contact is found via closest wrist-to-ball approach.
3. **Event classification** (`events/`) — labels each touch as a
   serve/set/spike/block/dig, via either a heuristic (trajectory shape +
   jump correlation) or a YOLO classifier fine-tuned on professional match
   footage.
4. **Timing assessment** (`timing/`) — jump apex = the temporal midpoint of
   takeoff and landing; timing offset = contact vs. that apex, in
   milliseconds; verdict = early / on-time / late against a configurable
   tolerance.

## Why this is worth reading past the demo

Every phase's README documents what was actually **measured**, not assumed:
real accuracy numbers against hand-labeled ground truth, real bugs found and
fixed (including in the vendored third-party model repo), and an honest
account of a fine-tuning attempt that didn't transfer across camera domains.
See `ball/README.md`, `pose/README.md`, and `events/README.md` for the
specifics — including what's still unreliable and why.

## Run it locally

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run webapp/app.py
```

Opens a full interactive UI: pick a clip (or upload one), run the live
pipeline, see every phase's overlay video and metrics. See `webapp/README.md`.

To run phases individually from the command line instead, each has its own
entry point: `ball/run.py`, `pose/run.py`, `events/run.py`, `timing/run.py`
(`--help` on any of them for usage).

## Project layout

```
ball/     ball detection + tracking + apex
pose/     pose estimation + takeoff/landing/contact detection
events/   event timestamping
timing/   the timing assessment
videoio/  video read/write, annotation overlay, JSON export
config/   thresholds and paths (config.yaml -- nothing is hardcoded in code)
webapp/   Streamlit UI (live local app + static public demo gallery)
```

`CLAUDE.md` has the full domain rules and design rationale this project was
built against.
