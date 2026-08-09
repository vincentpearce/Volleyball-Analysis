# Web UI

Simple Streamlit front-end over the pipeline -- pick a clip, run it, see
results. Not a reimplementation: it shells out to the same CLI scripts
(`ball/run.py`, `pose/run.py`, `events/run.py`, `timing/run.py`) used from
the command line, so it can't drift out of sync with how the pipeline
actually behaves when run manually.

## Run it

```
.venv/bin/streamlit run webapp/app.py
```

Opens at http://localhost:8501.

## How it works

- Pick a clip from `data/samples/` (shortest files listed first) or upload
  one (`data/uploads/`).
- Pick the event classifier: `heuristic` (default, this project's own
  footage) or `learned` (fine-tuned on professional footage -- see
  `events/README.md`).
- "Run analysis" runs all four phases in sequence as subprocesses, showing
  live status per phase with an expandable log. Results (overlay videos,
  metrics, the events table, the timing verdict) render in tabs afterward.

**Keep the browser tab open for the whole run** (a few minutes even for a
30s clip). This is a synchronous design by choice, matching "simple" scope --
closing the tab disconnects the Streamlit session partway through and stops
the pipeline wherever it was.
