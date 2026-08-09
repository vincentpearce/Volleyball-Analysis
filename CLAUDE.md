# Volleyball Timing Analyzer — Project Context

This file is read automatically at the start of every Claude Code session. It defines
what we're building, the domain rules that are easy to get wrong, and how I want you
to work. Read it before proposing changes.

---

## What this project does

A computer-vision app that analyses volleyball video. Given a video file, it:

1. Tracks the ball across frames and finds the ball's apex (highest point).
2. Tracks a target hitter's body and detects their jump.
3. Timestamps key events (serve, set, spike/attack, block, dig).
4. Assesses whether the hitter contacts the ball at the correct time (see the
   **Timing** section — this is the core deliverable, not an afterthought).

Output is (a) structured JSON of events/measurements and (b) an annotated output video.

---

## How I want you to work

- **Build phase by phase. MVP first.** Do not attempt ball tracking + pose + event
  detection + timing all at once. Get each phase runnable and validated before moving on.
- **Explain model/architecture choices** before committing to them, especially the
  accuracy-vs-speed tradeoff. Ask before large structural decisions.
- **Prefer well-maintained open-source models** over training from scratch. Flag clearly
  if a phase genuinely requires fine-tuning and what data that would need.
- **Start read-only.** When exploring, explain the plan in words first; don't write code
  until the approach is agreed.
- **Every phase ends with a validation step** — how to check the output on a sample clip.

---

## Environment & conventions

- Python, runnable locally on a single GPU (CPU fallback at reduced fps).
- Input: mp4/mov, single mostly-fixed camera.
- **Never hardcode fps** — read it from the video. All timing math converts frames↔seconds
  using the video's actual fps.
- **Config-driven thresholds.** The timing tolerance window, smoothing windows, and
  detection confidences live in a config file, not scattered as magic numbers.
- Modular layout — keep stages separable so models can be swapped:
  ```
  ball/     ball detection + tracking + apex
  pose/     pose estimation + takeoff/landing/contact detection
  events/   event timestamping
  timing/   the timing assessment
  videoio/  video read/write, annotation overlay, JSON export
            (named videoio/, not io/ -- Python's stdlib io isn't a package,
            so "from io.x import y" always raises ModuleNotFoundError)
  config/   thresholds and paths
  ```

---

## Domain rules (get these right)

### Coordinate convention
Image **y increases downward**. The ball's or a body part's *highest point in space* is
the **minimum** y value. Detect apexes as local minima of a smoothed y-signal, never maxima.

### Jump apex = temporal midpoint of the jump
A jump follows symmetric projectile motion, so the highest point of the jump occurs
exactly **halfway in time between takeoff and landing**:

```
apex_frame = (takeoff_frame + landing_frame) / 2
```

This is the definition we use — do **not** compute jump apex by tracking the hip/ankle
vertical position and looking for a peak. The midpoint method only needs two discrete
events, which is more robust than tracking a noisy smooth signal.

- **Takeoff** = the frame the hitter's feet leave the ground.
- **Landing** = the frame the feet return to the ground.
- Detect these from the foot/ankle keypoints. Establish a "ground baseline" for the feet
  from the stance in the frames just before the jump, then detect takeoff/landing as the
  feet crossing away from / back to that baseline (use vertical velocity of the ankles to
  disambiguate). Surface the detected takeoff and landing frames so I can verify them.

### Correct hitting timing (core metric)
The hitter has **correct timing** if ball **contact** happens at the jump apex.

```
timing_offset = contact_frame - apex_frame        # in frames
timing_offset_ms = timing_offset / fps * 1000     # in milliseconds
```

- **contact_frame** = the frame the hitting hand meets the ball. Detect from the ball
  trajectory + hitting-wrist keypoint (closest approach / sharp ball-velocity change near
  the hand).
- Classify with a **configurable tolerance window** (e.g. ±N ms, default set in config):
  - `offset ≈ 0` (within tolerance) → **on time**
  - `offset < 0` → **early** (contacting while still rising)
  - `offset > 0` → **late** (contacting on the way down)
- **Always surface the raw numbers** (takeoff, landing, apex, contact frames and the
  offset in ms), not just the label. This is a measurement tool; the early/on-time/late
  label is a threshold on top of the measurements, and I want to validate the underlying
  values.

*Optional secondary signal (future, not MVP):* also check the arm is near full extension
at contact. Keep it separate from the primary midpoint-based metric.

### Events to timestamp
serve, set, spike/attack, block, dig. Output each as
`{ type, start_time, end_time, confidence }`. Start with **heuristics** derived from the
ball trajectory + pose (velocity changes, direction reversals, apex crossings) before
considering a learned temporal action model.

---

## Pipeline (phased)

**Phase 1 — Ball detection & tracking (MVP).**
The ball is small, fast, and motion-blurred, which breaks generic detectors. Evaluate
TrackNet (purpose-built for small fast balls) vs a fine-tuned YOLO + ByteTrack/BoT-SORT.
Output per-frame ball (x, y), interpolate gaps, mark apex frames. Overlay trajectory on
the output video.

**Phase 2 — Pose estimation.**
Add pose (evaluate RTMPose/MMPose, YOLO-pose, or MediaPipe). Track the target hitter,
detect takeoff / landing / contact per the rules above. Handle multiple people — let me
select the target player (nearest to ball at contact, or manual box selection).

**Phase 3 — Event timestamping.**
Heuristic events from trajectory + pose. Output the events JSON.

**Phase 4 — Timing assessment.**
Compute apex from the takeoff/landing midpoint, compute the timing offset against contact,
emit the on-time/early/late label plus raw numbers.

---

## Output schema (target)

```json
{
  "video": { "path": "...", "fps": 0, "frame_count": 0 },
  "events": [
    { "type": "spike", "start_time": 0.0, "end_time": 0.0, "confidence": 0.0 }
  ],
  "hits": [
    {
      "takeoff_frame": 0,
      "landing_frame": 0,
      "apex_frame": 0,
      "contact_frame": 0,
      "timing_offset_ms": 0.0,
      "verdict": "early | on_time | late"
    }
  ]
}
```

---

## Open questions to confirm with me before building

- **Camera:** fixed vs moving/broadcast? (Fixed is the MVP assumption — pixel coords map
  consistently to the court. Flag if footage is broadcast.)
- **Single vs multi-player** target selection preference.
- **Target fps / resolution** of the source clips.
