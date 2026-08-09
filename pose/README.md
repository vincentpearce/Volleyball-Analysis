# Pose estimation & jump/contact detection (Phase 2)

Model: YOLO11-pose, medium variant (`pose/weights/yolo11m-pose.pt`), off-the-shelf
COCO weights, not fine-tuned. The small variant (yolo11s-pose) was tried first and
completely missed a real, camera-visible hitter -- players far from camera can be
too small for it to detect a person at all, even at very low confidence. The medium
model picks up players the small one misses; hasn't been tested against the large
variant yet, which would be the next thing to try if distant players are still
getting missed on other footage.

YOLO11-pose was chosen over MediaPipe (less proven multi-person support) and
RTMPose/MMPose (fragile non-CUDA dependency stack -- exactly the kind of
multi-hour setup fight already hit once with TensorFlow-Metal in Phase 1).

## Design: every tracked person, not one "target hitter"

CLAUDE.md describes selecting a single target hitter per clip. That assumption
broke on real footage: a 30-second match clip contains several different plays,
each with a different hitter -- checking wrist-to-ball proximity at every ball
apex frame showed a different closest person almost every time. There's no
single correct "target" for a clip like that.

So `pose/run.py` runs jump + contact detection on *every* sufficiently-tracked
person (`pose.min_track_frames`) and reports whichever ones have at least one
detected jump, rather than picking one upfront. Pass `--target_track_id` to
restrict to one specific person if you already know who you care about (found
by inspecting the overlay video's `id=` labels).

## How jump detection works

Ankle y-trajectory (COCO keypoints, y increases downward) is compared against
a rolling-median "ground baseline" (`pose.ground_baseline_window` frames wide
-- wide enough that standing frames dominate the window even mid-jump).
`baseline - ankle_y >= jump_height_threshold_px` marks a frame as airborne;
a takeoff/landing pair is recorded when that state starts/ends.

**Track loss mid-jump is handled specially.** A fast spike swing is exactly
when pose tracking is most likely to fail (motion blur, self-occlusion) --
in testing, the real hitter's track ended 2 frames after crossing the jump
threshold, well short of `min_jump_frames`. A jump cut short by the track
literally ending (not by the deviation genuinely dropping back to baseline)
is kept regardless of length, but marked `"landing_confirmed": false` so
downstream code knows that frame is a guess, not an observed landing.

## Known limitations (MVP, not yet fully validated)

- No fine-tuning, no labeled ground truth for takeoff/landing/contact frames
  the way `ball/`'s thresholds were validated -- `jump_height_threshold_px`
  and `min_jump_frames` are unverified guesses, only checked against one real
  jump so far.
- `jump_height_threshold_px` is a pixel threshold, not a real-world unit --
  scale-dependent on camera distance/resolution.
- Track IDs can churn on occlusion (ByteTrack losing/reacquiring a person may
  assign a new ID) -- a dropped/reassigned ID mid-jump leaves gaps in that
  jump's data rather than being bridged across the ID change.
- Contact detection only uses closest wrist-to-ball distance, not the
  secondary "sharp ball-velocity change" signal CLAUDE.md mentions.
- `landing_confirmed: false` jumps have a genuinely unknown landing frame --
  the recorded one is just "last frame we saw them," not a real detection.
  Downstream consumers (Phase 4's apex/offset math) should treat these as
  lower-confidence.

Always check the overlay video against what a human sees happening at that
timestamp before trusting these numbers -- same lesson as every other phase
so far: a screenshot is easy to misread (ceiling lights, blockers vs.
attackers, a small distant player a model doesn't even see).
