"""Heuristic event classification (serve/set/spike/block/dig) per ball touch
-- CLAUDE.md's explicit MVP approach (heuristics from trajectory + pose before
any learned model). Unvalidated against labeled ground truth -- see
events/README.md.

Each continuously-tracked run of the ball (ball/postprocess.py's
tracked_segments) is first split at detected touch points (events/
touch_detection.py -- velocity discontinuities, not just tracking gaps: a
real hit doesn't require the ball to have gone briefly untracked). Each
resulting sub-segment is one touch's resulting arc. Classification rules, in
priority order:

1. Segment starts a new rally (long gap since the previous *tracked* segment
   ended, or it's the first segment in the clip) -> serve. Sub-segments
   produced by a touch split (not a tracking gap) never qualify -- they're
   continuous with what came before by construction.
2. Segment start coincides with a pose-detected jump's contact frame -> a
   jumping hit. Large horizontal displacement (ball sent across court) ->
   spike; small displacement (deflected near where it was touched) -> block.
3. Otherwise: a large vertical rise to the apex (controlled, high arc) ->
   set; anything flatter -> dig (the catch-all for defensive touches).
"""

from ball.postprocess import tracked_segments
from events.touch_detection import split_segment


def _near_jump_contact(start, end, hitters, tolerance_frames):
    """True if a pose-detected jump's contact frame falls within this segment
    (+/- tolerance) -- checked against the whole segment range, not just its
    start, since touch_detection's velocity-kink estimate and contact_detection's
    wrist-proximity estimate are independent signals that won't necessarily
    agree on the exact frame."""
    for hitter in hitters:
        for jump in hitter["jumps"]:
            contact = jump.get("contact")
            if contact and start - tolerance_frames <= contact["contact_frame"] <= end + tolerance_frames:
                return True
    return False


def _segment_features(track, start, end, gap_before, hitters, config):
    ys = [track[i]["y"] for i in range(start, end + 1)]
    apex_offset = min(range(len(ys)), key=lambda k: ys[k])
    return {
        "start_frame": start,
        "end_frame": end,
        "apex_frame": start + apex_offset,
        "horizontal_displacement": track[end]["x"] - track[start]["x"],
        "apex_rise": track[start]["y"] - ys[apex_offset],  # y decreases upward
        "gap_before": gap_before,
        "near_jump_contact": _near_jump_contact(
            start, end, hitters, config["events"]["jump_contact_tolerance_frames"]
        ),
    }


def classify_segment(features, config):
    ev = config["events"]

    if features["gap_before"] is None or features["gap_before"] >= ev["serve_gap_frames"]:
        return "serve", 0.6

    if features["near_jump_contact"]:
        if abs(features["horizontal_displacement"]) >= ev["spike_min_horizontal_displacement_px"]:
            return "spike", 0.6
        return "block", 0.4

    if features["apex_rise"] >= ev["set_min_apex_rise_px"]:
        return "set", 0.5

    return "dig", 0.3


def classify_ball_track(track, hitters, config, fps):
    min_len = config["ball"]["min_flight_segment_frames"]
    events = []
    prev_tracked_end = None
    for seg_start, seg_end in tracked_segments(track):
        if seg_end - seg_start + 1 < min_len:
            prev_tracked_end = seg_end
            continue

        for i, (start, end) in enumerate(split_segment(track, seg_start, seg_end, config)):
            if end - start + 1 < min_len:
                continue
            if i == 0:
                gap_before = (start - prev_tracked_end - 1) if prev_tracked_end is not None else None
            else:
                gap_before = 0  # touch-split, not a tracking gap -- continuous with what came before
            features = _segment_features(track, start, end, gap_before, hitters, config)
            event_type, confidence = classify_segment(features, config)
            events.append({
                "type": event_type,
                "start_frame": start,
                "end_frame": end,
                "start_time": start / fps,
                "end_time": end / fps,
                "confidence": confidence,
            })
        prev_tracked_end = seg_end
    return events
