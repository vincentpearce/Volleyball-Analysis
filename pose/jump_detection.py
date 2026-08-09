"""Takeoff/landing detection from a target player's ankle trajectory.

Per CLAUDE.md: establish a ground baseline from the stance just before a jump,
then find takeoff/landing as the feet crossing away from / back to that
baseline. This is offline batch analysis (not real-time), so the baseline is a
rolling median over ground_baseline_window centered on each frame -- as long as
the window is wider than a real jump's duration, standing frames dominate the
window and the median stays close to true ground level even while a jump is
in progress, without needing strict causality.

Coordinate convention: y increases downward, so a jump makes ankle y decrease.
deviation = baseline_y - ankle_y is positive while airborne.
"""

import statistics


def _ankle_y_series(target_series, frame_count, min_keypoint_confidence):
    """target_series: {frame_idx: keypoints dict}. Returns a list (len=frame_count)
    of ankle y (average of left/right, whichever are confident) or None."""
    series = [None] * frame_count
    for frame_idx, keypoints in target_series.items():
        ys = []
        for side in ("left_ankle", "right_ankle"):
            x, y, conf = keypoints[side]
            if conf >= min_keypoint_confidence:
                ys.append(y)
        if ys:
            series[frame_idx] = sum(ys) / len(ys)
    return series


def _smooth(series, window):
    half = window // 2
    smoothed = [None] * len(series)
    for i in range(len(series)):
        if series[i] is None:
            continue
        lo, hi = max(0, i - half), min(len(series), i + half + 1)
        vals = [v for v in series[lo:hi] if v is not None]
        smoothed[i] = sum(vals) / len(vals)
    return smoothed


def _rolling_median_baseline(series, window):
    half = window // 2
    baseline = [None] * len(series)
    for i in range(len(series)):
        if series[i] is None:
            continue
        lo, hi = max(0, i - half), min(len(series), i + half + 1)
        vals = [v for v in series[lo:hi] if v is not None]
        baseline[i] = statistics.median(vals)
    return baseline


def detect_jumps(target_series, frame_count, config):
    min_conf = config["pose"]["min_keypoint_confidence"]
    smoothing_window = config["pose"]["smoothing_window"]
    baseline_window = config["pose"]["ground_baseline_window"]
    threshold = config["pose"]["jump_height_threshold_px"]
    min_jump_frames = config["pose"]["min_jump_frames"]

    raw = _ankle_y_series(target_series, frame_count, min_conf)
    smoothed = _smooth(raw, smoothing_window)
    baseline = _rolling_median_baseline(raw, baseline_window)

    deviation = [
        (baseline[i] - smoothed[i]) if (smoothed[i] is not None and baseline[i] is not None)
        else None
        for i in range(frame_count)
    ]

    # A track ending while still elevated (deviation[i] becomes None, not a
    # genuine return to baseline) usually means the person was lost mid-air --
    # fast swing motion during a real spike is exactly when pose tracking is
    # most likely to fail. That's different from a segment that actually
    # dips back below threshold within the tracked window: the latter is
    # good evidence of a complete, real jump; the former is a jump we only
    # partially observed. Only the complete case is held to min_jump_frames --
    # an incomplete one is kept regardless of its (possibly short) observed
    # length, but flagged as such since its landing frame is a guess.
    jumps = []
    airborne_start = None
    for i in range(frame_count):
        d = deviation[i]
        is_airborne = d is not None and d >= threshold
        data_ran_out = d is None and i > 0 and deviation[i - 1] is not None
        if is_airborne and airborne_start is None:
            airborne_start = i
        elif not is_airborne and airborne_start is not None:
            takeoff, landing = airborne_start, i - 1
            complete = not data_ran_out
            if not complete or landing - takeoff + 1 >= min_jump_frames:
                jumps.append({
                    "takeoff_frame": takeoff, "landing_frame": landing,
                    "landing_confirmed": complete,
                })
            airborne_start = None
    if airborne_start is not None:
        jumps.append({
            "takeoff_frame": airborne_start, "landing_frame": frame_count - 1,
            "landing_confirmed": False,
        })

    return jumps
