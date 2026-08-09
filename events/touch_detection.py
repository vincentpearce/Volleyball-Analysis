"""Detects touch points (hits) within a continuously-tracked ball trajectory
by finding discontinuities in velocity -- a real hit changes the ball's
velocity abruptly (new direction and/or speed); free flight under gravity
alone doesn't, including at the apex. Under gravity, horizontal velocity is
constant and vertical velocity changes *linearly*, so two symmetric points
around a natural apex have nearly equal speed and, whenever the trajectory
has real horizontal travel (which most touches impart), nearly the same
direction too -- an apex is a smooth minimum, not a kink. A real touch
produces both a sharper angle change and a speed jump than that.

This is the piece CLAUDE.md's "velocity changes, direction reversals" heuristic
actually needs -- events/classify.py previously only split on ball-tracking
gaps, which don't reliably align with real touches when detection recall is
high (the ball stays trackable straight through a hit).
"""

import math


def _velocity(track, i, j):
    pi, pj = track[i], track[j]
    dt = j - i
    return (pj["x"] - pi["x"]) / dt, (pj["y"] - pi["y"]) / dt


def _dedupe_nearby(frames, min_gap):
    """Collapse clusters of adjacent flagged frames (a real kink often trips
    several frames in a row near it) into one representative frame each."""
    if not frames:
        return []
    groups = [[frames[0]]]
    for f in frames[1:]:
        if f - groups[-1][-1] <= min_gap:
            groups[-1].append(f)
        else:
            groups.append([f])
    return [g[len(g) // 2] for g in groups]


def detect_touch_frames(track, start, end, config):
    ev = config["events"]
    window = ev["touch_window_frames"]
    angle_threshold = ev["touch_angle_threshold_deg"]
    speed_ratio_threshold = ev["touch_speed_ratio_threshold"]
    min_speed = ev["touch_min_speed_px"]

    flagged = []
    for i in range(start + window, end - window + 1):
        vb = _velocity(track, i - window, i)
        va = _velocity(track, i, i + window)
        speed_b, speed_a = math.hypot(*vb), math.hypot(*va)
        if speed_b < min_speed or speed_a < min_speed:
            continue

        dot = vb[0] * va[0] + vb[1] * va[1]
        cos_angle = max(-1.0, min(1.0, dot / (speed_b * speed_a)))
        angle_deg = math.degrees(math.acos(cos_angle))
        speed_ratio = max(speed_a, speed_b) / min(speed_a, speed_b)

        if angle_deg >= angle_threshold or speed_ratio >= speed_ratio_threshold:
            flagged.append(i)

    return _dedupe_nearby(flagged, window)


def split_segment(track, start, end, config):
    """Splits [start, end] into sub-ranges at detected touch points. Returns
    a list of (sub_start, sub_end) inclusive pairs covering the whole range."""
    touches = detect_touch_frames(track, start, end, config)
    bounds = [start] + touches + [end]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
