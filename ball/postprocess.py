"""Gap interpolation and apex-frame detection on a raw per-frame ball track.

Coordinate convention: image y increases downward, so the ball's highest point
in space is the *minimum* y, not the maximum. Never flip this.
"""


def interpolate_gaps(raw_track, max_gap_frames):
    """raw_track: list of {"frame", "visibility", "x", "y"} for every frame, in
    order, x/y meaningless when visibility is 0.

    Fills gaps of up to max_gap_frames between two real detections by linear
    interpolation. Longer gaps, and gaps missing a detection on either side
    (start/end of the clip), are left as untracked.

    Returns the same shape with "interpolated" added, and "x"/"y" set to None
    where still untracked.
    """
    n = len(raw_track)
    track = [dict(p) for p in raw_track]
    for p in track:
        p["interpolated"] = False
        if p["visibility"] != 1:
            p["x"] = None
            p["y"] = None

    i = 0
    while i < n:
        if track[i]["x"] is not None:
            i += 1
            continue
        gap_start = i
        while i < n and track[i]["x"] is None:
            i += 1
        gap_end = i  # exclusive; track[gap_end] is the next real point, if any
        gap_len = gap_end - gap_start
        if gap_start == 0 or gap_end == n:
            continue  # unbounded on one side -- can't interpolate
        if gap_len > max_gap_frames:
            continue

        before, after = track[gap_start - 1], track[gap_end]
        span = gap_end - (gap_start - 1)
        for j in range(gap_start, gap_end):
            t = (j - (gap_start - 1)) / span
            track[j]["x"] = before["x"] + t * (after["x"] - before["x"])
            track[j]["y"] = before["y"] + t * (after["y"] - before["y"])
            track[j]["interpolated"] = True

    return track


def _smooth(values, window):
    if window <= 1:
        return values
    smoothed = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def tracked_segments(track):
    """Yields (start_idx, end_idx) [inclusive] of contiguous frames that have a
    position (real or interpolated), i.e. one flight arc each."""
    n = len(track)
    i = 0
    while i < n:
        if track[i]["x"] is None:
            i += 1
            continue
        start = i
        while i < n and track[i]["x"] is not None:
            i += 1
        yield start, i - 1


def detect_apex_frames(track, smoothing_window, min_flight_segment_frames):
    """Returns a sorted list of frame indices, one per flight segment long
    enough to trust -- the frame where the ball is highest (min y) in that
    segment's smoothed trajectory."""
    apex_frames = []
    for start, end in tracked_segments(track):
        length = end - start + 1
        if length < min_flight_segment_frames:
            continue
        ys = [track[i]["y"] for i in range(start, end + 1)]
        smoothed = _smooth(ys, smoothing_window)
        offset = min(range(length), key=lambda k: smoothed[k])
        apex_frames.append(start + offset)
    return apex_frames
