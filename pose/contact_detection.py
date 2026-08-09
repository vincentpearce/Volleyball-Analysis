"""Contact-frame detection: closest approach between either wrist and the ball,
searched within a jump's [takeoff, landing] window (contact happens mid-air for
a spike/block, which is the case this targets).

Only using closest-approach distance for now, not the "sharp ball-velocity
change" secondary signal CLAUDE.md mentions -- worth adding if closest-approach
alone proves too noisy on real footage.
"""

import math


def detect_contact(jump, target_series, ball_by_frame, min_keypoint_confidence, max_contact_distance_px):
    best = None
    for frame_idx in range(jump["takeoff_frame"], jump["landing_frame"] + 1):
        keypoints = target_series.get(frame_idx)
        ball = ball_by_frame.get(frame_idx)
        if keypoints is None or ball is None:
            continue
        for hand, name in (("left", "left_wrist"), ("right", "right_wrist")):
            x, y, conf = keypoints[name]
            if conf < min_keypoint_confidence:
                continue
            dist = math.hypot(x - ball["x"], y - ball["y"])
            if best is None or dist < best["distance_px"]:
                best = {"contact_frame": frame_idx, "distance_px": dist, "hand": hand}
    if best is not None and best["distance_px"] > max_contact_distance_px:
        return None
    return best
