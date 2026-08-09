"""Draws the ball trajectory (recent trail + apex markers) onto video frames."""

import cv2

BALL_COLOR = (0, 0, 255)  # red, BGR
INTERPOLATED_COLOR = (255, 128, 0)  # orange, for gap-filled points
APEX_COLOR = (0, 255, 255)  # yellow
TRAIL_LENGTH = 8


def draw_ball_frame(frame, frame_idx, track_by_frame, apex_frames_set):
    """track_by_frame: {frame_idx: {"x":, "y":, "interpolated": bool}}"""
    out = frame.copy()

    for i in range(max(0, frame_idx - TRAIL_LENGTH), frame_idx + 1):
        point = track_by_frame.get(i)
        if point is None:
            continue
        color = INTERPOLATED_COLOR if point["interpolated"] else BALL_COLOR
        radius = 8 if i == frame_idx else 4
        cv2.circle(out, (int(point["x"]), int(point["y"])), radius, color, -1)

    point = track_by_frame.get(frame_idx)
    if point is not None and frame_idx in apex_frames_set:
        cv2.circle(out, (int(point["x"]), int(point["y"])), 14, APEX_COLOR, 2)
        cv2.putText(
            out, "APEX", (int(point["x"]) + 16, int(point["y"])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, APEX_COLOR, 2, cv2.LINE_AA,
        )

    cv2.putText(
        out, f"Frame: {frame_idx}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out


JUMPER_COLOR = (0, 255, 0)  # green -- has at least one detected jump somewhere in the clip
OTHER_COLOR = (128, 128, 128)  # gray
EVENT_COLOR = (0, 255, 255)  # yellow

SKELETON = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


def _draw_skeleton(frame, keypoints, color, min_conf):
    for a, b in SKELETON:
        xa, ya, ca = keypoints[a]
        xb, yb, cb = keypoints[b]
        if ca >= min_conf and cb >= min_conf:
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)
    for x, y, c in keypoints.values():
        if c >= min_conf:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)


def draw_pose_frame(frame, frame_idx, people, jumper_ids, min_keypoint_confidence, events_by_track=None):
    """events_by_track: {track_id: "TAKEOFF"|"LANDING"|"CONTACT"} for this frame."""
    events_by_track = events_by_track or {}
    out = frame.copy()
    for person in people:
        track_id = person["track_id"]
        is_jumper = track_id in jumper_ids
        color = JUMPER_COLOR if is_jumper else OTHER_COLOR
        _draw_skeleton(out, person["keypoints"], color, min_keypoint_confidence)
        x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 if is_jumper else 1)
        cv2.putText(
            out, f"id={track_id}", (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
        label = events_by_track.get(track_id)
        if label:
            cv2.putText(
                out, f"{label} id={track_id}", (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, EVENT_COLOR, 2, cv2.LINE_AA,
            )

    cv2.putText(
        out, f"Frame: {frame_idx}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out
