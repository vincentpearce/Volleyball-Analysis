"""Alternative to classify.py's heuristic rules: classifies each ball-flight
touch-segment using the YOLO11s-cls model fine-tuned on Roboflow's
professional-footage dataset (events/prepare_action_dataset.py).

Scored 94.7% on that dataset's own held-out test set but only ~11.5% on our
own casual/club-level footage (see events/README.md) -- a real domain gap.
Use this path for professional/high-level footage that looks more like the
model's training data; use the heuristic (classify.py, the default) for
footage like this project's own sample clips.

Segmentation (touch_detection.py) is shared with the heuristic path so
results are comparable -- only the per-segment classification differs.
"""

import math
import os

import cv2
from ultralytics import YOLO

from ball.postprocess import tracked_segments
from events.touch_detection import split_segment

CROP_MARGIN_FRAC = 0.15
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "action_classifier.pt")


def _find_touching_player(ball, people, min_conf):
    best, best_dist = None, float("inf")
    for p in people:
        for name in ("left_wrist", "right_wrist"):
            x, y, conf = p["keypoints"][name]
            if conf < min_conf:
                continue
            d = math.hypot(x - ball["x"], y - ball["y"])
            if d < best_dist:
                best, best_dist = p, d
    return best


def _crop_person(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * CROP_MARGIN_FRAC, bh * CROP_MARGIN_FRAC
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return frame[y1:y2, x1:x2]


def classify_ball_track_learned(track, video_path, pose_detector, config, fps, model_path=None):
    """track: the interpolated per-frame ball trajectory (list indexed by frame).
    pose_detector: a fresh pose.detector.PoseDetector (stateless use here --
    each call seeks to an arbitrary frame, so persistent tracking IDs across
    calls aren't meaningful and are ignored)."""
    classifier = YOLO(model_path or DEFAULT_MODEL_PATH)
    min_len = config["ball"]["min_flight_segment_frames"]
    min_conf = config["pose"]["min_keypoint_confidence"]
    cap = cv2.VideoCapture(video_path)

    events = []
    for seg_start, seg_end in tracked_segments(track):
        if seg_end - seg_start + 1 < min_len:
            continue
        for start, end in split_segment(track, seg_start, seg_end, config):
            if end - start + 1 < min_len:
                continue

            ball = track[start]
            if ball["x"] is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            ret, frame = cap.read()
            if not ret:
                continue

            people = pose_detector.detect(frame)
            target = _find_touching_player(ball, people, min_conf)
            if target is None:
                continue
            crop = _crop_person(frame, target["bbox"])
            if crop.size == 0:
                continue

            result = classifier.predict(crop, verbose=False)[0]
            predicted = result.names[result.probs.top1]
            confidence = float(result.probs.top1conf)
            if predicted == "none":
                continue

            events.append({
                "type": predicted,
                "start_frame": start,
                "end_frame": end,
                "start_time": start / fps,
                "end_time": end / fps,
                "confidence": confidence,
            })

    cap.release()
    return events
