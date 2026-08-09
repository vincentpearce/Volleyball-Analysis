"""One-off check: does the trained action classifier (trained on Roboflow's
Graz Uni footage) hold up on our own camera's footage, not just Roboflow's
held-out test set? Runs it on the player closest to the ball at each segment
we already have human labels for (from events/label_events.py's CSV output).

Usage:
    .venv/bin/python events/evaluate_classifier.py \\
        --video_path data/samples/clip.mp4 --ball_json output/phase1/clip_ball.json \\
        --labeled_csv output/phase3/clip_events_labeled.csv
"""

import argparse
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from ultralytics import YOLO

from config.loader import load_config
from pose.detector import PoseDetector

CROP_MARGIN_FRAC = 0.15


def crop_person(frame, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * CROP_MARGIN_FRAC, bh * CROP_MARGIN_FRAC
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return frame[y1:y2, x1:x2]


def find_touching_player(ball, people, min_conf=0.3):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--ball_json", required=True)
    parser.add_argument("--labeled_csv", required=True)
    parser.add_argument("--classifier_path", default="events/weights/action_classifier.pt")
    args = parser.parse_args()

    config = load_config()
    with open(args.ball_json) as f:
        ball_data = json.load(f)
    ball_by_frame = {p["frame"]: p for p in ball_data["ball"]["trajectory"] if p["x"] is not None}

    with open(args.labeled_csv) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["human_label"] not in ("merged", "unsure")]

    pose_detector = PoseDetector(config["pose"]["model_path"], config["pose"]["detection_confidence"])
    classifier = YOLO(args.classifier_path)
    cap = cv2.VideoCapture(args.video_path)

    correct = 0
    for row in rows:
        # The touch/action happens at the segment's START, not its middle --
        # segments are ball-flight-after-a-touch by construction (see
        # events/touch_detection.py), so start_frame is where the hit is.
        target_frame = int(row["start_frame"])
        ball = ball_by_frame.get(target_frame)
        if ball is None:
            for offset in range(1, 10):
                ball = ball_by_frame.get(target_frame + offset) or ball_by_frame.get(target_frame - offset)
                if ball:
                    break
        if ball is None:
            print(f"frame {target_frame}: no ball data nearby, skipping")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue
        people = pose_detector.detect(frame)
        target = find_touching_player(ball, people)
        if target is None:
            print(f"frame {target_frame}: no player found near ball, skipping")
            continue

        crop = crop_person(frame, target["bbox"])
        if crop.size == 0:
            continue
        result = classifier.predict(crop, verbose=False)[0]
        predicted = result.names[result.probs.top1]
        human = row["human_label"]
        match = "OK" if predicted == human else "  "
        if predicted == human:
            correct += 1
        print(f"[{match}] frame {target_frame}: predicted={predicted:6s} human={human:6s} (heuristic said {row['predicted_type']})")

    cap.release()
    print(f"\nAccuracy: {correct}/{len(rows)} ({correct/len(rows):.1%})")


if __name__ == "__main__":
    main()
