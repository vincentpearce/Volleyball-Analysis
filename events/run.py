"""Phase 3 entry point: event classification (serve/set/spike/block/dig).

Usage:
    .venv/bin/python events/run.py --video_path data/samples/clip.mp4 \\
        --ball_json output/phase1/clip_ball.json --pose_json output/phase2/clip_pose.json \\
        --output_dir output/phase3

Two classifiers, pick with --classifier:
  heuristic (default) -- rules on trajectory shape + jump correlation.
    Measured at 30.8% on our own footage, below a majority-class baseline.
    Needs --pose_json (jump-contact correlation).
  learned -- YOLO11s-cls fine-tuned on professional match footage (Roboflow).
    94.7% on that footage's own test set, ~11.5% on our club-level clips --
    use this for footage that looks more like professional play, not for
    footage like this project's own samples. Doesn't need --pose_json.
See events/README.md for the full story on both.

Writes <clip>_events.json and <clip>_events_overlay.mp4.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
from tqdm import tqdm

from config.loader import load_config
from events.classify import classify_ball_track
from events.learned_classify import classify_ball_track_learned
from pose.detector import PoseDetector
from videoio.export import write_json
from videoio.overlay import draw_ball_frame
from videoio.video import frame_iterator, open_video, open_writer

EVENT_LABEL_COLOR = (255, 255, 255)


def event_for_frame(frame_idx, events):
    for event in events:
        if event["start_frame"] <= frame_idx <= event["end_frame"]:
            return event
    return None


def write_overlay_video(video_path, info, track, apex_frames, events, output_path):
    cap, _ = open_video(video_path)
    writer = open_writer(output_path, info)
    track_by_frame = {p["frame"]: p for p in track if p["x"] is not None}
    apex_set = set(apex_frames)

    for frame_idx, frame in tqdm(
        frame_iterator(cap), total=info.frame_count, desc="Writing overlay"
    ):
        out = draw_ball_frame(frame, frame_idx, track_by_frame, apex_set)
        event = event_for_frame(frame_idx, events)
        if event:
            label = f"{event['type'].upper()} ({event['confidence']:.1f})"
            cv2.putText(out, label, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, EVENT_LABEL_COLOR, 2, cv2.LINE_AA)
        writer.write(out)
    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--ball_json", required=True)
    parser.add_argument(
        "--pose_json", default=None,
        help="Required for --classifier heuristic; unused for --classifier learned",
    )
    parser.add_argument("--classifier", choices=["heuristic", "learned"], default="heuristic")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    os.makedirs(args.output_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(args.video_path))[0]

    with open(args.ball_json) as f:
        ball_data = json.load(f)

    track = ball_data["ball"]["trajectory"]
    apex_frames = ball_data["ball"]["apex_frames"]
    fps = ball_data["video"]["fps"]

    if args.classifier == "learned":
        pose_detector = PoseDetector(config["pose"]["model_path"], config["pose"]["detection_confidence"])
        events = classify_ball_track_learned(
            track, args.video_path, pose_detector, config, fps,
            model_path=config["events"]["learned_classifier_path"],
        )
    else:
        if not args.pose_json:
            parser.error("--pose_json is required for --classifier heuristic")
        with open(args.pose_json) as f:
            pose_data = json.load(f)
        events = classify_ball_track(track, pose_data["pose"]["hitters"], config, fps)

    print(f"Classified {len(events)} events:")
    for e in events:
        print(f"  {e['type']:6s} frames {e['start_frame']}-{e['end_frame']} "
              f"({e['start_time']:.2f}s-{e['end_time']:.2f}s) confidence={e['confidence']:.1f}")

    cap, info = open_video(args.video_path)
    cap.release()
    overlay_path = os.path.join(args.output_dir, f"{clip_name}_events_overlay.mp4")
    write_overlay_video(args.video_path, info, track, apex_frames, events, overlay_path)

    result = {"video": ball_data["video"], "events": events}
    json_path = os.path.join(args.output_dir, f"{clip_name}_events.json")
    write_json(result, json_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {overlay_path}")


if __name__ == "__main__":
    main()
