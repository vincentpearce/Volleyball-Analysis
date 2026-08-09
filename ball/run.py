"""Phase 1 entry point: ball detection, gap interpolation, apex marking.

Usage:
    .venv/bin/python ball/run.py --video_path data/samples/clip.mp4 --output_dir output/phase1

Writes <output_dir>/<clip_name>_ball.json and <output_dir>/<clip_name>_overlay.mp4.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm

from ball.model import BallDetector
from ball.postprocess import detect_apex_frames, interpolate_gaps
from config.loader import load_config
from videoio.export import write_json
from videoio.overlay import draw_ball_frame
from videoio.video import frame_iterator, open_video, open_writer


def track_ball(video_path, config):
    cap, info = open_video(video_path)
    model_path = os.path.join(
        os.path.dirname(__file__), "..", config["ball"]["model_path"]
    )
    detector = BallDetector(model_path, config["ball"]["min_detection_confidence"])

    raw_track = []
    for frame_idx, frame in tqdm(
        frame_iterator(cap), total=info.frame_count, desc="Detecting ball"
    ):
        point = detector.detect(frame)
        if point is None:
            raw_track.append({"frame": frame_idx, "visibility": 0, "x": None, "y": None})
        else:
            x, y = point
            raw_track.append({"frame": frame_idx, "visibility": 1, "x": x, "y": y})
    cap.release()
    return raw_track, info


def write_overlay_video(video_path, info, track, apex_frames, output_path):
    cap, _ = open_video(video_path)
    writer = open_writer(output_path, info)
    track_by_frame = {p["frame"]: p for p in track if p["x"] is not None}
    apex_set = set(apex_frames)

    for frame_idx, frame in tqdm(
        frame_iterator(cap), total=info.frame_count, desc="Writing overlay"
    ):
        writer.write(draw_ball_frame(frame, frame_idx, track_by_frame, apex_set))
    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    os.makedirs(args.output_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(args.video_path))[0]

    raw_track, info = track_ball(args.video_path, config)
    track = interpolate_gaps(raw_track, config["ball"]["max_gap_interpolation_frames"])
    apex_frames = detect_apex_frames(
        track,
        config["ball"]["trajectory_smoothing_window"],
        config["ball"]["min_flight_segment_frames"],
    )

    detected = sum(1 for p in raw_track if p["visibility"] == 1)
    interpolated = sum(1 for p in track if p["interpolated"])
    print(
        f"Detected: {detected}/{info.frame_count} frames "
        f"({detected/info.frame_count:.1%}), "
        f"interpolated: {interpolated}, apex frames: {len(apex_frames)}"
    )

    output_path = os.path.join(args.output_dir, f"{clip_name}_overlay.mp4")
    write_overlay_video(args.video_path, info, track, apex_frames, output_path)

    result = {
        "video": {
            "path": args.video_path,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "width": info.width,
            "height": info.height,
        },
        "ball": {
            "trajectory": track,
            "apex_frames": apex_frames,
        },
    }
    json_path = os.path.join(args.output_dir, f"{clip_name}_ball.json")
    write_json(result, json_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
