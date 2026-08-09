"""Phase 2 entry point: pose estimation, jump/contact detection for every
tracked person (not just one "target hitter" -- a match clip usually contains
several different plays with different hitters; see pose/README.md).

Usage:
    .venv/bin/python pose/run.py --video_path data/samples/clip.mp4 \\
        --ball_json output/phase1/clip_ball.json --output_dir output/phase2

Requires Phase 1's ball JSON (contact detection needs the ball trajectory).
Writes <clip>_pose.json and <clip>_pose_overlay.mp4.

Pass --target_track_id to restrict analysis to one specific person (found by
inspecting the overlay video's id= labels) instead of everyone.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm

from config.loader import load_config
from pose.contact_detection import detect_contact
from pose.detector import PoseDetector
from pose.jump_detection import detect_jumps
from videoio.export import write_json
from videoio.overlay import draw_pose_frame
from videoio.video import frame_iterator, open_video, open_writer


def run_pose_detection(video_path, config):
    cap, info = open_video(video_path)
    model_path = os.path.join(os.path.dirname(__file__), "..", config["pose"]["model_path"])
    detector = PoseDetector(model_path, config["pose"]["detection_confidence"])

    pose_by_frame = {}
    for frame_idx, frame in tqdm(
        frame_iterator(cap), total=info.frame_count, desc="Detecting pose"
    ):
        pose_by_frame[frame_idx] = detector.detect(frame)
    cap.release()
    return pose_by_frame, info


def build_series_by_track(pose_by_frame):
    series_by_track = {}
    for frame_idx, people in pose_by_frame.items():
        for person in people:
            series_by_track.setdefault(person["track_id"], {})[frame_idx] = person["keypoints"]
    return series_by_track


def events_by_frame(hitters):
    """{frame_idx: {track_id: label}} for overlay rendering."""
    events = {}
    for hitter in hitters:
        track_id = hitter["track_id"]
        for jump in hitter["jumps"]:
            events.setdefault(jump["takeoff_frame"], {})[track_id] = "TAKEOFF"
            events.setdefault(jump["landing_frame"], {})[track_id] = (
                "LANDING" if jump["landing_confirmed"] else "LANDING?"
            )
            if jump.get("contact"):
                events.setdefault(jump["contact"]["contact_frame"], {})[track_id] = "CONTACT"
    return events


def write_overlay_video(video_path, info, pose_by_frame, jumper_ids, hitters, output_path):
    cap, _ = open_video(video_path)
    writer = open_writer(output_path, info)
    events = events_by_frame(hitters)

    for frame_idx, frame in tqdm(
        frame_iterator(cap), total=info.frame_count, desc="Writing overlay"
    ):
        people = pose_by_frame.get(frame_idx, [])
        frame_events = events.get(frame_idx, {})
        writer.write(draw_pose_frame(frame, frame_idx, people, jumper_ids, 0.0, frame_events))
    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--ball_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--target_track_id", type=int, default=None,
        help="Restrict analysis to one person instead of everyone tracked",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    os.makedirs(args.output_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(args.video_path))[0]

    with open(args.ball_json) as f:
        ball_data = json.load(f)
    ball_by_frame = {
        p["frame"]: p for p in ball_data["ball"]["trajectory"] if p["x"] is not None
    }

    pose_by_frame, info = run_pose_detection(args.video_path, config)
    series_by_track = build_series_by_track(pose_by_frame)

    min_kp_conf = config["pose"]["min_keypoint_confidence"]
    min_track_frames = config["pose"]["min_track_frames"]
    track_ids = (
        [args.target_track_id] if args.target_track_id is not None
        else [tid for tid, series in series_by_track.items() if len(series) >= min_track_frames]
    )

    hitters = []
    for track_id in track_ids:
        series = series_by_track.get(track_id, {})
        jumps = detect_jumps(series, info.frame_count, config)
        for jump in jumps:
            jump["contact"] = detect_contact(
                jump, series, ball_by_frame, min_kp_conf,
                config["pose"]["max_contact_distance_px"],
            )
        if jumps:
            hitters.append({
                "track_id": track_id,
                "frames_tracked": len(series),
                "jumps": jumps,
            })

    total_jumps = sum(len(h["jumps"]) for h in hitters)
    print(f"Analyzed {len(track_ids)} tracked people, found {total_jumps} jump(s) across {len(hitters)} of them")
    for h in hitters:
        print(f"  track_id={h['track_id']}: {h['jumps']}")

    jumper_ids = {h["track_id"] for h in hitters}
    overlay_path = os.path.join(args.output_dir, f"{clip_name}_pose_overlay.mp4")
    write_overlay_video(args.video_path, info, pose_by_frame, jumper_ids, hitters, overlay_path)

    result = {
        "video": {
            "path": args.video_path,
            "fps": info.fps,
            "frame_count": info.frame_count,
            "width": info.width,
            "height": info.height,
        },
        "pose": {"hitters": hitters},
    }
    json_path = os.path.join(args.output_dir, f"{clip_name}_pose.json")
    write_json(result, json_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {overlay_path}")


if __name__ == "__main__":
    main()
