"""Review tool for events/classify.py's output: plays each classified segment
back in a loop (with the ball trail overlaid) and asks for the correct label.
Produces ground truth to measure classifier accuracy against -- the same role
imgLabel.py plays for ball position, applied to event types instead.

A static single frame turned out to not give enough motion context for
confident judgment (first version of this tool, most segments came back
"unsure") -- this plays the segment back at roughly real speed, looping
until you press a key.

Usage:
    .venv/bin/python events/label_events.py --video_path data/samples/clip.mp4 \\
        --events_json output/phase3/clip_events.json --ball_json output/phase1/clip_ball.json \\
        --output_csv output/phase3/clip_events_labeled.csv

Controls: s=serve e=set p=spike b=block d=dig m=merged (multiple real events
in this segment) u=unsure q=quit and save what's labeled so far.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2

from videoio.overlay import draw_ball_frame
from videoio.video import open_video

TYPE_KEYS = {
    ord("s"): "serve",
    ord("e"): "set",
    ord("p"): "spike",
    ord("b"): "block",
    ord("d"): "dig",
    ord("m"): "merged",
    ord("u"): "unsure",
}


def annotate(frame, frame_idx, event, i, total):
    out = frame.copy()
    cv2.putText(out, f"Segment {i+1}/{total}: frames {event['start_frame']}-{event['end_frame']} (frame {frame_idx})",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"Predicted: {event['type']} (confidence {event['confidence']:.1f})",
                (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "s=serve e=set p=spike b=block d=dig m=merged u=unsure q=quit",
                (20, out.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--events_json", required=True)
    parser.add_argument("--ball_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    with open(args.events_json) as f:
        events = json.load(f)["events"]
    with open(args.ball_json) as f:
        track = json.load(f)["ball"]["trajectory"]
    track_by_frame = {p["frame"]: p for p in track if p["x"] is not None}

    cap, info = open_video(args.video_path)
    delay_ms = max(1, int(1000 / info.fps))

    labeled = []
    for i, event in enumerate(events):
        start, end = event["start_frame"], event["end_frame"]
        human_label = None

        while human_label is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for frame_idx in range(start, end + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                display = draw_ball_frame(frame, frame_idx, track_by_frame, set())
                display = annotate(display, frame_idx, event, i, len(events))
                cv2.imshow("label_events", display)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    write_csv(args.output_csv, labeled)
                    print(f"Quit early. Wrote {len(labeled)} labels to {args.output_csv}")
                    return
                if key in TYPE_KEYS:
                    human_label = TYPE_KEYS[key]
                    break
            # hold the last frame briefly between loop replays so it's clear
            # it's about to restart, rather than jump-cutting straight back
            if human_label is None:
                cv2.waitKey(300)

        labeled.append({**event, "human_label": human_label})
        print(f"Segment {i+1}: predicted={event['type']}, human={human_label}")

    cap.release()
    cv2.destroyAllWindows()
    write_csv(args.output_csv, labeled)

    scored = [l for l in labeled if l["human_label"] not in ("merged", "unsure")]
    correct = sum(1 for l in scored if l["human_label"] == l["type"])
    if scored:
        print(f"Accuracy: {correct}/{len(scored)} ({correct/len(scored):.1%}) "
              f"(excluding {len(labeled)-len(scored)} merged/unsure segments)")
    print(f"Wrote {args.output_csv}")


def write_csv(path, labeled):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "start_frame", "end_frame", "predicted_type", "confidence", "human_label"
        ])
        writer.writeheader()
        for l in labeled:
            writer.writerow({
                "start_frame": l["start_frame"], "end_frame": l["end_frame"],
                "predicted_type": l["type"], "confidence": l["confidence"],
                "human_label": l["human_label"],
            })


if __name__ == "__main__":
    main()
