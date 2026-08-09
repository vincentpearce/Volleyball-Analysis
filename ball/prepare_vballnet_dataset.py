"""
Converts a video + ball-position CSV (Frame,Visibility,X,Y — the format imgLabel.py
produces) into the frame/CSV layout vball-net's train_v1.py expects:

    third_party/vball-net/data/frames/<mode>/<track_id>/<frame_idx>.png
    third_party/vball-net/data/frames/<mode>/<track_id>_ball.csv

Frames are resized to the model's fixed input size (512x288) and coordinates are
rescaled to match.

Labeling sessions rarely review a clip end-to-end in one pass -- imgLabel.py's fast
jump keys make it easy to sparse-sample or leave stretches untouched, and an
unreviewed frame is indistinguishable in the CSV from a reviewed "no ball visible"
frame (both are Visibility=0). Feeding unreviewed frames to training as confirmed
negatives would teach the model to suppress real detections it never actually saw.

So by default this script auto-segments the CSV into separate tracks, one per
contiguous run of Visibility=1 frames (--min_run_length, default 3, since seq=3
training windows need at least that many consecutive frames). Runs shorter than
that or isolated single clicks are dropped as unusable rather than guessed at.
Each run becomes its own track with its own train/test split, since splitting
tracks across unrelated moments in the clip would train the model to expect
continuity that doesn't exist between them.

Pass --start_frame/--end_frame explicitly to instead process one manually-chosen
range as a single track (e.g. when you know a whole stretch was reviewed,
including confirmed-empty frames within it).
"""

import argparse
import os

import cv2
import pandas as pd

IMG_WIDTH, IMG_HEIGHT = 512, 288
VBALLNET_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "third_party", "vball-net", "data", "frames"
)


def rescale(x, y, orig_w, orig_h):
    new_x = min(int(x * IMG_WIDTH / orig_w), IMG_WIDTH - 1)
    new_y = min(int(y * IMG_HEIGHT / orig_h), IMG_HEIGHT - 1)
    return new_x, new_y


def find_visible_runs(df, min_run_length):
    visible = df[df["Visibility"] == 1]["Frame"].tolist()
    if not visible:
        return []
    runs = []
    start = prev = visible[0]
    for f in visible[1:]:
        if f != prev + 1:
            runs.append((start, prev + 1))
            start = f
        prev = f
    runs.append((start, prev + 1))
    return [(s, e) for s, e in runs if e - s >= min_run_length]


def process(cap, orig_w, orig_h, df, track_id, mode, frame_range):
    out_dir = os.path.join(VBALLNET_DATA_DIR, mode, track_id)
    os.makedirs(out_dir, exist_ok=True)

    start, end = frame_range
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    rows = []
    for out_idx, frame_idx in enumerate(range(start, end)):
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        cv2.imwrite(os.path.join(out_dir, f"{out_idx}.png"), resized)

        row = df.iloc[frame_idx]
        if int(row["Visibility"]) == 1:
            nx, ny = rescale(row["X"], row["Y"], orig_w, orig_h)
            rows.append({"Frame": out_idx, "Visibility": 1, "X": nx, "Y": ny})
        else:
            rows.append({"Frame": out_idx, "Visibility": 0, "X": 0, "Y": 0})

    pd.DataFrame(rows).to_csv(
        os.path.join(VBALLNET_DATA_DIR, mode, f"{track_id}_ball.csv"), index=False
    )
    print(f"[{mode}] {track_id}: wrote {len(rows)} frames -> {out_dir}")


def split_and_process(cap, orig_w, orig_h, df, track_id, start, end, test_frac):
    n = end - start
    split = start + max(1, int(n * (1 - test_frac)))
    if split >= end:
        split = end - 1  # guarantee at least one test frame
    process(cap, orig_w, orig_h, df, track_id, "train", (start, split))
    process(cap, orig_w, orig_h, df, track_id, "test", (split, end))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video_path", required=True)
    p.add_argument("--csv_path", required=True, help="Frame,Visibility,X,Y CSV")
    p.add_argument(
        "--track_id",
        default=None,
        help="Track name prefix. In auto-segment mode this is a prefix "
        "(track_id_0, track_id_1, ...); required with --start_frame/--end_frame.",
    )
    p.add_argument(
        "--test_frac",
        type=float,
        default=0.15,
        help="Fraction of frames (from the end of each range) held out for validation",
    )
    p.add_argument(
        "--start_frame",
        type=int,
        default=None,
        help="Manual mode: first frame (inclusive) of a single reviewed range.",
    )
    p.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="Manual mode: last frame (exclusive) of a single reviewed range.",
    )
    p.add_argument(
        "--min_run_length",
        type=int,
        default=3,
        help="Auto-segment mode: minimum consecutive Visibility=1 frames to keep "
        "a run (seq=3 windows need at least 3).",
    )
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.video_path}")
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    manual = args.start_frame is not None or args.end_frame is not None
    if manual:
        if not args.track_id:
            raise ValueError("--track_id is required with --start_frame/--end_frame")
        start = args.start_frame if args.start_frame is not None else 0
        end = args.end_frame if args.end_frame is not None else len(df)
        split_and_process(cap, orig_w, orig_h, df, args.track_id, start, end, args.test_frac)
    else:
        runs = find_visible_runs(df, args.min_run_length)
        prefix = args.track_id or "track"
        print(f"Auto-segmented into {len(runs)} runs (min_run_length={args.min_run_length})")
        for i, (start, end) in enumerate(runs):
            split_and_process(
                cap, orig_w, orig_h, df, f"{prefix}_{i}", start, end, args.test_frac
            )

    cap.release()


if __name__ == "__main__":
    main()
