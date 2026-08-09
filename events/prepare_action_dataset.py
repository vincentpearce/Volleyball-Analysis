"""Converts the downloaded Roboflow "Volleyball Activity Dataset" (COCO format,
per-player bounding boxes + action labels) into Ultralytics' image-classification
folder layout: <output_dir>/<split>/<class>/<crop>.jpg

Maps Roboflow's 7 real categories onto our 6 target classes (serve/set/spike/
block/dig/none) -- 'reception' and 'Defense-Move' both become 'dig' (both are
non-attacking defensive touches), 'stand' becomes 'none' (a negative class so
idle players don't get misclassified as some action). 'Volleyball-Players' is
a supercategory placeholder with zero real annotations, dropped entirely.

Usage:
    .venv/bin/python events/prepare_action_dataset.py \\
        --roboflow_dir /path/to/Volleyball-Activity-Dataset-4 \\
        --output_dir data/action_dataset
"""

import argparse
import json
import os

import cv2
from tqdm import tqdm

CATEGORY_MAP = {
    "service": "serve",
    "setting": "set",
    "attack": "spike",
    "block": "block",
    "reception": "dig",
    "Defense-Move": "dig",
    "stand": "none",
}

# Roboflow's "valid" split maps to Ultralytics' expected "val" directory name.
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

# Small margin around each bbox so the crop shows a bit of surrounding context
# (ball, nearby net/other players) rather than just the person silhouette.
CROP_MARGIN_FRAC = 0.15


def prepare_split(roboflow_dir, split, output_dir):
    json_path = os.path.join(roboflow_dir, split, "_annotations.coco.json")
    with open(json_path) as f:
        data = json.load(f)

    cats = {c["id"]: c["name"] for c in data["categories"]}
    images = {img["id"]: img for img in data["images"]}

    out_split = SPLIT_MAP[split]
    counts = {}
    for ann in tqdm(data["annotations"], desc=f"{split}"):
        raw_name = cats[ann["category_id"]]
        mapped = CATEGORY_MAP.get(raw_name)
        if mapped is None:
            continue

        img_info = images[ann["image_id"]]
        img_path = os.path.join(roboflow_dir, split, img_info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        x, y, bw, bh = ann["bbox"]
        mx, my = bw * CROP_MARGIN_FRAC, bh * CROP_MARGIN_FRAC
        x1, y1 = max(0, int(x - mx)), max(0, int(y - my))
        x2, y2 = min(w, int(x + bw + mx)), min(h, int(y + bh + my))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2, x1:x2]

        class_dir = os.path.join(output_dir, out_split, mapped)
        os.makedirs(class_dir, exist_ok=True)
        counts[mapped] = counts.get(mapped, 0) + 1
        crop_path = os.path.join(class_dir, f"{ann['id']}.jpg")
        cv2.imwrite(crop_path, crop)

    print(f"{split} -> {out_split}: {counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roboflow_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    for split in ("train", "valid", "test"):
        prepare_split(args.roboflow_dir, split, args.output_dir)


if __name__ == "__main__":
    main()
