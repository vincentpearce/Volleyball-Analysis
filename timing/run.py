"""Phase 4 entry point: timing assessment from Phase 2's pose/jump output.

Usage:
    .venv/bin/python timing/run.py --pose_json output/phase2/clip_pose.json \\
        --output_dir output/phase4
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.loader import load_config
from timing.assess import assess_all
from videoio.export import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()
    os.makedirs(args.output_dir, exist_ok=True)
    clip_name = os.path.splitext(os.path.basename(args.pose_json))[0].removesuffix("_pose")

    with open(args.pose_json) as f:
        pose_data = json.load(f)

    fps = pose_data["video"]["fps"]
    tolerance_ms = config["timing"]["tolerance_ms"]
    hits = assess_all(pose_data["pose"]["hitters"], fps, tolerance_ms)

    for hit in hits:
        if hit["verdict"] is None:
            print(f"track_id={hit['track_id']}: takeoff={hit['takeoff_frame']} "
                  f"landing={hit['landing_frame']} apex={hit['apex_frame']:.1f} "
                  f"-- no contact found, can't assess timing")
        else:
            print(f"track_id={hit['track_id']}: takeoff={hit['takeoff_frame']} "
                  f"landing={hit['landing_frame']} apex={hit['apex_frame']:.1f} "
                  f"contact={hit['contact_frame']} "
                  f"offset={hit['timing_offset_ms']:+.0f}ms -> {hit['verdict'].upper()}")

    result = {"video": pose_data["video"], "hits": hits}
    json_path = os.path.join(args.output_dir, f"{clip_name}_hits.json")
    write_json(result, json_path)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
