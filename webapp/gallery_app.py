"""Public demo gallery: browses pre-computed pipeline results, no live
processing. Built for deployment on free/constrained hosting (e.g. Streamlit
Community Cloud's 1GB RAM, CPU-only tier) -- the full pipeline (torch, YOLO,
ONNX models) doesn't fit that budget, so this app doesn't import any of it.
For the real, live, run-it-on-your-own-footage version, see webapp/app.py
and run locally (README has instructions).

Run with:
    .venv/bin/streamlit run webapp/gallery_app.py
"""

import json
import os

import streamlit as st

DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "demo_data")

CLIPS = {
    "clip_00-42-20": {
        "label": "Rally with a jump-spike (contact not confirmed)",
        "note": (
            "Shows a real detected jump (takeoff/landing) where the contact frame "
            "couldn't be confirmed -- the hitter's wrist tracking was lost during "
            "the fast swing, a known, documented limitation rather than a silent "
            "failure. The pipeline reports this honestly instead of guessing."
        ),
    },
    "clip2_00-48-00": {
        "label": "Rally with a confirmed on-time hit",
        "note": "A full successful run: jump, landing, and confirmed contact -- classified ON TIME.",
    },
}


def load_json(clip_key, name):
    with open(os.path.join(DEMO_DIR, clip_key, f"{name}.json")) as f:
        return json.load(f)


def video_path(clip_key, name):
    return os.path.join(DEMO_DIR, clip_key, f"{name}.mp4")


def show_clip(clip_key):
    ball_data = load_json(clip_key, "ball")
    pose_data = load_json(clip_key, "pose")
    events_data = load_json(clip_key, "events")
    hits_data = load_json(clip_key, "hits")

    tab_ball, tab_pose, tab_events, tab_timing = st.tabs(
        ["Ball tracking", "Pose & jumps", "Events", "Timing verdict"]
    )

    with tab_ball:
        traj = ball_data["ball"]["trajectory"]
        detected = sum(1 for p in traj if p["visibility"] == 1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Frames detected", f"{detected}/{len(traj)}", f"{detected/len(traj):.0%}")
        c2.metric("Interpolated", sum(1 for p in traj if p["interpolated"]))
        c3.metric("Apex frames", len(ball_data["ball"]["apex_frames"]))
        st.video(video_path(clip_key, "ball_overlay"))

    with tab_pose:
        hitters = pose_data["pose"]["hitters"]
        total_jumps = sum(len(h["jumps"]) for h in hitters)
        st.metric("Jumps detected", total_jumps, f"across {len(hitters)} tracked people")
        for h in hitters:
            for jump in h["jumps"]:
                confirmed = "confirmed" if jump["landing_confirmed"] else "unconfirmed (lost track)"
                contact = jump["contact"]["contact_frame"] if jump["contact"] else "not found"
                st.write(
                    f"**track_id={h['track_id']}** -- takeoff {jump['takeoff_frame']}, "
                    f"landing {jump['landing_frame']} ({confirmed}), contact: {contact}"
                )
        st.video(video_path(clip_key, "pose_overlay"))

    with tab_events:
        events = events_data["events"]
        st.metric("Events classified", len(events))
        st.dataframe(
            [{"type": e["type"], "start (s)": round(e["start_time"], 2),
              "end (s)": round(e["end_time"], 2), "confidence": round(e["confidence"], 2)}
             for e in events],
            use_container_width=True,
        )
        st.video(video_path(clip_key, "events_overlay"))

    with tab_timing:
        hits = hits_data["hits"]
        if not hits:
            st.info("No jumps with a detected contact frame -- nothing to assess.")
        for hit in hits:
            if hit["verdict"] is None:
                st.warning(
                    f"track_id={hit['track_id']}: apex at frame {hit['apex_frame']:.1f}, "
                    f"but no contact frame found -- can't compute a verdict."
                )
            else:
                color = {"on_time": "green", "early": "orange", "late": "orange"}[hit["verdict"]]
                st.markdown(
                    f"track_id={hit['track_id']}: contact {hit['timing_offset_ms']:+.0f}ms "
                    f"from apex -- :{color}[**{hit['verdict'].upper()}**]"
                )
        st.json(hits_data, expanded=False)


def main():
    st.set_page_config(page_title="Volleyball Timing Analyzer -- Demo", layout="wide")
    st.title("Volleyball Timing Analyzer")
    st.caption("Ball tracking -> pose/jump detection -> event classification -> timing assessment")
    st.info(
        "This is a **static demo gallery** of pre-computed results (the full ML "
        "pipeline needs more compute than free hosting provides -- clone the repo "
        "to run it live on your own footage)."
    )

    clip_key = st.selectbox(
        "Choose an example", list(CLIPS.keys()),
        format_func=lambda k: CLIPS[k]["label"],
    )
    st.caption(CLIPS[clip_key]["note"])
    show_clip(clip_key)


if __name__ == "__main__":
    main()
