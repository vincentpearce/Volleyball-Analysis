"""Simple Streamlit UI over the ball -> pose -> events -> timing pipeline.

Run with:
    .venv/bin/streamlit run webapp/app.py

Each phase is invoked as the same CLI script used from the command line
(ball/run.py, pose/run.py, events/run.py, timing/run.py) via subprocess, so
this UI can't drift out of sync with how the pipeline actually behaves when
run manually -- it's a thin presentation layer, not a reimplementation.
"""

import json
import os
import subprocess
import sys
import time

import cv2
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "data", "samples")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output", "webapp")

LONG_CLIP_WARNING_SECONDS = 60

st.set_page_config(page_title="Volleyball Timing Analyzer", layout="wide")


def list_sample_clips():
    if not os.path.isdir(SAMPLES_DIR):
        return []
    files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".mp4", ".mov"))]
    # Smallest file first -- a reasonable proxy for "shortest clip" that doesn't
    # depend on any naming convention, so trimmed demo clips default ahead of
    # full match recordings (which trigger the long-clip warning below).
    return sorted(files, key=lambda f: os.path.getsize(os.path.join(SAMPLES_DIR, f)))


def video_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps else 0
    return {"fps": fps, "frame_count": frame_count, "width": width, "height": height, "duration": duration}


def run_step(label, args):
    """Runs one pipeline script, showing a live status widget. Returns
    (success, log_text)."""
    with st.status(label, expanded=False) as status:
        result = subprocess.run(
            [PYTHON] + args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        log = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            status.update(label=f"{label} -- done", state="complete")
        else:
            status.update(label=f"{label} -- failed", state="error")
        with st.expander("Log"):
            st.code(log or "(no output)", language=None)
        return result.returncode == 0, log


def run_pipeline(video_path, clip_name, classifier, run_dir):
    phase1_dir = os.path.join(run_dir, "phase1")
    phase2_dir = os.path.join(run_dir, "phase2")
    phase3_dir = os.path.join(run_dir, "phase3")
    phase4_dir = os.path.join(run_dir, "phase4")

    ok, _ = run_step("Phase 1: ball tracking", [
        "ball/run.py", "--video_path", video_path, "--output_dir", phase1_dir,
    ])
    if not ok:
        return None
    ball_json = os.path.join(phase1_dir, f"{clip_name}_ball.json")

    ok, _ = run_step("Phase 2: pose, jumps, contact", [
        "pose/run.py", "--video_path", video_path, "--ball_json", ball_json,
        "--output_dir", phase2_dir,
    ])
    if not ok:
        return None
    pose_json = os.path.join(phase2_dir, f"{clip_name}_pose.json")

    events_args = [
        "events/run.py", "--video_path", video_path, "--ball_json", ball_json,
        "--output_dir", phase3_dir, "--classifier", classifier,
    ]
    if classifier == "heuristic":
        events_args += ["--pose_json", pose_json]
    ok, _ = run_step("Phase 3: event classification", events_args)
    if not ok:
        return None

    ok, _ = run_step("Phase 4: timing assessment", [
        "timing/run.py", "--pose_json", pose_json, "--output_dir", phase4_dir,
    ])
    if not ok:
        return None

    return {
        "ball_json": ball_json,
        "ball_overlay": os.path.join(phase1_dir, f"{clip_name}_overlay.mp4"),
        "pose_json": pose_json,
        "pose_overlay": os.path.join(phase2_dir, f"{clip_name}_pose_overlay.mp4"),
        "events_json": os.path.join(phase3_dir, f"{clip_name}_events.json"),
        "events_overlay": os.path.join(phase3_dir, f"{clip_name}_events_overlay.mp4"),
        "hits_json": os.path.join(phase4_dir, f"{clip_name}_hits.json"),
    }


def show_results(results):
    with open(results["ball_json"]) as f:
        ball_data = json.load(f)
    with open(results["pose_json"]) as f:
        pose_data = json.load(f)
    with open(results["events_json"]) as f:
        events_data = json.load(f)
    with open(results["hits_json"]) as f:
        hits_data = json.load(f)

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
        st.video(results["ball_overlay"])

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
        st.video(results["pose_overlay"])

    with tab_events:
        events = events_data["events"]
        st.metric("Events classified", len(events))
        st.dataframe(
            [{"type": e["type"], "start (s)": round(e["start_time"], 2),
              "end (s)": round(e["end_time"], 2), "confidence": round(e["confidence"], 2)}
             for e in events],
            use_container_width=True,
        )
        st.video(results["events_overlay"])

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
    st.title("Volleyball Timing Analyzer")
    st.caption("Ball tracking -> pose/jump detection -> event classification -> timing assessment")

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    with st.sidebar:
        st.header("Input")
        sample_clips = list_sample_clips()
        source = st.radio("Video source", ["Sample clip", "Upload"], horizontal=True)

        video_path = None
        if source == "Sample clip":
            if sample_clips:
                choice = st.selectbox("Choose a clip", sample_clips)
                video_path = os.path.join(SAMPLES_DIR, choice)
            else:
                st.warning(f"No clips found in {SAMPLES_DIR}")
        else:
            uploaded = st.file_uploader("Upload a short clip (~30s works best)", type=["mp4", "mov"])
            if uploaded:
                video_path = os.path.join(UPLOADS_DIR, uploaded.name)
                with open(video_path, "wb") as f:
                    f.write(uploaded.getbuffer())

        st.header("Event classifier")
        classifier = st.radio(
            "Classifier", ["heuristic", "learned"],
            help=(
                "heuristic: rule-based, default, measured at 30.8% on club-level footage.\n\n"
                "learned: YOLO model fine-tuned on professional match footage -- use for "
                "polished/high-level play, not casual footage."
            ),
        )

        run_clicked = st.button("Run analysis", type="primary", disabled=video_path is None)

    if video_path is None:
        st.info("Pick a sample clip or upload one to get started.")
        return

    info = video_info(video_path)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{info['duration']:.1f}s")
    c2.metric("Resolution", f"{info['width']}x{info['height']}")
    c3.metric("FPS", f"{info['fps']:.0f}")
    c4.metric("Frames", info["frame_count"])
    if info["duration"] > LONG_CLIP_WARNING_SECONDS:
        st.warning(
            f"This clip is {info['duration']:.0f}s long -- the full pipeline can take "
            f"several minutes even on a short clip, so this may be slow. A trimmed "
            f"~30s clip is recommended."
        )

    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    run_dir = os.path.join(OUTPUT_ROOT, f"{clip_name}_{classifier}")

    if run_clicked:
        start = time.time()
        with st.spinner("Running pipeline -- this takes a few minutes..."):
            results = run_pipeline(video_path, clip_name, classifier, run_dir)
        if results is None:
            st.error("Pipeline failed -- check the logs above for the failing step.")
        else:
            st.success(f"Done in {time.time() - start:.0f}s")
            st.session_state["results"] = results

    if "results" in st.session_state:
        show_results(st.session_state["results"])


if __name__ == "__main__":
    main()
