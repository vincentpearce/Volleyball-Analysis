"""Video read/write helpers. fps is always read from the source file -- never
hardcoded -- since all frame<->time conversion elsewhere depends on it being
the real value."""

from dataclasses import dataclass

import cv2


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    info = VideoInfo(
        path=path,
        fps=cap.get(cv2.CAP_PROP_FPS),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    if info.fps <= 0:
        raise ValueError(f"Video reports invalid fps ({info.fps}): {path}")
    return cap, info


def frame_iterator(cap):
    """Yields (frame_index, frame) until the video is exhausted."""
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_idx, frame
        frame_idx += 1


def open_writer(path, info):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, info.fps, (info.width, info.height))
