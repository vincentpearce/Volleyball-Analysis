"""YOLO11-pose wrapper: multi-person detection + tracking, one call per frame.

COCO 17-keypoint order (fixed by the model, not configurable):
0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear,
5 left_shoulder, 6 right_shoulder, 7 left_elbow, 8 right_elbow,
9 left_wrist, 10 right_wrist, 11 left_hip, 12 right_hip,
13 left_knee, 14 right_knee, 15 left_ankle, 16 right_ankle.
"""

from ultralytics import YOLO

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class PoseDetector:
    def __init__(self, model_path, detection_confidence):
        self.model = YOLO(model_path)
        self.conf = detection_confidence

    def detect(self, frame_bgr):
        """Call once per frame, in order -- persist=True keeps ByteTrack state
        across calls. Returns a list of persons: {"track_id", "bbox", "keypoints"}
        where keypoints is {name: (x, y, conf)}."""
        results = self.model.track(
            frame_bgr, persist=True, verbose=False, conf=self.conf
        )
        r = results[0]
        people = []
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            return people

        track_ids = r.boxes.id.tolist()
        boxes = r.boxes.xyxy.tolist()
        kp_xy = r.keypoints.xy.tolist()
        kp_conf = r.keypoints.conf.tolist() if r.keypoints.conf is not None else None

        for i, track_id in enumerate(track_ids):
            keypoints = {}
            for j, name in enumerate(KEYPOINT_NAMES):
                x, y = kp_xy[i][j]
                conf = kp_conf[i][j] if kp_conf is not None else 1.0
                keypoints[name] = (x, y, conf)
            people.append({
                "track_id": int(track_id),
                "bbox": boxes[i],
                "keypoints": keypoints,
            })
        return people
