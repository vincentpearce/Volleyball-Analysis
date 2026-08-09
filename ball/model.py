"""ONNX inference wrapper for the fine-tuned vball-net ball detector.

Preprocessing/postprocessing here matches third_party/vball-net/src/inference_onnx.py
exactly (3-frame sliding window in, heatmap-per-frame out, take the newest frame's
heatmap each step) -- that's the exact behavior ball/README.md's accuracy numbers
were measured against, so it isn't reimplemented differently here.
"""

from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

MODEL_INPUT_WIDTH = 512
MODEL_INPUT_HEIGHT = 288


class BallDetector:
    def __init__(self, model_path, min_detection_confidence):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = min_detection_confidence
        self.frame_buffer = deque(maxlen=3)

    def _preprocess(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT))
        return resized.astype(np.float32) / 255.0

    def _postprocess(self, heatmap):
        _, binary = cv2.threshold(heatmap, self.threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        m = cv2.moments(largest)
        if m["m00"] == 0:
            return None
        return m["m10"] / m["m00"], m["m01"] / m["m00"]

    def detect(self, frame_bgr):
        """Feed one frame in sequence; returns (x, y) in this frame's pixel
        coordinates, or None if no ball detected. Call once per frame, in order --
        it's stateful (keeps the last 3 frames for the model's temporal window)."""
        processed = self._preprocess(frame_bgr)
        self.frame_buffer.append(processed)
        while len(self.frame_buffer) < 3:
            self.frame_buffer.append(processed)

        input_tensor = np.concatenate(list(self.frame_buffer), axis=2)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.transpose(input_tensor, (0, 3, 1, 2)).astype(np.float32)

        output = self.session.run(None, {self.input_name: input_tensor})[0]
        heatmap = output[0, 2, :, :]  # newest frame's heatmap
        point = self._postprocess(heatmap)
        if point is None:
            return None

        frame_h, frame_w = frame_bgr.shape[:2]
        x = point[0] * frame_w / MODEL_INPUT_WIDTH
        y = point[1] * frame_h / MODEL_INPUT_HEIGHT
        return x, y
