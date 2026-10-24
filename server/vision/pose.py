"""OpenCV 5 pose estimation: person detector + 33-keypoint pose model, both run with cv2.dnn."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np

from .mp_persondet import MPPersonDet
from .mp_pose import MPPose

MODELS = Path(__file__).resolve().parents[2] / "models"

# BlazePose keypoint indices we use
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]


@dataclass
class Pose:
    points: np.ndarray       # (33, 2) pixel coords
    visibility: np.ndarray   # (33,) 0..1
    confidence: float
    image_size: tuple        # (w, h)
    latency_ms: float

    def to_json(self) -> dict:
        w, h = self.image_size
        return {
            "points": [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in self.points],
            "visibility": [round(float(v), 3) for v in self.visibility],
            "confidence": round(float(self.confidence), 3),
            "latency_ms": round(self.latency_ms, 1),
        }


class PoseEstimator:
    def __init__(self, models_dir: Path = MODELS):
        self.detector = MPPersonDet(
            str(models_dir / "person_detection_mediapipe_2023mar.onnx"),
            nmsThreshold=0.3, scoreThreshold=0.35, topK=5000,
        )
        self.pose = MPPose(str(models_dir / "pose_estimation_mediapipe_2023mar.onnx"), confThreshold=0.6)

    def estimate(self, image: np.ndarray) -> Pose | None:
        t0 = time.perf_counter()
        h, w = image.shape[:2]
        persons = self.detector.infer(image)
        if persons is None or len(persons) == 0:
            return None
        # keep the most confident person only
        person = persons[np.argmax(persons[:, -1])]
        result = self.pose.infer(image, person)
        if result is None:
            return None
        _, landmarks, _, _, _, conf = result
        pts = landmarks[:33, :2].astype(np.float32)
        vis = landmarks[:33, 3].astype(np.float32)
        return Pose(pts, vis, float(conf), (w, h), (time.perf_counter() - t0) * 1000)


def decode_jpeg(data: bytes) -> np.ndarray | None:
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv.imdecode(arr, cv.IMREAD_COLOR)


def draw_pose(image: np.ndarray, pose: Pose, bad_joints: set[int] | None = None) -> np.ndarray:
    """Annotated frame for traces/evaluation (black bones, red joints where form is wrong)."""
    out = image.copy()
    bad_joints = bad_joints or set()
    p = pose.points.astype(int)
    for a, b in SKELETON:
        if pose.visibility[a] > 0.3 and pose.visibility[b] > 0.3:
            cv.line(out, tuple(p[a]), tuple(p[b]), (20, 20, 20), 4, cv.LINE_AA)
    for i in {j for e in SKELETON for j in e}:
        if pose.visibility[i] > 0.3:
            color = (0, 0, 255) if i in bad_joints else (255, 255, 255)
            cv.circle(out, tuple(p[i]), 6, color, -1, cv.LINE_AA)
            cv.circle(out, tuple(p[i]), 6, (20, 20, 20), 1, cv.LINE_AA)
    return out
