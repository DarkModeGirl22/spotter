"""Turns OpenCV pose keypoints into form measurements and faults for each exercise.

Every analyzer is a pure function of one Pose -> Observation, so it can be unit-tested
with synthetic keypoints. Timing / persistence is handled by the coach agent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pose import (L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
                   R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST, Pose)

VIS = 0.35  # minimum keypoint visibility we trust


@dataclass
class Observation:
    exercise: str
    person: bool                     # a person was found
    in_position: bool                # person is in the start position of this exercise
    faults: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    bad_joints: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "exercise": self.exercise, "person": self.person, "in_position": self.in_position,
            "faults": self.faults, "metrics": self.metrics, "bad_joints": self.bad_joints,
        }


def angle(a, b, c) -> float:
    """Angle at b (degrees) formed by a-b-c."""
    ba, bc = np.asarray(a) - b, np.asarray(c) - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _side(pose: Pose) -> str:
    left = pose.visibility[[L_SHOULDER, L_HIP, L_KNEE, L_ANKLE]].mean()
    right = pose.visibility[[R_SHOULDER, R_HIP, R_KNEE, R_ANKLE]].mean()
    return "left" if left >= right else "right"


def _joints(pose: Pose, side: str) -> dict:
    idx = {
        "left": dict(S=L_SHOULDER, E=L_ELBOW, W=L_WRIST, H=L_HIP, K=L_KNEE, A=L_ANKLE),
        "right": dict(S=R_SHOULDER, E=R_ELBOW, W=R_WRIST, H=R_HIP, K=R_KNEE, A=R_ANKLE),
    }[side]
    return {k: (i, pose.points[i].astype(float), float(pose.visibility[i])) for k, i in idx.items()}


# ---------------------------------------------------------------- plank (side view)
PLANK_SAG = 0.07     # hip below the shoulder-ankle line by > 7% of body length
PLANK_PIKE = -0.09   # hip above the line by > 9% of body length
PLANK_KNEE = 150.0   # knee angle below this = knees bent / on the floor


def analyze_plank(pose: Pose | None) -> Observation:
    if pose is None:
        return Observation("plank", person=False, in_position=False, faults=["no_person"])
    side = _side(pose)
    j = _joints(pose, side)
    if min(j[k][2] for k in "SHA") < VIS:
        return Observation("plank", True, False, ["body_not_visible"], {"side": side})

    S, H, K, A, E = (j[k][1] for k in "SHKAE")
    body = A - S
    length = float(np.linalg.norm(body)) + 1e-9
    tilt = abs(float(np.degrees(np.arctan2(body[1], abs(body[0]) + 1e-9))))  # 0 = horizontal

    # signed vertical offset of the hip from the shoulder->ankle line (image y grows downward)
    t = np.clip(np.dot(H - S, body) / length**2, 0.0, 1.0)
    line_y = S[1] + t * body[1]
    hip_dev = float((H[1] - line_y) / length)  # + = hips sagging, - = hips piked
    hip_angle = angle(S, H, K)
    knee_angle = angle(H, K, A) if j["K"][2] >= VIS else 180.0
    # shoulders far above elbows = still holding; close = collapsed to the floor
    shoulder_lift = float((E[1] - S[1]) / length) if j["E"][2] >= VIS else None

    metrics = {
        "side": side, "body_tilt_deg": round(tilt, 1), "hip_deviation": round(hip_dev, 3),
        "hip_angle_deg": round(hip_angle, 1), "knee_angle_deg": round(knee_angle, 1),
        "shoulder_lift": None if shoulder_lift is None else round(shoulder_lift, 3),
    }
    in_position = tilt < 35.0
    obs = Observation("plank", True, in_position, [], metrics)
    if not in_position:
        obs.faults.append("not_in_plank_position")
        return obs

    # collapsed = chest dropped to the floor; a piked plank also puts shoulders low, so exclude it
    if shoulder_lift is not None and shoulder_lift < 0.05 and hip_dev > PLANK_PIKE:
        obs.faults.append("collapsed")
        obs.bad_joints += [j["S"][0], j["H"][0]]
    if knee_angle < PLANK_KNEE:
        obs.faults.append("knees_down")
        obs.bad_joints.append(j["K"][0])
    elif hip_dev > PLANK_SAG:
        obs.faults.append("hips_low")
        obs.bad_joints.append(j["H"][0])
    elif hip_dev < PLANK_PIKE:
        obs.faults.append("hips_high")
        obs.bad_joints.append(j["H"][0])
    return obs


# ---------------------------------------------------------------- squat (front or side)
def analyze_squat(pose: Pose | None) -> Observation:
    if pose is None:
        return Observation("squat", False, False, ["no_person"])
    lk = angle(pose.points[L_HIP], pose.points[L_KNEE], pose.points[L_ANKLE])
    rk = angle(pose.points[R_HIP], pose.points[R_KNEE], pose.points[R_ANKLE])
    vis_ok = pose.visibility[[L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE]].min() >= VIS
    if not vis_ok:
        return Observation("squat", True, False, ["legs_not_visible"])
    knee = (lk + rk) / 2
    knee_gap = abs(pose.points[L_KNEE][0] - pose.points[R_KNEE][0])
    ankle_gap = abs(pose.points[L_ANKLE][0] - pose.points[R_ANKLE][0]) + 1e-9
    obs = Observation("squat", True, True, [], {
        "knee_angle_deg": round(knee, 1), "knee_ankle_ratio": round(float(knee_gap / ankle_gap), 2),
    })
    # knees caving in: only meaningful in a front view (ankles clearly apart) while bent
    if knee < 140 and ankle_gap > 0.08 * pose.image_size[0] and knee_gap / ankle_gap < 0.75:
        obs.faults.append("knees_in")
        obs.bad_joints += [L_KNEE, R_KNEE]
    return obs


# ---------------------------------------------------------------- other moves: activity only
ACTIVITY_JOINTS = [L_WRIST, R_WRIST, L_ANKLE, R_ANKLE, L_HIP, R_HIP]


def analyze_activity(exercise: str, pose: Pose | None) -> Observation:
    if pose is None:
        return Observation(exercise, False, False, ["no_person"])
    w, h = pose.image_size
    pts = pose.points[ACTIVITY_JOINTS] / np.array([w, h])
    return Observation(exercise, True, True, [], {"activity_points": np.round(pts, 4).tolist()})


ANALYZERS = {"plank": analyze_plank, "squat": analyze_squat}


def analyze(exercise: str, pose: Pose | None) -> Observation:
    fn = ANALYZERS.get(exercise)
    return fn(pose) if fn else analyze_activity(exercise, pose)
