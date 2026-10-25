"""Which exercise is the person actually doing?

Pose-feature activity recognition: each frame is reduced to a few body-geometry features
(posture, joint angles, hand and foot positions, all normalised by torso length). A window of the
last ~4 s of features is matched against the signature of every exercise in the app. The coach
compares the result with the exercise the user is supposed to be doing.
"""
from __future__ import annotations

import numpy as np

from .exercises import VIS, angle
from .pose import (L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE, R_ANKLE, R_ELBOW,
                   R_HIP, R_KNEE, R_SHOULDER, R_WRIST, Pose)

NAMES = {"plank": "a plank", "bridge": "a glute bridge", "squat": "squats", "lunge": "lunges",
         "legraise": "side leg raises", "armcircles": "arm circles", "press": "overhead presses",
         "wallpush": "wall push-ups"}
FLOOR = {"plank", "bridge"}
WINDOW = 10          # frames (~4 s at one frame every 0.4 s)


def features(pose: Pose | None) -> dict | None:
    """Compact per-frame description of the body (kept in the agent state)."""
    if pose is None:
        return None
    P, V = pose.points.astype(float), pose.visibility
    if V[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]].max() < VIS:
        return None
    sh, hp = (P[L_SHOULDER] + P[R_SHOULDER]) / 2, (P[L_HIP] + P[R_HIP]) / 2
    torso = float(np.linalg.norm(hp - sh)) + 1e-6
    v = hp - sh
    tilt = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]) + 1e-9)))     # 0 = upright, 90 = lying
    knees = [angle(P[h], P[k], P[a]) for h, k, a in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))]
    elbows = [angle(P[s], P[e], P[w]) for s, e, w in ((L_SHOULDER, L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW, R_WRIST))]
    wrist_up = float(max(sh[1] - P[L_WRIST][1], sh[1] - P[R_WRIST][1]) / torso)       # + = above shoulders
    head_y = P[NOSE][1]
    wrist_over_head = float((head_y - min(P[L_WRIST][1], P[R_WRIST][1])) / torso)
    sh_w = float(np.linalg.norm(P[L_SHOULDER] - P[R_SHOULDER])) + 1e-6
    wrist_spread = float(abs(P[L_WRIST][0] - P[R_WRIST][0]) / sh_w)
    knee_y = (P[L_KNEE][1] + P[R_KNEE][1]) / 2
    ankle_y = max(P[L_ANKLE][1], P[R_ANKLE][1])
    return {
        "tilt": round(tilt, 1),
        "knee_l": round(knees[0], 1), "knee_r": round(knees[1], 1),
        "elbow": round(float(min(elbows)), 1),
        "wrist_up": round(wrist_up, 2), "wrist_over_head": round(wrist_over_head, 2),
        "wrist_spread": round(wrist_spread, 2),
        "front": round(sh_w / torso, 2),                       # ~0.7+ facing camera, small sideways
        "knee_above_hip": round(float((hp[1] - knee_y) / torso), 2),
        "shoulder_lift": round(float((max(P[L_ELBOW][1], P[R_ELBOW][1]) - sh[1]) / torso), 2),
        "hip_y": round(float(hp[1] / torso), 2),
        # ankles relative to the hips, so walking around the room does not look like a leg raise
        "ankle_lx": round(float((P[L_ANKLE][0] - hp[0]) / torso), 2),
        "ankle_rx": round(float((P[R_ANKLE][0] - hp[0]) / torso), 2),
        "ankle_gap": round(float(abs(P[L_ANKLE][0] - P[R_ANKLE][0]) / torso), 2),
        "ankle_low": round(float((ankle_y - hp[1]) / torso), 2),
    }


def _rng(xs):
    return float(max(xs) - min(xs)) if xs else 0.0


def recognize(window: list[dict]) -> tuple[str | None, float, dict]:
    """(exercise, confidence 0..1, evidence). None when the window is too short or nothing fits."""
    w = [f for f in window if f]
    if len(w) < 5:
        return None, 0.0, {}
    med = lambda k: float(np.median([f[k] for f in w]))
    frac = lambda cond: sum(1 for f in w if cond(f)) / len(w)
    lying = frac(lambda f: f["tilt"] > 55)
    upright = frac(lambda f: f["tilt"] < 30)
    ev = {"lying": round(lying, 2), "upright": round(upright, 2)}

    if lying >= 0.6:
        bridge = frac(lambda f: min(f["knee_l"], f["knee_r"]) < 125 and f["knee_above_hip"] > 0.15)
        plank = frac(lambda f: f["shoulder_lift"] > 0.05 and f["knee_above_hip"] < 0.2)
        ev.update(bridge=round(bridge, 2), plank=round(plank, 2))
        if bridge >= 0.5 and bridge > plank:
            return "bridge", bridge, ev
        if plank >= 0.5:
            return "plank", plank, ev
        return None, 0.0, ev

    lean = frac(lambda f: 15 <= f["tilt"] <= 60)
    if lean >= 0.6 and med("wrist_up") > -0.6:
        return "wallpush", lean, ev
    if upright < 0.6:
        return None, 0.0, ev

    knee_avg = [(f["knee_l"] + f["knee_r"]) / 2 for f in w]
    knee_rng = _rng(knee_avg)
    asym = float(np.median([abs(f["knee_l"] - f["knee_r"]) for f in w]))
    overhead = frac(lambda f: f["wrist_over_head"] > 0.1)
    arms_out = frac(lambda f: abs(f["wrist_up"]) < 0.45 and f["wrist_spread"] > 2.2)
    leg_side = max(_rng([f["ankle_lx"] for f in w]), _rng([f["ankle_rx"] for f in w]))
    split = frac(lambda f: f["ankle_gap"] > 0.9 and f["front"] < 0.45)
    ev.update(knee_range=round(knee_rng, 1), knee_asym=round(asym, 1), overhead=round(overhead, 2),
              arms_out=round(arms_out, 2), leg_side=round(leg_side, 2), split=round(split, 2))

    if overhead >= 0.3 and knee_rng < 30:
        return "press", min(1.0, overhead + 0.3), ev
    if arms_out >= 0.6 and knee_rng < 25:
        return "armcircles", arms_out, ev
    if knee_rng >= 35 and (asym > 25 or split >= 0.3):
        return "lunge", min(1.0, knee_rng / 60), ev
    if knee_rng >= 35:
        return "squat", min(1.0, knee_rng / 60), ev
    if leg_side >= 0.45 and knee_rng < 30:
        return "legraise", min(1.0, leg_side), ev
    return None, 0.0, ev


def compatible(expected: str, seen: str | None) -> bool:
    """Is what we see consistent with the expected exercise? Unknown never counts as wrong."""
    if seen is None or seen == expected:
        return True
    # a lunge and a squat can look alike for a few reps; only floor vs standing, or a clear other
    # signature, is a contradiction
    if {expected, seen} <= {"squat", "lunge"}:
        return True
    return False
