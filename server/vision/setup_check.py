"""Camera setup check (active perception): before a set starts, the coach makes sure it can
actually judge the exercise. It looks at light, sharpness, distance, framing, camera angle and,
for floor exercises, the start position, and tells the user what to change until all is good.
"""
from __future__ import annotations

import cv2 as cv
import numpy as np

from .exercises import VIS, analyze
from .pose import (L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, NOSE, R_ANKLE, R_HIP, R_KNEE,
                   R_SHOULDER, Pose)

SIDE_VIEW = {"plank", "lunge", "bridge", "wallpush"}       # judged from the side
FLOOR = {"plank", "bridge"}                                # start on the floor
BODY = [NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]

MESSAGES = {
    "too_dark": "It's too dark. Turn on a light or face a window.",
    "blurry": "The image is blurry. Wipe the camera and keep the phone still.",
    "no_person": "I can't see you yet. Step in front of the phone.",
    "cut_off": "Part of your body is out of the picture. Move back a little.",
    "too_far": "You're too far. Come a bit closer to the phone.",
    "too_close": "You're too close. Take a step back.",
    "need_side": "Turn sideways to the phone, so I can see your body line.",
    "need_front": "Face the phone, please.",
    "get_in_position": "Good spot! Now get into position.",
    "stand_up": "This one is done standing. Stand up, please.",
    "ok": "Perfect, I can see you well. Let's go!",
}


def frame_quality(image: np.ndarray) -> dict:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return {"brightness": round(float(gray.mean()), 1),
            "sharpness": round(float(cv.Laplacian(gray, cv.CV_64F).var()), 1)}


def check(exercise: str, image: np.ndarray, pose: Pose | None) -> dict:
    q = frame_quality(image)
    issue, m = None, dict(q)
    if q["brightness"] < 45:
        issue = "too_dark"
    elif q["sharpness"] < 15:
        issue = "blurry"
    elif pose is None:
        issue = "no_person"
    else:
        w, h = pose.image_size
        vis = pose.visibility[BODY]
        pts = pose.points[BODY] / np.array([w, h])
        seen = vis >= VIS
        inside = (pts[:, 0] > 0.01) & (pts[:, 0] < 0.99) & (pts[:, 1] > 0.01) & (pts[:, 1] < 0.99)
        span = pts[seen].max(axis=0) - pts[seen].min(axis=0) if seen.any() else np.zeros(2)
        size = float(span.max())
        sh = np.linalg.norm(pose.points[L_SHOULDER] - pose.points[R_SHOULDER])
        torso = np.linalg.norm((pose.points[L_SHOULDER] + pose.points[R_SHOULDER]) / 2 -
                               (pose.points[L_HIP] + pose.points[R_HIP]) / 2) + 1e-9
        width_ratio = float(sh / torso)          # ~0.7+ facing the camera, small when sideways
        m.update(body_size=round(size, 2), visible_joints=int((seen & inside).sum()),
                 shoulder_torso_ratio=round(width_ratio, 2))
        ok_joint = dict(zip(BODY, seen & inside))
        full_side = max(sum(ok_joint[j] for j in (L_SHOULDER, L_HIP, L_KNEE, L_ANKLE)),
                        sum(ok_joint[j] for j in (R_SHOULDER, R_HIP, R_KNEE, R_ANKLE)))
        if full_side < 4:                   # need at least one whole body side: shoulder, hip, knee, ankle
            issue = "cut_off"
        elif size < 0.35:
            issue = "too_far"
        elif size > 0.97:
            issue = "too_close"
        elif exercise not in FLOOR and not _upright(pose):
            issue = "stand_up"            # lying on the floor but the exercise is a standing one
        elif exercise in SIDE_VIEW and exercise not in FLOOR and width_ratio > 0.45:
            issue = "need_side"
        elif exercise not in SIDE_VIEW and width_ratio < 0.3:
            issue = "need_front"
        elif exercise in FLOOR:
            obs = analyze(exercise, pose)
            if exercise == "plank" and not obs.in_position:
                issue = "get_in_position"
            elif exercise == "bridge" and _upright(pose):
                issue = "get_in_position"
    issue = issue or "ok"
    return {"ok": issue == "ok", "issue": issue, "message": MESSAGES[issue], "metrics": m}


def _upright(pose: Pose) -> bool:
    s = (pose.points[L_SHOULDER] + pose.points[R_SHOULDER]) / 2
    hp = (pose.points[L_HIP] + pose.points[R_HIP]) / 2
    v = hp - s
    return abs(v[1]) > abs(v[0])               # torso vertical -> still standing, not lying down


