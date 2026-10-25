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

MESSAGES = {   # several phrasings each; the app avoids saying the same one twice in a row
    "too_dark": ["It's a bit dark for me. Could you turn on a light?",
                 "I need a little more light to see you well."],
    "blurry": ["The picture is blurry. Maybe wipe the camera?",
               "I can't see clearly. Keep the phone still for me."],
    "no_person": ["Whenever you're ready, step in front of the phone.",
                  "I don't see you yet. Take your time."],
    "cut_off": ["I can only see part of you. Try moving the phone a bit further away.",
                "Part of your body is out of the picture. A little further back, please."],
    "too_far": ["You're quite far. You can come a bit closer.",
                "Come a little closer to the phone."],
    "too_close": ["You're a bit close. Take a step back.",
                  "A little further away, please, so I see all of you."],
    "need_side": ["For this one, turn sideways to the phone so I see your body line.",
                  "Could you turn to the side? I need to see you in profile."],
    "need_front": ["Face the phone for this one, please.",
                   "Turn towards the phone so I can see you from the front."],
    "rotate_phone": ["For this one, turn your phone sideways, so your whole body fits.",
                     "Lay the phone horizontally, that gives me room to see you from head to toe."],
    "get_in_position": ["Good spot. Get into position whenever you're ready.",
                        "The phone is well placed. Get into position, I'll start when you're set."],
    "stand_up": ["This one is done standing. Stand up whenever you're ready.",
                 "Let's do this one standing up."],
    "ok": ["I can see you well. Let's go!", "Perfect, you're all set. Here we go!",
           "Great position. Let's start!"],
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
    elif exercise in FLOOR and image.shape[0] > image.shape[1] * 1.1:
        issue = "rotate_phone"           # floor exercise filmed in portrait: the body cannot fit
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
    return {"ok": issue == "ok", "issue": issue, "message": MESSAGES[issue][0], "messages": MESSAGES[issue],
            "metrics": m}


def _upright(pose: Pose) -> bool:
    s = (pose.points[L_SHOULDER] + pose.points[R_SHOULDER]) / 2
    hp = (pose.points[L_HIP] + pose.points[R_HIP]) / 2
    v = hp - s
    return abs(v[1]) > abs(v[0])               # torso vertical -> still standing, not lying down


