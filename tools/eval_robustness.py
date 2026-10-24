"""Stress-test plank fault detection under real-world capture conditions.

Takes the hand-labeled frames of a real plank video and re-runs the full OpenCV 5 pipeline on
transformed copies: mirrored (other side), phone tilted, far / close, dark room, blur, low quality,
feet cut off. Reports frame accuracy per condition and saves examples of every failure.

    python -m tools.eval_robustness --clips clips --out docs/eval/robustness
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

from server.vision.exercises import analyze
from server.vision.pose import PoseEstimator, draw_pose


def rotate(img, deg):
    h, w = img.shape[:2]
    m = cv.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv.warpAffine(img, m, (w, h), borderMode=cv.BORDER_REPLICATE)


def far(img, s=0.55):
    h, w = img.shape[:2]
    small = cv.resize(img, (int(w * s), int(h * s)), interpolation=cv.INTER_AREA)
    canvas = cv.resize(cv.blur(img, (51, 51)), (w, h))
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    canvas[y:y + small.shape[0], x:x + small.shape[1]] = small
    return canvas


def close(img, s=1.3):
    h, w = img.shape[:2]
    big = cv.resize(img, (int(w * s), int(h * s)))
    y, x = (big.shape[0] - h) // 2, (big.shape[1] - w) // 2
    return big[y:y + h, x:x + w]


def dark(img, g=0.35):
    return np.clip(img.astype(np.float32) * g + np.random.default_rng(0).normal(0, 4, img.shape), 0, 255).astype(np.uint8)


def jpeg(img, q=15):
    ok, buf = cv.imencode(".jpg", img, [cv.IMWRITE_JPEG_QUALITY, q])
    return cv.imdecode(buf, cv.IMREAD_COLOR)


def feet_cut(img, frac=0.12):
    # crop the side of the frame where the feet are (left in the test video), then pad back
    h, w = img.shape[:2]
    c = int(w * frac)
    return cv.copyMakeBorder(img[:, c:], 0, 0, 0, c, cv.BORDER_REPLICATE)


CONDITIONS = {
    "original": lambda im: im,
    "mirrored (other side)": lambda im: cv.flip(im, 1),
    "phone tilted 10°": lambda im: rotate(im, 10),
    "phone tilted -10°": lambda im: rotate(im, -10),
    "far away (55%)": far,
    "close (130%)": close,
    "dark room": dark,
    "motion blur": lambda im: cv.blur(im, (9, 1)),
    "low quality JPEG": jpeg,
    "feet cut off": feet_cut,
}


def load_frames(clips: Path):
    rows = list(csv.DictReader(open(clips / "labels.csv")))
    out = []
    for r in rows:
        cap = cv.VideoCapture(str(clips / r["clip"]))
        i, s, e = 0, int(r.get("start_frame") or 0), int(r["end_frame"]) if r.get("end_frame") else 10**9
        while True:
            ok, img = cap.read()
            if not ok or i > e:
                break
            if i >= s:
                out.append((r["exercise"], r["label"], i, img))
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--out", default="docs/eval/robustness")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    est, frames = PoseEstimator(), load_frames(Path(args.clips))
    res, fails = defaultdict(lambda: defaultdict(int)), defaultdict(list)
    for cond, fn in CONDITIONS.items():
        for ex, label, i, img in frames:
            im = fn(img)
            pose = est.estimate(im)
            obs = analyze(ex, pose)
            r = res[cond]
            r["frames"] += 1
            r["detected"] += pose is not None
            ok = (not obs.faults) if label == "good" else (label in obs.faults)
            r["correct"] += ok
            r[f"{label}_n"] += 1
            r[f"{label}_ok"] += ok
            if not ok:
                fails[cond].append({"frame": i, "truth": label, "got": obs.faults, "metrics": obs.metrics})
                if len(fails[cond]) <= 2:
                    vis = draw_pose(im, pose, set(obs.bad_joints)) if pose else im.copy()
                    cv.putText(vis, f"{cond}: truth={label} got={','.join(obs.faults) or 'good'}", (6, 18),
                               cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv.LINE_AA)
                    safe = "".join(c if c.isalnum() else "_" for c in cond)
                    cv.imwrite(str(out / f"fail_{safe}_{i:03d}.jpg"), vis)
    labels = ["good", "hips_low", "hips_high"]
    md = ["| Condition | Person found | All frames | good | hips_low | hips_high |", "|---|---|---|---|---|---|"]
    summary = {}
    for cond, r in res.items():
        acc = r["correct"] / r["frames"]
        summary[cond] = {"accuracy": round(acc, 3), "detected": round(r["detected"] / r["frames"], 3),
                         **{l: f"{r[l + '_ok']}/{r[l + '_n']}" for l in labels}}
        md.append(f"| {cond} | {100 * r['detected'] / r['frames']:.0f} % | **{100 * acc:.0f} %** "
                  f"({r['correct']}/{r['frames']}) | " + " | ".join(summary[cond][l] for l in labels) + " |")
    (out / "robustness.md").write_text("\n".join(md) + "\n")
    (out / "robustness.json").write_text(json.dumps({"summary": summary, "failures": fails}, indent=1, default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
