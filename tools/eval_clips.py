"""Evaluate Spotter on labeled video clips (evidence for the technical report).

Put clips in a folder and describe them in labels.csv:
    clip,exercise,label,start_frame,end_frame      (frame range is optional)
    plank_good_1.mp4,plank,good,,
    plank_sag_1.mp4,plank,hips_low
    plank_knees_1.mp4,plank,knees_down

For each clip the script samples frames like the app does (every 0.4 s), runs the OpenCV 5
pose pipeline + exercise analyzer, and replays the coach agent loop on the clip's timeline.
Outputs (in --out): results.csv, summary.json, agent traces, and annotated frames for every
misclassified clip (failure cases).

    python -m tools.eval_clips --clips clips --out eval_out
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

import cv2 as cv

from server.agent import coach
from server.vision.exercises import analyze
from server.vision.pose import PoseEstimator, draw_pose

STEP_S = 0.4


def run_clip(est: PoseEstimator, path: Path, exercise: str, start: int = 0, end: int | None = None,
             every_frame: bool = False):
    cap = cv.VideoCapture(str(path))
    fps = cap.get(cv.CAP_PROP_FPS) or 30
    stride = 1 if every_frame else max(1, round(fps * STEP_S))
    frames, i = [], 0
    state, trace, actions = coach.new_state(exercise, duration=10_000), [], []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i > (end if end is not None else 10**9):
            break
        if i >= start and (i - start) % stride == 0:
            img = cv.resize(img, (480, round(img.shape[0] * 480 / img.shape[1])))
            pose = est.estimate(img)
            obs = analyze(exercise, pose)
            state, acts, tr = coach.step(state, obs, (i - start) / fps, pose)
            actions += acts
            trace += tr
            frames.append((img, pose, obs))
        i += 1
    cap.release()
    return frames, trace, actions


def clip_label(frames) -> tuple[str, Counter]:
    counts = Counter()
    usable = [o for _, _, o in frames if o.in_position]
    for o in usable:
        counts.update(o.faults or ["good"])
    if not usable:
        return "not_detected", counts
    label, n = counts.most_common(1)[0]
    # a fault must be present in at least 30% of usable frames to label the clip with it
    faults = [(f, c) for f, c in counts.items() if f != "good" and c >= 0.3 * len(usable)]
    return (max(faults, key=lambda x: x[1])[0] if faults else "good"), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="clips")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--every-frame", action="store_true", help="analyze every frame instead of one every 0.4 s")
    args = ap.parse_args()
    clips, out = Path(args.clips), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    est = PoseEstimator()
    rows, latencies = [], []
    with open(clips / "labels.csv") as f:
        items = list(csv.DictReader(f))
    for it in items:
        start = int(it.get("start_frame") or 0)
        end = int(it["end_frame"]) if it.get("end_frame") else None
        frames, trace, actions = run_clip(est, clips / it["clip"], it["exercise"], start, end, args.every_frame)
        name = Path(it["clip"]).stem + (f"_{start}-{end}" if end is not None else "")
        usable = [o for _, _, o in frames if o.in_position]
        frame_ok = [(not o.faults) if it["label"] == "good" else (it["label"] in o.faults) for o in usable]
        latencies += [p.latency_ms for _, p, _ in frames if p]
        pred, counts = clip_label(frames)
        detected = sum(1 for _, p, _ in frames if p) / max(1, len(frames))
        cues = [a["text"] for a in actions if a["type"] == "say" and a.get("kind") == "correction"]
        fixes = [e for e in trace if e["event"] == "recheck" and e["result"] == "fixed"]
        escalations = [e for e in trace if e["event"] == "decide" and e["decision"] != "cue_and_recheck"]
        rows.append({"clip": name, "exercise": it["exercise"], "label": it["label"], "predicted": pred,
                     "correct": pred == it["label"], "frames": len(frames),
                     "frame_accuracy": round(sum(frame_ok) / len(frame_ok), 3) if frame_ok else None,
                     "person_detected_pct": round(100 * detected, 1), "frame_votes": dict(counts),
                     "corrections": len(cues), "fixes_confirmed": len(fixes), "escalations": len(escalations)})
        (out / f"{name}.trace.json").write_text(json.dumps(trace, default=str, indent=1))
        if pred != it["label"]:  # save failure-case frames
            fail_dir = out / "failures" / name
            fail_dir.mkdir(parents=True, exist_ok=True)
            for k, (img, pose, obs) in enumerate(frames[:: max(1, len(frames) // 6)]):
                vis = draw_pose(img, pose, set(obs.bad_joints)) if pose else img
                cv.putText(vis, ",".join(obs.faults) or "good", (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv.imwrite(str(fail_dir / f"{k:02d}.jpg"), vis)

    labels = sorted({r["label"] for r in rows} | {r["predicted"] for r in rows})
    per_label = {}
    for lab in labels:
        tp = sum(r["label"] == lab and r["predicted"] == lab for r in rows)
        fp = sum(r["label"] != lab and r["predicted"] == lab for r in rows)
        fn = sum(r["label"] == lab and r["predicted"] != lab for r in rows)
        per_label[lab] = {"precision": round(tp / (tp + fp), 2) if tp + fp else None,
                          "recall": round(tp / (tp + fn), 2) if tp + fn else None, "support": tp + fn}
    summary = {
        "clips": len(rows), "accuracy": round(sum(r["correct"] for r in rows) / max(1, len(rows)), 3),
        "per_label": per_label,
        "confusion": {f"{r['label']}->{r['predicted']}": sum(1 for x in rows if (x['label'], x['predicted']) ==
                                                              (r['label'], r['predicted'])) for r in rows},
        "pose_latency_ms": {"median": round(statistics.median(latencies), 1) if latencies else None,
                            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1) if latencies else None},
    }
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
