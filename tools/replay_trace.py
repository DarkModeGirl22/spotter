"""Replay a real video through the full pipeline (OpenCV 5 pose -> analyzer -> agent) and save the trace.

Shows, frame by frame, how OpenCV output changes the agent's next decision. For every agent event
it also saves the annotated frame (skeleton, wrong joints in red, the decision written on it).

    python -m tools.replay_trace --video clip.mp4 --exercise plank --out docs/eval/replay
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2 as cv

from server.agent import coach
from server.vision.exercises import analyze
from server.vision.pose import PoseEstimator, draw_pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--exercise", default="plank")
    ap.add_argument("--dt", type=float, default=0.4, help="seconds between analyzed frames")
    ap.add_argument("--every", type=int, default=1, help="analyze one frame out of N")
    ap.add_argument("--out", default="docs/eval/replay")
    ap.add_argument("--mode", default="live", choices=["live", "video"])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    est, cap = PoseEstimator(), cv.VideoCapture(args.video)
    state, t, i, rows, events = coach.new_state(args.exercise, duration=10_000, mode=args.mode), 0.0, 0, [], []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % args.every:
            continue
        pose = est.estimate(img)
        obs = analyze(args.exercise, pose)
        state, actions, trace = coach.step(state, obs, t, pose)
        rows.append({"frame": i - 1, "t": round(t, 2), "faults": obs.faults, "metrics": obs.metrics,
                     "pose_ms": round(pose.latency_ms, 1) if pose else None})
        said = [a["text"] for a in actions if a["type"] == "say"]
        agent_events = [e for e in trace if e["event"] != "perceive"]
        if agent_events or said:
            vis = draw_pose(img, pose, set(obs.bad_joints)) if pose else img.copy()
            label = (",".join(obs.faults) or "form ok").upper()
            cv.rectangle(vis, (0, 0), (vis.shape[1], 46), (255, 255, 255), -1)
            cv.putText(vis, f"t={t:4.1f}s  OpenCV: {label}", (8, 18), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 0, 255) if obs.faults else (0, 0, 0), 1, cv.LINE_AA)
            cv.putText(vis, f"Coach: {' / '.join(said)[:70]}", (8, 38), cv.FONT_HERSHEY_SIMPLEX, 0.45,
                       (0, 0, 0), 1, cv.LINE_AA)
            name = f"{len(events):02d}_t{t:05.1f}.jpg"
            cv.imwrite(str(out / name), vis)
            events.append({"t": round(t, 2), "frame": i - 1, "opencv_faults": obs.faults, "agent": agent_events,
                           "spoken": said, "image": name})
        t += args.dt
    (out / "trace.json").write_text(json.dumps({"video": Path(args.video).name, "exercise": args.exercise,
                                                "events": events, "frames": rows}, indent=1, default=str))
    for e in events:
        print(f"t={e['t']:5.1f}s  opencv={','.join(e['opencv_faults']) or 'ok':22s} "
              f"agent={[x['event'] + ':' + str(x.get('decision') or x.get('result') or x.get('action')) for x in e['agent']]}"
              f"  says={e['spoken']}")


if __name__ == "__main__":
    main()
