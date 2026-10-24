"""Unit tests with synthetic side-view keypoints (no camera needed)."""
import numpy as np

from server.agent import coach
from server.vision.exercises import analyze_plank
from server.vision.pose import (L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, Pose)


def plank_pose(hip_dy=0.0, knee_drop=False):
    """Person lying to the right, shoulder at x=100, ankle at x=500, floor at y=400."""
    pts = np.zeros((33, 2), np.float32)
    vis = np.full(33, 0.1, np.float32)
    S, A = np.array([100, 300]), np.array([500, 330])
    H = (S + A) / 2 + [0, hip_dy]
    K = H + (A - H) * 0.5
    if knee_drop:
        K = np.array([400, 395])
        A = np.array([500, 340])
    for i, p in {L_SHOULDER: S, L_ELBOW: [100, 395], L_WRIST: [160, 395],
                 L_HIP: H, L_KNEE: K, L_ANKLE: A}.items():
        pts[i] = p
        vis[i] = 0.95
    return Pose(pts, vis, 0.9, (640, 480), 5.0)


def test_good_plank_has_no_faults():
    obs = analyze_plank(plank_pose())
    assert obs.in_position and obs.faults == []


def test_hips_low_and_high():
    assert analyze_plank(plank_pose(hip_dy=50)).faults == ["hips_low"]
    assert analyze_plank(plank_pose(hip_dy=-60)).faults == ["hips_high"]


def test_knees_down():
    assert "knees_down" in analyze_plank(plank_pose(knee_drop=True)).faults


def test_standing_is_not_plank():
    pose = plank_pose()
    pose.points[L_HIP] = [105, 500]     # standing: whole body vertical
    pose.points[L_KNEE] = [108, 600]
    pose.points[L_ANKLE] = [110, 700]
    assert not analyze_plank(pose).in_position


def run(state, poses, t0=0.0, dt=0.5):
    acts, trace, t = [], [], t0
    for p in poses:
        state, a, tr = coach.step(state, analyze_plank(p), t)
        acts += a
        trace += tr
        t += dt
    return state, acts, trace, t


def test_loop_cue_then_recheck_fixed():
    state = coach.new_state("plank")
    state, acts, trace, t = run(state, [plank_pose()] * 2 + [plank_pose(hip_dy=50)] * 3)
    assert any(a["type"] == "say" and "hips" in a["text"].lower() for a in acts)
    state, acts, trace, t = run(state, [plank_pose()] * 3, t0=t)
    assert any(e["event"] == "recheck" and e["result"] == "fixed" for e in trace)


def test_loop_escalates_and_changes_plan():
    state = coach.new_state("plank")
    state, acts, trace, t = run(state, [plank_pose(knee_drop=True)] * 40)
    decisions = [e for e in trace if e["event"] == "decide" and e["decision"] != "cue_and_recheck"]
    assert decisions, "agent never escalated"
    assert state["variant"] == "knee" or state["pending_question"] or state["end_at"] < 60


def test_unanswered_question_pauses_safely():
    state = coach.new_state("plank")
    state["pending_question"] = {"question": "Stop?", "options": [], "asked_at": 5.0}
    state, acts, trace = coach.step(state, analyze_plank(plank_pose()), 16.0)
    assert any(a["type"] == "end_set" for a in acts)


def test_piked_plank_with_bent_knees_is_hips_high_not_knees_down():
    pose = plank_pose(hip_dy=-120)
    pose.points[L_KNEE] = [330, 250]    # knees slightly bent but far above the floor
    faults = analyze_plank(pose).faults
    assert "hips_high" in faults and "knees_down" not in faults


def test_setup_check_rejects_dark_frame_and_wrong_view():
    from server.vision.setup_check import check
    dark = np.full((240, 320, 3), 10, np.uint8)
    assert check("plank", dark, None)["issue"] == "too_dark"
    img = (np.random.default_rng(0).random((240, 320, 3)) * 255).astype(np.uint8)   # bright and sharp
    assert check("plank", img, None)["issue"] == "no_person"
    side = plank_pose()                       # lying sideways: fine for plank, wrong for squats
    assert check("plank", img, side)["issue"] == "ok"
    assert check("squat", img, side)["issue"] == "need_front"
