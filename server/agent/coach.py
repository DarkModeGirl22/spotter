"""Spotter's perception -> decision -> action -> re-check loop.

The server is stateless: the browser sends the session `state` back with every frame, so the
same code runs locally and on AWS Lambda. Each call returns new actions plus trace events.
"""
from __future__ import annotations

import random
from copy import deepcopy

from ..vision.exercises import Observation
from . import brain

CUES = {
    "hips_low": "Lift your hips! Squeeze your belly.",
    "hips_high": "Hips down a little. Flat back.",
    "knees_down": "Knees up! You can do it.",
    "collapsed": "Push the floor away, lift your chest.",
    "not_in_plank_position": "Get into your plank. Elbows under shoulders.",
    "body_not_visible": "I can't see your whole body. Move the phone back.",
    "legs_not_visible": "I can't see your legs. Move the phone back.",
    "no_person": "I can't see you. Come back in front of the camera.",
    "knees_in": "Push your knees out!",
    "shallow": "Go a little lower!",
    "not_moving": "Keep moving, don't stop now!",
}
KNEE_PLANK_IGNORES = {"knees_down"}          # in a knee plank, knees on the floor is correct
PRAISE = ["Perfect, hold it!", "Yes! That's it.", "Much better!", "Beautiful form!"]
MOTIVATION = ["You're strong, keep going!", "Breathe. Stay tight.", "Looking great, keep it up!"]

PERSIST_FRAMES = 2     # a fault must be seen in 2 of the last 3 frames before we react
CUE_COOLDOWN = 4.0     # seconds between two cues
FIX_WINDOW = 5.0       # seconds the user gets to fix a fault before the agent escalates
ESCALATE_AFTER = 2     # escalate when the same fault needs a 2nd failed cue
ASK_TIMEOUT = 10.0     # unanswered question -> safest choice (pause the set)
STOP_FRAMES = 3        # ~1.2 s of "not exercising" frames = the user stopped
MIN_ACTIVE = 3.0       # only after the user has really started
RESUME_FRAMES = 2      # back in position for 2 frames = resume
REST_AFTER = 20.0      # stopped this long without coming back = rest, end the set
STOP_SIGNS = {"not_in_plank_position", "collapsed", "no_person", "not_moving", "body_not_visible",
              "legs_not_visible"}


def new_state(exercise: str, duration: int = 60) -> dict:
    return {
        "exercise": exercise, "variant": "standard", "duration": duration, "end_at": float(duration),
        "recent": [], "awaiting_fix": None, "cue_counts": {}, "last_cue_t": -99.0,
        "good_since": None, "milestones": [], "pending_question": None, "paused": False,
        "squat": {"phase": "up", "min_knee": 180.0, "reps": 0},
        "activity": {"last": None, "still_since": None}, "step": 0,
        # stop detection: time actually spent working, consecutive "stopped" frames, open pause
        "active_time": 0.0, "last_t": None, "stop_frames": 0, "stopped": None,
    }


def _persistent_fault(state: dict) -> str | None:
    recent = state["recent"][-3:]
    counts: dict[str, int] = {}
    for faults in recent:
        for f in faults:
            counts[f] = counts.get(f, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1], default=(None, 0))
    return best[0] if best[1] >= PERSIST_FRAMES else None


def _say(actions, state, t, text, kind):
    actions.append({"type": "say", "text": text, "kind": kind})
    state["last_cue_t"] = t


def step(state: dict, obs: Observation, t: float) -> tuple[dict, list, list]:
    """One loop iteration. t = seconds since the set started. Returns (state, actions, trace)."""
    state = deepcopy(state)
    state["step"] += 1
    actions: list[dict] = []
    trace: list[dict] = [{"event": "perceive", "t": round(t, 2), "obs": obs.to_json()}]
    if state.get("ended"):          # set already over (user rested, pain, timeout): nothing more to do
        return state, actions, trace

    faults = [f for f in obs.faults if not (state["variant"] == "knee" and f in KNEE_PLANK_IGNORES)]
    faults += _extra_faults(state, obs, t)
    state["recent"] = (state["recent"] + [faults])[-5:]

    # ---- did the user stop? pause the timer at once, say how long they held, let them choose
    stopping = bool(set(faults) & STOP_SIGNS)
    if state["last_t"] is not None and not state["stopped"] and not stopping:
        state["active_time"] += t - state["last_t"]
    state["last_t"] = t
    state["stop_frames"] = state["stop_frames"] + 1 if stopping else 0
    if state["stopped"]:
        return _while_stopped(state, stopping, t, actions, trace)
    if state["stop_frames"] >= STOP_FRAMES and state["active_time"] >= MIN_ACTIVE and not state["pending_question"]:
        held = int(round(state["active_time"]))
        state["stopped"] = {"at": t, "held": held, "back_frames": 0}
        state["awaiting_fix"] = None
        actions.append({"type": "pause_timer", "held": held})
        _say(actions, state, t, f"You held {held} seconds. That's already good! "
             "If you can, get back in position and I'll restart the clock. If not, take a rest.", "question")
        state["pending_question"] = {"question": f"You held {held} s. Continue or rest?",
                                     "options": ["keep_going", "stop_set"], "asked_at": t, "soft": True}
        actions.append({"type": "ask", **state["pending_question"]})
        trace.append({"event": "decide", "t": round(t, 2), "fault": sorted(set(faults) & STOP_SIGNS)[0],
                      "decision": "pause_on_stop", "held_s": held, "decided_by": "rules"})
        return state, actions, trace

    # ---- human in the loop: a question is open
    q = state["pending_question"]
    if q:
        if t - q["asked_at"] > ASK_TIMEOUT:
            state["pending_question"] = None
            state["paused"] = True
            _say(actions, state, t, "No answer, so let's pause and rest. Take a breath.", "safety")
            state["ended"] = True
            actions.append({"type": "end_set", "reason": "no_answer_timeout"})
            trace.append({"event": "act", "t": round(t, 2), "action": "end_set", "why": "question timed out"})
        return state, actions, trace

    # ---- time milestones
    left = state["end_at"] - t
    for mark, text in ((30, "Halfway there!"), (10, "Ten seconds left, you've got this!")):
        if left <= mark and mark not in state["milestones"] and state["duration"] > mark + 5:
            state["milestones"].append(mark)
            _say(actions, state, t, text, "motivation")
    if left <= 0:
        state["ended"] = True
        actions.append({"type": "end_set", "reason": "time"})
        return state, actions, trace

    fault = _persistent_fault(state)
    waiting = state["awaiting_fix"]

    # ---- re-check: did the last correction work?
    if waiting:
        last2 = state["recent"][-2:]
        fixed = len(last2) == 2 and all(waiting["fault"] not in f for f in last2)
        still = not fixed
        if fixed and state["recent"][-1]:
            # the corrected fault is gone but something else is wrong now (or the user left): no praise,
            # close this correction and let the next frames handle the new problem
            state["awaiting_fix"] = None
            trace.append({"event": "recheck", "t": round(t, 2), "fault": waiting["fault"],
                          "result": "replaced_by", "now": state["recent"][-1]})
            state["last_cue_t"] = min(state["last_cue_t"], t - CUE_COOLDOWN)
            return state, actions, trace
        if fixed:
            state["awaiting_fix"] = None
            trace.append({"event": "recheck", "t": round(t, 2), "fault": waiting["fault"], "result": "fixed",
                          "seconds_to_fix": round(t - waiting["cued_at"], 2)})
            _say(actions, state, t, random.choice(PRAISE), "praise")
            state["good_since"] = t
            return state, actions, trace
        if still and t - waiting["cued_at"] >= FIX_WINDOW:
            trace.append({"event": "recheck", "t": round(t, 2), "fault": waiting["fault"], "result": "not_fixed"})
            waiting["failed"] += 1
            if waiting["failed"] >= ESCALATE_AFTER or waiting["fault"] in ("collapsed", "no_person"):
                _escalate(state, waiting["fault"], obs, t, actions, trace)
                state["awaiting_fix"] = None
            else:
                waiting["cued_at"] = t
                _say(actions, state, t, CUES.get(waiting["fault"], "Fix your form!"), "correction")
                trace.append({"event": "act", "t": round(t, 2), "action": "say", "why": "repeat cue"})
            return state, actions, trace

    # ---- new fault: quick spoken correction, then wait and re-check
    if fault and not waiting and t - state["last_cue_t"] >= CUE_COOLDOWN:
        state["cue_counts"][fault] = state["cue_counts"].get(fault, 0) + 1
        state["awaiting_fix"] = {"fault": fault, "cued_at": t, "failed": 0}
        state["good_since"] = None
        _say(actions, state, t, CUES.get(fault, "Fix your form!"), "correction")
        trace.append({"event": "decide", "t": round(t, 2), "fault": fault, "decision": "cue_and_recheck",
                      "decided_by": "rules"})
        return state, actions, trace

    # ---- good form streak -> encouragement
    if not fault:
        state["good_since"] = state["good_since"] if state["good_since"] is not None else t
        if t - state["good_since"] >= 12 and t - state["last_cue_t"] >= 10:
            _say(actions, state, t, random.choice(MOTIVATION), "motivation")
            state["good_since"] = t
    return state, actions, trace


def _extra_faults(state: dict, obs: Observation, t: float) -> list[str]:
    """Faults that need memory across frames: squat depth and 'stopped moving'."""
    out = []
    if obs.exercise == "squat" and obs.in_position:
        s, k = state["squat"], obs.metrics["knee_angle_deg"]
        s["min_knee"] = min(s["min_knee"], k)
        if s["phase"] == "up" and k < 140:
            s["phase"] = "down"
        elif s["phase"] == "down" and k > 160:
            s["phase"], s["reps"] = "up", s["reps"] + 1
            if s["min_knee"] > 110:
                out.append("shallow")
            s["min_knee"] = 180.0
    elif "activity_points" in obs.metrics:
        a = state["activity"]
        pts = obs.metrics["activity_points"]
        if a["last"] is not None:
            move = sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) for p, q in zip(pts, a["last"]))
            if move < 0.03:
                a["still_since"] = a["still_since"] if a["still_since"] is not None else t
            else:
                a["still_since"] = None
            if a["still_since"] is not None and t - a["still_since"] > 3:
                out.append("not_moving")
        a["last"] = pts
    return out


def _escalate(state, fault, obs, t, actions, trace):
    ctx = {
        "exercise": state["exercise"], "variant": state["variant"], "fault": fault,
        "fault_count": state["cue_counts"].get(fault, 0) + 1,
        "seconds_into_set": round(t, 1), "seconds_left": round(state["end_at"] - t, 1),
        "metrics": obs.metrics,
    }
    d = brain.decide(ctx)
    trace.append({"event": "decide", "t": round(t, 2), "fault": fault, "decision": d["tool"],
                  "input": d["input"], "decided_by": d["decided_by"], "latency_ms": d.get("latency_ms")})
    apply_decision(state, d["tool"], d["input"], t, actions, trace)


def apply_decision(state, tool, args, t, actions, trace):
    if tool == "switch_variant":
        state["variant"] = args.get("variant", "knee")
        actions.append({"type": "switch_variant", "variant": state["variant"]})
        _say(actions, state, t, args.get("text", "Let's make it easier."), "plan_change")
    elif tool == "shorten_set":
        secs = max(5, min(30, int(args.get("seconds", 10))))
        state["end_at"] = min(state["end_at"], t + secs)
        actions.append({"type": "shorten_set", "end_at": state["end_at"]})
        _say(actions, state, t, args.get("text", f"{secs} more seconds!"), "plan_change")
    elif tool == "ask_user":
        state["pending_question"] = {"question": args["question"], "options": args.get("options") or
                                     ["keep_going", "easier_version", "stop_set"], "asked_at": t}
        actions.append({"type": "ask", **state["pending_question"]})
        _say(actions, state, t, args["question"], "question")
    else:
        _say(actions, state, t, args.get("text", "Keep going!"), "motivation")
    trace.append({"event": "act", "t": round(t, 2), "action": tool, "args": args})


def answer(state: dict, choice: str, t: float) -> tuple[dict, list, list]:
    """The human's reply to an `ask` action."""
    state = deepcopy(state)
    state["pending_question"] = None
    actions, trace = [], [{"event": "human", "t": round(t, 2), "choice": choice}]
    if choice == "stop_set":
        state["stopped"] = None
    if choice == "keep_going" and state.get("stopped"):
        _say(actions, state, t, "Great! Get back in position, the clock starts when I see you.", "motivation")
        return state, actions, trace
    if choice == "stop_set":
        state["ended"] = True
        actions.append({"type": "end_set", "reason": "user_stopped"})
        _say(actions, state, t, "Good call. Rest now, you did great.", "safety")
    elif choice == "easier_version" and state["exercise"] == "plank":
        apply_decision(state, "switch_variant",
                       {"variant": "knee", "text": "Knee plank it is. Keep your back flat!"}, t, actions, trace)
    elif choice == "easier_version":
        apply_decision(state, "shorten_set", {"seconds": 10, "text": "Ten easy seconds, then we rest!"},
                       t, actions, trace)
    else:
        _say(actions, state, t, "Let's go! Stay strong!", "motivation")
    return state, actions, trace


def _while_stopped(state, stopping, t, actions, trace):
    """Timer is paused. Resume by itself when the camera sees the user back in position."""
    st = state["stopped"]
    st["back_frames"] = 0 if stopping else st["back_frames"] + 1
    if st["back_frames"] >= RESUME_FRAMES:
        pause = t - st["at"]
        state["end_at"] += pause                      # the paused time does not count
        state["stopped"], state["stop_frames"], state["recent"] = None, 0, []
        state["pending_question"] = None
        actions.append({"type": "resume_timer", "end_at": state["end_at"]})
        actions.append({"type": "close_ask"})
        _say(actions, state, t, "You're back! The clock is running again. Let's go!", "motivation")
        trace.append({"event": "recheck", "t": round(t, 2), "fault": "stopped", "result": "resumed",
                      "paused_s": round(pause, 1)})
    elif t - st["at"] > REST_AFTER:
        state["stopped"], state["pending_question"] = None, None
        state["ended"] = True
        actions.append({"type": "end_set", "reason": "rested"})
        _say(actions, state, t, f"Good job, {st['held']} seconds. Take your rest.", "safety")
        trace.append({"event": "act", "t": round(t, 2), "action": "end_set", "why": "user did not come back"})
    return state, actions, trace


# ---------------------------------------------------------------- voice: the user talks to the coach
INTENTS = [  # checked in order; first match wins (pain always first)
    ("pain", ("pain", "hurt", "injur", "ouch", "dizzy", "j'ai mal", "douleur", "mal au")),
    ("easier_version", ("easier", "too hard", "simpler", "plus facile", "trop dur")),
    ("stop_set", ("rest", "stop", "can't", "cannot", "can not", "enough", "break", "i'm done", "finish",
                  "repos", "arrête", "j'en peux plus", "pause")),
    ("keep_going", ("continue", "keep going", "go on", "i can", "let's go", "ready", "yes", "again",
                    "one more", "on y va", "je continue", "oui")),
]


def intent_of(text: str) -> str | None:
    t = " " + text.lower().strip() + " "
    for intent, words in INTENTS:
        if any(w in t for w in words):
            return intent
    return None


def hear(state: dict, text: str, t: float) -> tuple[dict, list, list]:
    """Something the user said (speech-to-text in the browser). Same effect as a button, plus 'it hurts'."""
    intent = intent_of(text)
    trace = [{"event": "human", "t": round(t, 2), "heard": text, "intent": intent}]
    if intent is None:
        if state.get("pending_question"):
            actions = [{"type": "say", "text": "Sorry, I didn't get that. Say continue, or rest.", "kind": "question"}]
            return state, actions, trace
        return state, [], trace
    if intent == "pain":
        state = deepcopy(state)
        state["pending_question"], state["stopped"], state["ended"] = None, None, True
        actions = [{"type": "close_ask"}, {"type": "end_set", "reason": "pain"},
                   {"type": "say", "kind": "safety",
                    "text": "Stop right now. Rest and breathe. If the pain doesn't go away, see a professional."}]
        trace.append({"event": "act", "t": round(t, 2), "action": "end_set", "why": "user reported pain"})
        return state, actions, trace
    state, actions, tr = answer(state, intent, t)
    actions.insert(0, {"type": "close_ask"})
    return state, actions, trace + tr[1:]
