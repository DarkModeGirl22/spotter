"""Spotter's perception -> decision -> action -> re-check loop.

The server is stateless: the browser sends the session `state` back with every frame, so the
same code runs locally and on AWS Lambda. Each call returns new actions plus trace events.
"""
from __future__ import annotations

import random
import re
from copy import deepcopy

from ..vision.exercises import Observation
from ..vision.recognize import NAMES, WINDOW, compatible, features, recognize
from . import brain

WRONG_FRAMES = 6       # ~2.4 s of a clear, different exercise before the coach says something

# Several phrasings per situation: the coach picks one it has not just said, so it sounds human.
CUES = {
    "hips_low": ["Lift your hips a little, squeeze your belly.",
                 "Bring your hips up in line with your shoulders.",
                 "Hips are dropping. Tighten your core and lift them.",
                 "Imagine a straight line from head to heels. Hips up."],
    "hips_high": ["Lower your hips a bit, flat back.",
                  "Hips are a little high. Bring them down in line.",
                  "Drop the hips slightly, make your body a straight plank.",
                  "Flatten your back, hips down a touch."],
    "knees_down": ["Knees up if you can. You've got this.",
                   "Try lifting your knees off the floor.",
                   "Push through your toes and lift your knees."],
    "collapsed": ["Push the floor away and lift your chest.",
                  "Press into your forearms, lift up again.",
                  "Come back up. Shoulders over elbows."],
    "not_in_plank_position": ["Get into your plank. Elbows under shoulders.",
                              "Down into your plank when you're ready."],
    "body_not_visible": ["I'm missing part of your body. Could you move the phone back a little?",
                         "I can only see part of you. Try placing the phone a bit further away."],
    "legs_not_visible": ["I can't see your legs. Could you move the phone back a bit?",
                         "Your legs are out of the picture. A little further away, please."],
    "no_person": ["I lost you for a second. Come back in front of the phone.",
                  "Where did you go? Step back into the picture."],
    "knees_in": ["Push your knees out, over your toes.",
                 "Knees are caving in. Open them up.",
                 "Keep your knees in line with your feet."],
    "shallow": ["Try going a little lower.",
                "A bit deeper on the next one.",
                "Sit back a little more, as if onto a chair."],
    "not_moving": ["Keep moving, you're doing great.",
                   "Don't stop now, a few more.",
                   "Stay with me, keep it going."],
}
KNEE_PLANK_IGNORES = {"knees_down"}          # in a knee plank, knees on the floor is correct
PRAISE = ["Perfect, hold it there.", "Yes, that's it.", "Much better.", "Beautiful form.",
          "That's exactly right.", "Nice correction."]
MOTIVATION = ["You're strong, keep going.", "Breathe, and stay tight.", "Looking great, keep it up.",
              "Steady and strong, nice.", "You're doing really well."]


def pick(options, state=None):
    """A phrasing not used in the last few lines, so the coach does not repeat itself."""
    if isinstance(options, str):
        return options
    recent = (state or {}).get("said", [])
    fresh = [o for o in options if o not in recent] or list(options)
    return random.choice(fresh)


PERSIST_FRAMES = 2     # a fault must be seen in 2 of the last 3 frames before we react
CUE_COOLDOWN = 4.0     # seconds between two cues
FIX_WINDOW = 5.0       # seconds the user gets to fix a fault before the agent escalates
ESCALATE_AFTER = 2     # escalate when the same fault needs a 2nd failed cue
ASK_TIMEOUT = 10.0     # unanswered question -> safest choice (pause the set)
STOP_FRAMES = 4        # ~1.6 s of "not exercising" frames = the user stopped
MIN_ACTIVE = 5.0       # only after the user has really started
RESUME_FRAMES = 2      # back in position for 2 frames = resume
REST_AFTER = 20.0      # stopped this long without coming back = rest, end the set
STOP_SIGNS = {"not_in_plank_position", "collapsed", "no_person", "not_moving", "body_not_visible",
              "legs_not_visible"}


def new_state(exercise: str, duration: int = 60, mode: str = "live") -> dict:
    return {
        "exercise": exercise, "variant": "standard", "duration": duration, "end_at": float(duration),
        "recent": [], "awaiting_fix": None, "cue_counts": {}, "last_cue_t": -99.0,
        "good_since": None, "milestones": [], "pending_question": None, "paused": False,
        "squat": {"phase": "up", "min_knee": 180.0, "reps": 0},
        "activity": {"last": None, "still_since": None}, "step": 0,
        # stop detection: time actually spent working, consecutive "stopped" frames, open pause
        "active_time": 0.0, "last_t": None, "stop_frames": 0, "stopped": None,
        # exercise recognition: recent body features, frames in a row that look like another exercise
        "recog": [], "wrong_frames": 0, "wrong_said": 0, "mode": mode,
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
    state["said"] = (state.get("said", []) + [text])[-6:]


def step(state: dict, obs: Observation, t: float, pose=None) -> tuple[dict, list, list]:
    """One loop iteration. t = seconds since the set started. Returns (state, actions, trace)."""
    state = deepcopy(state)
    state["step"] += 1
    actions: list[dict] = []
    trace: list[dict] = [{"event": "perceive", "t": round(t, 2), "obs": obs.to_json()}]
    if state.get("ended"):          # set already over (user rested, pain, timeout): nothing more to do
        return state, actions, trace

    # ---- is this even the right exercise? (pose-feature recognition over the last ~4 s)
    if pose is not None or obs.person is False:
        state["recog"] = (state.get("recog", []) + [features(pose)])[-WINDOW:]
    seen, conf, evidence = recognize(state.get("recog", []))
    trace[0]["recognized"] = {"exercise": seen, "confidence": round(conf, 2)}
    if not compatible(state["exercise"], seen) and conf >= 0.5 and not state["pending_question"]:
        state["wrong_frames"] = state.get("wrong_frames", 0) + 1
    else:
        state["wrong_frames"] = 0
    if state["wrong_frames"] >= WRONG_FRAMES:
        return _wrong_exercise(state, seen, evidence, t, actions, trace)

    faults = [f for f in obs.faults if not (state["variant"] == "knee" and f in KNEE_PLANK_IGNORES)]
    faults += _extra_faults(state, obs, t)
    state["recent"] = (state["recent"] + [faults])[-5:]
    state["last_obs"] = {"person": obs.person, "in_position": obs.in_position}

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
            _say(actions, state, t, pick(PRAISE, state), "praise")
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
                _say(actions, state, t, pick(CUES.get(waiting["fault"], "Check your form."), state), "correction")
                trace.append({"event": "act", "t": round(t, 2), "action": "say", "why": "repeat cue"})
            return state, actions, trace

    # ---- new fault: quick spoken correction, then wait and re-check
    # (once the user has started, "stopped" signs are handled by the pause logic above, not nagged)
    if fault in STOP_SIGNS and state["active_time"] >= MIN_ACTIVE:
        fault = None
    if fault and not waiting and t - state["last_cue_t"] >= CUE_COOLDOWN:
        state["cue_counts"][fault] = state["cue_counts"].get(fault, 0) + 1
        state["awaiting_fix"] = {"fault": fault, "cued_at": t, "failed": 0}
        state["good_since"] = None
        _say(actions, state, t, pick(CUES.get(fault, "Check your form."), state), "correction")
        trace.append({"event": "decide", "t": round(t, 2), "fault": fault, "decision": "cue_and_recheck",
                      "decided_by": "rules"})
        return state, actions, trace

    # ---- good form streak -> encouragement
    if not fault:
        state["good_since"] = state["good_since"] if state["good_since"] is not None else t
        if t - state["good_since"] >= 12 and t - state["last_cue_t"] >= 10:
            _say(actions, state, t, pick(MOTIVATION, state), "motivation")
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
    q_prev = state.get("pending_question")
    state["pending_question"] = None
    actions, trace = [], [{"event": "human", "t": round(t, 2), "choice": choice}]
    if choice == "switch_exercise":
        to = (q_prev or {}).get("switch_to")
        if to:
            state["exercise"], state["variant"] = to, "standard"
            state["recog"], state["wrong_frames"], state["wrong_said"], state["recent"] = [], 0, 0, []
            actions.append({"type": "switch_exercise", "to": to})
            _say(actions, state, t, f"Okay, coaching {NAMES.get(to, to)} now.", "plan_change")
            trace.append({"event": "act", "t": round(t, 2), "action": "switch_exercise", "args": {"to": to}})
        return state, actions, trace
    if choice == "keep_going" and q_prev and q_prev.get("switch_to"):
        state["wrong_said"] = 0
        state["recog"] = []
        _say(actions, state, t, "Alright, let's keep going with this one.", "motivation")
        return state, actions, trace
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
INTENTS = [  # checked in order; first match wins (pain always first). Matched on whole words.
    ("pain", ("pain", "hurts?", "hurting", "injur\\w*", "ouch", "dizzy", "j'ai mal", "douleur", "mal au")),
    ("form_check", ("am i doing (it|this|that|ok|okay|good|well|right|correctly)", "doing it right", "doing this right",
                    "how'?s my form", "how is my form", "how'?s my (plank|squat|position|posture)", "check my form",
                    "is (this|that|it|my form) (good|right|ok|okay|correct)", "how am i doing", "how do i look",
                    "is my (back|body|position|posture) (straight|ok|okay|good|right)", "correct form", "good form\\??$",
                    "est-ce que c'est bien", "c'est bon")),
    ("time_left", ("how long", "how much (time|longer)", "time left", "seconds left", "how many seconds",
                   "combien de temps")),
    ("reps", ("how many (reps|squats|have i done)", "rep count", "count")),
    ("what_next", ("what'?s next", "what is next", "what comes next", "what next", "next exercise\\?",
                   "which exercise", "what are we doing")),
    ("end_workout", ("end (the |my )?(workout|session|training)", "stop (the |my )?(workout|session|training)",
                     "finish (the |my )?(workout|session|training)", "quit", "end it", "i'?m done with (the |this )?workout",
                     "end", "terminate", "fin", "j'arr[eê]te tout")),
    ("skip", ("skip", "next exercise", "next one", "move on", "go to the next", "suivant")),
    ("pause", ("pause", "wait", "hold on", "one (sec|second|moment|minute)", "give me a (sec|second|moment|minute)",
               "time out", "attends")),
    ("resume", ("resume", "unpause", "i'?m back", "i'?m ready", "go ahead", "start again", "let'?s go", "go",
                "reprends", "c'est reparti")),
    ("easier_version", ("easier", "too hard", "simpler", "plus facile", "trop dur")),
    ("stop_set", ("rest", "stop", "can'?t", "cannot", "can not", "enough", "break", "i'?m done", "finish",
                  "repos", "arr[eê]te", "j'en peux plus")),
    ("keep_going", ("continue", "keep going", "go on", "i can", "ready", "yes", "yeah", "again", "one more",
                    "on y va", "je continue", "oui")),
]
_INTENT_RE = [(i, re.compile(r"(?<![\w'])(" + "|".join(ws) + r")(?![\w'])")) for i, ws in INTENTS]
QUESTION_START = ("what", "how", "is ", "am i", "are ", "can ", "should", "do i", "does", "why", "when", "where")

FAULT_SAYS = {   # what the coach says is wrong, in plain words (answer to "am I doing it right?")
    "hips_low": "your hips are dropping", "hips_high": "your hips are too high",
    "knees_down": "your knees are on the floor", "collapsed": "you've sunk down onto the floor",
    "knees_in": "your knees are caving in", "shallow": "you're not going very deep",
    "not_moving": "you've stopped moving",
}
GOOD_SAYS = {
    "plank": "Your body is in a straight line from shoulders to heels.",
    "squat": "Good depth, and your knees stay over your toes.",
}


def intent_of(text: str) -> str | None:
    t = " ".join(text.lower().replace("\u2019", "'").split())
    for intent, rx in _INTENT_RE:
        if rx.search(t):
            return intent
    return None


def form_answer(state: dict) -> str:
    """'Am I doing it right?' -> yes/no and what to change, from what OpenCV saw in the last frames."""
    recent = (state.get("recent") or [])[-3:]
    last = state.get("last_obs") or {}
    if not recent or last.get("person") is False:
        return "I can't see you right now. Get in front of the camera and ask me again."
    counts: dict[str, int] = {}
    for faults in recent:
        for f in faults:
            counts[f] = counts.get(f, 0) + 1
    sure = [f for f, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= max(1, len(recent) - 1)]
    if "no_person" in sure or "body_not_visible" in sure or "legs_not_visible" in sure:
        f = next(f for f in sure if f in ("no_person", "body_not_visible", "legs_not_visible"))
        return "I can't check yet. " + CUES[f][0]
    if any(f.startswith("not_in_") for f in sure) or last.get("in_position") is False:
        name = NAMES.get(state.get("exercise"), "the exercise")
        return f"Not yet, you're not in position for {name}. " + CUES["not_in_plank_position"][0] \
            if state.get("exercise") == "plank" else f"Not yet, I don't see you doing {name}. Get into position."
    bad = [f for f in sure if f in FAULT_SAYS and not (state.get("variant") == "knee" and f in KNEE_PLANK_IGNORES)]
    if not bad:
        good = GOOD_SAYS.get(state.get("exercise"), "Nice and controlled.")
        return pick(["Yes, that's right. ", "Yes, you're doing it well. ", "Yes, that looks good. "], state) + good
    text = f"Not quite: {FAULT_SAYS[bad[0]]}. {CUES[bad[0]][0]}"
    if len(bad) > 1:
        text += f" Also, {FAULT_SAYS[bad[1]]}. {CUES[bad[1]][0]}"
    return text


def hear(state: dict | None, text: str, t: float, phase: str = "work", left: float | None = None,
         next_name: str | None = None) -> tuple[dict, list, list]:
    """Something the user said (browser speech-to-text), at any moment of the workout.
    Commands (pause, resume, skip, end), questions (form, time, reps, what's next) and answers to the
    coach's open question. The client owns the timers, so pause/skip/end come back as actions."""
    state = deepcopy(state) if state else {"exercise": None, "pending_question": None, "said": []}
    intent = intent_of(text)
    q = state.get("pending_question") or {}
    low = " " + text.lower() + " "
    if q.get("switch_to") and intent not in ("pain", "end_workout", "pause", "skip", "form_check"):
        if any(w in low for w in (" yes", "switch", "sure", "okay", " ok ", "please", "oui", "change")):
            intent = "switch_exercise"
        elif any(w in low for w in (" no ", " no,", "keep", "stay", " non")):
            intent = "keep_going"
    if q and intent == "resume":                  # "go" while the coach asks "continue or rest?"
        intent = "keep_going"
    trace = [{"event": "human", "t": round(t, 2), "heard": text, "intent": intent, "phase": phase}]
    actions: list[dict] = []

    def say(line, kind="answer"):
        _say(actions, state, t, line, kind)

    if intent == "pain":
        state["pending_question"], state["stopped"], state["ended"] = None, None, True
        actions += [{"type": "close_ask"}, {"type": "end_set", "reason": "pain"}]
        say("Stop right now. Rest and breathe. If the pain doesn't go away, see a professional.", "safety")
        trace.append({"event": "act", "t": round(t, 2), "action": "end_set", "why": "user reported pain"})
    elif intent == "end_workout":
        state["pending_question"], state["ended"] = None, True
        actions += [{"type": "close_ask"}, {"type": "end_workout"}]
        say("Okay, ending the workout here. Well done today!", "plan_change")
        trace.append({"event": "act", "t": round(t, 2), "action": "end_workout", "why": "user asked"})
    elif intent == "skip":
        state["pending_question"], state["ended"] = None, True
        actions += [{"type": "close_ask"}, {"type": "skip"}]
        say("Sure, moving on." if phase != "rest" else "Okay, let's start right away.", "plan_change")
        trace.append({"event": "act", "t": round(t, 2), "action": "skip", "why": "user asked"})
    elif intent == "pause":
        state["pending_question"] = None
        state["last_t"] = None                    # paused time doesn't count as holding time
        actions += [{"type": "close_ask"}, {"type": "pause_workout"}]
        say(pick(["Sure, take a moment. Say 'go' when you're ready.",
                  "Paused. Tell me 'resume' when you want to continue."], state), "plan_change")
        trace.append({"event": "act", "t": round(t, 2), "action": "pause_workout", "why": "user asked"})
    elif intent == "resume":
        state["last_t"], state["stop_frames"] = None, 0
        actions.append({"type": "resume_workout"})
        say(pick(["Let's go!", "Back to it, you've got this.", "Here we go."], state), "motivation")
        trace.append({"event": "act", "t": round(t, 2), "action": "resume_workout", "why": "user asked"})
    elif intent == "form_check":
        if phase != "work":
            say("We haven't started this one yet. Ask me again once the timer is running.")
        else:
            line = form_answer(state)
            say(line, "answer")
            trace.append({"event": "act", "t": round(t, 2), "action": "answer_form", "args": {"text": line},
                          "why": "user asked; answered from the last OpenCV frames"})
    elif intent == "time_left":
        secs = left if left is not None else (state.get("end_at", 0) - t if phase == "work" else None)
        if secs is None:
            say("We're about to start.")
        else:
            s = max(0, int(round(secs)))
            what = "of rest" if phase == "rest" else "left" if phase == "work" else "before we start"
            say(f"{s} seconds {what}." + (" You've got this!" if phase == "work" and s > 0 else ""))
    elif intent == "reps":
        reps = (state.get("extra") or {}).get("squat", {}).get("reps")
        say(f"That's {reps} so far." if reps is not None and state.get("exercise") == "squat"
            else "This one is about time, not reps. Just hold good form.")
    elif intent == "what_next":
        say(f"Next is {next_name}." if next_name else "This is the last exercise. Finish strong!")
    elif intent is None:
        if q:
            say("Sorry, I didn't get that. Say continue, or rest.", "question")
        elif low.strip().startswith(QUESTION_START) or text.strip().endswith("?"):
            say("I can tell you if you're doing it right, how long is left, or what's next. "
                "You can also say pause, skip, or end.")
    elif phase != "work":                        # answers like "continue" / "rest" only mean something mid-set
        if intent == "keep_going" and phase == "rest":
            actions.append({"type": "skip"})
            say("Okay, let's start right away.", "plan_change")
    else:
        state, acts, tr = answer(state, intent, t)
        return state, [{"type": "close_ask"}] + acts, trace + tr[1:]
    return state, actions, trace


def _wrong_exercise(state, seen, evidence, t, actions, trace):
    """The camera shows a different exercise than planned: say so, then let the human decide."""
    expected = state["exercise"]
    state["wrong_frames"], state["awaiting_fix"] = 0, None
    state["wrong_said"] = state.get("wrong_said", 0) + 1
    trace.append({"event": "decide", "t": round(t, 2), "fault": "wrong_exercise", "decision": "wrong_exercise",
                  "expected": expected, "seen": seen, "evidence": evidence, "decided_by": "rules"})
    if state.get("mode") == "video" or state["wrong_said"] >= 2:
        q = (f"This looks like {NAMES.get(seen, seen)}, not {NAMES.get(expected, expected)}. "
             f"Should I coach it as {NAMES.get(seen, seen)}?")
        state["pending_question"] = {"question": q, "options": ["switch_exercise", "keep_going"],
                                     "asked_at": t, "switch_to": seen}
        actions.append({"type": "ask", **{k: v for k, v in state["pending_question"].items() if k != "switch_to"}})
        _say(actions, state, t, q, "question")
    else:
        floor_now, floor_planned = seen in ("plank", "bridge"), expected in ("plank", "bridge")
        hint = (" Let's stand up for it." if floor_now and not floor_planned
                else " Let's get down on the floor for it." if floor_planned and not floor_now else "")
        _say(actions, state, t, pick([
            f"That looks like {NAMES.get(seen, seen)}, but this one is {NAMES.get(expected, expected)}.{hint}",
            f"Hmm, I see {NAMES.get(seen, seen)}. Right now we're doing {NAMES.get(expected, expected)}.{hint}",
        ], state), "correction")
    trace.append({"event": "act", "t": round(t, 2), "action": "flag_wrong_exercise", "args": {"seen": seen}})
    return state, actions, trace
