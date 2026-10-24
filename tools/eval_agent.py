"""Evaluate the coach agent loop on scripted scenarios (task success, failure handling, human control).

Each scenario feeds a timeline of vision observations (what OpenCV would report every 0.4 s) into
the real agent loop and checks the outcome. Run:  python -m tools.eval_agent --out docs/eval
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from server.agent import brain, coach
from server.vision.exercises import Observation

DT = 0.4


def obs(exercise, *faults, in_position=True, person=True):
    return Observation(exercise, person, in_position, list(faults))


def timeline(*parts):
    """parts = (seconds, observation) pairs, expanded to one observation every DT seconds."""
    out = []
    for secs, o in parts:
        out += [o] * max(1, round(secs / DT))
    return out


def run(exercise, frames, answer_at=None, answer=None, duration=60):
    state, actions, trace, t = coach.new_state(exercise, duration), [], [], 0.0
    for o in frames:
        if answer_at is not None and t >= answer_at and state["pending_question"]:
            state, a, tr = coach.answer(state, answer, t)
            actions += a
            trace += tr
            if any(x["type"] == "end_set" for x in a):
                break
        state, a, tr = coach.step(state, o, t)
        actions += a
        trace += [e for e in tr if e["event"] != "perceive"]
        t += DT
        if any(x["type"] == "end_set" for x in a):
            break
    return state, actions, trace


def kinds(actions, kind):
    return [a for a in actions if a["type"] == "say" and a.get("kind") == kind]


GOOD, LOW, KNEES, GONE = obs("plank"), obs("plank", "hips_low"), obs("plank", "knees_down"), \
    obs("plank", "no_person", person=False, in_position=False)

SCENARIOS = [
    ("good form, whole set",
     timeline((60, GOOD)),
     lambda s, a, t: not kinds(a, "correction") and len(kinds(a, "motivation")) >= 2,
     "no corrections, only encouragement"),
    ("one-frame glitches (noise)",
     timeline((5, GOOD), (0.4, LOW), (3, GOOD), (0.4, KNEES), (3, GOOD)),
     lambda s, a, t: not kinds(a, "correction"),
     "single-frame faults are ignored (no false alarm)"),
    ("hips sag, user fixes after cue",
     timeline((5, GOOD), (3, LOW), (6, GOOD)),
     lambda s, a, t: any(e["event"] == "recheck" and e["result"] == "fixed" for e in t),
     "cue spoken, re-check confirms the fix, praise"),
    ("hips sag, user never fixes",
     timeline((5, GOOD), (30, LOW)),
     lambda s, a, t: any(e["event"] == "decide" and e["decision"] != "cue_and_recheck" for e in t)
     and (s["variant"] == "knee" or s["end_at"] < 60 or bool(s["pending_question"])),
     "cue, repeat, then the agent changes the plan"),
    ("knees drop repeatedly",
     timeline((5, GOOD), (25, KNEES)),
     lambda s, a, t: s["variant"] == "knee" and kinds(a, "correction")[-1]["text"] != coach.CUES["knees_down"]
     or (s["variant"] == "knee" and a.index(kinds(a, "plan_change")[0]) > a.index(kinds(a, "correction")[-1])),
     "switches to knee plank; knees on floor stop being a fault"),
    ("user leaves the camera, answers 'stop'",
     timeline((5, GOOD), (20, GONE)),
     lambda s, a, t: any(e["event"] == "human" and e["choice"] == "stop_set" for e in t)
     and any(x["type"] == "end_set" for x in a),
     "agent asks; the human decides to stop"),
    ("user leaves the camera, no answer",
     timeline((5, GOOD), (40, GONE)),
     lambda s, a, t: any(x["type"] == "end_set" and x["reason"] == "no_answer_timeout" for x in a),
     "unanswered question -> safe pause after 10 s"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/eval")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, traces = [], {}
    for name, frames, check, expect in SCENARIOS:
        answer_at = 13.0 if "answers 'stop'" in name else None
        s, a, t = run("plank", frames, answer_at, "stop_set")
        ok = bool(check(s, a, t))
        rows.append({"scenario": name, "expected": expect, "passed": ok,
                     "corrections": len(kinds(a, "correction")),
                     "plan_changes": sum(1 for e in t if e["event"] == "act" and e.get("action") in
                                         ("switch_variant", "shorten_set")),
                     "questions": sum(1 for x in a if x["type"] == "ask"),
                     "decided_by": sorted({e["decided_by"] for e in t if e["event"] == "decide"})})
        traces[name] = t

    # failure handling: Bedrock configured but unreachable -> rules take over, loop keeps going
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "invalid")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "invalid")
    brain.MODEL_ID, brain.REGION = "invalid-model-id", "eu-north-1"
    d = brain.decide({"exercise": "plank", "variant": "standard", "fault": "hips_low", "fault_count": 2,
                      "seconds_left": 40, "metrics": {}})
    rows.append({"scenario": "Bedrock unavailable", "expected": "fallback policy decides, no crash",
                 "passed": d["decided_by"] == "rules" and "error" in d, "corrections": "-", "plan_changes": "-",
                 "questions": "-", "decided_by": [d["decided_by"] + f" ({d.get('error')})"]})
    brain.MODEL_ID = ""

    (out / "agent_scenarios.json").write_text(json.dumps({"results": rows, "traces": traces}, indent=1, default=str))
    md = ["| Scenario | Expected behaviour | Pass | Corrections | Plan changes | Questions |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['scenario']} | {r['expected']} | {'✅' if r['passed'] else '❌'} | {r['corrections']} | "
           f"{r['plan_changes']} | {r['questions']} |" for r in rows]
    (out / "agent_scenarios.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n{sum(r['passed'] for r in rows)}/{len(rows)} scenarios passed")


if __name__ == "__main__":
    main()
