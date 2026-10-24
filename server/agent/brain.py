"""The coach's decision maker for plan changes.

When the vision loop sees that a quick cue did not fix a problem, it escalates here.
Amazon Bedrock (Claude, Converse API with tool use) picks ONE tool to call. If Bedrock is not
configured or fails, a deterministic policy picks instead, so the coach never stalls.
Every decision records who made it (`decided_by`) for the trace.
"""
from __future__ import annotations

import json
import os
import time

MODEL_ID = os.environ.get("SPOTTER_MODEL_ID", "")          # e.g. an Anthropic Claude model or inference profile id
REGION = os.environ.get("SPOTTER_BEDROCK_REGION", os.environ.get("AWS_REGION", "eu-north-1"))

TOOLS = [
    {"name": "say", "description": "Speak one short coaching sentence (max 12 words) to the user.",
     "schema": {"text": {"type": "string"}}},
    {"name": "switch_variant", "description": "Make the exercise easier or harder, e.g. standard plank -> knee plank.",
     "schema": {"variant": {"type": "string", "enum": ["standard", "knee"]}, "text": {"type": "string"}}},
    {"name": "shorten_set", "description": "End the current set early after N more seconds (5-30).",
     "schema": {"seconds": {"type": "integer"}, "text": {"type": "string"}}},
    {"name": "ask_user", "description": "Ask the user to choose before changing the plan (human approval). "
                                         "Use when stopping or when the user may be in pain or exhausted.",
     "schema": {"question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string",
                            "enum": ["keep_going", "easier_version", "stop_set"]}}}},
]

SYSTEM = (
    "You are Spotter, a warm but demanding home-workout coach. A computer-vision system watches the user "
    "and reports form faults. A quick spoken cue already failed to fix the problem. Choose exactly ONE tool. "
    "Prefer keeping the user moving safely: repeated hips sagging or knees dropping in a plank usually means "
    "fatigue, so offer the easier variant; signs of collapse, or the user disappearing from view, call for "
    "asking the user before continuing. Never push through pain. Keep spoken text short and encouraging."
)


def _bedrock_tools():
    specs = []
    for t in TOOLS:
        specs.append({"toolSpec": {
            "name": t["name"], "description": t["description"],
            "inputSchema": {"json": {"type": "object", "properties": t["schema"],
                                     "required": list(t["schema"].keys())}},
        }})
    return {"tools": specs, "toolChoice": {"any": {}}}


def fallback_policy(ctx: dict) -> dict:
    """Deterministic version of the same decision, used when Bedrock is unavailable."""
    fault, count = ctx["fault"], ctx["fault_count"]
    if fault == "no_person":
        return {"tool": "ask_user", "input": {
            "question": "I lost you. Are you okay? Keep going or stop?",
            "options": ["keep_going", "stop_set"]}}
    if fault == "collapsed" or (ctx.get("seconds_left", 60) > 40 and count >= 4):
        return {"tool": "ask_user", "input": {
            "question": "You keep dropping. Keep going, switch to an easier version, or stop?",
            "options": ["keep_going", "easier_version", "stop_set"]}}
    if ctx["exercise"] == "plank" and ctx.get("variant") == "standard" and fault in ("hips_low", "knees_down"):
        return {"tool": "switch_variant", "input": {
            "variant": "knee", "text": "Let's switch to knee plank. Knees down, back straight!"}}
    if ctx.get("seconds_left", 60) > 15:
        return {"tool": "shorten_set", "input": {"seconds": 10, "text": "Ten more seconds, then we rest!"}}
    return {"tool": "say", "input": {"text": "Almost there, hold strong!"}}


def decide(ctx: dict) -> dict:
    """Returns {"tool", "input", "decided_by", "latency_ms"}."""
    t0 = time.perf_counter()
    if MODEL_ID:
        try:
            import boto3
            client = boto3.client("bedrock-runtime", region_name=REGION)
            resp = client.converse(
                modelId=MODEL_ID,
                system=[{"text": SYSTEM}],
                messages=[{"role": "user", "content": [{"text": "Vision report:\n" + json.dumps(ctx)}]}],
                toolConfig=_bedrock_tools(),
                inferenceConfig={"maxTokens": 200, "temperature": 0.2},
            )
            for block in resp["output"]["message"]["content"]:
                if "toolUse" in block:
                    tu = block["toolUse"]
                    return {"tool": tu["name"], "input": tu["input"], "decided_by": "bedrock",
                            "latency_ms": round((time.perf_counter() - t0) * 1000)}
        except Exception as exc:  # network, permissions, throttling -> safe fallback
            out = fallback_policy(ctx)
            out.update(decided_by="rules", error=type(exc).__name__,
                       latency_ms=round((time.perf_counter() - t0) * 1000))
            return out
    out = fallback_policy(ctx)
    out.update(decided_by="rules", latency_ms=round((time.perf_counter() - t0) * 1000))
    return out
