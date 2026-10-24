# Spotter

A talking AI workout coach. Your phone camera watches your form with **OpenCV 5**, an agent decides what to do about it, and the coach **tells you out loud**, re-checks, and changes the plan when your body needs it.

Built for the OpenCV AI Competition 2026 (Agentic Vision path).

## What it does

- Pick **Lower body** or **Upper body**: 4 exercises × 60 s, 30 s rest.
- Top half of the screen: the coach, an animated girl doing the move and talking.
- Bottom half: you, with your skeleton drawn on top. Wrong joints turn red.
- The agent loop, for example in a plank:
  1. **See:** OpenCV 5 finds 33 body keypoints and measures the shoulder–hip–ankle line.
  2. **Decide:** hips sagging for more than ~1 s means a correction is needed.
  3. **Act:** the coach says "Lift your hips! Squeeze your belly."
  4. **Re-check:** the next frames show whether you fixed it, then "Perfect, hold it!", or…
  5. **Escalate:** the correction failed twice, so the agent (Amazon Bedrock) changes the plan: switch to knee plank, shorten the set, or **ask you** ("Keep going, easier version, or stop?"). No answer in 10 s means it pauses you (safe default).

Every step is written to a trace (in the app and in CloudWatch Logs).

## Architecture

```
Phone browser ──JPEG every 0.4 s──▶ AWS Lambda (container, Function URL)
  camera + coach animation            ├─ OpenCV 5 dnn: person detector + pose (33 keypoints)
  speaks actions, asks questions      ├─ exercise analyzers → faults (hips_low, knees_down…)
                                      ├─ coach agent loop: cue → re-check → escalate
                                      │     └─ Amazon Bedrock (Claude, tool use) picks plan changes
                                      ├─ Amazon Polly: coach voice (cached MP3s)
                                      └─ CloudWatch Logs: one JSON line per agent event
```

The server is stateless: the browser sends the agent state back with each frame, so it scales on Lambda with no database.

| Part | Where |
|---|---|
| Pose models (OpenCV Zoo, Apache-2.0) | `models/` |
| Pose + drawing | `server/vision/pose.py` |
| Form rules per exercise | `server/vision/exercises.py` |
| Agent loop | `server/agent/coach.py` |
| Bedrock decision + fallback policy | `server/agent/brain.py` |
| Voice (Polly) | `server/agent/voice.py` |
| Web app + coach figure | `web/` |
| Evaluation on labeled clips | `tools/eval_clips.py` |

**Form checks today:** plank (hips low, hips high, knees down, collapse) and squat (depth, knees caving in). The other moves track whether you keep moving.

## Run locally

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --port 8080
# open http://localhost:8080  (camera works on localhost)
```

Without AWS credentials, the coach uses the browser's voice and the built-in fallback policy. With credentials:

```bash
export SPOTTER_MODEL_ID=<Bedrock Claude model or inference profile id>
export SPOTTER_BEDROCK_REGION=eu-north-1
```

## Test

```bash
pytest -q                                              # unit tests (synthetic keypoints)
python -m tools.eval_clips --clips clips --out eval_out   # evaluation on labeled clips
```

## Responsible use

Frames are analyzed in memory and discarded. Nothing is recorded. The agent never pushes through signs of collapse: it asks, and pauses if you don't answer. Spotter is not medical advice.
