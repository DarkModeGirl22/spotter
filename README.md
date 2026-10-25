# Spotter

**A talking workout coach that watches your form with OpenCV 5, tells you out loud how to fix it, checks that you did, and changes the plan, with your approval, when your body needs it.**

OpenCV AI Competition 2026 (powered by AWS) · Agentic Vision path · Team: Hind Taouis

🟢 **Live on AWS Lambda:** https://3iblqdeawzjnsky7ihd6gu3jn40giofz.lambda-url.eu-north-1.on.aws/
(open it on a phone or in Chrome, allow camera and microphone)

---

## Submission checklist

| # | Required item | Where |
|---|---|---|
| 1 | Technical report | [docs/REPORT.md](docs/REPORT.md) |
| 2 | Code repository | this repo |
| 3 | Pinned dependencies + build, deploy, test instructions | [requirements.txt](requirements.txt) · [Run](#run-locally) · [Deploy](#deploy-to-aws) · [Test](#test-and-reproduce-the-evaluation) |
| 4 | Architecture diagram (OpenCV 5 + AWS + agent) | [docs/img/architecture.png](docs/img/architecture.png) |
| 5 | Working web endpoint | live link above · health check: [/api/health](https://3iblqdeawzjnsky7ihd6gu3jn40giofz.lambda-url.eu-north-1.on.aws/api/health) |
| 6 | Video (≤ 5 min) | link on the Devpost page |
| 7 | Evaluation with failure cases | [REPORT §6](docs/REPORT.md#6-evaluation) · [docs/eval/](docs/eval/) |
| + | Agentic Vision: agent workflow diagram | [docs/img/agent_loop.png](docs/img/agent_loop.png) |
| + | Agentic Vision: trace where OpenCV output changes a later decision | [trace strip](docs/img/trace_strip.jpg) · [replay trace](docs/eval/replay_plank/) |
| + | Agentic Vision: evaluation of task success, failure handling, observability, human control | [REPORT §6.2](docs/REPORT.md#62-agent-task-success-failure-handling-human-control) |

---

## The problem

Most people train at home alone, following videos. A video can't see you: it can't tell you that your hips are sagging in a plank (which loads the lower back instead of the core), and it keeps going while your form falls apart. Spotter is the coach that sees you, using only a phone.

## What the user experiences

**Spotter is made for practising live:** you train in front of your phone and the coach reacts to you in real time. *Upload a video* exists only so the loop can be demonstrated and tested on recorded footage (for judges, and for our own evaluation); the coaching logic is the same.

1. **Home:** *Practice live* (camera, the real product) or *Upload a video* (demo and testing).
2. **Workout:** Lower body (squats, reverse lunges, glute bridge, side leg raises) or Upper body (plank, arm circles, overhead press, wall push-ups). 4 × 60 s, 30 s rest. These 8 exercises are a starting set to show how the coach works; see [What's next](#whats-next).
3. **Camera check before every move:** the coach waits while you set the phone down, then tells you what to fix ("Turn sideways to the phone", "It's too dark", "Turn the phone sideways for floor exercises"). The set starts only once she can actually judge the exercise.
4. **During the set:** top half, the coach (an original animated character drawn in code) does the move and talks; bottom half, your camera with your skeleton, wrong joints in red.
5. **You talk to her, like a real coach, at any moment:**

| You say | She does |
|---|---|
| "pause" / "wait" · "go" / "resume" | pauses / restarts the timer |
| "skip" / "next exercise" · "end the workout" | moves on / ends and shows the summary |
| **"Am I doing it right?"** | answers from the last camera frames: "Yes, your body is in a straight line from shoulders to heels." or "Not quite: your hips are dropping. Lift your hips a little." |
| "How long is left?" · "How many reps?" · "What's next?" | answers |
| "make it easier" · "rest" | knee plank or a shorter set · ends the set |
| **"It hurts"** | stops the set at once, with a safety message |

6. **Summary:** corrections, confirmed fixes, time to fix, stops, plan changes, plus a downloadable agent trace.

---

## How it scores on the judging criteria

### Agentic Vision: a real perceive → decide → act → re-check loop

![Agent loop](docs/img/agent_loop.png)

| Stage | What happens | Code |
|---|---|---|
| **Perceive** | OpenCV 5 finds 33 body keypoints; geometry turns them into faults (`hips_low`, `knees_down`, `knees_in`…) | `server/vision/` |
| **Filter** | a fault must persist in 2 of the last 3 frames (no false alarms from one bad frame) | `coach._persistent_fault` |
| **Act** | a short spoken cue, varied so she doesn't repeat herself | `coach.step` |
| **Re-check** | next frames decide: **fixed** → praise · **replaced by another fault** → no praise, handle the new one · **not fixed** after 5 s → repeat once | `coach.step` |
| **Decide** | second failed cue → choose a tool: `switch_variant` (knee plank), `shorten_set`, `ask_user`, `say`. Amazon Bedrock (Claude, tool use) when enabled; a deterministic policy otherwise and whenever Bedrock fails | `agent/brain.py` |
| **Human control** | questions are answered by voice or tap; no answer in 10 s → safe pause, never "push through" | `coach.answer`, `coach.hear` |
| **Notice stops** | you drop out of position → timer pauses at once, "You held 32 seconds. Continue or rest?"; back in position → resumes by itself, paused time not counted | `coach._while_stopped` |
| **Notice the wrong exercise** | pose-signature recognition over ~4 s: "This looks like a plank, not squats. Should I coach it as a plank?" | `vision/recognize.py` |
| **Active perception** | before each move, checks light, sharpness (`cv.Laplacian`), framing, distance and camera angle, and guides the user | `vision/setup_check.py` |
| **Observability** | every event (perceive, decide, act, recheck, human, setup) goes to the in-app trace panel, the downloadable JSON trace and CloudWatch Logs | `server/app.py` |

#### Evidence: the image changes what the coach does next

The rules ask that *"the visual evidence must change what the system does next."* Here is a real replay ([`tools/replay_trace.py`](tools/replay_trace.py), full log in [docs/eval/replay_plank/trace.json](docs/eval/replay_plank/trace.json)): each frame is what OpenCV 5 saw, and the line under it is the action that frame caused.

![OpenCV 5 output changes the next action](docs/img/trace_strip.jpg)

| Time | Image → OpenCV 5 says | Agent decision | Coach does |
|---|---|---|---|
| 12.0 s | form OK | keep going | "You're strong, keep going!" |
| 16.0 s | **hips_low** (red hip joint) | `cue_and_recheck` | "Lift your hips! Squeeze your belly." |
| 20.8 s | new image: form OK | re-check → **fixed** | "Beautiful form!" |
| 25.2 s | **hips_high** | `cue_and_recheck` | "Hips down a little. Flat back." |
| 29.6 s | new image: form OK | re-check → **fixed** | "Perfect, hold it!" |

Each correction is verified on the next images, not assumed. When the next images still show the fault, the decision changes instead: repeat the cue, then a plan change (knee plank, shorter set) or a question to the user. Knees down repeatedly → switch to knee plank, after which `knees_down` stops counting as a fault. Person gone → timer paused and a question. All of these paths are tested in [REPORT §6.2](docs/REPORT.md#62-agent-task-success-failure-handling-human-control).

### Technical execution: substantive OpenCV 5

| Step | OpenCV 5 API |
|---|---|
| Decode browser frames | `cv.imdecode` |
| Person detection + 33-keypoint pose (OpenCV Zoo MediaPipe ONNX models) | `cv.dnn.readNet`, `forward` |
| Align the body before the pose model | `cv.getRotationMatrix2D`, `cv.warpAffine`, `cv.copyMakeBorder`, `cv.resize` |
| Dark frames | CLAHE on the LAB lightness channel (`cv.createCLAHE`, `cv.cvtColor`) |
| Small or far person | crop + `cv.resize` around the body, detect again |
| Camera check | mean grey level, `cv.Laplacian` variance for blur |
| Evidence images, offline evaluation | `cv.line`, `cv.circle`, `cv.putText`, `cv.imwrite`, `cv.VideoCapture` |

Form rules are geometry normalised by body length, so they don't depend on distance: plank (hips low, hips high, knees down, collapse), squat (depth via a rep state machine, knees caving in), motion tracking for the other moves. **OpenCV 5 port:** the new DNN engine returns detector outputs in a different order than 4.x; we match them by tensor shape.

### Results (evaluation)

| Test | Result |
|---|---|
| Plank faults on real video, every frame labeled | **74/74 frames correct**, 4/4 segments, person found in 98.7 % of frames |
| Agent scenarios (good form, noise, fix, no fix, knees drop, stop and return, rest, no answer, Bedrock outage) | **10/10 pass** |
| Robustness: far away / dark room / blur / low JPEG, before → after fixes | 64→**81 %** · 67→**83 %** · 88→**91 %** · 77→**84 %** |
| Exercise recognition on real plank, squat, lunge videos | **0 false alarms**; wrong exercise detected in 4–8 s |
| Latency on Lambda | OpenCV 5: **35–44 ms** warm; browser round trip **≈ 0.1 s** |
| Unit tests | **13 pass** (`pytest`) |

### Real-world impact

Home training with a phone, no equipment, no coach. Spotter targets the faults that hurt people (sagging hips, knees caving in), adapts instead of pushing (easier variant, shorter set), and stops at once on "it hurts".

### User experience

Voice in both directions (Amazon Polly neural voice, browser speech recognition), human-like pacing (time to set up, no praise before she can see you, no repeated sentences), clean 2018-style app in white/black/red, works on phones (portrait and landscape), separate pages with back buttons, buttons kept as a fallback for noisy rooms.

### Responsible cloud delivery

- **Serverless, pay per request:** AWS Lambda + Function URL, no server running 24/7; Bedrock is called only on escalations, never per frame; a $20 monthly spend limit on the account.
- **Privacy:** frames are analysed in memory and discarded; nothing is stored; traces contain only numbers and labels.
- **Stateless:** the agent state travels with each request, so no database and nothing about the user kept on the server.
- **Least-privilege role** in `deploy/deploy.sh` (logs, Polly, Bedrock only); 14-day log retention.
- **Not medical advice;** stop if something hurts.

---

## Architecture

![Architecture](docs/img/architecture.png)

```
Phone browser ──JPEG every 0.4 s + agent state──▶ AWS Lambda (Python 3.13, 2 GB, Function URL)
  camera, coach animation                          ├─ OpenCV 5 dnn: person detector + pose (33 keypoints)
  speech recognition (user talks)                  ├─ camera check · form analyzers · exercise recognition
  plays the voice, shows questions                 ├─ coach agent: cue → re-check → decide → ask
                                                   │     └─ Amazon Bedrock (Claude, tool use), rules fallback
                                                   ├─ Amazon Polly: coach voice (cached MP3s)
                                                   └─ CloudWatch Logs: one JSON line per agent event
```

| Part | Where |
|---|---|
| Pose models (OpenCV Zoo, Apache-2.0) | `models/` |
| Pose, CLAHE, zoom re-detect, drawing | `server/vision/pose.py` |
| Form rules per exercise | `server/vision/exercises.py` |
| Camera check | `server/vision/setup_check.py` |
| Exercise recognition | `server/vision/recognize.py` |
| Agent loop, voice commands, form answers | `server/agent/coach.py` |
| Bedrock decision + fallback policy | `server/agent/brain.py` |
| Voice (Polly) | `server/agent/voice.py` |
| API + Lambda entry | `server/app.py`, `server/lambda_handler.py` |
| Web app + coach figure | `web/` |
| Evaluation tools | `tools/`, `eval/` |

---

## How we built it: what failed and how we fixed it

We tested on real footage and on the phone after every change. Each failure became a fix and a test.

| What went wrong | Why | What we did |
|---|---|---|
| OpenCV 5 never found a person | new DNN engine returns outputs in a different order | match outputs by shape; detector threshold 0.5 → 0.35 |
| Piked plank read as "collapsed" or "knees down" | low shoulders / bent knees alone are ambiguous | "collapsed" only when hips aren't piked; "knees down" also needs knees near the floor |
| Coach said "Beautiful form!" when the user walked away | a vanished fault counted as fixed | re-check passes only on a clean frame, otherwise `replaced_by` |
| Far away: 64 % accuracy; dark room: person lost in 16 % of frames | few pixels, low contrast | zoom re-detect (→ 81 %), CLAHE (→ 83 %) |
| Blur and compression flagged good form | one bad ankle bends the body line; threshold inside the noise | average both body sides, knee fallback, thresholds moved out of the noise band |
| Coach started before she could see the user, and praised too early | timer-based start | camera check: 8 s to set up, starts only after several good frames |
| Timer kept running when the user stopped | no stop detection | timer pauses, "You held 45 s. Continue or rest?", resumes by itself |
| **Failed test:** plank video uploaded as squats, coach said "Form OK" | only checked squat geometry, never *which* exercise | exercise recognition from pose signatures; asks to switch ([REPORT §6.4](docs/REPORT.md#64-is-it-the-right-exercise-a-failed-test-then-a-fix)) |
| Answering by buttons felt robotic | | voice answers, then voice commands and questions at any time ("pause", "end", "am I doing it right?") |
| No voice on phones | mobile browsers block audio until a tap | unlock one audio element on the first tap |
| Stale frame re-paused the set after the user answered | an in-flight request came back late | state version counter drops stale responses |
| Coach repeated the same sentences | single phrase per cue | phrase lists, never the same line twice in a row |

**Cloud path:** Bedrock model access stayed locked (use-case form), so the deterministic policy decides and Bedrock remains an option. CloudShell was blocked by account verification, so `deploy.sh` couldn't run there; we deployed the same zip through the Lambda console via S3. To avoid re-uploading on every change, the function now loads `server/`, `web/` and `models/` from this repo on each cold start (`SPOTTER_CODE_REPO`), falls back to the bundled code if GitHub is unreachable, and `/api/health` reports which code is live.

## What's next

The 8 exercises are a first set, chosen to show the full loop (form checks, corrections, plan changes, voice). The goal is a complete coach that covers everything a person needs to reach their goal:
- many more exercises, each with its own form checks (not only plank and squat);
- full programmes built around a goal (strength, mobility, weight loss, back health) that progress week after week;
- the coach remembering your history: your usual faults, how long you can hold, when to make it harder;
- more languages for the voice, and tests on many more people, bodies and rooms.

## Limitations

- Plank and squat have detailed form checks; the other six moves check movement only.
- Vision accuracy comes from one public test video (one person, one camera); fairness across bodies, skin tones and clothing isn't measured yet.
- Recognition of bridge, leg raises, arm circles, press and wall push-up is tested on synthetic poses only.
- 2D camera: no depth; planks need a side view.
- Voice understanding is keyword-based (English, basic French) on the browser's speech recognition (Chrome, Edge, Safari).

Full list: [REPORT §6.6](docs/REPORT.md#66-failure-cases-and-limitations).

---

## Run locally

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # all versions pinned (opencv-python-headless==5.0.0.93)
uvicorn server.app:app --port 8080
# open http://localhost:8080  (camera works on localhost)
```

Without AWS credentials the coach uses the browser's voice and the fallback policy. Optional:

```bash
export SPOTTER_MODEL_ID=<Bedrock Claude model or inference profile id>
export SPOTTER_BEDROCK_REGION=eu-north-1
```

## Deploy to AWS

**Option A, one command** (AWS CloudShell, region eu-north-1):

```bash
git clone https://github.com/DarkModeGirl22/spotter && cd spotter && bash deploy/deploy.sh
```

Builds the Lambda zip (OpenCV 5 for Python 3.13), uploads it to S3, creates a least-privilege role, the function (2 GB, 30 s) and a public HTTPS Function URL, and prints the link.

**Option B, console** (what the live demo uses): build the zip with `deploy/deploy.sh` locally, upload it to S3, create a Python 3.13 function from it (handler `server.lambda_handler.handler`, 2048 MB, 30 s), add a Function URL (auth NONE), and set:

| Variable | Value |
|---|---|
| `SPOTTER_CODE_REPO` | `DarkModeGirl22/spotter@main` (load the latest code on cold start) |
| `SPOTTER_CODE_REV` | any value; change it to force a restart with new code |
| `SPOTTER_WEB_REPO` | `DarkModeGirl22/spotter@main` (serve the web app from the repo) |

## Test and reproduce the evaluation

```bash
pytest -q                                                   # 13 unit tests
python -m tools.eval_agent --out docs/eval                  # 10 agent scenarios
bash eval/fetch_test_data.sh                                # public plank test video
python -m tools.eval_clips --clips clips --out eval_out --every-frame
python -m tools.eval_robustness --clips clips --out eval_out/robustness
python -m tools.replay_trace --video clips/plank.gif --exercise plank --out eval_out/replay
```

## Credits

Pose and person-detection models and wrapper code: OpenCV Zoo (Apache-2.0) · plank test video: Exercise-Correction by Ngô Hồng Quốc Bảo (MIT) · coach character: original, drawn in code (`web/figure.js`).
