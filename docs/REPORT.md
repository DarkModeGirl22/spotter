# Spotter: technical report

**A talking workout coach that watches your form with OpenCV 5 and changes the plan when your body needs it.**
OpenCV AI Competition 2026 · Agentic Vision path · Team: Hind Taouis
Demo video: https://youtu.be/iXDpT73NZRU · Live app: https://3iblqdeawzjnsky7ihd6gu3jn40giofz.lambda-url.eu-north-1.on.aws/

---

## 1. Problem and users

Most people train at home alone, following videos. A video cannot see you, so it cannot tell you that your hips are sagging in a plank. That fault puts load on the lower back instead of the core. When you get tired, a video also cannot adapt: it keeps going while your form falls apart.

**Users:** beginners and intermediate people training at home with only a phone, and the coaches who would like to give them feedback between sessions.

**Goal:** a coach that (1) sees form faults in real time, (2) tells you out loud how to fix them, (3) checks that you did, and (4) changes the workout, with your approval, when corrections stop working.

## 2. What the user experiences

Spotter is made for **practising live** in front of a phone. *Upload a video* exists only to demonstrate and test the same coaching on recorded footage. The 8 exercises below are a starting set chosen to show the full loop; the goal is a complete coach with many more exercises and goal-based programmes.

1. Pick **Lower body** (squats, reverse lunges, glute bridge, side leg raises) or **Upper body** (plank, arm circles, overhead press, wall push-ups). Each is 4 × 60 s with 30 s of rest.
2. Top half of the screen: the coach, an animated character (drawn in code, original design) doing the move and talking. Bottom half: the user's camera with their skeleton drawn on top. Wrong joints turn red.
3. The coach speaks short cues ("Lift your hips!"), praises fixes ("Perfect, hold it!"), motivates ("Ten seconds left!"), and asks questions when a plan change is needed. The user answers by voice (or taps), and can talk to her at any moment: pause, skip, end, "am I doing it right?".
4. At the end: a summary (corrections, confirmed fixes, average time to fix, plan changes, questions asked) and a downloadable agent trace.

## 3. Architecture

![Architecture](img/architecture.png)

- **Browser** (phone or laptop): captures a 480 px JPEG every 0.4 s and sends it with the agent state. It plays the coach animation and voice and shows the questions.
- **AWS Lambda** (Python 3.13, 2 GB, Function URL over HTTPS) runs the OpenCV 5 pipeline and the agent. The server is **stateless**: the agent state travels with each request, so Lambda can scale with no database.
- **Plan-change decisions:** a deterministic policy (`brain.fallback_policy`) decides by default, and that is what the live demo runs. **Amazon Bedrock** (Claude, Converse API with tool use) is wired in and takes over when `SPOTTER_MODEL_ID` is set; any Bedrock error falls back to the policy. Bedrock model access was still pending (use-case form) at submission time, so Bedrock is not used in the live demo.
- **S3** holds the deployment package; on each cold start the function loads the latest `server/`, `web/` and `models/` from the GitHub repo (§7).
- **Amazon Polly** (neural voice) speaks the coach's lines. Audio is cached per phrase.
- **CloudWatch Logs** receives one JSON line per agent event (perceive → decide → act → re-check → human), which gives observability.

## 4. OpenCV 5 implementation

| Step | OpenCV 5 API | Details |
|---|---|---|
| Decode frame | `cv.imdecode` | JPEG from the browser |
| Find the person | `cv.dnn.readNet` + `forward` | OpenCV Zoo MediaPipe person detector (ONNX, 224×224, anchors + NMS) |
| Body keypoints | `cv.dnn` | OpenCV Zoo MediaPipe pose model (ONNX, 256×256): 33 keypoints with visibility |
| Crop and rotate person | `cv.getRotationMatrix2D`, `cv.warpAffine`, `cv.copyMakeBorder`, `cv.resize` | Aligns the body upright before the pose model |
| Visual evidence | `cv.line`, `cv.circle`, `cv.putText`, `cv.imwrite` | Annotated frames for traces and failure cases |
| Video evaluation | `cv.VideoCapture` | Offline replay of test clips |

**OpenCV 5 port note:** OpenCV 5's new DNN graph engine returns the person detector's outputs in a different order than 4.x. We match the outputs by shape. That is the only change to the OpenCV Zoo code (`server/vision/mp_persondet.py`).

**Form analysis (geometry on keypoints, `server/vision/exercises.py`):**

- **Plank (side view).** It picks the more visible body side and builds the shoulder → ankle line.
  - **Hip deviation** = signed vertical distance of the hip from that line ÷ body length. Above +0.07 → `hips_low`; below −0.09 → `hips_high`.
  - **Knee angle** (hip-knee-ankle) below 150° → `knees_down`.
  - **Shoulder lift** (shoulder above elbow ÷ body length) below 0.05, with hips not piked → `collapsed`.
  - **Body tilt** above 35° → `not_in_plank_position`.
- **Squat.** Knee angle per rep, tracked by a state machine: `shallow` if the deepest knee angle stays above 110°. In front view, knee width ÷ ankle width below 0.75 while bent → `knees_in`.
- **Other moves:** keypoint motion energy, so `not_moving` fires after 3 s still.

Thresholds are normalized by body length, so they don't depend on distance to the camera.

## 5. The agent (Agentic Vision)

![Agent loop](img/agent_loop.png)

*↻ = back to ① with the next frame.*

| Stage | What happens | Code |
|---|---|---|
| Perceive | OpenCV 5 output → list of faults | `exercises.py` |
| Filter | A fault must appear in 2 of the last 3 frames (anti-flicker) | `coach._persistent_fault` |
| Act | Spoken cue for that fault | `coach.step` |
| Re-check | The next frames decide: **fixed** (praise), **replaced by another fault** (no praise, handle the new one), or **not fixed** after 5 s (repeat the cue once) | `coach.step` |
| Decide | Second failed cue → one tool is chosen: `switch_variant`, `shorten_set`, `ask_user` or `say`. A deterministic policy chooses by default (live demo); Amazon Bedrock (Claude tool use) chooses when enabled, with the policy as fallback | `brain.decide` |
| Human control | `ask_user` asks out loud and shows the options (keep going / easier / stop); the user answers by voice or tap. No answer in 10 s → safe default: pause the set | `coach.answer` |
| Stop detection | The user stops (drops out of position, stands up, leaves the frame, stops moving) for about 1.2 s → **the timer pauses at once**, the coach says "You held 14 seconds. That's already good!" and offers *I'll continue / Rest now*. The camera keeps watching: back in position → timer resumes by itself ("You're back!"), with paused time not counted. Gone for 20 s → the set ends as rest | `coach._while_stopped` |
| Talking back | The user talks to the coach at any moment (camera check, set, rest), like with a human coach. Browser speech recognition → text → intent: commands **pause**, **go/resume**, **skip / next exercise**, **end the workout**; questions **"am I doing it right?"** (answered *yes* or *not quite + what to change* from the faults OpenCV saw in the last 3 frames, e.g. "Not quite: your hips are dropping. Lift your hips a little"), **"how long is left?"**, **"how many reps?"**, **"what's next?"**; answers **continue / rest / easier**, and **"it hurts"**, which always ends the set with a safety message. The mic is ignored while the coach speaks, so it never hears itself. Buttons remain as a fallback for noisy rooms | `coach.hear`, `coach.form_answer`, `/api/hear` |

**Active perception: camera check before every exercise.** In live mode the set does not start on a timer. Every 0.6 s the agent checks, with OpenCV, whether it *can* judge this exercise from where the phone is:
- light (mean grey level);
- sharpness (variance of `cv.Laplacian`);
- a person found;
- full body inside the frame;
- distance (body size in the frame);
- camera angle (shoulder width ÷ torso length: side view for plank, lunges, bridge and wall push-ups; front view for the others);
- for floor exercises, the start position.

It says what to change ("Turn sideways to the phone", "It's too dark, turn on a light", "Good spot! Now get into position"). It starts only after 3 good frames in a row: "Perfect, I can see you well. Let's go!" Each check is logged (`setup_check` events). This guards against the weaknesses measured in §6.3 instead of only documenting them.

**How vision changes the next action:** the same exercise leads to different actions depending only on what OpenCV sees. Hips back in line → praise and continue. Hips still low → repeat the cue, then a Bedrock decision. Knees on the floor repeatedly → switch to knee plank; after that, `knees_down` stops being a fault. Person gone from view → ask "Are you okay?"

### Trace from a real video

Real plank video (public, MIT licence, [Exercise-Correction](https://github.com/NgoQuocBao1010/Exercise-Correction), which shows the original project's own coloured overlay underneath ours) replayed through the full pipeline with `tools/replay_trace.py`:

![Trace](img/trace_strip.jpg)

| t | OpenCV 5 output | Agent event | Spoken |
|---|---|---|---|
| 12.0 s | form ok (12 s streak) | encourage | "Looking great, keep it up!" |
| 16.0 s | `hips_low` (hip deviation +0.10) | decide: cue_and_recheck | "Lift your hips! Squeeze your belly." |
| 20.8 s | form ok | **re-check: fixed** | "Beautiful form!" |
| 25.2 s | `hips_high` (hip deviation −0.26) | decide: cue_and_recheck | "Hips down a little. Flat back." |
| 29.6 s | form ok | **re-check: fixed** | "Perfect, hold it!" |

Full trace with per-frame metrics: `docs/eval/replay_plank/trace.json`.

## 6. Evaluation

### 6.1 Vision: plank faults on real video

`eval/fetch_test_data.sh` then `python -m tools.eval_clips --clips clips --every-frame`. Segments are labeled by hand (transition frames between postures are excluded).

| Segment | Truth | Frames | Person detected | Frames correct | Clip verdict |
|---|---|---|---|---|---|
| 0–34 | good | 35 | 97.1 % | 34/34 | good ✅ |
| 41–49 | hips_low | 9 | 100 % | 9/9 | hips_low ✅ |
| 57–69 | hips_high | 13 | 100 % | 13/13 | hips_high ✅ |
| 82–99 | good | 18 | 100 % | 18/18 | good ✅ |
| **Total** | | **75** | **98.7 %** | **74/74 (100 %)** | **4/4** |

Before the fixes in §6.5 this was 73/74 (one piked frame read as "knees down").

Pose latency (person detector + pose model, CPU, one core of the development container): **median 33.8 ms, p95 49.0 ms**. On Lambda: 35–44 ms warm (§7).

### 6.2 Agent: task success, failure handling, human control

`python -m tools.eval_agent` runs the real agent loop on 9 scripted observation timelines plus a Bedrock-outage test (`docs/eval/agent_scenarios.md`):

| Scenario | Expected behaviour | Pass | Corrections | Plan changes | Questions |
|---|---|---|---|---|---|
| good form, whole set | no corrections, only encouragement | ✅ | 0 | 0 | 0 |
| one-frame glitches (noise) | single-frame faults are ignored (no false alarm) | ✅ | 0 | 0 | 0 |
| hips sag, user fixes after cue | cue spoken, re-check confirms the fix, praise | ✅ | 1 | 0 | 0 |
| hips sag, user never fixes | cue, repeat, then the agent changes the plan | ✅ | 4 | 2 | 0 |
| knees drop repeatedly | switches to knee plank; knees on floor stop being a fault | ✅ | 2 | 1 | 0 |
| user stops after 20 s, comes back | timer paused, 'you held 20 s', resumes by itself; paused time not counted | ✅ | 1 | 0 | 1 |
| user stops, chooses 'rest' | agent asks continue/rest; the human decides to rest | ✅ | 1 | 0 | 1 |
| user stops, no answer, never comes back | after 20 s paused, the set ends as rest (safe default) | ✅ | 1 | 0 | 1 |
| question unanswered while still exercising | escalation never blocks: plan change or question | ✅ | 6 | 2 | 0 |
| Bedrock unavailable | fallback policy decides, no crash | ✅ | - | - | - |

Plus 13 unit tests (`pytest`).

These scenarios are not a first attempt: most were written after a failure we found on real footage or on the phone (praise when the user walked away, the timer running while the user had stopped, a late frame re-pausing the set, Bedrock being unavailable). Each failure, its cause and its fix are listed in §6.5; the scenarios keep them from coming back.

### 6.3 Robustness: the same frames under real capture conditions

A clean test video says little about a phone propped on a living-room floor. `python -m tools.eval_robustness` re-runs every labeled frame through the full OpenCV 5 pipeline after transforming it: mirrored, phone tilted, far away, close, dark room, motion blur, heavy JPEG compression, feet cut off. Frame accuracy, **before → after** the fixes in §6.5:

| Condition | Before | After | Remaining weakness |
|---|---|---|---|
| original | 96 % | **99 %** | one transition frame with no person found |
| mirrored (other side) | 96 % | **99 %** | |
| phone tilted 10° / −10° | 95 % / 99 % | **95 % / 99 %** | 3 hips_high frames missed at +10° |
| far away (55 % size) | 64 % | **81 %** | hips_low missed: 2/9 (keypoints too coarse) |
| close (130 %) | 95 % | **97 %** | |
| dark room | 67 % | **83 %** | person lost in 7 % of frames; hips_low 3/9 |
| motion blur | 88 % | **91 %** | hips_low 2/9 (blurred ankles) |
| low-quality JPEG | 77 % | **84 %** | hips_high 7/13 |
| feet cut off | 93 % | **97 %** | |

Full tables and an annotated picture of each failure: `docs/eval/robustness/`.

**What this means for users:** hip sag is the subtlest fault, and it is the one that degrades first with distance, darkness and blur. So the camera check (§5) refuses to start in a dark room or with the user too far, and tells them how to fix it. The agent's 2-of-3-frames rule absorbs isolated misses, but not a sustained one (a far, dark camera can miss a real sag).

### 6.4 Is it the right exercise? (a failed test, then a fix)

**Failed test (25 Oct, live on AWS):** we uploaded a *plank* video and told the app it was *squats*. The coach kept saying "Form OK" and counted down. It only checked squat geometry and never asked whether this *was* a squat. A human coach would have noticed at once.

**Fix: exercise recognition** (`server/vision/recognize.py`). Each frame becomes a compact pose description: posture tilt, knee and elbow angles, hands relative to shoulders and head, feet relative to hips, all normalised by torso length. The last ~4 s are matched against the signature of each of the 8 exercises: lying prone with shoulders lifted = plank; lying with knees bent and up = bridge; upright with large symmetric knee motion = squat; asymmetric or split stance = lunge; wrists over the head = press; arms out at shoulder height = arm circles; one foot moving sideways = leg raise; leaning with hands forward = wall push-up. If a *clearly different* exercise persists for about 2.4 s:
- **live:** the coach says so ("That looks like a plank, but this one is squats. Let's stand up for it."). If it continues, she asks whether to switch;
- **uploaded video:** she asks right away ("This looks like a plank, not squats. Should I coach it as a plank?"). "Yes" (spoken or tapped) switches the exercise, the animation and the form checks.

"Not sure" never counts as wrong, and squat ⇄ lunge are not flagged (they look alike for a few reps).

| Video (real footage) | Analysed windows | Recognised as | Wrong-exercise alarms when the right exercise is selected |
|---|---|---|---|
| plank | 127 | plank 107 · not sure 20 | 0 |
| squats | 97 | squat 82 · not sure 15 | 0 |
| reverse lunges | 100 | lunge 81 · squat 3 · not sure 16 | 0 |

| Wrong pairing | Result |
|---|---|
| plank video, *squats* selected (upload) | question after 4.0 s |
| squat video, *plank* selected (live) | spoken remark after 5.6 s, question after 8.0 s |

Limits: glute bridge, leg raises, arm circles, press and wall push-up signatures are checked only on synthetic poses, not on real footage yet.

### 6.5 Bugs found by testing, and fixes

| # | Found by | Symptom | Cause | Fix |
|---|---|---|---|---|
| 1 | first real photo | No person ever detected | OpenCV 5's new DNN engine returns the detector outputs in a different order than 4.x | Match outputs by tensor shape (`mp_persondet.py`) |
| 2 | real photo | Clear full-body photo, still no person | Detector score threshold 0.5 too strict (best score 0.496) | Threshold 0.35 |
| 3 | real plank video | Piked plank ("hips high") also flagged as "collapsed" | Low shoulders in a pike look like a collapse | "Collapsed" only when hips are not piked |
| 4 | replay trace | Coach said "Beautiful form!" when the user left the camera | Fault disappearing counted as "fixed" | Re-check passes only on a clean frame; otherwise `replaced_by`, no praise |
| 5 | robustness test | Pike with slightly bent knees read as "knees down" (also 1 frame in the clean video) | Knee angle alone | Knees down also requires the knees near floor level |
| 6 | robustness, far away | Small person → garbage keypoints (64 %) | Pose crop from a few pixels | When the body is small, zoom on it (crop + `cv.resize`) and detect again (→ 81 %) |
| 7 | robustness, dark room | Person lost in 16 % of frames | Low contrast | CLAHE on the LAB lightness channel when the frame is dark (→ 7 %) |
| 8 | robustness, blur | Misplaced ankle made a flat body look tilted, so "not in position" | One bad keypoint bends the body line | Average both body sides when they agree; fall back to shoulder→knee line when the ankle disagrees with the torso |
| 9 | robustness, compression | Good form flagged hips_high at −0.09 | Threshold inside the noise band | Pike threshold −0.12 (real pikes measure −0.26 to −0.33), sag threshold 0.06 |
| 10 | user test on the phone | Saying "pause" or "end" did nothing; the coach could only answer its own questions | Only 4 voice intents, and the mic was only used during the set | Voice commands and questions at any time, with whole-word matching (`coach.hear`); tests `test_voice_commands_work_any_time`, `test_am_i_doing_it_right_uses_what_the_camera_sees` |

Every fix is covered by a test (`tests/`) or measured by an evaluation script.

### 6.6 Failure cases and limitations

- **Transitions are missed:** frames 5 and 75 of the test video (the person moving between postures) return no person. The agent treats `no_person` like any fault: it reacts only if it persists, so short dropouts cause no false cue.
- **One test video, one person:** the 98.6 % comes from one subject, one camera angle and one lighting. It shows the method works but is not a population-level accuracy. More clips (different bodies, clothes, floors) are the next step, and `tools/eval_clips.py` is ready for them.
- **Side view needed for planks.** Seen from the front, hip sag is not measurable. The app tells the user where to put the phone before each exercise.
- **Depth.** A 2D camera cannot see twisting toward or away from the camera.
- **Form checks today:** plank (4 faults) and squat (2). The other six moves only check that the user keeps moving.
- **Network.** Feedback arrives after one round trip; cues are not frame-instant.
- **Voice understanding** is keyword-based (English, plus basic French) on top of the browser's speech recognition (Chrome/Edge/Safari). Unclear questions get a short list of what she can answer.
- **Live camera:** the demo uses the presenter's own recorded and live attempts. Smart glasses (Meta's Wearables Device Access Toolkit gives camera access) are future work. A first-person view cannot see the body, so they would suit other exercises.

## 7. AWS deployment

**Scripted path:** one command in AWS CloudShell, `bash deploy/deploy.sh`. It:

1. Builds a Lambda zip (OpenCV 5 wheels for Python 3.13, trimmed to 226 MB unzipped, under the 250 MB limit). The app's start-up was checked with exactly that package.
2. Uploads the zip to S3.
3. Creates a least-privilege IAM role: CloudWatch Logs, `polly:SynthesizeSpeech`, `bedrock:InvokeModel`.
4. Creates or updates the function (2 GB, 30 s timeout).
5. Opens a public HTTPS Function URL.
6. Sets log retention to 14 days and prints the link.

Re-running the script updates the function in place.

**What we actually did for the live demo:** CloudShell was blocked on our new account (pending account verification), so the script could not run there. We built the same zip, uploaded it to S3, and created the function and its Function URL in the Lambda console with the same settings. Two differences from the script remain: the live function uses a broader role created in the console (not the script's least-privilege role), and log retention is the account default. Moving the live function to the script's role is the first item on the to-do list.

**Status: live** at https://3iblqdeawzjnsky7ihd6gu3jn40giofz.lambda-url.eu-north-1.on.aws/ (eu-north-1, Python 3.13, x86_64, 2048 MB, 30 s, code from S3). `/api/health` reports OpenCV 5.0.0.

Measured on the deployed function from a laptop in Lille, one 480×360 JPEG per request:

| | OpenCV 5 time on Lambda | Round trip from the browser |
|---|---|---|
| first request (models loading) | 420 ms | 477 ms |
| warm requests | 35–44 ms | 93–103 ms |

So a frame every 0.4 s leaves plenty of margin.

**Updating without re-uploading:** the zip carries the heavy dependencies (OpenCV 5, NumPy, FastAPI). With `SPOTTER_CODE_REPO=DarkModeGirl22/spotter@main`, each cold start downloads the repo's `server/`, `web/` and `models/` (about 2 s) and runs them. Bumping `SPOTTER_CODE_REV` forces a fresh start. If GitHub is unreachable, the code bundled in the zip runs instead. `/api/health` reports which code is live.

**Cost:** pay-per-request only, with no server running 24/7. Bedrock is called only on escalations, not per frame. A $20 monthly spend limit is set on the account.

## 8. Responsible use and security

- **Privacy:** frames are analyzed in memory and discarded. No image is stored or logged; the trace contains only numbers and labels.
- **Human control:** the agent never stops or changes a set because a model said so alone. Stopping goes through the user, and silence means a safe pause, never "push through."
- **Safety tone:** motivation is encouraging, never shaming. Repeated failures lead to an easier variant, not more pressure. The home screen says to stop if something hurts.
- **Not medical advice;** not for rehabilitation without a professional.
- **Security:** HTTPS only (Function URL). No AWS credentials in the browser; the browser only talks to our API. The function is stateless and stores nothing. Inputs are bounded: frames up to ~1.5 MB, spoken text cut to 200 characters, voice text to 300 characters (Polly audio is cached, so repeated phrases cost nothing). The endpoint is public without login so judges can open it; a $20 account spend limit caps abuse. Least-privilege IAM role in `deploy/deploy.sh` (see §7 for the live function).
- **Fairness:** accuracy across body types, skin tones and clothing has not been measured yet (see 6.6). That is the main evaluation gap.

## 9. How to reproduce

```bash
pip install -r requirements.txt
pytest -q                                   # unit tests
python -m tools.eval_agent --out docs/eval  # agent scenarios
bash eval/fetch_test_data.sh                # public plank test video
python -m tools.eval_clips --clips clips --out eval_out --every-frame
python -m tools.eval_robustness --clips clips --out eval_out/robustness
python -m tools.replay_trace --video clips/plank.gif --exercise plank --out eval_out/replay
uvicorn server.app:app --port 8080          # app at http://localhost:8080
```

All dependencies are pinned in `requirements.txt`.

## 10. Credits

- Pose and person-detection models and wrapper code: OpenCV Zoo (Apache-2.0).
- Plank test video: Ngô Hồng Quốc Bảo, Exercise-Correction (MIT).
- Coach character: original, drawn in code (`web/figure.js`).
