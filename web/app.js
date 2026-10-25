// Spotter web app: runs the workout, streams camera frames to the coach agent, speaks its actions.
"use strict";

const WORKOUTS = {
  lower: { name: "Lower body", items: [
    { id: "squat", name: "Squats", move: "squat", tip: "Face the phone. Whole body in view." },
    { id: "lunge", name: "Reverse lunges", move: "lunge", tip: "Turn sideways to the phone." },
    { id: "bridge", name: "Glute bridge", move: "bridge", tip: "Lie on your back, side to the phone." },
    { id: "legraise", name: "Side leg raises", move: "legraise", tip: "Face the phone, hands on hips." },
  ] },
  upper: { name: "Upper body", items: [
    { id: "plank", name: "Plank", move: "plank", tip: "Phone on the floor, to your side." },
    { id: "armcircles", name: "Arm circles", move: "armcircles", tip: "Face the phone, arms out." },
    { id: "press", name: "Overhead press", move: "press", tip: "Face the phone. Push your hands to the sky." },
    { id: "wallpush", name: "Wall push-ups", move: "wallpush", tip: "Side to the phone, hands on the wall." },
  ] },
};
const WORK = 60, REST = 30, PREP = 6, FRAME_MS = 400;
const ASK_LABELS = { keep_going: "I'll continue", easier_version: "Easier version", stop_set: "Rest now" };
const FAULT_LABELS = {
  hips_low: "HIPS LOW", hips_high: "HIPS HIGH", knees_down: "KNEES DOWN", collapsed: "COLLAPSED",
  not_in_plank_position: "GET IN POSITION", body_not_visible: "MOVE BACK", legs_not_visible: "MOVE BACK",
  no_person: "NO ONE IN VIEW", knees_in: "KNEES IN", shallow: "GO LOWER", not_moving: "KEEP MOVING",
};

const $ = (id) => document.getElementById(id);
const session = Math.random().toString(36).slice(2, 10);
let coach, plan, idx, phase, phaseEnd, state, paused = false, pausedAt = 0, inflight = false;
let workStart = 0, traceLog = [], stats, sourceFile = null;

// ---------- home
function renderHome() {
  const cards = $("cards");
  cards.innerHTML = "";
  for (const [key, w] of Object.entries(WORKOUTS)) {
    const el = document.createElement("div");
    el.className = "card";
    el.innerHTML = `<div class="card-head"><h2>${w.name}</h2><span>4 MOVES · 5:30</span></div>
      <ol>${w.items.map((i) => `<li>${i.name}<span class="dur">${WORK}s</span></li>`).join("")}</ol>
      <button class="btn btn-red">Start workout</button>`;
    el.querySelector("button").onclick = () => start(key);
    cards.appendChild(el);
  }
}
function chooseMode(m) {
  $("modeLive").classList.toggle("on", m === "live");
  $("modeVideo").classList.toggle("on", m === "video");
  $("livePanel").hidden = m !== "live";
  $("videoPanel").hidden = m !== "video";
}
$("modeLive").onclick = () => { sourceFile = null; chooseMode("live"); };
$("modeVideo").onclick = () => chooseMode("video");
for (const w of Object.values(WORKOUTS)) for (const it of w.items) {
  const o = document.createElement("option");
  o.value = it.id; o.textContent = `${it.name} (${w.name.toLowerCase()})`;
  $("exSelect").appendChild(o);
}
$("exSelect").value = "plank";
$("fileInput").onchange = (e) => { sourceFile = e.target.files[0] || null; $("btnAnalyze").disabled = !sourceFile; };
$("btnAnalyze").onclick = () => {
  const ex = Object.values(WORKOUTS).flatMap((w) => w.items).find((i) => i.id === $("exSelect").value);
  start({ name: "Video: " + ex.name, items: [ex], video: true });
};

// ---------- camera / video source
async function openSource() {
  const v = $("video");
  if (sourceFile) {
    v.srcObject = null; v.src = URL.createObjectURL(sourceFile); v.loop = false;
    document.querySelector(".cam-panel").classList.remove("mirror");
    await new Promise((ok) => { if (v.readyState >= 1) ok(); else v.onloadedmetadata = ok; });
    v.onended = () => { if (phase === "work") endExercise(); };
    v.currentTime = 0;
    return;   // the video starts playing when the set starts
  } else {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } }, audio: false });
    v.srcObject = stream;
    document.querySelector(".cam-panel").classList.add("mirror");
  }
  await v.play();
}
function closeSource() {
  const v = $("video");
  if (v.srcObject) v.srcObject.getTracks().forEach((t) => t.stop());
  v.pause(); v.srcObject = null; v.removeAttribute("src");
}

// ---------- voice: Amazon Polly through /api/voice, browser speech as fallback
let audio = null;
async function speak(text, kind = "") {
  const sub = $("subtitle");
  sub.textContent = text;
  sub.className = "subtitle" + (kind === "correction" || kind === "question" ? " correction" : "");
  try {
    if (audio) { audio.pause(); audio = null; }
    window.speechSynthesis && speechSynthesis.cancel();
    const r = await fetch("api/voice?text=" + encodeURIComponent(text));
    if (r.status === 200) {
      audio = new Audio(URL.createObjectURL(await r.blob()));
      await audio.play();
      return;
    }
  } catch (_) { /* fall through */ }
  if (window.speechSynthesis) {
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "en-US"; u.rate = 1.05;
    speechSynthesis.speak(u);
  }
}

// ---------- workout flow
async function start(key) {
  plan = typeof key === "string" ? WORKOUTS[key] : key;
  frozen = null;
  stats = { stops: 0, corrections: 0, fixed: 0, fixTimes: [], planChanges: 0, questions: 0, frames: 0, latency: [] };
  traceLog = [];
  $("home").hidden = true; $("workout").hidden = false; $("done").hidden = true;
  if (!coach) coach = new window.Coach($("coachCanvas"));
  try { await openSource(); } catch (e) {
    alert("Camera not available: " + e.message); return finish();
  }
  idx = 0;
  enterPrep();
}

function setPhase(name, seconds) { phase = name; phaseEnd = performance.now() + seconds * 1000; }
function current() { return plan.items[idx]; }

function enterPrep() {
  const ex = current();
  $("progress").textContent = `${idx + 1} / ${plan.items.length}`;
  $("exName").textContent = ex.name; $("exPhase").textContent = "GET READY";
  $("nextUp").textContent = ""; clearOverlay(); setChip("WAITING", false);
  coach.setMove(ex.move);
  if (plan.video) { setPhase("prep", 3); speak(`${ex.name}. Watching your video.`); return; }
  // live: the coach checks the camera view first and starts only when it can judge the exercise
  phase = "setup"; setup = { okStreak: 0, lastIssue: null, lastSpoke: performance.now(), t0: performance.now() };
  $("exPhase").textContent = "CAMERA CHECK"; $("timer").textContent = "SETUP";
  speak(`${ex.name}. ${ex.tip}`);
}

let setup = null;
const SETUP_LABELS = {
  too_dark: "TOO DARK", blurry: "BLURRY", no_person: "NO ONE IN VIEW", cut_off: "MOVE BACK",
  too_far: "COME CLOSER", too_close: "STEP BACK", need_side: "TURN SIDEWAYS", need_front: "FACE THE PHONE",
  get_in_position: "GET IN POSITION", stand_up: "STAND UP",
};
function grabFrame() {
  const v = $("video");
  if (!v.videoWidth) return null;
  const w = 480, h = Math.round((v.videoHeight / v.videoWidth) * w);
  grab.width = w; grab.height = h;
  grab.getContext("2d").drawImage(v, 0, 0, w, h);
  return grab.toDataURL("image/jpeg", 0.7);
}
async function sendSetupFrame() {
  const image = grabFrame();
  if (!image) return;
  inflight = true;
  try {
    const r = await post("api/setup", { session, exercise: current().id, image });
    if (phase !== "setup") return;
    drawPose(r.pose, { faults: r.ok ? [] : [r.issue], bad_joints: [] });
    setChip(r.ok ? "CAMERA OK" : (SETUP_LABELS[r.issue] || r.issue.toUpperCase()), !r.ok);
    $("meta").textContent = `light ${r.metrics.brightness} · sharp ${r.metrics.sharpness}` +
      (r.metrics.body_size != null ? ` · size ${r.metrics.body_size}` : "");
    const now = performance.now();
    if (r.ok) {
      setup.okStreak += 1;
      if (setup.okStreak >= 3) {
        traceLog.push({ exercise: current().id, event: "setup", t: 0, result: "ok", metrics: r.metrics });
        addTraceItem({ event: "setup", t: 0, result: "camera ok", ...r.metrics });
        phase = "starting";
        speak(r.message);
        setTimeout(enterWork, 2200);
      }
      return;
    }
    setup.okStreak = 0;
    // speak when the problem changes, or remind every 8 s (never talk over the intro line)
    if (now - setup.t0 > 3500 && ((r.issue !== setup.lastIssue && now - setup.lastSpoke > 2500) || now - setup.lastSpoke > 8000)) {
      speak(r.message, "correction");
      traceLog.push({ exercise: current().id, event: "setup", t: 0, result: r.issue, metrics: r.metrics });
      addTraceItem({ event: "setup", t: 0, result: r.issue });
      setup.lastIssue = r.issue; setup.lastSpoke = now;
    }
  } catch (e) {
    $("meta").textContent = "connection problem, retrying…";
  } finally { inflight = false; }
}

async function enterWork() {
  const ex = current();
  const v = $("video");
  const dur = plan.video ? Math.max(5, Math.ceil(v.duration || WORK)) : WORK;
  const r = await post("api/start", { exercise: ex.id, duration: dur });
  state = r.state;
  $("exPhase").textContent = "GO";
  setPhase("work", dur);
  if (plan.video) { v.currentTime = 0; v.play(); }
  workStart = performance.now();
  speak("Go!");
}

function enterRest() {
  if (idx >= plan.items.length - 1) return finish();
  const nxt = plan.items[idx + 1];
  $("exName").textContent = "Rest"; $("exPhase").textContent = "BREATHE";
  $("nextUp").textContent = "NEXT: " + nxt.name.toUpperCase();
  coach.setMove("idle"); clearOverlay(); setChip("REST", false);
  setPhase("rest", REST);
  speak(`Rest ${REST} seconds. Next: ${nxt.name}.`);
}

function endExercise() {
  hideAsk();
  if (phase === "work") enterRest();
}

function nextAfterRest() { idx += 1; enterPrep(); }

function finish() {
  phase = "done"; closeSource(); hideAsk();
  $("workout").hidden = true; $("done").hidden = false; $("trace").hidden = true;
  const avg = (a) => (a.length ? (a.reduce((x, y) => x + y, 0) / a.length) : 0);
  const rows = [
    ["Corrections spoken", stats.corrections],
    ["Fixes confirmed by camera", stats.fixed],
    ["Average time to fix", stats.fixTimes.length ? avg(stats.fixTimes).toFixed(1) + " s" : "-"],
    ["Plan changes made by the agent", stats.planChanges],
    ["Questions asked to you", stats.questions],
    ["Frames analyzed", stats.frames],
    ["Average analysis time", stats.latency.length ? Math.round(avg(stats.latency)) + " ms" : "-"],
  ];
  $("summary").innerHTML = rows.map(([k, v], i) =>
    `<div class="stat${i < 2 ? " hl" : ""}"><b>${v}</b><span>${k}</span></div>`).join("");
  speak("Workout done. Great job!");
}

// ---------- frames -> agent
const grab = document.createElement("canvas");
async function sendFrame() {
  const image = grabFrame();
  if (!image) return;
  const t = (performance.now() - workStart) / 1000;
  const sentAt = performance.now();
  inflight = true;
  try {
    const r = await post("api/frame", { session, state, t, image });
    if (phase !== "work") return;
    state = r.state;
    stats.frames += 1; stats.latency.push(performance.now() - sentAt);
    drawPose(r.pose, r.obs);
    $("meta").textContent = `opencv ${r.server_ms} ms · round trip ${Math.round(performance.now() - sentAt)} ms`;
    handle(r.actions, r.trace);
  } catch (e) {
    $("meta").textContent = "connection problem, retrying…";
  } finally { inflight = false; }
}

function handle(actions, trace) {
  for (const ev of trace || []) {
    if (ev.event === "perceive") continue;
    traceLog.push({ exercise: current().id, ...ev });
    addTraceItem(ev);
    if (ev.event === "recheck" && ev.result === "fixed") { stats.fixed += 1; stats.fixTimes.push(ev.seconds_to_fix); }
    if (ev.event === "act" && ["switch_variant", "shorten_set"].includes(ev.action)) stats.planChanges += 1;
  }
  for (const a of actions || []) {
    if (a.type === "say") { if (a.kind === "correction") stats.corrections += 1; speak(a.text, a.kind); }
    else if (a.type === "switch_variant") { coach.setMove(a.variant === "knee" ? "kneeplank" : current().move); $("exPhase").textContent = "EASIER VERSION"; }
    else if (a.type === "shorten_set") { phaseEnd = workStart + a.end_at * 1000; }
    else if (a.type === "ask") { stats.questions += 1; showAsk(a); }
    else if (a.type === "pause_timer") { frozen = phaseEnd - performance.now(); $("exPhase").textContent = `PAUSED · YOU HELD ${a.held}s`; stats.stops += 1; }
    else if (a.type === "resume_timer") { phaseEnd = performance.now() + (frozen || 0); frozen = null; $("exPhase").textContent = "GO"; }
    else if (a.type === "close_ask") { hideAsk(); }
    else if (a.type === "end_set") { setTimeout(endExercise, 1500); }
  }
}

// ---------- human in the loop
let frozen = null, askSoft = false;
function showAsk(a) {
  askSoft = !!a.soft;
  $("ask").classList.toggle("soft", askSoft);
  $("askQ").textContent = a.question;
  const box = $("askBtns"); box.innerHTML = "";
  for (const opt of a.options) {
    const b = document.createElement("button");
    b.textContent = ASK_LABELS[opt] || opt;
    if (opt === "stop_set") b.className = "stop";
    b.onclick = () => answer(opt);
    box.appendChild(b);
  }
  $("ask").hidden = false;
}
function hideAsk() { $("ask").hidden = true; askSoft = false; }
async function answer(choice) {
  hideAsk();
  const t = (performance.now() - workStart) / 1000;
  const r = await post("api/answer", { session, state, t, choice });
  state = r.state;
  handle(r.actions, r.trace);
}

// ---------- drawing the user's skeleton
const BONES = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24],[23,25],[25,27],[24,26],[26,28]];
function clearOverlay() { const c = $("overlay"); c.getContext("2d").clearRect(0, 0, c.width, c.height); }
function drawPose(pose, obs) {
  const c = $("overlay"), v = $("video"), ctx = c.getContext("2d");
  c.width = c.clientWidth; c.height = c.clientHeight;
  ctx.clearRect(0, 0, c.width, c.height);
  const faults = (obs && obs.faults) || [];
  const shown = faults.filter((f) => !(state && state.variant === "knee" && f === "knees_down"));
  setChip(shown.length ? FAULT_LABELS[shown[0]] || shown[0].toUpperCase() : "FORM OK", shown.length > 0);
  if (!pose || !v.videoWidth) return;
  const s = Math.min(c.width / v.videoWidth, c.height / v.videoHeight);
  const ox = (c.width - v.videoWidth * s) / 2, oy = (c.height - v.videoHeight * s) / 2;
  const P = pose.points.map(([x, y]) => [ox + x * v.videoWidth * s, oy + y * v.videoHeight * s]);
  const bad = new Set((obs && obs.bad_joints) || []);
  ctx.lineCap = "round"; ctx.lineWidth = 5; ctx.strokeStyle = "#000";
  for (const [a, b] of BONES) {
    if (pose.visibility[a] < 0.3 || pose.visibility[b] < 0.3) continue;
    ctx.beginPath(); ctx.moveTo(...P[a]); ctx.lineTo(...P[b]); ctx.stroke();
  }
  for (const i of new Set(BONES.flat())) {
    if (pose.visibility[i] < 0.3) continue;
    ctx.fillStyle = bad.has(i) ? "#FF0000" : "#FFFFFF";
    ctx.beginPath(); ctx.arc(...P[i], bad.has(i) ? 9 : 6, 0, Math.PI * 2); ctx.fill();
    ctx.lineWidth = 2; ctx.stroke(); ctx.lineWidth = 5;
  }
}
function setChip(text, bad) { const c = $("chip"); c.textContent = text; c.className = "chip" + (bad ? " bad" : ""); }

// ---------- trace panel
function addTraceItem(ev) {
  const li = document.createElement("li");
  li.className = ev.event;
  const detail = ev.event === "decide" ? `${ev.fault} → ${ev.decision} (${ev.decided_by})`
    : ev.event === "recheck" ? `${ev.fault}: ${ev.result}${ev.seconds_to_fix ? " in " + ev.seconds_to_fix + "s" : ""}`
    : ev.event === "act" ? `${ev.action} ${ev.args ? JSON.stringify(ev.args) : ev.why || ""}`
    : ev.event === "human" ? `user chose ${ev.choice}`
    : ev.event === "setup" ? `camera check: ${ev.result}` : JSON.stringify(ev);
  li.innerHTML = `<span class="ev">${current().id} t=${ev.t}s ${ev.event.toUpperCase()}</span><br>${detail}`;
  $("traceList").prepend(li);
}
function downloadTrace() {
  const blob = new Blob([JSON.stringify({ session, workout: plan && plan.name, stats, trace: traceLog }, null, 2)],
    { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `spotter-trace-${session}.json`; a.click();
}

// ---------- controls
$("btnPause").onclick = () => {
  paused = !paused;
  $("btnPause").textContent = paused ? "RESUME" : "PAUSE";
  if (paused) { pausedAt = performance.now(); $("video").pause(); speechSynthesis && speechSynthesis.cancel(); }
  else { const d = performance.now() - pausedAt; phaseEnd += d; workStart += d; $("video").play(); }
};
$("btnSkip").onclick = () => { if (phase === "work") endExercise(); else if (phase === "rest") nextAfterRest(); else if (phase === "prep" || phase === "setup") { phase = "starting"; enterWork(); } };
$("btnEnd").onclick = () => finish();
$("btnTrace").onclick = () => { $("trace").hidden = !$("trace").hidden; };
$("btnTraceX").onclick = () => { $("trace").hidden = true; };
$("btnTraceDl").onclick = downloadTrace;
$("btnDl").onclick = downloadTrace;
$("btnHome").onclick = () => { $("done").hidden = true; $("home").hidden = false; };

async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

// ---------- main clock
let lastFrame = 0;
function tick() {
  requestAnimationFrame(tick);
  if (!phase || phase === "done" || paused) return;
  if (phase === "setup" || phase === "starting") {
    if (phase === "setup" && performance.now() - lastFrame > 600 && !inflight) { lastFrame = performance.now(); sendSetupFrame(); }
    return;
  }
  if (frozen !== null && phase === "work") phaseEnd = performance.now() + frozen;   // timer stopped
  const left = Math.max(0, (phaseEnd - performance.now()) / 1000);
  const shown = Math.ceil(left);
  $("timer").textContent = `${Math.floor(shown / 60)}:${String(shown % 60).padStart(2, "0")}`;
  $("timer").classList.toggle("low", phase === "work" && left <= 10);
  const total = phase === "work" ? (phaseEnd - workStart) / 1000 : 0;
  $("setBar").style.width = total > 0 ? `${100 * (1 - left / total)}%` : "0%";
  if (left <= 0) {
    if (phase === "prep") return void (phase = "starting", enterWork());
    if (phase === "work") return endExercise();
    if (phase === "rest") return nextAfterRest();
  }
  // keep watching during a "soft" question (user stopped): the camera can see them come back
  if (phase === "work" && !inflight && ($("ask").hidden || askSoft) && performance.now() - lastFrame > FRAME_MS) {
    lastFrame = performance.now();
    sendFrame();
  }
}
renderHome();
requestAnimationFrame(tick);
