"""Spotter API + web app. Runs locally with uvicorn and on AWS Lambda through Mangum."""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import cv2 as cv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import brain, coach, voice
from .vision.exercises import analyze
from .vision.setup_check import check as setup_check
from .vision.pose import PoseEstimator, decode_jpeg

log = logging.getLogger("spotter")
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(title="Spotter")
WEB = Path(__file__).resolve().parents[1] / "web"
_estimator: PoseEstimator | None = None


def estimator() -> PoseEstimator:
    global _estimator
    if _estimator is None:
        _estimator = PoseEstimator()
    return _estimator


def _log_trace(session: str, trace: list):
    # one JSON line per agent event -> CloudWatch Logs on AWS (observability)
    for ev in trace:
        log.info(json.dumps({"session": session, **ev}, default=str))


class StartIn(BaseModel):
    exercise: str
    duration: int = 60


class FrameIn(BaseModel):
    session: str
    state: dict
    t: float
    image: str  # base64 JPEG


class SetupIn(BaseModel):
    session: str
    exercise: str
    image: str  # base64 JPEG


class AnswerIn(BaseModel):
    session: str
    state: dict
    t: float
    choice: str


@app.get("/api/health")
def health():
    return {"opencv": cv.__version__, "bedrock_model": brain.MODEL_ID or None, "polly": voice.ENABLED}


@app.post("/api/start")
def start(body: StartIn):
    return {"state": coach.new_state(body.exercise, body.duration)}


@app.post("/api/frame")
def frame(body: FrameIn):
    t0 = time.perf_counter()
    try:
        img = decode_jpeg(base64.b64decode(body.image.split(",")[-1]))
    except Exception:
        img = None
    if img is None:
        raise HTTPException(400, "bad image")
    pose = estimator().estimate(img)
    obs = analyze(body.state["exercise"], pose)
    state, actions, trace = coach.step(body.state, obs, body.t)
    _log_trace(body.session, trace)
    return {
        "state": state, "actions": actions, "trace": trace, "obs": obs.to_json(),
        "pose": pose.to_json() if pose else None,
        "server_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.post("/api/setup")
def setup(body: SetupIn):
    """Active perception before a set: can the camera judge this exercise from where it is?"""
    t0 = time.perf_counter()
    try:
        img = decode_jpeg(base64.b64decode(body.image.split(",")[-1]))
    except Exception:
        img = None
    if img is None:
        raise HTTPException(400, "bad image")
    pose = estimator().estimate(img)
    result = setup_check(body.exercise, img, pose)
    _log_trace(body.session, [{"event": "setup_check", "exercise": body.exercise, "issue": result["issue"],
                               "metrics": result["metrics"]}])
    return {**result, "pose": pose.to_json() if pose else None,
            "server_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/api/answer")
def answer(body: AnswerIn):
    state, actions, trace = coach.answer(body.state, body.choice, body.t)
    _log_trace(body.session, trace)
    return {"state": state, "actions": actions, "trace": trace}


class HearIn(BaseModel):
    session: str
    state: dict
    t: float
    text: str


@app.post("/api/hear")
def hear(body: HearIn):
    """The user spoke to the coach (browser speech-to-text): understand it and act."""
    state, actions, trace = coach.hear(body.state, body.text[:200], body.t)
    _log_trace(body.session, trace)
    return {"state": state, "actions": actions, "trace": trace}


@app.get("/api/voice")
def speak(text: str):
    audio = voice.synthesize(text)
    if audio is None:
        return Response(status_code=204)
    return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.exception_handler(Exception)
def on_error(_, exc):
    log.exception("unhandled")
    return JSONResponse({"error": type(exc).__name__}, status_code=500)


app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
