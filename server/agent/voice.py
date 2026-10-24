"""Coach voice with Amazon Polly (neural). Phrases are cached so repeated cues play instantly."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

VOICE = os.environ.get("SPOTTER_VOICE", "Joanna")        # Polly neural en-US female voice
REGION = os.environ.get("SPOTTER_POLLY_REGION", "eu-west-1")
CACHE = Path(os.environ.get("SPOTTER_VOICE_CACHE", "/tmp/spotter-voice"))
ENABLED = os.environ.get("SPOTTER_POLLY", "1") == "1"


def synthesize(text: str) -> bytes | None:
    """MP3 bytes, or None when Polly is unavailable (the browser then uses its own speech)."""
    text = text.strip()[:300]
    if not ENABLED or not text:
        return None
    key = hashlib.sha1(f"{VOICE}|{text}".encode()).hexdigest()
    path = CACHE / f"{key}.mp3"
    if path.exists():
        return path.read_bytes()
    try:
        import boto3
        polly = boto3.client("polly", region_name=REGION)
        resp = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=VOICE, Engine="neural")
        audio = resp["AudioStream"].read()
    except Exception:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    return audio
