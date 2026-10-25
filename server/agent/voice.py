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
    audio = None
    import boto3
    for region, engine in ((REGION, "neural"), (os.environ.get("AWS_REGION", "eu-north-1"), "standard")):
        try:
            polly = boto3.client("polly", region_name=region)
            resp = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=VOICE, Engine=engine)
            audio = resp["AudioStream"].read()
            break
        except Exception as exc:  # logged to CloudWatch; the browser voice takes over
            print(f"polly {region}/{engine} failed: {type(exc).__name__}: {exc}")
    if audio is None:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audio)
    return audio
