"""AWS Lambda entry point (zip package, behind a Lambda Function URL).

Live code: when SPOTTER_CODE_REPO is set (e.g. "DarkModeGirl22/spotter@main"), each cold start
downloads that branch of the public repo and runs its `server/`, `web/` and `models/` instead of
the copy bundled in the zip, so pushing to GitHub + changing SPOTTER_CODE_REV (forces a cold start)
deploys new code without uploading a new package. Heavy dependencies (OpenCV 5, NumPy, FastAPI)
always come from the zip. Any failure falls back to the bundled code.
"""
import io
import os
import sys
import tarfile
import time
import urllib.request

LIVE_DIR = "/tmp/spotter-live"
CODE_SOURCE = "bundled"


def _load_live_code(spec: str) -> bool:
    repo, _, ref = spec.partition("@")
    url = f"https://codeload.github.com/{repo}/tar.gz/{ref or 'main'}"
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=15) as r:
        data = r.read()
    keep = ("server/", "web/", "models/")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1].startswith(keep) or not (m.isfile() or m.isdir()):
                continue
            if ".." in parts[1]:
                continue
            m.name = parts[1]
            tar.extract(m, LIVE_DIR)
    print(f"live code: {spec} loaded in {time.time() - t0:.1f}s ({len(data) // 1024} KB)")
    return True


def _boot():
    global CODE_SOURCE
    spec = os.environ.get("SPOTTER_CODE_REPO", "")
    if spec:
        try:
            _load_live_code(spec)
            # forget the bundled `server` package (except this module) and import the live one
            for name in [n for n in sys.modules if n == "server" or n.startswith("server.")]:
                if name != __name__:
                    del sys.modules[name]
            sys.path.insert(0, LIVE_DIR)
            import importlib
            pkg = importlib.import_module("server")
            if not pkg.__file__.startswith(LIVE_DIR):
                raise RuntimeError(f"imported {pkg.__file__}, not the live copy")
            app_module = importlib.import_module("server.app")
            CODE_SOURCE = f"github:{spec}@rev{os.environ.get('SPOTTER_CODE_REV', '0')}"
            return app_module.app
        except Exception as exc:  # network, GitHub, bad push... -> keep the service up
            print(f"live code failed, using bundled code: {type(exc).__name__}: {exc}")
            if LIVE_DIR in sys.path:
                sys.path.remove(LIVE_DIR)
            for name in [n for n in sys.modules if n.startswith("server.") and n != __name__]:
                del sys.modules[name]
    from .app import app
    return app


from mangum import Mangum  # noqa: E402

app = _boot()
os.environ["SPOTTER_CODE_SOURCE"] = CODE_SOURCE
handler = Mangum(app, lifespan="off")
