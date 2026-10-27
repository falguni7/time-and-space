"""The web app: an animated pitch next to every viewer's verified explanation of the same moment.

    bin/pl python -m presslens.web.server            # http://localhost:8000

Serves demo/app.json (the recorded demo texts with viewer labels and the moment list) and the
tracking clips in demo/clips, both written by presslens.web.clips, plus the single-page front end in
web/static. It imports nothing but FastAPI, so it runs offline in a slim image with no model key and
no match data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..paths import ROOT

STATIC = Path(__file__).resolve().parent / "static"
DEMO_DIR = Path(os.environ.get("PRESSLENS_DEMO_DIR", ROOT / "demo"))


def _feed_path(demo_dir: Path, match_id: str) -> Path:
    live_dir = (demo_dir / "live").resolve()
    path = (live_dir / f"{match_id}.jsonl").resolve()
    if path.parent != live_dir or not path.exists():
        raise HTTPException(404, "no live feed for this match")
    return path


@lru_cache(maxsize=4)
def load_app_data(demo_dir: Path) -> dict:
    return json.loads((demo_dir / "app.json").read_text())


def create_app(demo_dir: Path = DEMO_DIR) -> FastAPI:
    app = FastAPI(title="Time and Space", docs_url=None, redoc_url=None)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    clip_dir = demo_dir / "clips"

    @app.get("/api/demo")
    def demo() -> dict:
        return load_app_data(demo_dir)

    @app.get("/api/clips/{clip_id}")
    def clip(clip_id: str) -> FileResponse:
        path = (clip_dir / f"{clip_id}.json").resolve()
        if path.parent != clip_dir.resolve() or not path.exists():
            raise HTTPException(404, "no such clip")
        return FileResponse(path, media_type="application/json")

    @app.get("/api/live/{match_id}")
    def live_feed(match_id: str) -> FileResponse:
        """The recorded live feed (JSON Lines): every event, running state and key moment, in match order."""
        return FileResponse(_feed_path(demo_dir, match_id), media_type="application/x-ndjson")

    @app.get("/api/live/{match_id}/stream")
    async def live_stream(match_id: str, speed: float = 30.0, start_frame: int = 0) -> StreamingResponse:
        """The same feed as server-sent events, released in match time (speed 1 = real time)."""
        path = _feed_path(demo_dir, match_id)
        speed = min(max(speed, 0.1), 1000.0)

        async def events():
            t0, first = time.monotonic(), None
            with path.open() as f:
                for line in f:
                    frame = json.loads(line).get("frame") or 0
                    if frame < start_frame:
                        continue
                    first = frame if first is None else first
                    wait = (frame - first) / 10 / speed - (time.monotonic() - t0)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    yield f"data: {line.strip()}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    # The page reads the same files a static host (e.g. GitHub Pages) would serve under data/.
    for sub in ("clips", "live"):
        if (demo_dir / sub).is_dir():
            app.mount(f"/data/{sub}", StaticFiles(directory=demo_dir / sub), name=f"data-{sub}")

    @app.get("/data/app.json")
    def app_json() -> dict:
        return load_app_data(demo_dir)
    return app


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description="Serve the Time and Space web app")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = ap.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
