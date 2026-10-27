"""Build the web app as a static website (no server needed): the page plus the recorded demo files.

    python -m presslens.web.site                   # -> build/site/  (open with any static host)

Live mode, the clips and every text play in the browser from these files, so the site can be served by
GitHub Pages, Azure Static Web Apps or `python -m http.server`. The FastAPI server is only needed for
the streaming endpoint (/api/live/<match>/stream).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ..paths import ROOT

STATIC = Path(__file__).resolve().parent / "static"
DEMO_DIR = ROOT / "demo"


def build_site(out: Path, demo_dir: Path = DEMO_DIR) -> Path:
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    shutil.copytree(STATIC, out / "static")
    shutil.copy(STATIC / "index.html", out / "index.html")
    shutil.copy(demo_dir / "app.json", out / "data" / "app.json")
    for sub in ("clips", "live"):
        shutil.copytree(demo_dir / sub, out / "data" / sub)
    (out / ".nojekyll").write_text("")          # GitHub Pages: serve files as they are
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the static website")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "site")
    args = ap.parse_args()
    out = build_site(args.out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"static site -> {out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
