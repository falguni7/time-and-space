"""Repository locations, defined once (src layout: <repo>/src/presslens/paths.py)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
