"""Tracking clips for the web app: a few seconds of all 22 players and the ball around each demo
moment, plus what the pitch overlays draw at the moment's key frame (pressure on the passer, every
passing option with its xPass, chosen vs best, the league's pressure-threat curve for that third).

    bin/pl python -m presslens.web.clips          # writes demo/clips/*.json and demo/app.json

Moments closer together than a clip are merged into one clip with several key frames, so the 81'
chain (press won, press broken, goal) plays as one continuous move. No model is involved.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..agents.demo import DEMO_DIR
from ..agents.facts import _league_zone
from ..engine import cube, data
from ..mcp_server import tools as T

CLIP_DIR = DEMO_DIR / "clips"
BEFORE, AFTER = 60, 40          # frames (10 per second) shown before / after a key frame
PASS_TYPES = ("press_trap", "press_broken", "missed_option")
SHOT_TYPES = ("goal", "big_chance")


def _windows(frames: list[int]) -> list[tuple[int, int, list[int]]]:
    """Merge overlapping [f - BEFORE, f + AFTER] windows: (start, end, key frames)."""
    out: list[tuple[int, int, list[int]]] = []
    for f in sorted(frames):
        if out and f - BEFORE <= out[-1][1]:
            s, _, keys = out[-1]
            out[-1] = (s, f + AFTER, [*keys, f])
        else:
            out.append((f - BEFORE, f + AFTER, [f]))
    return out


def _club(meta: dict) -> dict:
    return {"id": meta["id"], "name": meta["name"], "color": meta["colors"]["primary"],
            "secondary": meta["colors"]["secondary"]}


def _league_curve(third: str) -> dict:
    return {band: round(_league_zone(third, band)["mean_xt"], 4) for band in ("none", "some", "high")}


def _speed(match_id: str, frame: int, kind: str) -> dict:
    """Distance and ball speed of the pass or shot starting at a frame (as in the live feed)."""
    ev = _events(match_id)
    hit = ev[(ev.type == kind) & (ev.frame >= frame) & (ev.frame <= frame + 10)]
    if hit.empty:
        return {}
    e = hit.iloc[0]
    dist = float(np.hypot(e.end_x - e.x, e.end_y - e.y))
    secs = (e.end_frame - e.frame) / 10
    return {"distance_m": round(dist, 1), "ball_speed_kmh": round(3.6 * dist / secs) if secs > 0 else None}


@lru_cache(maxsize=4)
def _events(match_id: str) -> pd.DataFrame:
    return pd.read_parquet(data.MATCHES / match_id / "events.parquet")


def _overlay(m: data.MatchArrays, card: dict) -> tuple[int, dict]:
    """(key frame, overlay) for one moment."""
    kind, f = card["type"], int(card["frame"])

    def slot(pid: str | None) -> int:
        return m.player_index(f, pid) if pid else -1

    if kind in PASS_TYPES:
        opts = T.pass_options(card["match_id"], f)
        f = int(opts["frame"])
        pr = T.pressure_at(card["match_id"], f)
        with cube.connect_cube() as con:
            row = con.execute("select third, pressure_band from passes where match_id = ? and frame = ?",
                              [card["match_id"], f]).fetchone()
        third, band = row or (None, None)
        return f, {
            "kind": "pass", "passer": slot(opts["passer"]["player_id"]),
            "presser": slot((pr.get("nearest_defender") or {}).get("player_id")),
            "pressure": opts["pressure"], "presser_m": opts["carrier_proximity_m"],
            "decision_gap": opts["decision_gap"], "xt_added": opts["xt_added"], "outcome": opts["outcome"],
            "options": [{"slot": slot(o["player_id"]), "name": o["receiver"], "xpass": o["xpass"],
                         "value": o["value"], "marking_m": o["marking_m"], "chosen": o["chosen"], "best": o["best"]}
                        for o in opts["options"]],
            "third": third, "band": band, "league": _league_curve(third) if third else None,
            **_speed(card["match_id"], f, "pass"),
        }
    if kind in SHOT_TYPES:
        metrics = card.get("metrics", {})
        return f, {"kind": "shot", "shooter": slot(card["player_id"]), "xg": metrics.get("provider_xg"),
                   "outcome": metrics.get("outcome"), **_speed(card["match_id"], f, "shot")}
    return f, {"kind": "other"}


def build_clips(match_id: str, moment_ids: list[str], out_dir: Path = CLIP_DIR) -> dict:
    m = data.load_arrays(match_id)
    cards = {mid: T.get_moment(mid) for mid in moment_ids}
    keyed = {mid: _overlay(m, c) for mid, c in cards.items()}
    by_frame: dict[int, list[str]] = {}
    for mid, (f, _) in keyed.items():
        by_frame.setdefault(f, []).append(mid)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for start, end, keys in _windows(list(by_frame)):
        start, end = max(start, 0), min(end, len(m.xy) - 1)
        span = slice(start, end + 1)
        xy = np.round(m.xy[span] * 10).astype(int)            # decimetres keep the file small
        objects = []
        for k in range(22):
            pid = m.names[m.ids[keys[0], k]]
            p = T._player(pid) or {}
            objects.append({"team": 0 if k < 11 else 1, "player_id": pid, "name": p.get("name", pid),
                            "number": p.get("number"), "position": p.get("position")})
        clip_id = f"{match_id}_{start}"
        moments = []
        for f in keys:
            for mid in by_frame[f]:
                c = cards[mid]
                moments.append({"moment_id": mid, "type": c["type"], "clock": c["clock"], "index": f - start,
                                "team": c["team"], "overlay": keyed[mid][1]})
                index[mid] = {"clip": clip_id, "index": f - start, "frame": f, "type": c["type"],
                              "clock": c["clock"], "team": c["team"]}
        clip = {"id": clip_id, "match_id": match_id, "fps": int(m.meta["fps"]), "start_frame": start,
                "pitch": m.meta["pitch"], "home": _club(m.meta["home"]), "away": _club(m.meta["away"]),
                "objects": objects, "xy": xy.reshape(len(xy), -1).tolist(),
                "ball_z": np.round(m.z[span], 1).tolist(), "carrier": m.carrier[span].tolist(),
                "moments": moments}
        (out_dir / f"{clip_id}.json").write_text(json.dumps(clip, separators=(",", ":")))
    return index


LIVE_STEP = 2      # full-match tracking for the Live view at 5 frames per second (the page interpolates)


def build_live_tracking(match_id: str, out_dir: Path = DEMO_DIR / "live") -> Path:
    """All 22 players and the ball for the whole match, every LIVE_STEP frames, plus who is in each
    slot when (substitutions), for the Live view's pitch."""
    m = data.load_arrays(match_id)
    idx = np.arange(0, len(m.xy), LIVE_STEP)
    xy = np.round(m.xy[idx] * 10).astype(int).reshape(len(idx), -1)
    slots, last = [], [None] * 22
    for f in idx:
        for k in range(22):
            pid = m.names[m.ids[f, k]]
            if pid != last[k]:
                p = T._player(pid) or {}
                slots.append({"frame": int(f), "slot": k, "player_id": pid, "name": p.get("name", pid),
                              "number": p.get("number")})
                last[k] = pid
    out = {"match_id": match_id, "step": LIVE_STEP, "fps": int(m.meta["fps"]), "xy": xy.tolist(),
           "carrier": m.carrier[idx].tolist(), "slots": slots}
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{match_id}_tracking.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    return path


def demo_moment_ids(demo: dict, match_id: str) -> list[str]:
    md = demo["matches"][match_id]
    ids = {s["facts"]["moment_id"] for s in md["key_moments"]}
    ids |= {i["moment_id"] for f in md["feeds"] for i in f["items"]}
    return sorted(ids)


LANGUAGES = {"en": "", "de": "German", "es": "Spanish"}


def viewer_label(p: dict) -> dict:
    """A short name and a one-line description for a viewer profile."""
    kind = p["kind"]
    if kind == "supporter":
        name = f"{T._clubs().get(p['focus_club'], p['focus_club'])} fan"
    elif kind == "player_follower":
        name = f"{T._name(p['focus_player'])} follower"
    else:
        name = {"casual": "Casual fan", "analyst": "Analyst"}.get(kind, kind)
    extras = [LANGUAGES.get(p.get("language", "en"), p.get("language")), "deep detail" if p.get("detail") == "deep"
              else "", "plain language" if p.get("accessibility") == "plain" else ""]
    return {"id": p["id"], "name": name, "kind": kind, "language": p.get("language", "en"),
            "extras": [e for e in extras if e], "focus_player": p.get("focus_player"),
            "focus_club": p.get("focus_club")}


def app_data(demo: dict, clips: dict) -> dict:
    """Everything the web page shows, with names resolved, so the server needs no match data."""
    matches = {}
    for match_id, md in demo["matches"].items():
        s = T.match_summary(match_id)
        meta = data.load_arrays(match_id).meta
        moments = sorted(({"moment_id": mid, **c} for mid, c in clips.items() if mid.startswith(match_id)),
                         key=lambda c: c["frame"])
        key_ids = [st["facts"]["moment_id"] for st in md["key_moments"]]
        matches[match_id] = {
            "home": s["home"], "away": s["away"], "score": s.get("score"),
            "viewers": [viewer_label(p) for p in md["profiles"]],
            "moments": moments, "key_moments": key_ids, "stories": {st["facts"]["moment_id"]: st
                                                                   for st in md["key_moments"]},
            "feeds": {f["profile_id"]: f["items"] for f in md["feeds"]},
            "recaps": {r["profile_id"]: r for r in md["recaps"]},
            "players": {pid: p["name"] for pid, p in T._roster().items()
                        if pid.split("-")[0] in (s["home"]["club_id"], s["away"]["club_id"])},
            "colors": {c["id"]: _club(c) for c in (meta["home"], meta["away"])},
            "end_frame": int(meta["total_frames"]),
        }
    return {"model": demo["model"], "matches": matches, "answers": demo["answers"]}


def main() -> None:
    demo = json.loads((DEMO_DIR / "demo.json").read_text())
    clips = {}
    for match_id in demo["matches"]:
        index = build_clips(match_id, demo_moment_ids(demo, match_id))
        clips.update(index)
        print(f"{match_id}: {len(index)} moments in {len({v['clip'] for v in index.values()})} clips -> {CLIP_DIR}")
    (CLIP_DIR / "index.json").write_text(json.dumps(clips, indent=1))
    for match_id in demo["matches"]:
        print(f"live tracking -> {build_live_tracking(match_id)}")
    (DEMO_DIR / "app.json").write_text(json.dumps(app_data(demo, clips), ensure_ascii=False))
    print(f"page data -> {DEMO_DIR / 'app.json'}")


if __name__ == "__main__":
    main()
