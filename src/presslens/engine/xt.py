"""Expected threat (xT): the value of having the ball in each zone, learned from public events.

Standard Markov formulation on a 12 x 8 grid in the attacking frame:
    xT(z) = shot_rate(z) * shot_quality(z) + move_rate(z) * sum_z' P(z -> z') * xT(z')
Moves are completed passes and carries; failed passes count as moves that lose the ball (value 0).
Shot quality is the mean provider xG of shots from the zone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..sim.pitch import HALF_L, HALF_W, LENGTH, WIDTH
from .data import MODELS

NX, NY = 12, 8
ZX, ZY = LENGTH / NX, WIDTH / NY
XT_PATH = MODELS / "xt.json"


def zone_index(ax: np.ndarray, ay: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ix = np.clip(((np.asarray(ax) + HALF_L) / ZX).astype(int), 0, NX - 1)
    iy = np.clip(((np.asarray(ay) + HALF_W) / ZY).astype(int), 0, NY - 1)
    return ix, iy


def zone_centres() -> np.ndarray:
    """(NX, NY, 2) centres in the attacking frame."""
    xs = -HALF_L + ZX * (np.arange(NX) + 0.5)
    ys = -HALF_W + ZY * (np.arange(NY) + 0.5)
    return np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1)


@dataclass
class XTModel:
    grid: np.ndarray            # (NX, NY)
    shot_rate: np.ndarray
    move_rate: np.ndarray
    shot_quality: np.ndarray
    counts: np.ndarray          # actions per zone (shots + moves), for sample sizes
    trained_on: list[str]

    def zone_value(self, ax, ay) -> np.ndarray:
        ix, iy = zone_index(ax, ay)
        return self.grid[ix, iy]

    def value(self, ax, ay) -> np.ndarray:
        """Bilinear interpolation between zone centres (smooth, for exact points)."""
        ax, ay = np.asarray(ax, dtype=float), np.asarray(ay, dtype=float)
        fx = np.clip((ax + HALF_L) / ZX - 0.5, 0, NX - 1)
        fy = np.clip((ay + HALF_W) / ZY - 0.5, 0, NY - 1)
        x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        x1, y1 = np.minimum(x0 + 1, NX - 1), np.minimum(y0 + 1, NY - 1)
        tx, ty = fx - x0, fy - y0
        g = self.grid
        return ((1 - tx) * (1 - ty) * g[x0, y0] + tx * (1 - ty) * g[x1, y0]
                + (1 - tx) * ty * g[x0, y1] + tx * ty * g[x1, y1])

    def opponent_value(self, ax, ay) -> np.ndarray:
        """What the ball at (ax, ay) is worth to the other team (their frame is mirrored)."""
        return self.value(-np.asarray(ax), -np.asarray(ay))

    def save(self, path: Path = XT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "grid": self.grid.round(6).tolist(), "shot_rate": self.shot_rate.round(5).tolist(),
            "move_rate": self.move_rate.round(5).tolist(), "shot_quality": self.shot_quality.round(5).tolist(),
            "counts": self.counts.astype(int).tolist(), "trained_on": self.trained_on,
        }) + "\n")

    @classmethod
    def load(cls, path: Path = XT_PATH) -> XTModel:
        d = json.loads(path.read_text())
        return cls(np.array(d["grid"]), np.array(d["shot_rate"]), np.array(d["move_rate"]),
                   np.array(d["shot_quality"]), np.array(d["counts"]), d["trained_on"])


def fit(con, match_ids: list[str], iterations: int = 100) -> XTModel:
    ids = ", ".join(f"'{m}'" for m in match_ids)
    ev = con.execute(f"""
        select type, outcome, ax, ay, end_ax, end_ay, provider_xg, restart
        from events_a
        where match_id in ({ids}) and type in ('pass', 'carry', 'shot')
          and coalesce(restart, '') not in ('penalty', 'kickoff')""").df()
    shots = ev[ev.type == "shot"]
    moves = ev[ev.type.isin(["pass", "carry"])]
    ok = moves[(moves.type == "carry") | (moves.outcome == "complete")]

    sx, sy = zone_index(shots.ax, shots.ay)
    n_shot = np.zeros((NX, NY))
    xg_sum = np.zeros((NX, NY))
    np.add.at(n_shot, (sx, sy), 1)
    np.add.at(xg_sum, (sx, sy), shots.provider_xg.fillna(0).to_numpy())

    mx, my = zone_index(moves.ax, moves.ay)
    n_move = np.zeros((NX, NY))
    np.add.at(n_move, (mx, my), 1)

    trans = np.zeros((NX * NY, NX * NY))
    ox, oy = zone_index(ok.ax, ok.ay)
    ex, ey = zone_index(ok.end_ax, ok.end_ay)
    np.add.at(trans, (ox * NY + oy, ex * NY + ey), 1)

    total = n_shot + n_move
    safe = np.maximum(total, 1)
    shot_rate = n_shot / safe
    move_rate = n_move / safe
    # Shot quality: zone mean, shrunk towards the overall mean where there are few shots.
    prior = xg_sum.sum() / max(n_shot.sum(), 1)
    shot_quality = (xg_sum + 2 * prior * (zone_centres()[..., 0] > 0)) / (n_shot + 2)
    T = trans / np.maximum(n_move.reshape(-1, 1), 1)   # failed moves leave probability mass at 0

    xt = np.zeros(NX * NY)
    base = (shot_rate * shot_quality).ravel()
    for _ in range(iterations):
        xt = base + move_rate.ravel() * (T @ xt)
    return XTModel(xt.reshape(NX, NY), shot_rate, move_rate, shot_quality, total, list(match_ids))
