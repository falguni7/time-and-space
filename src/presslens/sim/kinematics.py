"""Player movement with physical limits.

Each frame every player steers towards a target point. Speed is capped by the
player's top speed (and a per-frame speed cap from tactics, e.g. jogging vs sprinting),
acceleration is capped by the player's max acceleration, and players brake before
reaching their target so nobody overshoots or turns instantly.
"""

from __future__ import annotations

import numpy as np

from .pitch import clip_to_pitch

BRAKE_FACTOR = 0.8  # fraction of amax used for braking when arriving at a target


def step(
    pos: np.ndarray,
    vel: np.ndarray,
    target: np.ndarray,
    vmax: np.ndarray,
    amax: np.ndarray,
    dt: float,
    speed_cap: np.ndarray | None = None,
    accel_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance positions (n, 2) and velocities (n, 2) by one time step towards targets."""
    cap = vmax if speed_cap is None else np.minimum(speed_cap, vmax)
    delta = target - pos
    dist = np.linalg.norm(delta, axis=1)
    direction = np.divide(delta, dist[:, None], out=np.zeros_like(delta), where=dist[:, None] > 1e-6)

    desired_speed = np.minimum(cap, np.sqrt(2.0 * BRAKE_FACTOR * amax * dist))
    desired_vel = direction * desired_speed[:, None]

    dv = desired_vel - vel
    dv_norm = np.linalg.norm(dv, axis=1)
    max_dv = amax * dt
    scale = np.where(dv_norm > max_dv, max_dv / np.maximum(dv_norm, 1e-9), 1.0)
    new_vel = vel + dv * scale[:, None]

    if accel_noise > 0.0 and rng is not None:
        new_vel = new_vel + rng.normal(0.0, accel_noise * dt, size=new_vel.shape)

    speed = np.linalg.norm(new_vel, axis=1)
    over = speed > vmax
    new_vel[over] *= (vmax[over] / speed[over])[:, None]

    new_pos = clip_to_pitch(pos + new_vel * dt)
    # A player stopped by the touchline loses the velocity component into it.
    blocked = new_pos != pos + new_vel * dt
    new_vel[blocked] = 0.0
    return new_pos, new_vel
