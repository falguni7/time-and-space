"""Matplotlib helpers for pitch plots and match animations."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

from .pitch import (
    CENTRE_CIRCLE_R,
    GOAL_AREA_DEPTH,
    GOAL_AREA_WIDTH,
    GOAL_WIDTH,
    HALF_L,
    HALF_W,
    PENALTY_DEPTH,
    PENALTY_SPOT,
    PENALTY_WIDTH,
)

PITCH_GREEN = "#3a7d44"
LINE = "#f1f5f9"


def draw_pitch(ax) -> None:
    ax.set_facecolor(PITCH_GREEN)
    kw = {"fill": False, "color": LINE, "lw": 1.2}
    ax.add_patch(Rectangle((-HALF_L, -HALF_W), 2 * HALF_L, 2 * HALF_W, **kw))
    ax.plot([0, 0], [-HALF_W, HALF_W], color=LINE, lw=1.2)
    ax.add_patch(Circle((0, 0), CENTRE_CIRCLE_R, **kw))
    for s in (-1, 1):
        x_line = s * HALF_L
        ax.add_patch(Rectangle((x_line - s * PENALTY_DEPTH if s > 0 else x_line, -PENALTY_WIDTH / 2),
                               PENALTY_DEPTH, PENALTY_WIDTH, **kw))
        ax.add_patch(Rectangle((x_line - s * GOAL_AREA_DEPTH if s > 0 else x_line, -GOAL_AREA_WIDTH / 2),
                               GOAL_AREA_DEPTH, GOAL_AREA_WIDTH, **kw))
        ax.plot([x_line, x_line], [-GOAL_WIDTH / 2, GOAL_WIDTH / 2], color="white", lw=4)
        spot_x = x_line - s * PENALTY_SPOT
        ax.add_patch(Arc((spot_x, 0), 2 * CENTRE_CIRCLE_R, 2 * CENTRE_CIRCLE_R,
                         theta1=127 if s > 0 else -53, theta2=233 if s > 0 else 53, color=LINE, lw=1.2))
    ax.set_xlim(-HALF_L - 3, HALF_L + 3)
    ax.set_ylim(-HALF_W - 3, HALF_W + 3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_team(ax, xy, numbers, color, edge, pressing=None):
    artists = []
    for k, ((x, y), num) in enumerate(zip(xy, numbers)):
        ring = edge if pressing is None or not pressing[k] else "#fde047"
        artists.append(ax.scatter(x, y, s=150, c=color, edgecolors=ring, linewidths=2.2 if ring != edge else 1.2, zorder=3))
        artists.append(ax.text(x, y, str(num), ha="center", va="center", fontsize=6.5,
                               color=edge, zorder=4, fontweight="bold"))
    return artists


def draw_ball(ax, xy):
    return ax.scatter(xy[0], xy[1], s=45, c="white", edgecolors="black", linewidths=1, zorder=5)


def new_figure(n_cols: int = 1, n_rows: int = 1, scale: float = 5.0):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale * 1.5, n_rows * scale), squeeze=False)
    return fig, axes
