"""Visual and numeric checks for the simulation.

    bin/pl python -m presslens.sim.checks shape              # team shapes and pressure in live play
    bin/pl python -m presslens.sim.checks squads             # squads and sample player cards
    bin/pl python -m presslens.sim.checks match RIV NOR      # one full match: report, overview, first goal
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import paths
from .config import load_clubs, load_players, squad
from .match import DT, HELD, Match, MatchResult
from .pitch import HALF_L, third

OUT_DIR = paths.DATA_DIR / "checks"
SHAPE_MINUTES = 15.0   # per half for the shape check
CLIP_S = 30.0
RESTART_EVENTS = ("corner", "free_kick", "penalty", "throw_in", "goal_kick", "kickoff", "gk")


@dataclass
class ShapeRun:
    """A real simulated match; `attack` (home, team 0) is the side whose build-up we study."""

    result: MatchResult
    attack: str
    defend: str


def run_shape(attack_id: str, defend_id: str, seed: int = 0, minutes: float = SHAPE_MINUTES) -> ShapeRun:
    return ShapeRun(Match(attack_id, defend_id, seed=seed, period_minutes=minutes).run(), attack_id, defend_id)


def _ball_in_attack_frame(r: MatchResult, t: int, team: int) -> np.ndarray:
    return r.ball[t, :2] * r.directions[r.period[t] - 1][team]


def _carrier_slot(r: MatchResult, t: int, team: int) -> int | None:
    slots = np.where(r.ids[t, team] == r.carrier[t])[0]
    return int(slots[0]) if len(slots) else None


def proximity_summary(run: ShapeRun) -> dict[str, dict[str, float]]:
    """Mean nearest-defender distance to the carrier and to his outfield team-mates, per ball third."""
    r = run.result
    rows: dict[str, list[tuple[float, float]]] = {}
    for t in np.where((r.ball_state == HELD) & (r.possession == 0))[0][::5]:
        c = _carrier_slot(r, t, 0)
        if c is None:
            continue
        d = np.linalg.norm(r.pos[t, 0][:, None, :] - r.pos[t, 1, 1:][None, :, :], axis=2).min(axis=1)
        others = [i for i in range(1, 11) if i != c]
        rows.setdefault(third(_ball_in_attack_frame(r, t, 0)[0]), []).append((d[c], d[others].mean()))
    return {
        k: {"carrier_m": float(np.mean([a for a, _ in v])), "options_m": float(np.mean([b for _, b in v])),
            "frames": len(v)}
        for k, v in rows.items()
    }


def _numbers(r: MatchResult, players) -> dict[int, int]:
    return {code: players[pid].number for code, pid in enumerate(r.player_ids)}


def draw_frame(ax, r: MatchResult, t: int, clubs, numbers: dict[int, int], mark_pressure: bool = False) -> None:
    from .plotting import draw_ball, draw_pitch

    draw_pitch(ax)
    for team, club_id in enumerate((r.home, r.away)):
        c = clubs[club_id].colors
        for s in range(11):
            x, y = r.pos[t, team, s]
            ax.scatter(x, y, s=150, c=c["primary"], edgecolors=c["secondary"], linewidths=1.2, zorder=3)
            ax.text(x, y, str(numbers[int(r.ids[t, team, s])]), ha="center", va="center",
                    fontsize=6.5, color=c["secondary"], zorder=4, fontweight="bold")
    b = r.ball[t]
    draw_ball(ax, b[:2])
    if b[2] > 0.3:
        ax.text(b[0] + 1.2, b[1] + 1.2, f"{b[2]:.0f} m", fontsize=6, color="white")
    if mark_pressure and r.ball_state[t] == HELD:
        team = int(r.possession[t])
        c = _carrier_slot(r, t, team)
        if c is not None:
            carrier = r.pos[t, team, c]
            opp = r.pos[t, 1 - team, 1:]
            j = int(np.argmin(np.linalg.norm(opp - carrier, axis=1)))
            ax.plot([carrier[0], opp[j, 0]], [carrier[1], opp[j, 1]], color="#fde047", lw=1.5, zorder=2)
            ax.text((carrier[0] + opp[j, 0]) / 2, (carrier[1] + opp[j, 1]) / 2 + 1.5,
                    f"{np.linalg.norm(opp[j] - carrier):.1f} m", fontsize=7, color="#fde047", zorder=5)


def _pick_frame(r: MatchResult, region: str, min_minute: float = 4.0) -> int | None:
    """A frame where the home side has had the ball for >= 1 s in the given region."""
    held = (r.ball_state == HELD) & (r.possession == 0)
    for t in range(int(min_minute * 60 / DT), len(r.period)):
        if not held[max(0, t - 10):t + 1].all():
            continue
        x, y = _ball_in_attack_frame(r, t, 0)
        in_box = x > HALF_L - 16.5 and abs(y) < 20.16
        if (region == "box" and in_box) or (region != "box" and not in_box and third(x) == region):
            return t
    return None


def plot_snapshots(runs: list[ShapeRun], path: Path) -> None:
    from .plotting import new_figure

    clubs, players = load_clubs(), load_players()
    labels = {"defensive": "Build-up (own third)", "middle": "Middle third",
              "attacking": "Final third", "box": "Box"}
    fig, axes = new_figure(n_cols=len(labels), n_rows=len(runs), scale=3.2)
    for row, run in enumerate(runs):
        r = run.result
        numbers = _numbers(r, players)
        for col, (region, label) in enumerate(labels.items()):
            ax = axes[row][col]
            t = _pick_frame(r, region)
            if t is None:
                ax.axis("off")
                continue
            draw_frame(ax, r, t, clubs, numbers, mark_pressure=True)
            ax.set_title(f"{label}, {int(r.clock[t] // 60) + 1}' of half {r.period[t]}\n"
                         f"{clubs[run.attack].name} in possession vs {clubs[run.defend].name}", fontsize=8)
    fig.suptitle("Shape check from live play (yellow line = nearest defender to the ball carrier)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def animate_clip(r: MatchResult, path: Path, t0: int, seconds: float = CLIP_S, every: int = 2, title: str = "") -> None:
    from matplotlib import animation

    from .plotting import new_figure

    clubs, players = load_clubs(), load_players()
    numbers = _numbers(r, players)
    fig, axes = new_figure(scale=4.5)
    ax = axes[0][0]
    t1 = min(len(r.period) - 1, t0 + int(seconds / DT))

    def frame(t):
        ax.clear()
        draw_frame(ax, r, t, clubs, numbers)
        ax.set_title(f"{title}   {int(r.clock[t] // 60)}:{r.clock[t] % 60:04.1f} (half {r.period[t]})", fontsize=10)

    anim = animation.FuncAnimation(fig, frame, frames=range(t0, t1, every))
    anim.save(path, writer=animation.PillowWriter(fps=int(1 / (DT * every))))


def cmd_shape() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clubs = load_clubs()
    runs = [run_shape("EAS", "RIV"), run_shape("RIV", "EAS")]
    for run in runs:
        d = clubs[run.defend].tactics
        print(f"\n{clubs[run.attack].name} in possession vs {clubs[run.defend].name} "
              f"(press_intensity {d.press_intensity}, {d.press_height} press)")
        print(f"  {'ball third':<12}{'carrier':>10}{'options':>10}   (mean nearest-defender distance, m, live play)")
        for k in ("defensive", "middle", "attacking"):
            s = proximity_summary(run).get(k)
            if s:
                print(f"  {k:<12}{s['carrier_m']:>10.1f}{s['options_m']:>10.1f}")
    plot_snapshots(runs, OUT_DIR / "shape_snapshots.png")
    for run in runs:
        t0 = _pick_frame(run.result, "defensive") or 0
        animate_clip(run.result, OUT_DIR / f"shape_clip_{run.attack}_vs_{run.defend}.gif", max(0, t0 - 20),
                     title=f"{clubs[run.attack].name} vs {clubs[run.defend].name}")
    print(f"\nWrote plots to {OUT_DIR}")


def cmd_squads(card_ids: list[str]) -> None:
    clubs, players = load_clubs(), load_players()
    for club in clubs.values():
        t = club.tactics
        print(f"\n{club.name} ({club.tier}, {t.formation}, press {t.press_intensity} {t.press_height})")
        for p in squad(players, club.id):
            g = p.group_ratings
            star = "XI " if p.slot is not None else "   "
            styles = ", ".join(p.playstyles)
            print(f"  {star}{p.number:>2} {p.name:<24}{p.position:<4}{p.overall:>3}  "
                  f"PAC {g['pace']:>2} SHO {g['shooting']:>2} PAS {g['passing']:>2} "
                  f"DRI {g['dribbling']:>2} DEF {g['defending']:>2} PHY {g['physical']:>2}  {styles}")
    for pid in card_ids:
        print("\n" + players[pid].card())


def plot_match_overview(r, clubs, path: Path) -> None:
    """Shot map (size = xG), where each team's players spent the match, possession over time."""
    from .match import DEAD
    from .plotting import draw_pitch, new_figure

    fig, axes = new_figure(n_cols=3, n_rows=1, scale=3.6)
    ax = axes[0][0]
    draw_pitch(ax)
    for e in r.events:
        if e["type"] != "shot":
            continue
        team = clubs[e["team"]]
        t = 0 if e["team"] == r.home else 1
        # Home attacks right, away attacks left, in both halves.
        flip = r.directions[e["period"] - 1][t] * (1 if t == 0 else -1)
        face = team.colors["primary"] if e["outcome"] == "goal" else "none"
        ax.scatter(e["x"] * flip, e["y"] * flip, s=40 + 900 * e["xg"], facecolors=face,
                   edgecolors=team.colors["primary"], linewidths=1.5, zorder=4)
    ax.set_title(f"Shots: {clubs[r.home].name} →, {clubs[r.away].name} ← (size = xG, filled = goal)", fontsize=9)

    ax = axes[0][1]
    draw_pitch(ax)
    for t, club_id in enumerate((r.home, r.away)):
        xy = r.pos[:, t, 1:, :].reshape(-1, 2)[::37].copy()  # stride coprime with 10 cycles all players
        first_half = np.repeat(r.period == 1, 10)[::37]
        xy[~first_half] *= -1  # show both halves attacking the same way as the first half
        ax.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.08, c=clubs[club_id].colors["primary"])
    ax.set_title("Outfield positions (1st-half direction)", fontsize=9)

    ax = axes[0][2]
    in_play = r.ball_state != DEAD
    window = int(5 * 60 / DT)
    minutes, share = [], []
    for start in range(0, len(r.period) - window, window // 5):
        sl = slice(start, start + window)
        m = in_play[sl]
        if m.sum():
            minutes.append((start + window / 2) * DT / 60)
            share.append(100 * np.mean(r.possession[sl][m] == 0))
    ax.plot(minutes, share, color=clubs[r.home].colors["primary"], lw=2)
    ax.axhline(50, color="grey", lw=0.8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("minute")
    ax.set_ylabel(f"{clubs[r.home].name} possession % (5-min window)")
    for e in r.events:
        if e["type"] == "goal":
            ax.axvline(e["frame"] * DT / 60, color=clubs[e["team"]].colors["primary"], ls="--", lw=1)
    ax.set_title("Possession over time (dashed = goal)", fontsize=9)
    ax.set_aspect("auto")
    fig.suptitle(f"{clubs[r.home].name} {r.score[0]} - {r.score[1]} {clubs[r.away].name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def animate_goal_at(r, goal: dict, clubs, players, path: Path, before_s: float = 20.0, after_s: float = 3.0,
                    every: int = 2, subtitle: str = "") -> None:
    """Animate the build-up to `goal` (one of r.events)."""
    from matplotlib import animation

    from .plotting import new_figure

    shot = max((e for e in r.events if e["type"] == "shot" and e["frame"] <= goal["frame"]), key=lambda e: e["frame"])
    # Never open on a dead-ball wait: start at most 3 s before the last restart before the shot.
    t0 = max(0, shot["frame"] - int(before_s / DT))
    restarts = [e["frame"] for e in r.events if e["type"] in RESTART_EVENTS and t0 <= e["frame"] <= shot["frame"]]
    if restarts:
        t0 = max(t0, restarts[-1] - int(3.0 / DT))
    t1 = min(len(r.period) - 1, goal["frame"] + int(after_s / DT))
    numbers = _numbers(r, players)
    fig, axes = new_figure(scale=4.5)
    ax = axes[0][0]
    scorer = players[goal["player_id"]].name
    minute = int(goal["minute"]) + 1
    kind = {"penalty": " (penalty)", "free_kick": " (free kick)"}.get(shot.get("restart"), "")
    match = f"{clubs[r.home].name} vs {clubs[r.away].name}"

    def frame(t):
        ax.clear()
        draw_frame(ax, r, t, clubs, numbers)
        title = f"{clubs[goal['team']].name}: {scorer} {minute}'{kind}  ({(t - shot['frame']) * DT:+.1f} s)"
        if subtitle:
            title += f"\n{match}: {subtitle}"
        ax.set_title(title, fontsize=9)

    anim = animation.FuncAnimation(fig, frame, frames=range(t0, t1, every))
    anim.save(path, writer=animation.PillowWriter(fps=int(1 / (DT * every))))


def animate_goal(r, clubs, players, path: Path, **kw) -> bool:
    goals = [e for e in r.events if e["type"] == "goal"]
    if not goals:
        return False
    animate_goal_at(r, goals[0], clubs, players, path, **kw)
    return True


def cmd_match(home: str, away: str, seed: int) -> None:
    import pickle

    from .match import Match
    from .report import print_report

    clubs, players = load_clubs(), load_players()
    r = Match(home, away, seed=seed).run()
    print_report(r, {k: c.name for k, c in clubs.items()}, players)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"match_{home}_{away}_s{seed}"
    with open(OUT_DIR / f"{stem}.pkl", "wb") as f:
        pickle.dump(r, f)
    plot_match_overview(r, clubs, OUT_DIR / f"{stem}_overview.png")
    if animate_goal(r, clubs, players, OUT_DIR / f"{stem}_first_goal.gif"):
        print(f"\nWrote {stem}_overview.png and {stem}_first_goal.gif to {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulation checks")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("shape")
    sq = sub.add_parser("squads")
    sq.add_argument("--cards", nargs="*", default=[])
    mt = sub.add_parser("match")
    mt.add_argument("home")
    mt.add_argument("away")
    mt.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.cmd == "shape":
        cmd_shape()
    elif args.cmd == "match":
        cmd_match(args.home, args.away, args.seed)
    else:
        cmd_squads(args.cards)


if __name__ == "__main__":
    main()
