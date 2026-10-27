"""One simulated match: ball physics, carrier decisions, duels, restarts, substitutions.

Outcomes of passes and shots are drawn from the hidden truth models at the moment of
the kick; the ball then physically travels to the receiver, interceptor or out of play.
Everything is recorded at 10 Hz together with an event log and a log of every carrier
decision (the options, their true values and the choice), which step 3 turns into files.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import decisions
from .config import POSITION_ROLE, Player, load_clubs, load_players, player_params, squad
from .decisions import Context, Option
from .pitch import (
    GOAL_AREA_DEPTH,
    HALF_L,
    HALF_W,
    from_team_frame,
    unit,
)
from .tactics import kickoff_positions, offside_line
from .team import Overrides, TeamState, step_teams
from .truth import AIR_DRAG, GROUND_DRAG, pass_speed, pressure, travel_time

FPS = 10
DT = 1.0 / FPS
PERIOD_MINUTES = 45

HELD, FLIGHT, LOOSE, DEAD = 0, 1, 2, 3
BALL_STATES = {HELD: "held", FLIGHT: "flight", LOOSE: "loose", DEAD: "dead"}

RESTART_DELAY_S = {
    "kickoff": (45.0, 60.0), "throw_in": (13.0, 25.0), "goal_kick": (22.0, 36.0),
    "corner": (32.0, 46.0), "free_kick": (25.0, 45.0), "gk": (6.0, 12.0), "penalty": (45.0, 70.0),
}
BOX_TACKLE = 8.0          # challenge rate near goal (x the base duel rate); zero elsewhere
CLEAR_PROB = 0.6          # chance a defender winning the ball in his danger zone clears it
TACKLE_RANGE_M = 1.5      # a lost carry counts as a tackle only if a defender is this close
KEEPER_SPREAD = 0.8       # how much a keeper's save rating moves the chance of conceding vs xG
FOUL_BASE = 0.2           # chance a close challenge is a foul (plus more for aggressive players)
FOUL_CARE_BOX = 0.06      # ...multiplied by this inside the defender's own penalty area
WALL_RANGE_M = 32.0       # free kicks closer than this to goal get a defensive wall
BLOCK_PER_DEFENDER = 0.15  # extra chance a shot is blocked per defender in the shooting cone
BOX_TACKLE_RANGE_M = 40.0  # challenges start ramping up this far from goal (full strength 15 m closer)
CONTROL_RADIUS_M = 1.0    # a player controls a ground pass that comes this close
CONTROL_HEIGHT_M = 2.3    # ...and is no higher than this
AERIAL_REACH_M = 2.0      # high balls: chest, head or raised foot
AERIAL_HEIGHT_M = 2.8
ACCEL_ALLOWANCE_S = 0.35  # time to get up to speed when chasing a pass
# When a defender gets to a pass first, how often he gets clean control (else the ball runs loose).
CLEAN_INTERCEPTION = 0.5
CLEAN_DUEL_WIN = 0.4
RUN_ROLE_FACTOR = {"ST": 1.0, "W": 1.0, "AM": 0.9, "WM": 0.6, "WB": 0.6, "FB": 0.4,
                   "CM": 0.4, "DM": 0.05, "CB": 0.02, "GK": 0.0}
def line_dir(team: TeamState) -> np.ndarray:
    """Unit vector towards the goal the team attacks (global frame)."""
    return np.array([float(team.direction), 0.0])


CORNER_SPOTS = np.array([[HALF_L - 6.0, -2.5], [HALF_L - 6.0, 2.5], [HALF_L - 11.0, 0.0],
                         [HALF_L - 9.0, -6.0], [HALF_L - 9.0, 6.0]])


@dataclass
class Flight:
    kind: str                       # pass | shot
    start: np.ndarray               # global
    end: np.ndarray                 # global
    t0: int                         # frame the ball was kicked
    duration: float                 # seconds
    lofted: bool
    apex: float
    outcome: str
    receiver: tuple[int, int] | None = None   # (team, slot) who controls the ball at the end
    after: dict | None = None       # what happens at the end (restart, loose ball, ...)
    u: np.ndarray | None = None     # direction of travel (passes)
    v0: float = 0.0                 # kick speed (horizontal), m/s
    intended: tuple[int, int] | None = None  # (team, slot) the pass was meant for
    event: dict | None = None       # the pass event, corrected if nobody reaches the ball


@dataclass
class Restart:
    kind: str
    team: int
    spot: np.ndarray                # global
    taker: int
    resume: int                     # frame from which the restart may be taken


@dataclass
class MatchResult:
    match_id: str
    home: str
    away: str
    seed: int
    score: list[int]
    period_frames: list[int]
    directions: list[list[int]]     # [period][team] attack direction
    player_ids: list[str]           # index -> player id, for the per-frame id codes
    # Per-frame arrays (T = total frames)
    period: np.ndarray              # (T,)
    clock: np.ndarray               # (T,) seconds since the start of the period
    ball: np.ndarray                # (T, 3)
    ball_state: np.ndarray          # (T,)
    possession: np.ndarray          # (T,) team index, -1 if dead ball
    carrier: np.ndarray             # (T,) player code, -1 if nobody
    pos: np.ndarray                 # (T, 2, 11, 2)
    ids: np.ndarray                 # (T, 2, 11) player codes
    events: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)


class Match:
    def __init__(self, home_id: str, away_id: str, seed: int = 0, match_id: str | None = None,
                 clubs=None, players=None, period_minutes: float = PERIOD_MINUTES):
        self.rng = np.random.default_rng(seed)
        self.period_minutes = period_minutes
        self.seed = seed
        clubs = clubs or load_clubs()
        players = players or load_players()
        self.match_id = match_id or f"{home_id}-{away_id}-{seed}"
        self.teams = [TeamState.kickoff(clubs[home_id], players, +1),
                      TeamState.kickoff(clubs[away_id], players, -1)]
        self.bench: list[list[Player]] = [
            [p for p in squad(players, t.club.id) if p.slot is None] for t in self.teams]
        self.sub_minutes = [sorted(self.rng.uniform(55.0, 82.0, size=int(self.rng.integers(2, 4))).tolist())
                            for _ in self.teams]
        self.stoppage = [float(self.rng.uniform(1.0, 4.0)) for _ in range(2)]
        all_ids = [p.id for t in self.teams for p in squad(players, t.club.id)]
        clubs_all = clubs
        self.avg_save = float(np.mean([player_params(p, clubs_all[p.club_id].tactics).save
                                       for p in players.values() if p.position == "GK"]))
        self.code = {pid: k for k, pid in enumerate(all_ids)}
        self.player_ids = all_ids

        self.score = [0, 0]
        self.frame = 0
        self.period = 1
        self.period_frame = 0
        self.ball = np.zeros(3)
        self.ball_vel = np.zeros(2)
        self.state = DEAD
        self.poss = 0
        self.carrier: int | None = None
        self.last_touch = 0
        self.action = "hold"
        self.action_target: np.ndarray | None = None
        self.next_decision = 0
        self.last_decision = -100
        self.header = False
        self.kick_header = False
        self.loss_frame: int | None = None   # frame at which the current carry/hold loses the ball
        self.gained_frame = 0                # frame the current carrier received the ball
        self.runs: dict[int, tuple[np.ndarray, int]] = {}
        self.flight: Flight | None = None
        self.restart: Restart | None = None
        self.ball_rest: tuple[np.ndarray, int] | None = None  # (where the ball lies, until frame) after a goal
        self.events: list[dict] = []
        self.decision_log: list[dict] = []
        self.directions: list[list[int]] = []
        self.period_frames: list[int] = []
        self._rec: dict[str, list] = {k: [] for k in
                                      ("period", "clock", "ball", "ball_state", "possession", "carrier", "pos", "ids")}

    # ------------------------------------------------------------------ helpers
    @property
    def minute(self) -> float:
        return (self.period - 1) * self.period_minutes + self.period_frame * DT / 60.0

    def _frames(self, seconds: float) -> int:
        return max(1, round(seconds * FPS))

    def _pid(self, team: int, slot: int) -> str:
        return self.teams[team].players[slot].id

    def _event(self, kind: str, team: int | None = None, slot: int | None = None, **data) -> dict:
        ev = {"frame": self.frame, "period": self.period, "clock": round(self.period_frame * DT, 1),
              "minute": round(self.minute, 2), "type": kind,
              "team": self.teams[team].club.id if team is not None else None,
              "player_id": self._pid(team, slot) if team is not None and slot is not None else None}
        for k, v in data.items():
            ev[k] = [round(float(x), 2) for x in v] if isinstance(v, np.ndarray) else v
        self.events.append(ev)
        return ev

    def _goal_x(self, team: int) -> float:
        """x of the goal the team attacks, global frame."""
        return HALF_L * self.teams[team].direction

    # ------------------------------------------------------------------ match flow
    def run(self) -> MatchResult:
        for period in (1, 2):
            self._start_period(period)
            stoppage = self.stoppage[period - 1] if self.period_minutes == PERIOD_MINUTES else 0.0
            n = int((self.period_minutes + stoppage) * 60 * FPS)
            for pf in range(n):
                self.period_frame = pf
                self._tick()
                self._record()
                self.frame += 1
            self._event("period_end")
            self.period_frames.append(n)
        r = self._rec
        return MatchResult(
            match_id=self.match_id, home=self.teams[0].club.id, away=self.teams[1].club.id,
            seed=self.seed, score=self.score, period_frames=self.period_frames,
            directions=self.directions, player_ids=self.player_ids,
            period=np.array(r["period"], dtype=np.int8), clock=np.array(r["clock"], dtype=np.float32),
            ball=np.array(r["ball"], dtype=np.float32), ball_state=np.array(r["ball_state"], dtype=np.int8),
            possession=np.array(r["possession"], dtype=np.int8), carrier=np.array(r["carrier"], dtype=np.int16),
            pos=np.array(r["pos"], dtype=np.float32), ids=np.array(r["ids"], dtype=np.int16),
            events=self.events, decisions=self.decision_log,
        )

    def _start_period(self, period: int) -> None:
        self.period = period
        self.period_frame = 0
        if period == 2:
            for t in self.teams:
                t.direction *= -1
        self.directions.append([t.direction for t in self.teams])
        for t in self.teams:
            t.pos = from_team_frame(kickoff_positions(t.club.tactics), t.direction)
            t.vel[:] = 0.0
        if period == 1:
            self.kickoff_team = int(self.rng.integers(2))
            team = self.kickoff_team
        else:
            team = 1 - self.kickoff_team
        self._event("period_start")
        self._set_dead("kickoff", team, np.zeros(2), delay_s=1.0)
        taker = self.restart.taker
        self.teams[team].pos[taker] = np.array([-0.5 * self.teams[team].direction, 0.0])

    def _tick(self) -> None:
        if self.frame % 30 == 0:
            for t in self.teams:
                t.refresh_mark_error(self.rng)
        if self.state == DEAD:
            self._tick_dead()
        elif self.state == HELD:
            self._tick_held()
        elif self.state == FLIGHT:
            self._tick_flight()
        else:
            self._tick_loose()
        self._update_runs()
        self._move_players()

    def _record(self) -> None:
        r = self._rec
        r["period"].append(self.period)
        r["clock"].append(self.period_frame * DT)
        r["ball"].append(self.ball.copy())
        r["ball_state"].append(self.state)
        r["possession"].append(self.poss if self.state != DEAD else -1)
        carrier = -1
        if self.state == HELD and self.carrier is not None:
            carrier = self.code[self._pid(self.poss, self.carrier)]
        r["carrier"].append(carrier)
        r["pos"].append(np.stack([t.pos for t in self.teams]))
        r["ids"].append([[self.code[p.id] for p in t.players] for t in self.teams])

    # ------------------------------------------------------------------ movement
    def _move_players(self) -> None:
        att, dfn = self.teams[self.poss], self.teams[1 - self.poss]
        att_over: Overrides = {}
        def_over: Overrides = {}
        carrier = self.carrier
        ball2 = self.ball[:2]
        allow_press = True

        if self.state == DEAD and self.restart is not None:
            r = self.restart
            carrier = r.taker
            ball2 = r.spot
            allow_press = False
            att_over[r.taker] = (r.spot, att.params[r.taker].vmax * 0.6)
            if r.kind == "penalty":
                self._penalty_positions(att, dfn, r, att_over, def_over)
            elif r.kind in ("free_kick", "corner"):
                self._wall_and_distance(att, dfn, r, def_over)
            if r.kind == "corner":
                forwards = np.argsort([-self._slot_depth(self.poss, i) for i in range(11)])
                k = 0
                for i in forwards:
                    if i in (0, r.taker) or k >= len(CORNER_SPOTS):
                        continue
                    att_over[int(i)] = (from_team_frame(CORNER_SPOTS[k], att.direction), att.params[i].vmax * 0.6)
                    k += 1
        elif self.state == HELD and self.carrier is not None:
            if self.action == "carry" and self.action_target is not None:
                me = att.params[self.carrier]
                att_over[self.carrier] = (self.action_target, me.vmax * (0.6 + 0.25 * me.dribble))
            else:
                att_over[self.carrier] = (att.pos[self.carrier], att.params[self.carrier].vmax * 0.3)
        elif self.state == FLIGHT and self.flight is not None:
            carrier = None
            f = self.flight
            if f.kind == "pass" and f.receiver is None and f.intended is None:
                # A clearance in the air: the nearest player of each side goes for the drop.
                for team, over in ((self.poss, att_over), (1 - self.poss, def_over)):
                    i = int(np.argmin(np.linalg.norm(self.teams[team].pos - f.end, axis=1)))
                    over[i] = (f.end, self.teams[team].params[i].vmax)
            elif f.kind == "pass":
                # Whoever is going for the ball runs to meet it while it is still moving.
                for chaser in {f.receiver, f.intended} - {None}:
                    team, slot = chaser
                    over = att_over if team == self.poss else def_over
                    vmax = self.teams[team].params[slot].vmax
                    over[slot] = (self._intercept_point(f, self.teams[team].pos[slot], vmax), vmax)
            elif f.receiver is not None:
                team, slot = f.receiver
                over = att_over if team == self.poss else def_over
                over[slot] = (f.end, self.teams[team].params[slot].vmax)
            if f.kind == "shot":
                gk_over = def_over
                gk_target = np.array([self._goal_x(self.poss) - np.sign(self._goal_x(self.poss)) * 0.8,
                                      np.clip(f.end[1], -3.5, 3.5)])
                gk_over[0] = (gk_target, dfn.params[0].vmax)
        elif self.state == LOOSE:
            carrier = None
            for team, over in ((self.poss, att_over), (1 - self.poss, def_over)):
                d = np.linalg.norm(self.teams[team].pos - ball2, axis=1)
                for i in np.argsort(d)[:2]:
                    over[int(i)] = (ball2, self.teams[team].params[i].vmax)

        runs = {i: tgt for i, (tgt, _) in self.runs.items()} if self.state in (HELD, FLIGHT) else None
        step_teams(att, dfn, ball2, carrier, DT, self.rng, minute=self.minute,
                   attack_overrides=att_over, defend_overrides=def_over, runs=runs, allow_press=allow_press)

    def _penalty_positions(self, att: TeamState, dfn: TeamState, r: Restart,
                           att_over: Overrides, def_over: Overrides) -> None:
        """Everyone but the taker and the keeper waits outside the box."""
        edge = HALF_L - 18.0
        for team, over in ((att, att_over), (dfn, def_over)):
            for i in range(11):
                if (team is att and i == r.taker) or (team is dfn and i == 0):
                    continue
                xy = att.frame(team.pos[i])
                if xy[0] > edge - 1.0 and abs(xy[1]) < 22.0:
                    xy = np.array([edge - 1.0, np.clip(xy[1], -20.0, 20.0)])
                over[i] = (from_team_frame(xy, att.direction), team.params[i].vmax * 0.5)
        def_over[0] = (from_team_frame(np.array([HALF_L - 0.3, 0.0]), att.direction), dfn.params[0].vmax * 0.5)

    def _wall_and_distance(self, att: TeamState, dfn: TeamState, r: Restart, def_over: Overrides) -> None:
        """Defenders keep 9.15 m away; close to goal some of them form a wall on the line to goal."""
        goal = from_team_frame(np.array([HALF_L, 0.0]), att.direction)
        to_goal = float(np.linalg.norm(goal - r.spot))
        taken = set()
        if r.kind == "free_kick" and to_goal < WALL_RANGE_M:
            n_wall = 2 + int(np.clip((WALL_RANGE_M - to_goal) / 6.0, 0.0, 3.0))
            line = unit(goal - r.spot)
            normal = np.array([-line[1], line[0]])
            centre = r.spot + line * 9.15
            spots = [centre + normal * (k - (n_wall - 1) / 2.0) * 0.7 for k in range(n_wall)]
            d = np.linalg.norm(dfn.pos[None, 1:, :] - np.array(spots)[:, None, :], axis=2)
            for k, spot in enumerate(spots):
                order = [int(j) + 1 for j in np.argsort(d[k]) if int(j) + 1 not in taken]
                taken.add(order[0])
                def_over[order[0]] = (spot, dfn.params[order[0]].vmax * 0.7)
        for i in range(1, 11):
            if i in taken:
                continue
            gap = dfn.pos[i] - r.spot
            if np.linalg.norm(gap) < 9.6:
                def_over[i] = (r.spot + unit(gap if np.linalg.norm(gap) > 0.1 else -line_dir(att)) * 9.6,
                               dfn.params[i].vmax * 0.7)

    def _slot_depth(self, team: int, slot: int) -> float:
        from .formations import FORMATIONS
        return FORMATIONS[self.teams[team].club.tactics.formation][slot][1]

    def _update_runs(self) -> None:
        self.runs = {i: v for i, v in self.runs.items() if v[1] > self.frame}
        if self.state != HELD:
            return
        att, dfn = self.teams[self.poss], self.teams[1 - self.poss]
        own = att.frame(att.pos)
        opp = dfn.frame(dfn.pos)
        last_line = float(np.sort(opp[1:, 0])[-1])
        line = offside_line(opp, att.frame(self.ball[:2]))
        for i in range(1, 11):
            if i == self.carrier or i in self.runs or own[i, 0] > line:
                continue
            p = att.params[i]
            rate = p.run_rate / 60.0 * DT * RUN_ROLE_FACTOR[POSITION_ROLE[att.players[i].position]]
            if self.rng.random() < rate:
                x = min(max(own[i, 0] + 10.0, last_line + 5.0), HALF_L - 8.0)
                y = float(np.clip(own[i, 1] + self.rng.normal(0.0, 5.0), -HALF_W + 3.0, HALF_W - 3.0))
                self.runs[i] = (from_team_frame(np.array([x, y]), att.direction), self.frame + self._frames(3.5))

    # ------------------------------------------------------------------ ball states
    def _tick_held(self) -> None:
        att, dfn = self.teams[self.poss], self.teams[1 - self.poss]
        c = self.carrier
        pos = att.pos[c]
        speed = np.linalg.norm(att.vel[c])
        offset = unit(att.vel[c]) * 0.5 if speed > 0.5 else np.zeros(2)
        self.ball[:2] = pos + offset
        self.ball[2] = 0.0

        if (self.action == "carry" and self.action_target is not None
                and np.linalg.norm(self.action_target - pos) < 1.0):
            self.action = "hold"

        if self.loss_frame is not None and self.frame >= self.loss_frame:
            self._lose_ball()
            return
        if self._duels():
            return

        press = float(pressure(np.linalg.norm(dfn.pos - self.ball[:2], axis=1).min()))
        rushed = press > 0.6 and self.frame - self.last_decision >= 5 and self.rng.random() < 0.3
        if self.frame >= self.next_decision or rushed:
            self._decide()

    def _lose_ball(self) -> None:
        """The carry/hold failed (drawn from its true keep probability when it was chosen)."""
        att, dfn = self.teams[self.poss], self.teams[1 - self.poss]
        c = self.carrier
        self.loss_frame = None
        d = np.linalg.norm(dfn.pos - att.pos[c], axis=1)
        d[0] = np.inf
        j = int(np.argmin(d))
        if TACKLE_RANGE_M <= d[j] < 4.0:
            # Lost control under pressure without a challenge: the ball runs loose towards the defender.
            self._event("dispossessed", self.poss, c, x=self.ball[0], y=self.ball[1],
                        pressured_by=self._pid(1 - self.poss, j))
            self.last_touch = self.poss
            toward = unit(dfn.pos[j] - self.ball[:2]) + self.rng.normal(0.0, 0.4, size=2)
            self._loose(self.ball[:2].copy(), unit(toward) * self.rng.uniform(3.0, 6.0))
            return
        if d[j] < TACKLE_RANGE_M:
            aggression = dfn.players[j].attributes["aggression"]
            p_foul = (FOUL_BASE + 0.07 + 0.2 * max(0.0, (aggression - 75) / 25.0)) * self._foul_care()
            if self.rng.random() < p_foul:
                self._foul(j)
                return
            self._event("tackle", 1 - self.poss, j, x=self.ball[0], y=self.ball[1],
                        tackled_id=self._pid(self.poss, c))
            if self._deflected_out(self.ball[:2].copy(), defender_team=1 - self.poss):
                return
            if self.rng.random() < 0.7 + 0.2 * self._danger(self.ball[:2], 1 - self.poss):
                self._gain(1 - self.poss, j, how="tackle")
                return
            self.last_touch = 1 - self.poss
        else:
            self._event("miscontrol", self.poss, c, x=self.ball[0], y=self.ball[1])
            self.last_touch = self.poss
        self._scramble(self.ball[:2].copy(), 1 - self.poss, (4.0, 7.0))

    def _duels(self) -> bool:
        """Box defending: near their own goal defenders commit to challenges and blocks.

        Elsewhere losing the ball is already priced into the keep probability of the carrier's
        chosen carry/hold (drawn in `_decide`), so there is no second, per-frame tackle chance.
        Returns True if the ball changed hands or went dead.
        """
        att, dfn = self.teams[self.poss], self.teams[1 - self.poss]
        c = self.carrier
        me = att.params[c]
        to_goal = float(np.linalg.norm(att.frame(att.pos[c]) - np.array([HALF_L, 0.0])))
        urgency = float(np.clip((BOX_TACKLE_RANGE_M - to_goal) / 15.0, 0.0, 1.0))
        if urgency <= 0.0:
            return False
        d = np.linalg.norm(dfn.pos - att.pos[c], axis=1)
        reach = 1.5 + 0.5 * urgency
        for j in np.where(d < reach)[0]:
            if j == 0:
                continue
            opp = dfn.params[j]
            shield = 1.0 - 0.4 * me.strength * (self.action == "hold")
            hazard = 0.3 * opp.tackle * (1.0 - 0.7 * me.dribble) * shield * (0.6 + 0.8 * opp.press)
            hazard *= BOX_TACKLE * urgency
            if self.rng.random() >= 1.0 - np.exp(-hazard * DT):
                continue
            aggression = dfn.players[j].attributes["aggression"]
            p_foul = (FOUL_BASE + 0.15 * max(0.0, (aggression - 75) / 25.0)) * self._foul_care()
            if self.rng.random() < p_foul:
                self._foul(int(j))
                return True
            danger = self._danger(self.ball[:2], 1 - self.poss)
            if self.rng.random() < CLEAR_PROB * danger:
                # A block or a foot in that sends the ball away: a clearance, not a tackle.
                self._clearance(1 - self.poss, int(j), header=False, challenge=True)
                return True
            self._event("tackle", 1 - self.poss, int(j), x=self.ball[0], y=self.ball[1],
                        tackled_id=self._pid(self.poss, c))
            if self.rng.random() < 0.7 + 0.2 * danger:
                self._gain(1 - self.poss, int(j), how="tackle_kept")
            else:
                self.last_touch = 1 - self.poss
                self._scramble(self.ball[:2].copy(), 1 - self.poss, (4.0, 7.0))
            return True
        return False

    def _in_attacking_box(self) -> bool:
        """Is the ball in the penalty area the team in possession attacks?"""
        x, y = self.teams[self.poss].frame(self.ball[:2])
        return x > HALF_L - 16.5 and abs(y) < 20.16

    def _foul_care(self) -> float:
        """Defenders are careful near their own goal, and very careful in their own box."""
        if self._in_attacking_box():
            return FOUL_CARE_BOX
        return 1.0 - 0.6 * self._danger(self.ball[:2], 1 - self.poss)

    def _foul(self, defender: int) -> None:
        att = self.teams[self.poss]
        penalty = self._in_attacking_box()
        self._event("foul", 1 - self.poss, defender, x=self.ball[0], y=self.ball[1],
                    fouled_id=self._pid(self.poss, self.carrier), penalty=penalty)
        if penalty:
            self._set_dead("penalty", self.poss, from_team_frame(np.array([HALF_L - 11.0, 0.0]), att.direction))
        else:
            self._set_dead("free_kick", self.poss, self.ball[:2].copy())

    @staticmethod
    def _flight_state(f: Flight, t: float) -> tuple[np.ndarray, float, float]:
        """Ball position, height and speed t seconds after a pass was kicked (no steering)."""
        if not f.lofted:
            k = GROUND_DRAG
            return f.start + f.u * (f.v0 / k * (1.0 - np.exp(-k * t))), 0.0, f.v0 * np.exp(-k * t)
        T = f.duration
        if t <= T:
            k = AIR_DRAG
            s = f.v0 / k * (1.0 - np.exp(-k * t))
            return f.start + f.u * s, 4.0 * f.apex * (t / T) * (1.0 - t / T), f.v0 * np.exp(-k * t)
        # Landed: bounce loses speed, then it rolls.
        s_land = f.v0 / AIR_DRAG * (1.0 - np.exp(-AIR_DRAG * T))
        v_land = 0.4 * f.v0 * np.exp(-AIR_DRAG * T)
        tt = t - T
        s = s_land + v_land / GROUND_DRAG * (1.0 - np.exp(-GROUND_DRAG * tt))
        return f.start + f.u * s, 0.0, v_land * np.exp(-GROUND_DRAG * tt)

    def _overrun(self, f: Flight, xy: np.ndarray) -> None:
        """Nobody reached the pass: record what actually happened (the truth keeps the original odds)."""
        if f.event is not None:
            f.event.update(outcome="overhit", receiver_id=None, end_x=round(float(xy[0]), 2),
                           end_y=round(float(xy[1]), 2))
        self._event("overrun", self.poss, None, x=xy[0], y=xy[1])

    def _intercept_point(self, f: Flight, pos: np.ndarray, vmax: float) -> np.ndarray:
        """First point on the ball's path the player can reach no later than the ball."""
        t_now = (self.frame - f.t0) * DT
        xy = pos
        for tau in np.arange(0.0, 4.0, 0.1):
            xy, z, _ = self._flight_state(f, t_now + tau)
            if z > (AERIAL_HEIGHT_M if f.lofted else CONTROL_HEIGHT_M):
                continue
            if np.linalg.norm(xy - pos) / vmax + ACCEL_ALLOWANCE_S <= tau:
                return xy
        return xy

    def _tick_flight(self) -> None:
        f = self.flight
        t = (self.frame - f.t0) * DT
        if f.kind == "shot":
            if t >= f.duration:
                self._resolve_flight()
                return
            k = 0.1
            frac = (1.0 - np.exp(-k * t)) / (1.0 - np.exp(-k * f.duration))
            u = t / f.duration
            self.ball[:2] = f.start + (f.end - f.start) * frac
            self.ball[2] = 4.0 * f.apex * u * (1.0 - u)
            return
        xy, z, speed = self._flight_state(f, t)
        self.ball[:2] = xy
        self.ball[2] = z
        if f.receiver is None:  # misplaced or overhit: runs to where it was going, then loose / out
            if t >= f.duration:
                self._resolve_flight()
            return
        team, slot = f.receiver
        if f.lofted:
            reach, height = AERIAL_REACH_M, AERIAL_HEIGHT_M
        else:
            reach, height = CONTROL_RADIUS_M, CONTROL_HEIGHT_M
            if team != self.poss:  # a defender cutting it out stretches as far as his rating allows
                reach = max(reach, self.teams[team].params[slot].intercept_reach_m)
        if z <= height and t >= 0.2 and np.linalg.norm(self.teams[team].pos[slot] - xy) <= reach:
            self._resolve_flight()
        elif abs(xy[0]) > HALF_L or abs(xy[1]) > HALF_W:
            self._overrun(f, xy)
            self._ball_out(xy.copy(), self.poss)
        elif t > f.duration + 2.5:
            # Nobody got there: the ball is just rolling now.
            self._overrun(f, xy)
            self._loose(xy.copy(), f.u * max(speed, 0.5))

    def _tick_loose(self) -> None:
        self.ball[:2] += self.ball_vel * DT
        self.ball[2] = 0.0
        self.ball_vel *= np.exp(-2.0 * GROUND_DRAG * DT)
        if abs(self.ball[0]) > HALF_L or abs(self.ball[1]) > HALF_W:
            self._ball_out(self.ball[:2].copy(), self.last_touch)
            return
        best = None
        for team in (0, 1):
            d = np.linalg.norm(self.teams[team].pos - self.ball[:2], axis=1)
            i = int(np.argmin(d))
            if d[i] < 1.0 and (best is None or d[i] < best[2]):
                best = (team, i, d[i])
        if best is not None:
            self._event("recovery", best[0], best[1], x=self.ball[0], y=self.ball[1])
            self._gain(best[0], best[1], how="recovery")

    def _tick_dead(self) -> None:
        r = self.restart
        resting = self.ball_rest is not None and self.frame < self.ball_rest[1]
        self.ball[:2] = self.ball_rest[0] if resting else r.spot
        self.ball[2] = 0.0
        taker_pos = self.teams[r.team].pos[r.taker]
        if self.frame >= r.resume and (np.linalg.norm(taker_pos - r.spot) < 2.0 or self.frame >= r.resume + 50):
            self.teams[r.team].pos[r.taker] = r.spot.copy()
            self.state = HELD
            self.poss = r.team
            self.carrier = r.taker
            self.gained_frame = self.frame
            self.last_touch = r.team
            self._event(r.kind, r.team, r.taker, x=r.spot[0], y=r.spot[1])
            self._decide(restart=r.kind)
            self.restart = None

    # ------------------------------------------------------------------ transitions
    def _gain(self, team: int, slot: int, how: str, lofted: bool = False) -> None:
        # Winning it near your own goal you often just get rid of it.
        wins_ball = how in ("tackle", "interception", "duel_won") or (how == "recovery" and team != self.last_touch)
        if wins_ball and self.rng.random() < CLEAR_PROB * self._danger(self.ball[:2], team):
            self._clearance(team, slot, header=lofted or self.ball[2] > 1.2)
            return
        if team != self.poss:
            self.runs = {}
        self.poss = team
        self.carrier = slot
        self.gained_frame = self.frame
        self.state = HELD
        self.last_touch = team
        self.flight = None
        self.action = "hold"
        self.action_target = None
        self.loss_frame = None
        own = self.teams[team].frame(self.teams[team].pos[slot])
        self.header = lofted and own[0] > HALF_L - 16.5 and abs(own[1]) < 20.16
        if self.header:
            delay = 0.1
        elif how == "recovery":
            delay = self.rng.uniform(0.5, 1.0)  # first touch on a bouncing, loose ball
        else:
            delay = self.rng.uniform(0.3, 0.6)
        self.next_decision = self.frame + self._frames(delay)

    def _clearance(self, team: int, slot: int, header: bool, challenge: bool = False) -> None:
        """Head or kick the ball high and away from goal; it lands as a loose ball (or goes out)."""
        t = self.teams[team]
        start = self.ball[:2].copy()
        direction = unit(np.array([t.direction, self.rng.normal(0.0, 0.7)]))
        dist = self.rng.uniform(15.0, 30.0) if header else self.rng.uniform(25.0, 45.0)
        end = start + direction * dist
        outcome, after = "clearance", None
        if abs(end[0]) > HALF_L or abs(end[1]) > HALF_W:
            outcome, after = "out", {"out_at": end.copy()}
        v0 = pass_speed(dist, True)
        duration = float(travel_time(dist, v0, True))
        apex = float(np.clip(0.15 * dist, 4.0, 14.0))
        self._event("clearance", team, slot, x=start[0], y=start[1], end_x=end[0], end_y=end[1], header=header,
                    challenge=challenge)
        if team != self.poss:
            self.runs = {}
        self.poss = team
        self.carrier = None
        self.last_touch = team
        self.flight = Flight("pass", start, end, self.frame, duration, True, apex, outcome, None, after,
                             u=direction, v0=v0, intended=None)
        self.state = FLIGHT

    def _danger(self, at: np.ndarray, defending_team: int) -> float:
        """0 far from the defending team's goal .. 1 inside ~20 m of it."""
        own_goal = np.array([-HALF_L * self.teams[defending_team].direction, 0.0])
        return float(np.clip((35.0 - np.linalg.norm(at - own_goal)) / 15.0, 0.0, 1.0))

    def _scramble(self, at: np.ndarray, defending_team: int, speed: tuple[float, float]) -> None:
        """A ball knocked loose. Near their own goal defenders clear it away; elsewhere it squirts anywhere."""
        danger = self._danger(at, defending_team)
        angle = self.rng.uniform(0, 2 * np.pi)
        random_dir = np.array([np.cos(angle), np.sin(angle)])
        away = np.array([self.teams[defending_team].direction, self.rng.normal(0.0, 0.6)])
        direction = unit((1.0 - danger) * random_dir + danger * unit(away))
        lo, hi = speed
        self._loose(at.copy(), direction * self.rng.uniform(lo + 5.0 * danger, hi + 6.0 * danger))

    def _deflected_out(self, at: np.ndarray, defender_team: int) -> bool:
        """A contested ball near a line is often knocked out of play off the defender."""
        to_side = HALF_W - abs(at[1])
        to_end = HALF_L - abs(at[0])
        p_out = 0.45 if to_side <= to_end else 0.2
        if min(to_side, to_end) > 12.0 or self.rng.random() > p_out:
            return False
        if to_side <= to_end:
            out_at = np.array([at[0], np.sign(at[1] or 1.0) * (HALF_W + 0.5)])
        else:
            out_at = np.array([np.sign(at[0]) * (HALF_L + 0.5), at[1]])
        self._event("deflection", defender_team, None, x=at[0], y=at[1])
        self._ball_out(out_at, defender_team)
        return True

    def _loose(self, at: np.ndarray, vel: np.ndarray) -> None:
        self.state = LOOSE
        self.carrier = None
        self.flight = None
        self.ball[:2] = at
        self.ball_vel = vel

    def _set_dead(self, kind: str, team: int, spot: np.ndarray, delay_s: float | None = None) -> None:
        self.state = DEAD
        self.carrier = None
        self.flight = None
        self.runs = {}
        self.poss = team
        lo, hi = RESTART_DELAY_S[kind]
        delay = self.rng.uniform(lo, hi) if delay_s is None else delay_s
        t = self.teams[team]
        if kind in ("goal_kick", "gk"):
            taker = 0
        elif kind == "kickoff":
            taker = int(np.argmax([self._slot_depth(team, i) for i in range(11)]))
        elif kind == "corner":
            cand = [i for i in range(1, 11) if POSITION_ROLE[t.players[i].position] not in ("CB", "GK")]
            taker = max(cand, key=lambda i: t.players[i].attributes["long_passing"])
        elif kind == "penalty":
            taker = max(range(1, 11), key=lambda i: t.players[i].attributes["finishing"])
        else:
            d = np.linalg.norm(t.pos - spot, axis=1)
            d[0] = np.inf
            taker = int(np.argmin(d))
        self.restart = Restart(kind, team, np.asarray(spot, dtype=float), taker, self.frame + self._frames(delay))
        self.ball[:2] = spot
        self.ball[2] = 0.0
        self._maybe_substitute()

    def _ball_out(self, at: np.ndarray, last_touch: int) -> None:
        self._event("ball_out", last_touch, None, x=at[0], y=at[1])
        other = 1 - last_touch
        if abs(at[1]) > HALF_W and abs(at[0]) <= HALF_L:
            spot = np.array([np.clip(at[0], -HALF_L + 1, HALF_L - 1), np.sign(at[1]) * HALF_W])
            self._set_dead("throw_in", other, spot)
            return
        side = np.sign(at[0])
        defending = 0 if self.teams[0].direction == -side else 1
        if last_touch == defending:
            spot = np.array([side * HALF_L, np.sign(at[1] or 1.0) * HALF_W])
            self._set_dead("corner", 1 - defending, spot)
        else:
            spot = np.array([side * (HALF_L - GOAL_AREA_DEPTH), np.sign(at[1] or 1.0) * 5.0])
            self._set_dead("goal_kick", defending, spot)

    def _maybe_substitute(self) -> None:
        for team in (0, 1):
            t = self.teams[team]
            while self.sub_minutes[team] and self.sub_minutes[team][0] <= self.minute and self.bench[team]:
                self.sub_minutes[team].pop(0)
                weights = np.array([p.fatigue for p in t.params[1:]])
                slot = 1 + int(self.rng.choice(10, p=weights / weights.sum()))
                out = t.players[slot]
                same = [p for p in self.bench[team] if p.position == out.position]
                role = [p for p in self.bench[team] if POSITION_ROLE[p.position] == POSITION_ROLE[out.position]]
                outfield = [p for p in self.bench[team] if p.position != "GK"]
                pool = same or role or outfield
                if not pool:
                    break
                incoming = pool[0]
                self.bench[team].remove(incoming)
                t.substitute(slot, incoming)
                self._event("substitution", team, slot, off_id=out.id)

    # ------------------------------------------------------------------ decisions
    def _decide(self, restart: str | None = None) -> None:
        att, dfn = self.teams[self.poss], self.teams[1 - self.poss]
        c = self.carrier
        ctx = Context(
            carrier=c, own=att.frame(att.pos), own_vel=att.frame(att.vel), own_params=att.params,
            opp=att.frame(dfn.pos), opp_params=dfn.params, ball=att.frame(self.ball[:2]),
            decision_quality=att.club.tactics.decision_quality, tempo=att.club.tactics.tempo,
            runs={i: att.frame(tgt) for i, (tgt, _) in self.runs.items()},
            restart=restart, header=self.header,
            time_on_ball=(self.frame - self.gained_frame) * DT,
            lead=self.score[self.poss] - self.score[1 - self.poss],
        )
        opts = decisions.options(ctx)
        press = decisions.carrier_pressure(ctx)
        me = att.params[c]
        temp = decisions.temperature(ctx.decision_quality, press, me.composure)
        k = decisions.choose(opts, temp, self.rng)
        chosen = opts[k]
        self.decision_log.append({
            "frame": self.frame, "period": self.period, "team": att.club.id, "player_id": self._pid(self.poss, c),
            "x": float(ctx.ball[0]), "y": float(ctx.ball[1]), "pressure": press, "composure": me.composure,
            "temperature": temp, "restart": restart, "header": self.header, "chosen": k,
            "options": [{"kind": o.kind, "receiver": self._pid(self.poss, o.receiver) if o.receiver is not None else None,
                         "tx": float(o.target[0]), "ty": float(o.target[1]), "lofted": o.lofted,
                         "through": o.through, "p_success": o.p_success, "value": o.value} for o in opts],
        })
        self.last_decision = self.frame
        self.kick_header = self.header  # remembered for the shot/pass this decision produces
        self.header = False
        self.loss_frame = None
        if chosen.kind == "pass":
            self._kick_pass(chosen, restart)
        elif chosen.kind == "shot":
            self._kick_shot(chosen, restart)
        else:
            self.action = chosen.kind
            self.action_target = from_team_frame(chosen.target, att.direction) if chosen.kind == "carry" else None
            tempo = att.club.tactics.tempo
            mean = (1.6 - 0.9 * tempo) * (1.0 - 0.4 * press)
            hold = self.rng.gamma(2.0, mean / 2.0) + 0.2
            if chosen.kind == "carry":
                hold = max(hold, CARRY_TIME_S)
            self.next_decision = self.frame + self._frames(hold)
            if self.rng.random() > chosen.p_success:
                self.loss_frame = self.frame + self._frames(hold * self.rng.uniform(0.3, 0.9))

    def _kick_pass(self, o: Option, restart: str | None) -> None:
        att = self.teams[self.poss]
        start = self.ball[:2].copy()
        info = o.info
        r = self.rng.random()
        receiver: tuple[int, int] | None
        after = None
        if r < info["p_complete"]:
            outcome, end, receiver = "complete", from_team_frame(o.target, att.direction), (self.poss, o.receiver)
        else:
            fails = np.array([info["p_exec_fail"], info["p_lane_fail"], info["p_contest_fail"]])
            mode = int(self.rng.choice(3, p=fails / fails.sum())) if fails.sum() > 0 else 0
            if mode == 1:
                outcome = "intercepted"
                end = from_team_frame(info["lane_point"], att.direction)
                receiver = (1 - self.poss, info["lane_defender"])
            elif mode == 2:
                outcome = "lost_duel"
                end = from_team_frame(o.target, att.direction)
                receiver = (1 - self.poss, info["contest_defender"])
            else:
                direction = unit(o.target - att.frame(start))
                normal = np.array([-direction[1], direction[0]])
                miss = o.target + normal * self.rng.normal(0.0, 8.0) + direction * self.rng.uniform(2.0, 15.0)
                end = from_team_frame(miss, att.direction)
                receiver = None
                if abs(end[0]) > HALF_L or abs(end[1]) > HALF_W:
                    outcome = "out"
                    after = {"out_at": end.copy()}
                    end = np.array([np.clip(end[0], -HALF_L - 1, HALF_L + 1), np.clip(end[1], -HALF_W - 1, HALF_W + 1)])
                else:
                    outcome = "misplaced"
                    after = {"loose_vel": unit(end - start) * 3.0}
        # The ball travels along the line it was struck on, weighted for the intended target;
        # an interception or lost duel happens where the defender meets that moving ball.
        aim = from_team_frame(o.target, att.direction) if receiver is not None else end
        dist = max(float(np.linalg.norm(aim - start)), 1.0)
        v0 = pass_speed(dist, o.lofted)
        duration = max(float(travel_time(dist, v0, o.lofted)), 0.2)
        apex = float(np.clip(0.12 * dist, 2.0, 12.0)) if o.lofted else 0.0
        direction = unit(aim - start)
        pass_event = self._event("pass", self.poss, self.carrier, x=start[0], y=start[1], end_x=end[0], end_y=end[1],
                    receiver_id=self._pid(*receiver) if receiver else None, outcome=outcome,
                    lofted=o.lofted, through=o.through, restart=restart, p_complete=info["p_complete"],
                    intended_id=self._pid(self.poss, o.receiver))
        self.flight = Flight("pass", start, end, self.frame, duration, o.lofted, apex, outcome, receiver, after,
                             u=direction, v0=v0, intended=(self.poss, o.receiver), event=pass_event)
        self.state = FLIGHT
        self.carrier = None
        self.last_touch = self.poss

    def _kick_shot(self, o: Option, restart: str | None) -> None:
        att, dfn = self.teams[self.poss], self.teams[1 - self.poss]
        start = self.ball[:2].copy()
        own_start = att.frame(start)
        xg = o.info["xg"]
        press = float(pressure(np.linalg.norm(dfn.pos - start, axis=1).min()))
        goal_x = self._goal_x(self.poss)
        sgn = np.sign(goal_x)
        r = self.rng.random()
        if restart == "penalty":
            p_block = 0.0
        elif restart == "free_kick":
            p_block = min(0.1 + 0.07 * o.info.get("cone", 0), 0.4)  # the wall
        else:
            p_block = min(0.08 + 0.2 * press + BLOCK_PER_DEFENDER * o.info.get("cone", 0), 0.7)
        # xG is the true chance of scoring against an average keeper. The block / on-target / save
        # steps below are chosen so that, on average, they multiply back to exactly that chance;
        # a better or worse keeper moves it up or down from there.
        gk = dfn.params[0]
        p_goal = float(np.clip(xg * (1.0 + KEEPER_SPREAD * (self.avg_save - gk.save)), 0.0, 0.97))
        p_block = min(p_block, max(0.0, 1.0 - p_goal / 0.9))
        p_on = 0.92 if restart == "penalty" else float(np.clip(0.3 + 1.3 * xg, 0.3, 0.9))
        p_on = min(max(p_on, p_goal / ((1.0 - p_block) * 0.97)), 1.0)
        p_goal_on = min(p_goal / ((1.0 - p_block) * p_on), 1.0)
        receiver, after, apex = None, None, float(self.rng.uniform(0.2, 2.2))
        if r < p_block:
            outcome = "blocked"
            end = start + unit(np.array([goal_x, 0.0]) - start) * 1.5
            after = {"loose_vel": unit(np.array([-sgn, self.rng.normal()])) * self.rng.uniform(8.0, 12.0)}
            apex = 0.2
        elif self.rng.random() < p_on:
            if self.rng.random() < p_goal_on:
                outcome = "goal"
                end = np.array([goal_x + sgn * 0.8, self.rng.uniform(-3.0, 3.0)])
            else:
                outcome = "saved"
                end = dfn.pos[0].copy()
                receiver = (1 - self.poss, 0)
        else:
            outcome = "off_target"
            side = self.rng.choice([-1.0, 1.0])
            end = np.array([goal_x + sgn * 2.0, side * self.rng.uniform(4.5, 12.0)])
            after = {"out_at": end.copy()}
        after = {**(after or {}), "shooter": self.carrier}
        speed = att.params[self.carrier].shot_speed
        duration = max(float(np.linalg.norm(end - start)) / speed, 0.2)
        self._event("shot", self.poss, self.carrier, x=start[0], y=start[1], end_x=end[0], end_y=end[1],
                    outcome=outcome, xg=xg, p_goal=p_goal, header=bool(self.kick_header),
                    restart=restart, distance=float(np.linalg.norm(np.array([HALF_L, 0.0]) - own_start)))
        self.flight = Flight("shot", start, end, self.frame, duration, False, apex, outcome, receiver, after)
        self.state = FLIGHT
        self.carrier = None
        self.last_touch = self.poss

    def _resolve_flight(self) -> None:
        f = self.flight
        if f.kind != "pass":
            self.ball[:2] = f.end
        self.ball[2] = 0.0
        if f.kind == "pass":
            if f.outcome == "complete":
                self._gain(f.receiver[0], f.receiver[1], how="pass", lofted=f.lofted)
            elif f.outcome in ("intercepted", "lost_duel"):
                if self._deflected_out(f.end, defender_team=f.receiver[0]):
                    return
                clean = CLEAN_INTERCEPTION if f.outcome == "intercepted" else CLEAN_DUEL_WIN
                clean += (1.0 - clean) * 0.5 * self._danger(self.ball[:2], f.receiver[0])
                if self.rng.random() < clean:
                    kind = "interception" if f.outcome == "intercepted" else "duel_won"
                    self._event(kind, f.receiver[0], f.receiver[1], x=f.end[0], y=f.end[1])
                    self._gain(f.receiver[0], f.receiver[1], how=kind, lofted=f.lofted)
                else:
                    # Got a touch but not control: the ball runs loose (deflection, knock-down, 50/50).
                    self._event("deflection", f.receiver[0], f.receiver[1], x=f.end[0], y=f.end[1], loose=True)
                    self.last_touch = f.receiver[0]
                    self._scramble(self.ball[:2].copy(), f.receiver[0], (2.0, 5.0))
            elif f.outcome == "out":
                self._ball_out(f.after["out_at"], self.poss)
            else:
                _, _, speed = self._flight_state(f, (self.frame - f.t0) * DT)
                self._loose(self.ball[:2].copy(), f.u * max(speed, 1.0))
            return
        shooter_team = self.poss
        if f.outcome == "goal":
            self.score[shooter_team] += 1
            self._event("goal", shooter_team, f.after["shooter"], x=f.end[0], y=f.end[1], score=list(self.score))
            self._set_dead("kickoff", 1 - shooter_team, np.zeros(2))
            self.ball_rest = (f.end.copy(), self.frame + self._frames(3.0))  # the ball sits in the net a moment
        elif f.outcome == "saved":
            if self.rng.random() < 0.6:
                self._event("save", 1 - shooter_team, 0, held=True)
                self._set_dead("gk", 1 - shooter_team, self.teams[1 - shooter_team].pos[0].copy())
            else:
                self._event("save", 1 - shooter_team, 0, held=False)
                if self.rng.random() < 0.5:
                    self.last_touch = 1 - shooter_team
                    self._ball_out(np.array([f.end[0] * 1.02, np.sign(f.end[1] or 1.0) * 8.0]), 1 - shooter_team)
                else:
                    self.last_touch = 1 - shooter_team
                    sgn = -np.sign(self._goal_x(shooter_team))
                    wide = np.sign(f.end[1] or self.rng.choice([-1.0, 1.0]))
                    self._loose(f.end.copy(), unit(np.array([sgn * 0.5, wide])) * self.rng.uniform(8.0, 10.0))
        elif f.outcome == "off_target":
            deflected = self.rng.random() < 0.15
            self._ball_out(f.after["out_at"], 1 - shooter_team if deflected else shooter_team)
        else:  # blocked
            self.last_touch = 1 - shooter_team
            self._loose(f.end.copy(), f.after["loose_vel"])


CARRY_TIME_S = decisions.CARRY_M / 5.0


def simulate(home_id: str, away_id: str, seed: int = 0, **kw) -> MatchResult:
    return Match(home_id, away_id, seed=seed, **kw).run()


__all__ = ["BALL_STATES", "DT", "FPS", "Match", "MatchResult", "simulate"]
