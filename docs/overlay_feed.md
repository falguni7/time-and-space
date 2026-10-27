# Overlay feed: timed, machine-readable match intelligence

Time and Space publishes everything it knows about a match as **timed JSON messages** that a broadcast
graphics system, a second-screen app or another agent can render in sync with the match feed. The
web app is one consumer of this feed; it uses nothing that another consumer could not.

| Feed | Where | What |
|---|---|---|
| **Live feed** | `demo/live/<match>.jsonl`, `GET /data/live/<match>.jsonl`, or streamed: `GET /api/live/<match>/stream?speed=1&start_frame=0` (server-sent events) | every event, the running state and the key moments, in match order |
| **Moment overlays** | `demo/clips/<clip>.json`, `GET /data/clips/<clip>.json` | a few seconds of tracking around each key moment, plus what to draw at its key frame |
| **Explanations** | `demo/app.json`, `GET /data/app.json` | every viewer's verified text for every moment, why they got it, and the facts behind it |
| **Full-match tracking** | `demo/live/<match>_tracking.json` | all 22 players and the ball at 5 Hz, with who plays in each slot when |

## Timing

All messages carry **`frame`**, the 10 Hz match timeline that also indexes the tracking data
(`frame / 10` = seconds since kick-off, both halves on one timeline), and **`clock`**, the broadcast
clock label (`"45+2'"`, `"81'"`). To sync with video, map one frame to its video timestamp; every
other message follows. The stream endpoint releases messages at `speed` × real time from `start_frame`,
so `speed=1` behaves like a live provider feed.

## Live feed messages

One JSON object per line, in this order within a frame: `moment`, then `event`, then `state`.

### `event`: a validated provider event, with quality metrics for passes and shots

```json
{"kind": "event", "frame": 10, "clock": "1'",
 "event": {"event_id": 2, "frame": 10, "end_frame": 26, "period": 1, "type": "pass", "team": "RIV",
           "player_id": "RIV-10", "x": 0.0, "y": 0.0, "end_x": -24.03, "end_y": 9.07,
           "outcome": "complete", "receiver_id": "RIV-13", "restart": "kickoff"},
 "metrics": {"distance_m": 25.7, "ball_speed_kmh": 58, "xpass": 0.86, "difficulty": "easy",
             "pressure": 0.07, "xt_added": -0.0058}}
```

| `metrics` field | Meaning |
|---|---|
| `distance_m` | pass or shot distance |
| `ball_speed_kmh` | average ball speed over the flight (passes and shots) |
| `xpass` | chance the chosen pass arrives, from the engine's xPass model |
| `difficulty` | xPass in words: easy ≥ 0.85 > medium ≥ 0.6 > hard ≥ 0.35 > very hard |
| `pressure` | pressure on the passer, 0–1 (1 = a defender within 1 m, 0 = nobody within 5 m) |
| `xt_added` | change in expected threat; negative when the ball was lost |

A shot carries `distance_m` and `ball_speed_kmh` (e.g. `95` km/h), and `event.provider_xg`.

Coordinates are metres from the centre spot (x along the 105 m length, y across the 68 m width), the
same frame as the tracking data. Event types: `pass`, `carry`, `shot`, `goal`, `save`, `pressure`,
`possession`, `tackle`, `interception`, `recovery`, `clearance`, `foul`, `free_kick`, `corner`,
`throw_in`, `goal_kick`, `substitution`, `kickoff`, `period_start`, `period_end` and a few more
(`stream/schema.py`). Every event is validated before use; a malformed one becomes:

```json
{"kind": "rejected", "frame": 1234, "reason": "Value error, unknown event type 'teleport'"}
```

### `state`: running statistics (every 30 s of match time and after each goal)

```json
{"kind": "state", "frame": 35504, "clock": "58'", "score": "2-0",
 "teams": {"RIV": {"goals": 2, "shots": 18, "xg": 1.93, "passes": 353, "completion_pct": 69,
                   "possession_pct": 58, "xt_created": 2.94, "pressures": 162,
                   "pressure_faced": 0.55, "presses_won": 0, "momentum_5min": -0.114},
           "STO": {"...": "same fields"}}}
```

`momentum_5min` is the net threat each team created over the last five minutes; `pressure_faced` is
the mean pressure on that team's passers. At full time these numbers equal the metrics cube's
(tested in `tests/test_stream.py`).

### `moment`: a key moment, announced when it happens, with the viewers it is for

```json
{"kind": "moment", "frame": 49361, "clock": "81'", "moment_id": "R01_RIV-STO-M044",
 "type": "press_trap", "team": "RIV",
 "for": ["casual", "analyst", "home_fan", "away_fan_de", "star_follower_es"]}
```

Types: `goal`, `big_chance`, `press_trap` (the press won the ball), `press_broken` (a pass beat the press),
`missed_option` (a clearly better pass was on), `momentum_swing`. `for` lists the viewer profiles whose
feed holds the moment (the relevance selector's choice); each one's text is in `app.json` under
`matches.<match>.stories` (key moments) or `matches.<match>.feeds.<viewer>`.

## Moment overlay (`clips/<clip>.json`)

| Field | Meaning |
|---|---|
| `fps`, `start_frame` | 10 Hz; the clip's first frame on the match timeline |
| `objects[22]` | per slot: team (0 home, 1 away), `player_id`, name, number, position |
| `xy[t]` | 46 integers per frame: x, y of the 22 players and the ball, in **decimetres** |
| `ball_z[t]`, `carrier[t]` | ball height (m); slot of the ball carrier (−1 = none) |
| `moments[]` | `moment_id`, `type`, `clock`, `index` (key frame within the clip) and `overlay` |

`overlay` for a pass moment (what the web app draws at the key frame):

```json
{"kind": "pass", "passer": 13, "presser": 8, "pressure": 0.99, "presser_m": 1.05,
 "decision_gap": 0.0098, "xt_added": -0.0051, "outcome": "lost_duel",
 "options": [{"slot": 16, "name": "Malik Mbatha", "xpass": 0.977, "value": 0.0033, "marking_m": 1.1,
              "chosen": false, "best": true}, "..."],
 "third": "defensive", "band": "high",
 "league": {"none": 0.0001, "some": -0.0002, "high": -0.0009},
 "distance_m": 29.5, "ball_speed_kmh": 66}
```

`league` is the league's average threat added per pass in the same third by pressure band: the
curve this pass is compared with. A shot overlay has `shooter`, `xg`, `outcome`, `distance_m`,
`ball_speed_kmh`.

## Producing the feeds

```bash
python -m presslens.stream feed       # live feed      -> demo/live/<match>.jsonl
python -m presslens.web.clips         # clips, tracking, app.json -> demo/
python -m presslens.stream replay R01_RIV-STO --speed 20   # watch the live interpreter in a terminal
```
