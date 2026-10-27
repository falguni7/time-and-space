# data/

Generated files only; nothing here is committed.

## matches/ (one simulated season, 56 matches)

```bash
bin/pl python -m presslens.sim generate            # writes data/matches/ (~45 s, ~1.7 GB)
```

```
matches/
  index.parquet                 one row per match: match_id, round, home, away, goals, seed
  R01_RIV-STO/
    meta.json                   clubs, formations, line-ups, substitutions, directions, score
    tracking.parquet            10 Hz positions of 22 players + ball (x, y, z, velocity, possession, carrier)
    events.parquet              provider-style events, incl. possession spells, carries,
                                pressure episodes and `provider_xg` on shots
    phases.parquet              phases of play for both teams
    truth/                      HIDDEN ground truth: validation only, the engine must not read it
      decisions.parquet         every carrier decision with all options, true values, the choice
      events.parquet            true xG / scoring chance per shot, true completion chance per pass
      settings.json             club tactics and players' simulation parameters
```

Load with `presslens.sim.io.load_match(match_id)`, which never touches `truth/`.

## checks/

Plots, animations and scratch calibration scripts from the simulation checks.
