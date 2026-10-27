"""Validate the engine against the hidden ground truth (the only engine module that reads truth/).

    bin/pl python -m presslens.engine.validate      # writes data/engine/validation_report.md

Models are fitted on rounds 1-10; every check that involves a fitted model uses rounds 11-14 only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .. import paths
from ..sim.config import load_clubs, load_players
from ..sim.io import load_truth
from ..sim.truth import true_value
from .data import MODELS, TEST_ROUNDS, match_ids
from .xt import XTModel, zone_centres

REPORT = MODELS / "validation_report.md"


def check_xt() -> dict:
    model = XTModel.load()
    centres = zone_centres().reshape(-1, 2)
    truth = true_value(centres)
    return {"pearson": pearsonr(model.grid.ravel(), truth)[0], "spearman": spearmanr(model.grid.ravel(), truth)[0]}


def check_xpass(test_ids: list[str]) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    from .xpass import FEATURES, XPassModel, pass_tables
    table = pass_tables(test_ids)
    pred = XPassModel.load().predict(table[FEATURES].to_numpy())
    truth = []
    for mid in test_ids:
        t = load_truth(mid)["events"]
        t = t[t.type == "pass"][["event_id", "true_p_complete"]].assign(match_id=mid)
        truth.append(t)
    joined = table.assign(pred=pred).merge(pd.concat(truth), on=["match_id", "event_id"])
    y = joined.complete.to_numpy()
    return {"passes": len(joined), "auc": roc_auc_score(y, joined.pred), "brier": brier_score_loss(y, joined.pred),
            "brier_baseline": brier_score_loss(y, np.full(len(y), y.mean())),
            "corr_true": float(np.corrcoef(joined.pred, joined.true_p_complete)[0, 1]),
            "mae_true": float(np.mean(np.abs(joined.pred - joined.true_p_complete)))}


def _truth_decisions(test_ids: list[str]) -> pd.DataFrame:
    """True gap of every pass decision: best true value minus the chosen option's true value."""
    rows = []
    for mid in test_ids:
        d = load_truth(mid)["decisions"]
        d = d[d.restart.isna()]
        best = d.groupby("decision_id").true_value.max()
        chosen = d[d.chosen].set_index("decision_id")
        chosen = chosen[chosen.kind == "pass"]
        rows.append(pd.DataFrame({
            "match_id": mid, "frame": chosen.frame, "player_id": chosen.player_id,
            "true_gap": best.loc[chosen.index] - chosen.true_value, "true_pressure": chosen.pressure,
            "true_temperature": chosen.temperature}))
    return pd.concat(rows, ignore_index=True)


def check_decisions(decisions: pd.DataFrame, test_ids: list[str]) -> dict:
    eng = decisions[decisions.match_id.isin(test_ids) & decisions.restart.isna()]
    truth = _truth_decisions(test_ids)
    joined = eng.merge(truth, on=["match_id", "frame", "player_id"])
    players = load_players()
    joined["composure"] = joined.player_id.map(lambda p: players[p].attributes["composure"])
    rho = spearmanr(joined.decision_gap, joined.true_gap)[0]
    # Pressure effect, as the engine sees it (public data only), split by public composure rating.
    joined["pressure_band"] = pd.cut(joined.pressure, [-0.01, 0.0, 0.5, 1.0], labels=["none", "some", "high"])
    joined["composure_band"] = np.where(joined.composure >= joined.composure.median(), "high", "low")
    effect = (joined.groupby(["composure_band", "pressure_band"], observed=True)
              .agg(engine_gap=("decision_gap", "mean"), true_gap=("true_gap", "mean"),
                   chose_best=("chose_best", "mean"), n=("decision_gap", "size")).reset_index())
    return {"linked": len(joined), "of": len(eng), "spearman_gap": rho, "effect": effect}


HIGH, MIDDLE, DEEP = "high (opponent's own third)", "middle third", "deep (own third)"
THIRD_FOR_DEFENDERS = {"defensive": HIGH, "middle": MIDDLE, "attacking": DEEP}


def check_team_pressing(decisions: pd.DataFrame) -> pd.DataFrame:
    """Pressure each club applies when defending, by where the ball is (all matches; no fitted model)."""
    clubs = load_clubs()
    d = decisions[decisions.restart.isna()].copy()
    d["zone_for_defenders"] = d.third.map(THIRD_FOR_DEFENDERS)
    table = d.pivot_table(index="opponent", columns="zone_for_defenders", values="pressure", aggfunc="mean")
    table["true_press_intensity"] = [clubs[c].tactics.press_intensity for c in table.index]
    table["true_press_height"] = [clubs[c].tactics.press_height for c in table.index]
    table.index.name = "club (defending)"
    return table.sort_values(MIDDLE, ascending=False)


def build_report() -> str:
    decisions = pd.read_parquet(MODELS / "decisions.parquet")
    test_ids = match_ids(TEST_ROUNDS)
    xt, xp = check_xt(), check_xpass(test_ids)
    dec = check_decisions(decisions, test_ids)
    press = check_team_pressing(decisions)
    rho_mid = spearmanr(press[MIDDLE], press.true_press_intensity)[0]
    high = press[press.true_press_height == "high"][HIGH]
    rest = press[press.true_press_height != "high"][HIGH]
    lines = [
        "# PressLens engine validation against hidden ground truth", "",
        f"_Models fitted on rounds 1–10; checks on rounds 11–14 ({len(test_ids)} matches) unless noted._", "",
        "| Engine estimate | Hidden truth | Result |", "|---|---|---|",
        f"| xT grid (96 zones) | true danger map | Pearson **{xt['pearson']:.3f}**, Spearman {xt['spearman']:.3f} |",
        (f"| xPass ({xp['passes']} test passes) | true completion chance | correlation **{xp['corr_true']:.3f}**, "
         f"mean abs diff {xp['mae_true']:.3f}; AUC {xp['auc']:.3f}, Brier {xp['brier']:.3f} "
         f"(baseline {xp['brier_baseline']:.3f}) |"),
        f"| Decision gap ({dec['linked']} open-play passes) | true decision gap | Spearman **{dec['spearman_gap']:.3f}** |",
        (f"| Pressure applied in the middle third, 8 clubs (all matches) | true press intensity | "
         f"Spearman **{rho_mid:.3f}** |"),
        (f"| Pressure applied in the opponent's own third | true press height | high-press clubs "
         f"{high.min():.2f}–{high.max():.2f} vs others {rest.min():.2f}–{rest.max():.2f} |"), "",
        "## Pressure effect on decisions (engine view vs truth)", "",
        "| composure (public card) | pressure | engine gap | true gap | engine: chose best | passes |",
        "|---|---|---|---|---|---|",
    ]
    for r in dec["effect"].itertuples():
        lines.append(f"| {r.composure_band} | {r.pressure_band} | {r.engine_gap:.4f} | {r.true_gap:.4f} | "
                     f"{100 * r.chose_best:.0f}% | {r.n} |")
    lines += ["", "## Pressure each club applies (engine pressure index on the passer) vs its hidden setting", "",
              ("| club (defending) | high (opponent's own third) | middle third | deep (own third) "
               "| true intensity | true height |"),
              "|---|---|---|---|---|---|"]
    for club, r in press.iterrows():
        lines.append(f"| {club} | {r[HIGH]:.3f} | {r[MIDDLE]:.3f} | {r[DEEP]:.3f} | "
                     f"{r.true_press_intensity:.2f} | {r.true_press_height} |")
    return "\n".join(lines)


def main() -> None:
    text = build_report()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text + "\n")
    print(text)
    print(f"\nWrote {REPORT}")


__all__ = ["build_report", "main", "paths"]

if __name__ == "__main__":
    main()
