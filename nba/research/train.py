"""
train.py - fits NBA Edge's margin model and backtests it on the latest season.

Walks every stored game in order (model.League), so each game's features are
what was known that morning. Fits the home margin by least squares on every
season before the latest one, scores the latest season (which the fit never
saw), then refits on everything and writes nba/model_weights.json.
"""

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
import store  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_weights.json")
WARMUP_UNTIL = "12-01"  # skip the first stored season's first weeks, before ratings have settled
BANDS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]


def build_rows():
    games, box = store.load_all()
    league = M.League()
    rows = []
    for g in games:
        feats = league.features_from_box(g, box.get(g["id"]))
        rows.append({"game": g, "x": [feats[k] for k in M.FEATURES],
                     "margin": g["home_pts"] - g["away_pts"], "has_box": g["id"] in box})
        league.update(g, box.get(g["id"]))
    return rows


def fit(rows):
    X = np.array([r["x"] for r in rows])
    y = np.array([r["margin"] for r in rows], dtype=float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    sigma = float(np.std(y - X @ coef))
    return {"features": M.FEATURES, "coef": {k: round(float(c), 4) for k, c in zip(M.FEATURES, coef)},
            "sigma": round(sigma, 3)}


def evaluate(weights, rows):
    preds = []
    for r in rows:
        feats = dict(zip(M.FEATURES, r["x"]))
        p_home, m = M.win_prob(weights, feats)
        home_won = r["margin"] > 0
        pick_home = p_home >= 0.5
        conf = p_home if pick_home else 1 - p_home
        preds.append({"date": r["game"]["date"], "p_home": p_home, "conf": conf, "correct": pick_home == home_won,
                      "home_won": home_won, "margin_err": abs(m - r["margin"]), "type": r["game"]["type"]})
    n = len(preds)
    eps = 1e-9
    log_loss = -sum(math.log(max(eps, p["p_home"] if p["home_won"] else 1 - p["p_home"])) for p in preds) / n
    brier = sum((p["p_home"] - p["home_won"]) ** 2 for p in preds) / n

    bands = []
    for lo, hi in BANDS:
        b = [p for p in preds if lo <= p["conf"] < hi]
        if b:
            bands.append({"range": [lo, min(hi, 1.0)], "n": len(b),
                          "predicted": round(sum(p["conf"] for p in b) / len(b), 4),
                          "actual": round(sum(p["correct"] for p in b) / len(b), 4)})

    by_day = defaultdict(list)
    for p in preds:
        by_day[p["date"]].append(p)
    top = [p for day in by_day.values() for p in sorted(day, key=lambda p: -p["conf"])[:3]]

    by_month = defaultdict(list)
    for p in preds:
        by_month[p["date"][:7]].append(p)
    months = [{"month": mo, "n": len(ps), "correct": sum(p["correct"] for p in ps),
               "predicted": round(sum(p["conf"] for p in ps) / len(ps), 4)} for mo, ps in sorted(by_month.items())]

    return {
        "games": n,
        "correct": sum(p["correct"] for p in preds),
        "accuracy": round(sum(p["correct"] for p in preds) / n, 4),
        "predicted_accuracy": round(sum(p["conf"] for p in preds) / n, 4),
        "log_loss": round(log_loss, 4),
        "brier": round(brier, 4),
        "margin_mae": round(sum(p["margin_err"] for p in preds) / n, 2),
        "home_win_rate": round(sum(p["home_won"] for p in preds) / n, 4),
        "top3": {"days": len(by_day), "picks": len(top), "correct": sum(p["correct"] for p in top),
                 "accuracy": round(sum(p["correct"] for p in top) / len(top), 4)},
        "bands": bands,
        "months": months,
    }


def main():
    rows = build_rows()
    seasons = sorted({r["game"]["season"] for r in rows})
    first, test_season = seasons[0], seasons[-1]
    usable = [r for r in rows if r["game"]["season"] != first or r["game"]["date"][5:] >= WARMUP_UNTIL
              or r["game"]["date"][5:7] < "07"]
    train = [r for r in usable if r["game"]["season"] < test_season]
    test = [r for r in rows if r["game"]["season"] == test_season]
    print(f"{len(rows)} games in seasons {seasons}; training on {len(train)}, testing on {len(test)} "
          f"({sum(r['has_box'] for r in rows)} with box scores)")

    w = fit(train)
    print("coefficients (fit before the test season):", w["coef"], "sigma", w["sigma"])
    test_regular = [r for r in test if r["game"]["type"] == "regular"]
    backtest = evaluate(w, test_regular)
    playoffs = [r for r in test if r["game"]["type"] != "regular"]
    backtest_playoffs = evaluate(w, playoffs) if playoffs else None

    # Baselines on the same games: Elo alone (with home court), and always the home team.
    elo_only = fit([{**r, "x": [r["x"][0], r["x"][1]] + [0.0] * (len(M.FEATURES) - 2)} for r in train])
    elo_bt = evaluate(elo_only, [{**r, "x": [r["x"][0], r["x"][1]] + [0.0] * (len(M.FEATURES) - 2)}
                                 for r in test_regular])

    print(json.dumps({k: v for k, v in backtest.items() if k != "months"}, indent=1))
    print("elo only:", elo_bt["accuracy"], elo_bt["log_loss"], "| home team:", backtest["home_win_rate"])
    if backtest_playoffs:
        print("postseason:", backtest_playoffs["accuracy"], backtest_playoffs["games"])

    final = fit(usable)
    final.update({
        "trained_on": f"{first - 1}-{str(first)[2:]} to {test_season - 1}-{str(test_season)[2:]}",
        "backtest_season": f"{test_season - 1}-{str(test_season)[2:]}",
        "backtest": backtest,
        "backtest_postseason": backtest_playoffs,
        "baselines": {"elo_only_accuracy": elo_bt["accuracy"], "elo_only_log_loss": elo_bt["log_loss"],
                      "home_team_accuracy": backtest["home_win_rate"]},
    })
    print("final coefficients:", final["coef"], "sigma", final["sigma"])
    with open(OUT, "w") as f:
        json.dump(final, f, indent=1)


if __name__ == "__main__":
    main()
