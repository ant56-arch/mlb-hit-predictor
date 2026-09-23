"""
train_model.py - fits the hit-probability model used by predict.py.

Logistic regression on research_gamelogs.json (one row per hitter per game he
batted in). Every feature is computed as of the morning of that game - the
batter's season and 14-day numbers from his earlier rows, the pitcher's from
the pull - so the model never learns from stats it couldn't have known.

It is scored on the last 20% of the season (fit only on the first 80%), then
refit on everything for the saved model. The headline check is the one the
site lives by: on each held-out day, how often did the model's top 10 get a hit?

Run from the repo root:
  pip install -r research/requirements.txt
  python research/train_model.py
"""

import json
import os
import sys
from collections import defaultdict, deque
from datetime import date, datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import features as F  # noqa: E402

RESEARCH_FILE = "research_gamelogs.json"
OUTPUT_FILE = "model_weights.json"

LEARNING_RATE = 0.1
ITERATIONS = 4000
L2_LAMBDA = 0.001
TOP_N = 10


def build_rows(raw_rows):
    by_player = defaultdict(list)
    for r in raw_rows:
        by_player[r["player_id"]].append(r)

    rows = []
    for games in by_player.values():
        games.sort(key=lambda r: (r["date"], r.get("game_pk", 0)))
        hits = ab = n_games = 0
        window = deque()
        recent_hits = recent_ab = 0
        for r in games:
            day = date.fromisoformat(r["date"])
            while window and window[0][0] < day - timedelta(days=F.RECENT_WINDOW_DAYS):
                _, h, a = window.popleft()
                recent_hits -= h
                recent_ab -= a

            feats = F.batter_features(hits, ab, n_games, recent_hits, recent_ab)
            feats.update(F.matchup_features(r.get("is_home"), r.get("batter_hand"),
                                            r.get("opp_pitcher_hand"), r.get("venue")))
            if "opp_pitcher_ip" in r:
                feats.update(F.pitcher_features(r.get("opp_pitcher_era"), r.get("opp_pitcher_whip"),
                                                r.get("opp_pitcher_k9"), r.get("opp_pitcher_ip")))
            else:
                # Older pulls only have one season-level line per pitcher.
                feats.update({"opp_pitcher_era": r["opp_pitcher_era"], "opp_pitcher_whip": r["opp_pitcher_whip"],
                              "opp_pitcher_k9": r["opp_pitcher_k9"]})

            if all(feats.get(f) is not None for f in F.CANDIDATE_FEATURES):
                rows.append({"date": r["date"], "got_hit": 1.0 if r["got_hit"] else 0.0, **feats})

            hits += r["hits"]
            ab += r["at_bats"]
            n_games += 1
            window.append((day, r["hits"], r["at_bats"]))
            recent_hits += r["hits"]
            recent_ab += r["at_bats"]
    rows.sort(key=lambda r: r["date"])
    return rows


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit(X, y):
    w = np.zeros(X.shape[1])
    b = float(np.log(y.mean() / (1 - y.mean())))
    for _ in range(ITERATIONS):
        err = sigmoid(X @ w + b) - y
        w -= LEARNING_RATE * ((X.T @ err) / len(y) + L2_LAMBDA * w)
        b -= LEARNING_RATE * err.mean()
    return w, b


def log_loss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def top_n_backtest(dates, y, p):
    """Hit rate of the model's top N per day vs. every hitter that day."""
    by_day = defaultdict(list)
    for d, yi, pi in zip(dates, y, p):
        by_day[d].append((pi, yi))
    picked, predicted = [], []
    for day_rows in by_day.values():
        if len(day_rows) < TOP_N * 3:
            continue
        top = sorted(day_rows, reverse=True)[:TOP_N]
        picked += [yi for _, yi in top]
        predicted += [pi for pi, _ in top]
    return {
        "days": len(picked) // TOP_N,
        "top_n": TOP_N,
        "top_n_hit_rate": float(np.mean(picked)) if picked else None,
        "top_n_predicted": float(np.mean(predicted)) if predicted else None,
        "all_hitters_hit_rate": float(np.mean(y)),
    }


def calibration(y, p, edges=(0, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0)):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi)
        if mask.sum() >= 30:
            out.append({"range": [lo, hi], "n": int(mask.sum()),
                        "predicted": float(p[mask].mean()), "actual": float(y[mask].mean())})
    return out


def main():
    with open(RESEARCH_FILE) as f:
        raw = json.load(f)
    rows = build_rows(raw["rows"])
    if len(rows) < 1000:
        print(f"Only {len(rows)} usable rows - not enough to train on.")
        return
    print(f"{len(rows)} rows, {rows[0]['date']} to {rows[-1]['date']}")

    X_all = np.array([[r[f] for f in F.CANDIDATE_FEATURES] for r in rows])
    keep = X_all.std(axis=0) > 1e-9
    feats = [f for f, k in zip(F.CANDIDATE_FEATURES, keep) if k]
    dropped = [f for f, k in zip(F.CANDIDATE_FEATURES, keep) if not k]
    if dropped:
        print(f"Dropped (never varies in this data): {', '.join(dropped)}")
    X_all = X_all[:, keep]
    y_all = np.array([r["got_hit"] for r in rows])
    dates = np.array([r["date"] for r in rows])

    split_date = dates[int(len(rows) * 0.8)]
    train, test = dates < split_date, dates >= split_date
    means, stds = X_all[train].mean(axis=0), X_all[train].std(axis=0)
    w, b = fit((X_all[train] - means) / stds, y_all[train])
    p_test = sigmoid(((X_all[test] - means) / stds) @ w + b)
    y_test = y_all[test]
    base = y_all[train].mean()

    backtest = top_n_backtest(dates[test], y_test, p_test)
    metrics = {
        "test_from": str(split_date),
        "test_rows": int(test.sum()),
        "test_log_loss": log_loss(y_test, p_test),
        "baseline_log_loss": log_loss(y_test, np.full_like(y_test, base)),
        "test_auc": auc(y_test, p_test),
        "calibration": calibration(y_test, p_test),
        "backtest": backtest,
    }

    print(f"\nHeld-out games from {split_date} ({metrics['test_rows']} rows):")
    print(f"  log loss {metrics['test_log_loss']:.4f} (always guessing the average: {metrics['baseline_log_loss']:.4f})")
    print(f"  AUC {metrics['test_auc']:.3f}")
    if backtest["top_n_hit_rate"] is not None:
        print(f"  top {TOP_N}/day over {backtest['days']} days: {backtest['top_n_hit_rate']:.1%} got a hit "
              f"(model said {backtest['top_n_predicted']:.1%}; every hitter: {backtest['all_hitters_hit_rate']:.1%})")
    for c in metrics["calibration"]:
        print(f"  predicted {c['range'][0]:.0%}-{c['range'][1]:.0%}: {c['n']:>5} rows, "
              f"said {c['predicted']:.1%}, actual {c['actual']:.1%}")

    means, stds = X_all.mean(axis=0), X_all.std(axis=0)
    w, b = fit((X_all - means) / stds, y_all)
    print("\nFinal coefficients (log-odds per +1 std dev, fit on all rows):")
    for f, wi in sorted(zip(feats, w), key=lambda x: -abs(x[1])):
        print(f"  {f:<18} {wi:+.3f}")

    model = {
        "features": feats,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": w.tolist(),
        "bias": float(b),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_rows": len(rows),
        "training_dates": [rows[0]["date"], rows[-1]["date"]],
        "metrics": metrics,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nSaved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
