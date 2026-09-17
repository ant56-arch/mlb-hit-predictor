"""
train_model.py — trains the hit-probability model used by predict.py.

Fits a logistic regression on real historical game rows (research_gamelogs.json,
one row per player per game actually played, hit or not — no survivorship bias)
to answer one question directly: given a player's recent form, season numbers,
the opposing pitcher, park, platoon split, and home/away, what is the
probability he gets at least one hit today?

This replaces the old approach of hand-picking factor weights (22%, 20%, 18%, ...)
and calling the result a "confidence score." That score was never a probability —
it was just each player's total divided by the top pick's total. This script fits
real coefficients against actual outcomes and calibrates them with a sigmoid, so
the number predict.py shows is an honest P(hit).

Run this manually whenever there's meaningfully more data to train on (e.g. once
picks_history.json has a full season of resolved raw_features rows). It writes
model_weights.json, which predict.py loads at runtime — predict.py itself has no
numpy/sklearn dependency, it just applies the saved coefficients.

Requires: pip install -r research/requirements.txt (numpy)
"""

import json
import math
from datetime import datetime

import numpy as np

RESEARCH_FILE = "research_gamelogs.json"
PICKS_FILE = "picks_history.json"
OUTPUT_FILE = "model_weights.json"

FEATURES = [
    "recent_form_avg",
    "season_avg",
    "season_ops",
    "is_home",
    "platoon",
    "park_factor",
    "opp_pitcher_era",
    "opp_pitcher_whip",
    "opp_pitcher_k9",
]

LEARNING_RATE = 0.1
ITERATIONS = 3000
L2_LAMBDA = 0.01


def load_research_rows():
    with open(RESEARCH_FILE) as f:
        data = json.load(f)
    rows = []
    for r in data.get("rows", []):
        if r.get("season_avg") is None or r.get("opp_pitcher_era") is None:
            continue
        recent = r.get("recent_form_avg")
        if recent is None:
            recent = r["season_avg"]
        rows.append({
            "date": r["date"],
            "got_hit": 1.0 if r["got_hit"] else 0.0,
            "recent_form_avg": recent,
            "season_avg": r["season_avg"],
            "season_ops": r.get("season_ops") or (r["season_avg"] * 2.6),
            "is_home": 1.0 if r.get("is_home") else 0.0,
            "platoon": 1.0 if (r.get("batter_hand") and r.get("opp_pitcher_hand")
                                and r["batter_hand"] != r["opp_pitcher_hand"]) else 0.0,
            "park_factor": r.get("park_factor") or 1.0,
            "opp_pitcher_era": r["opp_pitcher_era"],
            "opp_pitcher_whip": r["opp_pitcher_whip"],
            "opp_pitcher_k9": r["opp_pitcher_k9"],
        })
    return rows


def load_picks_rows():
    """Pull in resolved picks that were saved with raw_features (added going
    forward in predict.py) so retrains after this season don't need another
    expensive research backfill."""
    try:
        with open(PICKS_FILE) as f:
            picks = json.load(f)
    except FileNotFoundError:
        return []
    rows = []
    for p in picks.get("picks", []):
        if p.get("got_hit") is None or not p.get("raw_features"):
            continue
        row = {"date": p["date"], "got_hit": 1.0 if p["got_hit"] else 0.0}
        row.update(p["raw_features"])
        if any(row.get(f) is None for f in FEATURES):
            continue
        rows.append(row)
    return rows


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_logistic_regression(X, y):
    n, d = X.shape
    weights = np.zeros(d)
    bias = 0.0
    for _ in range(ITERATIONS):
        z = X @ weights + bias
        pred = sigmoid(z)
        error = pred - y
        grad_w = (X.T @ error) / n + (L2_LAMBDA * weights)
        grad_b = np.mean(error)
        weights -= LEARNING_RATE * grad_w
        bias -= LEARNING_RATE * grad_b
    return weights, bias


def log_loss(y, pred):
    eps = 1e-9
    pred = np.clip(pred, eps, 1 - eps)
    return float(-np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred)))


def auc(y, pred):
    pairs = sorted(zip(pred, y))
    ranks = {}
    i = 0
    n = len(pairs)
    sorted_scores = [p for p, _ in pairs]
    while i < n:
        j = i
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_pos = sum(ranks[k] for k in range(n) if pairs[k][1] == 1)
    n_pos = sum(1 for _, label in pairs if label == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    rows = load_research_rows() + load_picks_rows()
    if len(rows) < 200:
        print(f"Only {len(rows)} usable rows — need more data before training.")
        return

    rows.sort(key=lambda r: r["date"])
    print(f"Loaded {len(rows)} training rows "
          f"({rows[0]['date']} to {rows[-1]['date']}).")

    split = int(len(rows) * 0.8)
    train_rows, test_rows = rows[:split], rows[split:]

    X_train = np.array([[r[f] for f in FEATURES] for r in train_rows])
    y_train = np.array([r["got_hit"] for r in train_rows])
    X_test = np.array([[r[f] for f in FEATURES] for r in test_rows])
    y_test = np.array([r["got_hit"] for r in test_rows])

    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    stds[stds == 0] = 1.0

    X_train_std = (X_train - means) / stds
    X_test_std = (X_test - means) / stds

    weights, bias = train_logistic_regression(X_train_std, y_train)

    train_pred = sigmoid(X_train_std @ weights + bias)
    test_pred = sigmoid(X_test_std @ weights + bias)

    baseline_rate = float(y_train.mean())
    print(f"\nBaseline hit rate (train): {baseline_rate*100:.1f}%")
    print(f"Train log loss: {log_loss(y_train, train_pred):.4f}")
    print(f"Test  log loss: {log_loss(y_test, test_pred):.4f}  "
          f"(baseline: {log_loss(y_test, np.full_like(y_test, baseline_rate)):.4f})")
    test_auc = auc(y_test, test_pred)
    print(f"Test  AUC: {test_auc:.3f}" if test_auc else "Test AUC: n/a")

    print("\nLearned coefficients (standardized units, log-odds per +1 std dev):")
    for f, w in sorted(zip(FEATURES, weights), key=lambda x: -abs(x[1])):
        print(f"  {f:<18} {w:+.3f}")

    model = {
        "features": FEATURES,
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_train_rows": len(train_rows),
        "n_test_rows": len(test_rows),
        "train_date_range": [rows[0]["date"], rows[-1]["date"]],
        "test_auc": test_auc,
        "test_log_loss": log_loss(y_test, test_pred),
        "baseline_hit_rate": baseline_rate,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nSaved model to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
