"""
train_model.py - retrains the hit-probability model used by predict.py, and
only changes how it's built when the change is proven on recent games.

Logistic regression on research/data/<season>.json.gz (one row per hitter per
game he batted in, from pull_season_data.py). Every feature is computed as of
the morning of that game - the batter's season and 14-day numbers from his
earlier rows (with last season's average as the starting point, exactly like
predict.py), the pitcher's from the pull - so the model never learns from
stats it couldn't have known.

Each run:
  1. Holds out the most recent HOLDOUT_ROWS hitter-games (about five weeks).
  2. Fits every candidate recipe in RECIPES - how many years of games to learn from,
     how strongly to regularize, fixed or learned park factors - on the games
     before the holdout, including the recipe the live model uses, and scores
     each on the holdout it never saw.
  3. Switches recipe only if the best candidate beats the current recipe's
     log loss by SWITCH_MARGIN. Otherwise the current recipe stays.
  4. Refits the chosen recipe on every game, so it learns from the newest
     results, and saves it only if it passes the sanity checks in deploy_ok().
  5. Appends what happened to model_history.json, with the live picks' record,
     so there's a running log of how the model is doing.

Runs weekly from the "Research Data Pull + Retrain" workflow. It does nothing
when there are no games newer than the live model's (the offseason) unless
FORCE=1.

Run from the repo root:
  pip install -r research/requirements.txt
  python research/train_model.py
"""

import glob
import gzip
import json
import math
import os
import sys
from collections import defaultdict, deque
from datetime import date, datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import features as F  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_FILE = "model_weights.json"
HISTORY_FILE = "model_history.json"
PICKS_FILE = "picks_history.json"

HOLDOUT_ROWS = 8000
MIN_TRAIN_ROWS = 5000
SWITCH_MARGIN = 0.0005  # log loss a new recipe must win by
SANITY_MARGIN = 0.002  # how much worse than the live model the refit may score on the holdout
PARK_PRIOR_ROWS = 3000  # hitter-games of the fixed table's value blended into each learned park factor
TOP_N = 10

DEFAULT_RECIPE = {"seasons": 1, "l2": 0.001, "park": "table"}  # how the model was built before recipes existed
RECIPES = [{"seasons": s, "l2": l2, "park": park}
           for s in (1, 2, 3) for l2 in (0.001, 0.01) for park in ("table", "learned")]
PARK_COL = F.CANDIDATE_FEATURES.index("park_factor")


# ── Data ─────────────────────────────────────────────────────────────────────
def load_raw():
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.json.gz"))):
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        for r in d["rows"]:
            r["season"] = d["season"]
        rows += d["rows"]
    return rows


def build_rows(raw_rows):
    """Features as of each game, walking each hitter's season in order."""
    by_player = defaultdict(list)
    totals = defaultdict(lambda: [0, 0])
    for r in raw_rows:
        by_player[(r["player_id"], r["season"])].append(r)
        totals[(r["player_id"], r["season"])][0] += r["hits"]
        totals[(r["player_id"], r["season"])][1] += r["at_bats"]

    rows = []
    for (pid, season), games in by_player.items():
        prev_hits, prev_ab = totals.get((pid, season - 1), (0, 0))
        prior = (prev_hits + F.LEAGUE_AVG * 100) / (prev_ab + 100) if prev_ab else F.LEAGUE_AVG
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

            feats = F.batter_features(hits, ab, n_games, recent_hits, recent_ab, prior)
            feats.update(F.matchup_features(r.get("is_home"), r.get("batter_hand"),
                                            r.get("opp_pitcher_hand"), r.get("venue")))
            feats.update(F.pitcher_features(r.get("opp_pitcher_era"), r.get("opp_pitcher_whip"),
                                            r.get("opp_pitcher_k9"), r.get("opp_pitcher_ip")))
            if all(feats.get(f) is not None for f in F.CANDIDATE_FEATURES):
                rows.append({"date": r["date"], "season": season, "venue": r.get("venue") or "",
                             "got_hit": 1.0 if r["got_hit"] else 0.0, **feats})

            hits += r["hits"]
            ab += r["at_bats"]
            n_games += 1
            window.append((day, r["hits"], r["at_bats"]))
            recent_hits += r["hits"]
            recent_ab += r["at_bats"]
    rows.sort(key=lambda r: r["date"])
    return rows


class Data:
    """Every row as arrays. The park factor column is filled in per model,
    since each model can carry its own park table."""

    def __init__(self, rows):
        self.dates = np.array([r["date"] for r in rows])
        self.seasons = np.array([r["season"] for r in rows])
        self.y = np.array([r["got_hit"] for r in rows])
        self.X = np.array([[r[f] for f in F.CANDIDATE_FEATURES] for r in rows])
        self.venue_names, self.venue_idx = np.unique([r["venue"] for r in rows], return_inverse=True)

    def matrix(self, park_factors, mask=None):
        X = self.X.copy() if mask is None else self.X[mask].copy()
        table = np.array([park_factors.get(v, 1.0) for v in self.venue_names])
        X[:, PARK_COL] = table[self.venue_idx if mask is None else self.venue_idx[mask]]
        return X


# ── Fitting ──────────────────────────────────────────────────────────────────
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit(X, y, l2):
    """L2-regularized logistic regression by Newton's method."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    w[-1] = math.log(y.mean() / (1 - y.mean()))
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0
    for _ in range(50):
        p = sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / n + reg * w
        hess = (Xb.T * (p * (1 - p))) @ Xb / n + np.diag(reg)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w[:-1], float(w[-1])


def learned_park_factors(data, mask):
    """Each venue's hit rate relative to the league, blended with the fixed
    table (or 1.0 for a park it doesn't know) so a few games can't swing it.
    New and renamed ballparks pick up a factor from their own results."""
    y, v = data.y[mask], data.venue_idx[mask]
    league = y.mean()
    table = dict(F.PARK_FACTORS)
    for i, name in enumerate(data.venue_names):
        n = int((v == i).sum())
        if n == 0 or not name:
            continue
        observed = y[v == i].mean() / league
        prior = F.PARK_FACTORS.get(name, 1.0)
        table[name] = round((observed * n + prior * PARK_PRIOR_ROWS) / (n + PARK_PRIOR_ROWS), 4)
    return table


def train(recipe, data, mask):
    """Fit one recipe on the rows in mask. None if there's too little data."""
    # A "season" is the last 365 days of games, so a window never shrinks to a
    # few days of a new season.
    last = date.fromisoformat(str(data.dates[mask][-1]))
    mask = mask & (data.dates > (last - timedelta(days=365 * recipe["seasons"])).isoformat())
    if mask.sum() < MIN_TRAIN_ROWS:
        return None
    parks = learned_park_factors(data, mask) if recipe["park"] == "learned" else None
    X = data.matrix(parks or F.PARK_FACTORS, mask)
    keep = X.std(axis=0) > 1e-9
    X = X[:, keep]
    means, stds = X.mean(axis=0), X.std(axis=0)
    w, b = fit((X - means) / stds, data.y[mask], recipe["l2"])
    dates = data.dates[mask]
    return {
        "features": [f for f, k in zip(F.CANDIDATE_FEATURES, keep) if k],
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": w.tolist(),
        "bias": b,
        "park_factors": parks,
        "recipe": recipe,
        "training_rows": int(mask.sum()),
        "training_dates": [str(dates[0]), str(dates[-1])],  # rows are in date order
    }


def predict(model, data, mask):
    X = data.matrix(model.get("park_factors") or F.PARK_FACTORS, mask)
    cols = [F.CANDIDATE_FEATURES.index(f) for f in model["features"]]
    z = ((X[:, cols] - np.array(model["means"])) / np.array(model["stds"])) @ np.array(model["weights"])
    return sigmoid(z + model["bias"])


# ── Scoring ──────────────────────────────────────────────────────────────────
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


def evaluate(model, data, mask):
    p, y = predict(model, data, mask), data.y[mask]
    return {"log_loss": log_loss(y, p), "auc": auc(y, p), "calibration": calibration(y, p),
            "backtest": top_n_backtest(data.dates[mask], y, p)}


def short(ev):
    bt = ev["backtest"]
    return {"log_loss": round(ev["log_loss"], 5), "auc": round(ev["auc"], 4),
            "top10_hit_rate": round(bt["top_n_hit_rate"], 4) if bt["top_n_hit_rate"] is not None else None}


def recipe_name(r):
    return f"{r['seasons']} year{'s' if r['seasons'] > 1 else ''}, L2 {r['l2']}, {r['park']} parks"


def live_record(days=7):
    """How the live picks did over the last week, for the history log."""
    try:
        with open(PICKS_FILE) as f:
            picks = json.load(f)["picks"]
    except (OSError, ValueError, KeyError):
        return None
    since = (date.today() - timedelta(days=days)).isoformat()
    g = [p for p in picks if p["date"] >= since and p.get("got_hit") is not None and p.get("confidence") is not None]
    if not g:
        return None
    return {"days": days, "picks": len(g), "hit_rate": round(sum(p["got_hit"] for p in g) / len(g), 4),
            "predicted": round(sum(p["confidence"] for p in g) / len(g) / 100, 4)}


# ── The run ──────────────────────────────────────────────────────────────────
def deploy_ok(final, final_ev, live_ev, baseline_ll, chosen_ev):
    """Sanity checks before the refit replaces the live model: finite numbers,
    it beats guessing the average on games it never saw, and on the recent
    games it isn't clearly worse than the model it replaces."""
    numbers = final["weights"] + [final["bias"]] + final["means"] + final["stds"]
    if not all(math.isfinite(x) for x in numbers):
        return False, "the refit has non-finite coefficients"
    if chosen_ev["log_loss"] >= baseline_ll:
        return False, "the chosen recipe didn't beat guessing the average on held-out games"
    if live_ev and final_ev["log_loss"] > live_ev["log_loss"] + SANITY_MARGIN:
        return False, "the refit scored worse than the live model on recent games"
    return True, ""


def weights_snapshot(model):
    """Each factor's weight (per standard deviation of that factor), for the
    site's Model tab to show how the last retrain moved them."""
    if not model.get("features"):
        return None
    return {f: round(w, 4) for f, w in zip(model["features"], model["weights"])}


def append_history(entry):
    try:
        with open(HISTORY_FILE) as f:
            history = json.load(f)
    except (OSError, ValueError):
        history = {"runs": []}
    history["runs"].append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=1)


def summary(lines):
    text = "\n".join(lines)
    print(text)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(text + "\n")


def main():
    force = os.environ.get("FORCE", "").lower() in ("1", "true", "yes")
    live = {}
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE) as f:
            live = json.load(f)
    live_through = live.get("trained_through") or (live.get("training_dates") or [None, None])[1]
    current = live.get("recipe", DEFAULT_RECIPE)

    rows = build_rows(load_raw())
    if len(rows) < MIN_TRAIN_ROWS + HOLDOUT_ROWS:
        print(f"Only {len(rows)} usable rows - not enough to train on.")
        return
    data = Data(rows)
    latest = str(data.dates[-1])
    print(f"{len(rows)} rows, {data.dates[0]} to {latest}, seasons {sorted(set(data.seasons.tolist()))}")
    if live_through and latest <= live_through and not force:
        print(f"No games since the live model's last training day ({live_through}); nothing to learn from.")
        return

    days = np.unique(data.dates)[::-1]
    counts = {d: c for d, c in zip(*np.unique(data.dates, return_counts=True))}
    n, holdout_from = 0, days[0]
    for d in days:
        n += counts[d]
        holdout_from = d
        if n >= HOLDOUT_ROWS:
            break
    test = data.dates >= holdout_from
    before = ~test
    y_test = data.y[test]
    baseline_ll = log_loss(y_test, np.full(len(y_test), data.y[before].mean()))
    print(f"Holdout: {test.sum()} hitter-games from {holdout_from} on (guessing the average: {baseline_ll:.5f})")

    candidates = RECIPES + ([current] if current not in RECIPES else [])
    results = []
    for recipe in candidates:
        m = train(recipe, data, before)
        if m is None:
            continue
        ev = evaluate(m, data, test)
        results.append((recipe, m, ev))
        s = short(ev)
        print(f"  {recipe_name(recipe):<38} log loss {s['log_loss']:.5f}  AUC {s['auc']:.4f}  "
              f"top {TOP_N} {s['top10_hit_rate']}")
    if not results:
        print("No recipe had enough data to fit.")
        return

    by_recipe = {json.dumps(r, sort_keys=True): (r, m, ev) for r, m, ev in results}
    cur = by_recipe.get(json.dumps(current, sort_keys=True))
    best = min(results, key=lambda t: t[2]["log_loss"])
    switched = cur is None or (best[0] != current and best[2]["log_loss"] < cur[2]["log_loss"] - SWITCH_MARGIN)
    chosen = best if switched else cur
    chosen_recipe, _, chosen_ev = chosen

    live_ev = evaluate(live, data, test) if live.get("features") else None
    final = train(chosen_recipe, data, np.ones(len(data.y), dtype=bool))
    final_ev = evaluate(final, data, test)
    ok, why = deploy_ok(final, final_ev, live_ev, baseline_ll, chosen_ev)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if ok:
        final.update({
            "trained_at": now,
            "trained_through": latest,
            "metrics": {
                "test_from": str(holdout_from),
                "test_rows": int(test.sum()),
                "test_log_loss": chosen_ev["log_loss"],
                "baseline_log_loss": baseline_ll,
                "test_auc": chosen_ev["auc"],
                "calibration": chosen_ev["calibration"],
                "backtest": chosen_ev["backtest"],
            },
        })
        # The Accuracy page describes the backtest as fit before test_from and tested after.
        final["training_dates"][0] = chosen[1]["training_dates"][0]
        with open(MODEL_FILE, "w") as f:
            json.dump(final, f, indent=2)

    append_history({
        "run_at": now,
        "data_through": latest,
        "holdout": {"from": str(holdout_from), "rows": int(test.sum()), "baseline_log_loss": round(baseline_ll, 5)},
        "live_model": {"trained_at": live.get("trained_at"), "recipe": current,
                       **({"holdout": short(live_ev)} if live_ev else {})},
        "current_recipe": short(cur[2]) if cur else None,
        "best_recipe": {"recipe": best[0], **short(best[2])},
        "candidates": [{"recipe": r, **short(ev)} for r, _, ev in results],
        "switched_recipe": bool(switched and ok),
        "deployed": ok,
        "reason": why or ("new recipe beat the current one on held-out games" if switched
                          else "kept the recipe, refit with the newest games"),
        "live_picks_last_7_days": live_record(),
        "weights_before": weights_snapshot(live),
        "weights_after": weights_snapshot(final if ok else live),
    })

    lines = [f"### MLB hit model retrain ({latest})",
             f"- Holdout: {int(test.sum())} hitter-games from {holdout_from}",
             f"- Current recipe ({recipe_name(current)}): "
             + (f"log loss {cur[2]['log_loss']:.5f}" if cur else "couldn't be fit"),
             f"- Best candidate ({recipe_name(best[0])}): log loss {best[2]['log_loss']:.5f}",
             f"- Recipe: {'switched to ' + recipe_name(chosen_recipe) if switched else 'unchanged'}",
             f"- Deployed: {'yes, refit through ' + latest if ok else 'no - ' + why}"]
    summary(lines)


if __name__ == "__main__":
    main()
