"""
train.py - retrains the MLB team model, and only changes how it's built when
the change is proven on recent games.

Walks every stored game in order (model.League), so each game's features are
what was known that morning, and fits the home team's win chance by logistic
regression. Each run:

  1. Holds out the most recent HOLDOUT_GAMES games.
  2. Fits every candidate recipe in RECIPES - how fast Elo reacts (elo_k), how
     much of it carries over the winter (elo_carry), how much a starter's line
     is shrunk toward average (sp_shrink), and how many years of games to
     learn from - on the games before the holdout, including the recipe the
     live model uses, and scores each on the holdout it never saw.
  3. Switches recipe only if the best candidate beats the current recipe's log
     loss by SWITCH_MARGIN. Otherwise the current recipe stays.
  4. Refits the chosen recipe on every game, so it learns from the newest
     results, and saves it to teams/model_weights.json only if it passes the
     checks in deploy_ok(). The saved model also carries a backtest on the
     latest finished season (fit only on the seasons before it) for the site.
  5. Appends what happened to teams/model_history.json, with the live picks'
     record, so there's a running log of how the model is doing.

The daily workflow runs this after teams/predict.py. It only retrains once a
week, and only when at least MIN_NEW_GAMES have been played since the live
model's last training day, so it does nothing in the offseason. FORCE=1 skips
both checks (the research workflow sets it).
"""

import json
import math
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import model as M  # noqa: E402
import store  # noqa: E402

OUT = os.path.join(HERE, "model_weights.json")
HISTORY = os.path.join(HERE, "model_history.json")
PICKS = os.path.join(HERE, "picks_history.json")
WARMUP_UNTIL = "07-01"  # skip the first stored season's first half: no pitcher history yet, ratings unsettled
BANDS = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 1.01)]
L2 = 1.0  # ridge penalty on every coefficient but home field

HOLDOUT_GAMES = 300  # about two weeks of the regular season
MIN_TRAIN_GAMES = 1500
MIN_NEW_GAMES = 60
RETRAIN_EVERY_DAYS = 7
SWITCH_MARGIN = 0.001  # log loss a new recipe must win by
SANITY_MARGIN = 0.005  # how much worse than the live model the refit may score on the holdout

DEFAULT_RECIPE = {"elo_k": M.ELO_K, "elo_carry": M.ELO_CARRY, "sp_shrink": M.SP_SHRINK_IP, "years": 2}
RECIPES = [{"elo_k": k, "elo_carry": c, "sp_shrink": s, "years": y}
           for k in (3, 4, 6) for c in (0.5, 0.67) for s in (25, 40, 60) for y in (2, 3)]


def league_params(recipe):
    return {"elo_k": recipe["elo_k"], "elo_carry": recipe["elo_carry"], "sp_shrink": recipe["sp_shrink"]}


_walks = {}


def build_rows(params=None):
    """One row per stored game with that morning's features, walking the league
    with the given settings. Cached per setting."""
    key = json.dumps(params or {}, sort_keys=True)
    if key in _walks:
        return _walks[key]
    league = M.League(params)
    rows = []
    for g in store.load_all():
        feats = league.features_for_past(g)
        rows.append({"game": g, "x": [feats[k] for k in M.FEATURES],
                     "home_won": g["home_runs"] > g["away_runs"]})
        league.update(g)
    first = rows[0]["game"]["season"] if rows else None
    for r in rows:
        g = r["game"]
        r["usable"] = g["season"] != first or g["date"][5:] >= WARMUP_UNTIL
    _walks[key] = rows
    return rows


def fit(rows):
    """Ridge-penalized logistic regression by Newton's method."""
    X = np.array([r["x"] for r in rows])
    y = np.array([r["home_won"] for r in rows], dtype=float)
    pen = np.full(X.shape[1], L2)
    pen[M.FEATURES.index("home_field")] = 0.0
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) + pen * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"features": M.FEATURES, "coef": {k: round(float(c), 5) for k, c in zip(M.FEATURES, w)}}


def evaluate(weights, rows):
    preds = []
    for r in rows:
        p_home = M.win_prob(weights, dict(zip(M.FEATURES, r["x"])))
        pick_home = p_home >= 0.5
        preds.append({"date": r["game"]["date"], "p_home": p_home, "conf": p_home if pick_home else 1 - p_home,
                      "correct": pick_home == r["home_won"], "home_won": r["home_won"]})
    n = len(preds)
    log_loss = -sum(math.log(max(1e-9, p["p_home"] if p["home_won"] else 1 - p["p_home"])) for p in preds) / n
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
        "home_win_rate": round(sum(p["home_won"] for p in preds) / n, 4),
        "top3": {"days": len(by_day), "picks": len(top), "correct": sum(p["correct"] for p in top),
                 "accuracy": round(sum(p["correct"] for p in top) / len(top), 4)},
        "bands": bands,
        "months": months,
    }


def raw_log_loss(weights, rows):
    """Unrounded log loss, for comparing recipes whose scores are close."""
    total = 0.0
    for r in rows:
        p = M.win_prob(weights, dict(zip(M.FEATURES, r["x"])))
        total -= math.log(max(1e-9, p if r["home_won"] else 1 - p))
    return total / len(rows)


def home_rate_log_loss(rate, rows):
    return -sum(math.log(rate if r["home_won"] else 1 - rate) for r in rows) / len(rows)


def train_recipe(recipe, before_idx=None):
    """Fit a recipe on usable games before index before_idx (all games if
    None), within its window of years. None if there are too few games."""
    rows = build_rows(league_params(recipe))
    pool = rows if before_idx is None else rows[:before_idx]
    if not pool:
        return None
    last = date.fromisoformat(pool[-1]["game"]["date"])
    since = (last - timedelta(days=365 * recipe["years"])).isoformat()
    train = [r for r in pool if r["usable"] and r["game"]["date"] > since]
    if len(train) < MIN_TRAIN_GAMES:
        return None
    w = fit(train)
    w["league"] = league_params(recipe)
    w["recipe"] = recipe
    w["training_games"] = len(train)
    w["training_dates"] = [train[0]["game"]["date"], train[-1]["game"]["date"]]
    return w


def season_backtest(recipe):
    """The site's backtest: fit on the seasons before the latest finished one,
    then pick every game of that season."""
    rows = build_rows(league_params(recipe))
    seasons = sorted({r["game"]["season"] for r in rows})
    # A season is finished once we're past its World Series.
    finished = [s for s in seasons if s < seasons[-1] or date.today().isoformat() >= f"{s}-11-15"]
    if len(finished) < 2:
        return {}
    test_season = finished[-1]
    first = max(seasons[0], test_season - recipe["years"])
    train = [r for r in rows if r["usable"] and first <= r["game"]["season"] < test_season]
    test = [r for r in rows if r["game"]["season"] == test_season]
    if len(train) < MIN_TRAIN_GAMES:
        return {}
    w = fit(train)
    test_regular = [r for r in test if r["game"]["type"] == "regular"]
    post = [r for r in test if r["game"]["type"] != "regular"]
    bt = evaluate(w, test_regular)
    home_rate = float(np.mean([r["home_won"] for r in train]))

    def only(rs, keep):
        return [{**r, "x": [v if M.FEATURES[i] in keep else 0.0 for i, v in enumerate(r["x"])]} for r in rs]

    elo_keep = {"home_field", "elo"}
    elo_bt = evaluate(fit(only(train, elo_keep)), only(test_regular, elo_keep))
    return {
        "trained_on": f"{first} to {test_season - 1}",
        "backtest_season": str(test_season),
        "backtest": bt,
        "backtest_postseason": evaluate(w, post) if post else None,
        "baselines": {"elo_only_accuracy": elo_bt["accuracy"], "elo_only_log_loss": elo_bt["log_loss"],
                      "home_team_accuracy": bt["home_win_rate"],
                      "home_rate_log_loss": round(home_rate_log_loss(home_rate, test_regular), 4)},
    }


def load(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def live_record(days=14):
    picks = load(PICKS, {}).get("picks", [])
    since = (date.today() - timedelta(days=days)).isoformat()
    g = [p for p in picks if p["date"] >= since and p.get("correct") is not None and not p.get("void")]
    if not g:
        return None
    return {"days": days, "picks": len(g), "accuracy": round(sum(p["correct"] for p in g) / len(g), 4),
            "predicted": round(sum(p["prob"] for p in g) / len(g) / 100, 4)}


def deploy_ok(final, final_ll, live_ll, chosen_ll, baseline_ll):
    """Sanity checks before the refit replaces the live model: finite numbers,
    it beats always taking the usual home-team rate on games it never saw, and
    on the recent games it isn't clearly worse than the model it replaces."""
    if not all(math.isfinite(v) for v in final["coef"].values()):
        return False, "the refit has non-finite coefficients"
    if chosen_ll >= baseline_ll:
        return False, "the chosen recipe didn't beat the home-team rate on held-out games"
    if live_ll is not None and final_ll > live_ll + SANITY_MARGIN:
        return False, "the refit scored worse than the live model on recent games"
    return True, ""


def summary(lines):
    text = "\n".join(lines)
    print(text)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a") as f:
            f.write(text + "\n")


def recipe_name(r):
    return f"Elo K {r['elo_k']}, carry {r['elo_carry']}, SP shrink {r['sp_shrink']}, {r['years']} years"


def main():
    force = os.environ.get("FORCE", "").lower() in ("1", "true", "yes")
    live = load(OUT, {})
    history = load(HISTORY, {"runs": []})
    current = live.get("recipe", DEFAULT_RECIPE)

    rows = build_rows(league_params(current))
    if not rows:
        print("No stored games.")
        return
    latest = rows[-1]["game"]["date"]
    live_through = live.get("trained_through")
    new_games = sum(1 for r in rows if live_through is None or r["game"]["date"] > live_through)
    if not force:
        last_run = history["runs"][-1]["run_at"][:10] if history["runs"] else None
        if last_run and date.fromisoformat(last_run) > date.today() - timedelta(days=RETRAIN_EVERY_DAYS):
            print(f"Retrained {last_run}; next retrain once a week has passed.")
            return
        if new_games < MIN_NEW_GAMES:
            print(f"{new_games} games since the live model's last training day ({live_through}); "
                  f"waiting for {MIN_NEW_GAMES}.")
            return

    cut = len(rows) - HOLDOUT_GAMES
    holdout_from = rows[cut]["game"]["date"]
    print(f"{len(rows)} games through {latest}; holdout: last {HOLDOUT_GAMES} games from {holdout_from}")

    def holdout(recipe):
        return build_rows(league_params(recipe))[cut:]

    home_rate = float(np.mean([r["home_won"] for r in rows[:cut]]))
    baseline_ll = home_rate_log_loss(home_rate, holdout(current))

    candidates = RECIPES + ([current] if current not in RECIPES else [])
    results = []
    for recipe in candidates:
        w = train_recipe(recipe, cut)
        if w is None:
            continue
        ll = raw_log_loss(w, holdout(recipe))
        results.append((recipe, w, ll))
        print(f"  {recipe_name(recipe):<50} log loss {ll:.4f}")
    if not results:
        print("No recipe had enough games to fit.")
        return

    cur = next((t for t in results if t[0] == current), None)
    best = min(results, key=lambda t: t[2])
    switched = cur is None or (best[0] != current and best[2] < cur[2] - SWITCH_MARGIN)
    chosen = best if switched else cur
    chosen_recipe = chosen[0]

    live_ll = raw_log_loss(live, holdout(live.get("recipe", DEFAULT_RECIPE))) if live.get("coef") else None
    final = train_recipe(chosen_recipe)
    final_ll = raw_log_loss(final, holdout(chosen_recipe))
    ok, why = deploy_ok(final, final_ll, live_ll, chosen[2], baseline_ll)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if ok:
        final.update(season_backtest(chosen_recipe))
        final.update({
            "trained_at": now,
            "trained_through": latest,
            "recent_holdout": {"from": holdout_from, "games": HOLDOUT_GAMES,
                               **{k: v for k, v in evaluate(chosen[1], holdout(chosen_recipe)).items()
                                  if k not in ("months", "bands")}},
        })
        with open(OUT, "w") as f:
            json.dump(final, f, indent=1)

    history["runs"].append({
        "run_at": now,
        "data_through": latest,
        "holdout": {"from": holdout_from, "games": HOLDOUT_GAMES, "home_rate_log_loss": round(baseline_ll, 4)},
        "live_model": {"trained_at": live.get("trained_at"), "recipe": live.get("recipe", DEFAULT_RECIPE),
                       "holdout_log_loss": round(live_ll, 4) if live_ll is not None else None},
        "current_recipe_log_loss": round(cur[2], 4) if cur else None,
        "best_recipe": {"recipe": best[0], "log_loss": round(best[2], 4)},
        "candidates": [{"recipe": r, "log_loss": round(ll, 4)} for r, _, ll in results],
        "switched_recipe": bool(switched and ok),
        "deployed": ok,
        "reason": why or ("new recipe beat the current one on held-out games" if switched
                          else "kept the recipe, refit with the newest games"),
        "live_picks_last_14_days": live_record(),
        "weights_before": live.get("coef"),
        "weights_after": (final if ok else live).get("coef"),
    })
    with open(HISTORY, "w") as f:
        json.dump(history, f, indent=1)

    summary([f"### MLB team model retrain ({latest})",
             f"- Holdout: last {HOLDOUT_GAMES} games, from {holdout_from}",
             f"- Current recipe ({recipe_name(current)}): "
             + (f"log loss {cur[2]:.4f}" if cur else "couldn't be fit"),
             f"- Best candidate ({recipe_name(best[0])}): log loss {best[2]:.4f}",
             f"- Recipe: {'switched to ' + recipe_name(chosen_recipe) if switched else 'unchanged'}",
             f"- Deployed: {'yes, refit through ' + latest if ok else 'no - ' + why}"])


if __name__ == "__main__":
    main()
