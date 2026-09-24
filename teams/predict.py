"""
predict.py - the MLB team model's daily run (MLB Edge's Games tab). Every run
does three things, so any run catches up on whatever an earlier one missed:

  1. sync     - stores every finished day's final scores and starting
                pitchers (teams/data/days/, see store.py) that isn't stored yet
  2. grade    - settles pending picks from those final scores; a postponed or
                suspended game is voided rather than counted
  3. picks    - gives every game today a win chance and a pick, using the
                announced probable starters, plus a moneyline pick against
                the book price on ESPN's scoreboard (see moneyline.py). Picks
                are refreshed each run (a starter may be named or changed, a
                price may move) and lock at first pitch.

Writes teams/picks_history.json for build_site.py. MLB_TODAY=YYYY-MM-DD
overrides today's date for testing.
"""

import json
import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model as M  # noqa: E402
import moneyline  # noqa: E402
import statsapi  # noqa: E402
import store  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(HERE, "picks_history.json")
SYNC_FILE = os.path.join(HERE, "data", "sync.json")
MODEL_FILE = os.path.join(HERE, "model_weights.json")

NOW = datetime.now(statsapi.ET)
TODAY = os.environ.get("MLB_TODAY") or NOW.date().isoformat()


def load(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def save(path, data, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, **kw)


def in_season(day):
    """Late March through early November. December-February never has
    regular-season or postseason games."""
    return date.fromisoformat(day).month not in (12, 1, 2)


# ── 1. sync ──────────────────────────────────────────────────────────────────
def sync():
    through = load(SYNC_FILE, {}).get("through")
    if not through:
        games = store.load_all()
        through = games[-1]["date"] if games else f"{date.fromisoformat(TODAY).year}-02-28"
    day = date.fromisoformat(through) + timedelta(days=1)
    yesterday = date.fromisoformat(TODAY) - timedelta(days=1)
    while day <= yesterday:
        d = day.isoformat()
        if in_season(d):
            day_games = statsapi.schedule(d)  # can include a suspended game resumed today
            if any(not (g["final"] or g["postponed"]) for g in day_games if g["date"] == d):
                print(f"  {d}: not every game is final yet, will retry next run")
                break
            finals = [g for g in day_games if g["final"] and g["home_runs"] is not None]
            if finals:
                rows = []
                for g in finals:
                    try:
                        box = statsapi.boxscore(g["id"])
                    except Exception as e:  # the score still counts without the starters
                        print(f"  box score {g['id']} failed: {e}")
                        box = {}
                    rows.append(statsapi.stored(g, box))
                store.save_day(d, rows)
                print(f"  stored {d}: {len(finals)} games")
        through = d
        day += timedelta(days=1)
    save(SYNC_FILE, {"through": through}, indent=1)


# ── 2. grade ─────────────────────────────────────────────────────────────────
def grade(history, games_by_id):
    graded = 0
    stale = (date.fromisoformat(TODAY) - timedelta(days=3)).isoformat()
    for p in history["picks"]:
        if p.get("correct") is not None or p.get("void") or p["date"] >= TODAY:
            continue
        g = games_by_id.get(p["game_id"])
        if g and g["date"] == p["date"]:
            p["home_runs"], p["away_runs"] = g["home_runs"], g["away_runs"]
            p["winner"] = g["home"] if g["home_runs"] > g["away_runs"] else g["away"]
            p["correct"] = p["winner"] == p["pick"]
            graded += 1
        elif (g and g["date"] != p["date"]) or p["date"] < stale:
            p["void"] = True  # postponed or suspended: not played to a finish on its date
            graded += 1
    print(f"Graded {graded} picks")


def grade_moneylines(history):
    n = sum(moneyline.grade(p) for p in history["picks"])
    if n:
        print(f"Graded {n} moneyline picks")


# ── 3. picks ─────────────────────────────────────────────────────────────────
def starter(league, probable):
    if not probable:
        return {"id": None, "name": "TBD", "era": None, "ip": 0}
    ip, era = league.sp_line(probable[0])
    return {"id": probable[0], "name": probable[1], "era": era, "ip": ip}


def odds_for(g, slate, events, old):
    """Today's moneyline odds for a game from ESPN, or the last ones stored for
    it when ESPN has none right now."""
    ev = moneyline.match(events, g["away"], g["home"], g["away_name"], g["home_name"],
                         datetime.fromisoformat(g["start"]), moneyline.MLB_ALIASES)
    if ev:  # doubleheader: the event belongs to whichever of the two games starts closest to it
        same = [s for s in slate if (s["away"], s["home"]) == (g["away"], g["home"])]
        if min(same, key=lambda s: abs((datetime.fromisoformat(s["start"]) - ev["start"]).total_seconds())) is not g:
            ev = None
    if ev and ev["odds"]:
        return ev["odds"]
    ml = (old or {}).get("ml")
    return {"home": ml["home_ml"], "away": ml["away_ml"], "book": ml.get("book")} if ml else None


def make_picks(history, league, weights):
    slate = [g for g in statsapi.schedule(TODAY) if g["date"] == TODAY]
    print(f"{len(slate)} games today")
    events = moneyline.fetch("baseball/mlb", TODAY) if any(g["state"] == "pre" for g in slate) else []
    existing = {p["game_id"]: p for p in history["picks"] if p["date"] == TODAY}
    for g in slate:
        league.check_season(g["season"])  # new season: ratings regress, pitcher lines roll into history
        old = existing.get(g["id"])
        if g["postponed"]:
            if old and old.get("correct") is None:
                old["void"] = True
            continue
        if g["state"] != "pre":
            continue  # started or final: the pick made before first pitch stays as it was
        sp = {side: starter(league, g[f"{side}_probable"]) for side in ("home", "away")}
        feats = league.features(g, sp["home"]["id"], sp["away"]["id"])
        p_home = M.win_prob(weights, feats)
        pick_home = p_home >= 0.5
        pick = {
            "date": TODAY, "game_id": g["id"], "start": g["start"], "type": g["type"], "round": g["round"],
            "doubleheader": g["doubleheader"],
            "home": g["home"], "away": g["away"], "home_name": g["home_name"], "away_name": g["away_name"],
            "home_id": g["home_id"], "away_id": g["away_id"],
            "home_record": league.record(g["home"]), "away_record": league.record(g["away"]),
            "home_sp": sp["home"], "away_sp": sp["away"],
            "pick": g["home"] if pick_home else g["away"],
            "prob": round(100 * (p_home if pick_home else 1 - p_home), 1),
            "home_prob": round(100 * p_home, 1),
            "factors": {k: round(weights["coef"][k] * feats[k], 3) for k in weights["features"]},
            "model": weights.get("trained_at"),
            "correct": None,
        }
        try:  # the moneyline pick is extra: without odds, the game pick still goes out
            ml = moneyline.pick(g["home"], g["away"], pick["home_prob"], odds_for(g, slate, events, old), pick["pick"])
        except Exception as e:
            print(f"  moneyline for {g['away']} @ {g['home']} failed: {e}")
            ml = None
        if ml:
            pick["ml"] = ml
        if old:
            old.clear()
            old.update(pick)
        else:
            history["picks"].append(pick)
        print(f"  {g['away']} @ {g['home']} ({sp['away']['name']} vs {sp['home']['name']}): "
              f"{pick['pick']} {pick['prob']}%"
              + (f"; ML {moneyline.text(ml)} ({moneyline.detail(ml)})" if ml else ""))


def main():
    weights = load(MODEL_FILE, None)
    history = load(HISTORY_FILE, {"picks": []})
    print(f"MLB team model run for {TODAY}")
    if in_season(TODAY):
        sync()
    games = store.load_all()
    grade(history, {g["id"]: g for g in games})

    if not in_season(TODAY):
        store.roll_up_days()  # the season is over: its day files become one season file
    league = M.League((weights or {}).get("league"))
    for g in games:
        if g["date"] < TODAY:
            league.update(g)
    if weights and in_season(TODAY):
        make_picks(history, league, weights)
    grade_moneylines(history)

    history["picks"].sort(key=lambda p: (p["date"], p["start"], p["game_id"]))
    save(HISTORY_FILE, history, indent=1)


if __name__ == "__main__":
    main()
