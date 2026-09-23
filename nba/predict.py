"""
predict.py - NBA Edge's daily run. Every run does three things, so any run
catches up on whatever an earlier one missed:

  1. sync     - stores every finished day's scores and box scores
                (nba/data/days/, see store.py) that isn't stored yet
  2. grade    - settles pending picks from those final scores; a postponed
                game is voided rather than counted
  3. picks    - gives every game today a win chance and a pick. Picks are
                refreshed each run with the latest injury report, and lock
                once their game tips off.

Writes nba/picks_history.json for build_site.py. NBA_TODAY=YYYY-MM-DD
overrides today's date for testing.
"""

import json
import os
import sys
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import espn  # noqa: E402
import model as M  # noqa: E402
import store  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(HERE, "picks_history.json")
SYNC_FILE = os.path.join(HERE, "data", "sync.json")
MODEL_FILE = os.path.join(HERE, "model_weights.json")

NOW = datetime.now(espn.ET)
TODAY = os.environ.get("NBA_TODAY") or NOW.date().isoformat()
OUT_STATUSES = {"Out", "Doubtful"}


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
    """October through June. July-September never has regular-season or playoff games."""
    return date.fromisoformat(day).month not in (7, 8, 9)


# ── 1. sync ──────────────────────────────────────────────────────────────────
def sync():
    games, _ = store.load_all()
    through = load(SYNC_FILE, {}).get("through") or (games[-1]["date"] if games else "2026-06-30")
    day = date.fromisoformat(through) + timedelta(days=1)
    yesterday = date.fromisoformat(TODAY) - timedelta(days=1)
    while day <= yesterday:
        d = day.isoformat()
        if in_season(d):
            day_games = espn.scoreboard(d)
            if any(not (g["final"] or g["postponed"]) for g in day_games):
                print(f"  {d}: not every game is final yet, will retry next run")
                break
            finals = [g for g in day_games if g["final"] and g["home_pts"] is not None]
            if finals:
                box = {}
                for g in finals:
                    try:
                        box[g["id"]] = espn.boxscore(g["id"])
                    except Exception as e:  # scores still count without a box score
                        print(f"  box score {g['id']} failed: {e}")
                store.save_day(d, finals, box)
                print(f"  stored {d}: {len(finals)} games")
        through = d
        day += timedelta(days=1)
    save(SYNC_FILE, {"through": through}, indent=1)


# ── 2. grade ─────────────────────────────────────────────────────────────────
def grade(history, games_by_id):
    graded = 0
    for p in history["picks"]:
        if p.get("correct") is not None or p.get("void") or p["date"] >= TODAY:
            continue
        g = games_by_id.get(p["game_id"])
        if g:
            p["home_pts"], p["away_pts"] = g["home_pts"], g["away_pts"]
            p["winner"] = g["home"] if g["home_pts"] > g["away_pts"] else g["away"]
            p["correct"] = p["winner"] == p["pick"]
            graded += 1
        elif p["date"] < (date.fromisoformat(TODAY) - timedelta(days=3)).isoformat():
            p["void"] = True  # postponed and never played on its date
            graded += 1
    print(f"Graded {graded} picks")


# ── 3. picks ─────────────────────────────────────────────────────────────────
def record(league, team):
    m = league.margins[team]
    w = sum(1 for x in m if x > 0)
    return f"{w}-{len(m) - w}"


def make_picks(history, league, weights):
    slate = espn.scoreboard(TODAY)
    print(f"{len(slate)} games today")
    if not slate:
        return
    try:
        report = espn.injuries()
    except Exception as e:  # picks still go out, just without injury news
        print(f"  injury report failed: {e}")
        report = {}

    existing = {p["game_id"]: p for p in history["picks"] if p["date"] == TODAY}
    for g in slate:
        league._check_season(g["season"])  # new season: ratings regress, last season's rotations clear
        old = existing.get(g["id"])
        if g["state"] != "pre" or g["postponed"]:
            continue  # started, final or postponed: a pick already made stays as it was
        out = {}
        for side in ("home", "away"):
            team, team_id = g[side], str(g[f"{side}_id"])
            injured = {pid for pid, status in report.get(team_id, {}).items() if status in OUT_STATUSES}
            playing = league.rotation(team) - injured
            out[side] = (league.missing_value(team, playing), league.missing_players(team, playing))
        feats = league.features(g, out["home"][0], out["away"][0])
        p_home, margin = M.win_prob(weights, feats)
        pick_home = p_home >= 0.5
        pick = {
            "date": TODAY, "game_id": g["id"], "start": g["start"], "type": g["type"],
            "home": g["home"], "away": g["away"], "home_name": g["home_name"], "away_name": g["away_name"],
            "home_record": record(league, g["home"]), "away_record": record(league, g["away"]),
            "neutral": g["neutral"],
            "pick": g["home"] if pick_home else g["away"],
            "prob": round(100 * (p_home if pick_home else 1 - p_home), 1),
            "home_prob": round(100 * p_home, 1),
            "margin": round(abs(margin), 1),
            "home_out": [n for n, _ in out["home"][1]], "away_out": [n for n, _ in out["away"][1]],
            "rest": {"home": league.rest_days(g["home"], TODAY), "away": league.rest_days(g["away"], TODAY)},
            "correct": None,
        }
        if old:
            old.clear()
            old.update(pick)
        else:
            history["picks"].append(pick)
        print(f"  {g['away']} @ {g['home']}: {pick['pick']} {pick['prob']}% by {pick['margin']}")


def main():
    weights = load(MODEL_FILE, None)
    history = load(HISTORY_FILE, {"picks": []})
    print(f"NBA Edge run for {TODAY}")
    sync()
    games, box = store.load_all()
    grade(history, {g["id"]: g for g in games})

    league = M.League()
    for g in games:
        if g["date"] < TODAY:
            league.update(g, box.get(g["id"]))
    if weights and in_season(TODAY):
        make_picks(history, league, weights)

    history["picks"].sort(key=lambda p: (p["date"], p["start"], p["game_id"]))
    save(HISTORY_FILE, history, indent=1)


if __name__ == "__main__":
    main()
