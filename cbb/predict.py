"""
predict.py - CBB Edge's daily run. Every run does four things, so any run
catches up on whatever an earlier one missed:

  1. sync     - stores every finished day's scores and team box scores
                (cbb/data/days/, see store.py) that isn't stored yet
  2. grade    - settles pending picks from those final scores; a postponed
                game is voided rather than counted
  3. ratings  - every Division I team's KenPom-style ratings, four factors and
                strength of schedule as of this morning, with Bart Torvik's
                T-Rank alongside (cbb/ratings.json, the site's Ratings tab)
  4. picks    - gives every game today between two Division I teams a win
                chance, a projected score and a pick, plus a moneyline pick
                against the book price on ESPN's scoreboard (see
                moneyline.py). Picks are refreshed each run with the latest
                prices, and lock once their game tips off.

Writes cbb/picks_history.json for build_pages.py. CBB_TODAY=YYYY-MM-DD
overrides today's date for testing.
"""

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import espn  # noqa: E402
import model as M  # noqa: E402
import moneyline  # noqa: E402
import store  # noqa: E402
import torvik  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(HERE, "picks_history.json")
RATINGS_FILE = os.path.join(HERE, "ratings.json")
SYNC_FILE = os.path.join(HERE, "data", "sync.json")
MODEL_FILE = os.path.join(HERE, "model_weights.json")

NOW = datetime.now(espn.ET)
TODAY = os.environ.get("CBB_TODAY") or NOW.date().isoformat()


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
    """November through early April, plus late October for early tip-offs."""
    d = date.fromisoformat(day)
    return d.month in (11, 12, 1, 2, 3) or (d.month == 4 and d.day <= 15) or (d.month == 10 and d.day >= 25)


def season_of(day):
    """ESPN's season year: 2027 for the 2026-27 season."""
    d = date.fromisoformat(day)
    return d.year + 1 if d.month >= 7 else d.year


# ── 1. sync ──────────────────────────────────────────────────────────────────
def sync():
    games, _ = store.load_all()
    through = load(SYNC_FILE, {}).get("through") or (games[-1]["date"] if games else "2026-10-24")
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
                        b = espn.boxscore(g["id"])
                        if b:
                            box[g["id"]] = b
                    except Exception as e:  # scores still count without a box score
                        print(f"  box score {g['id']} failed: {e}")
                store.save_day(d, finals, box)
                print(f"  stored {d}: {len(finals)} games, {len(box)} box scores")
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
            p["winner"] = p["home"] if g["home_pts"] > g["away_pts"] else p["away"]
            p["correct"] = p["winner"] == p["pick"]
            graded += 1
        elif p["date"] < (date.fromisoformat(TODAY) - timedelta(days=3)).isoformat():
            p["void"] = True  # postponed and never played on its date
            graded += 1
    n = sum(moneyline.grade(p) for p in history["picks"])
    print(f"Graded {graded} picks" + (f", {n} moneyline picks" if n else ""))


# ── 3. ratings ───────────────────────────────────────────────────────────────
def write_ratings(league, season, games, final=False):
    """final: the offseason, when the table is last season's final ratings."""
    day = None if final else TODAY
    table = M.team_table(league, day)
    if not table:
        return
    trank = torvik.fetch(season)
    matched = torvik.attach(table, trank)
    snapshot = torvik.by_team_id(trank, store.name_to_id(games, torvik.normalize))
    if snapshot and not final:  # the model learns from these once enough games have one
        store.save_trank(season, TODAY, snapshot)
    r = league.ratings(day)
    save(RATINGS_FILE, {
        "date": games[-1]["date"] if final else TODAY, "season": season, "final": final, "teams": table,
        "average": {"eff": round(r["mu"], 1), "tempo": round(r["mu_t"], 1),
                    "home_court": round(2 * r["h"] * r["mu_t"] / 100, 1)},
        "trank_matched": matched,
    }, indent=1)
    print(f"Ratings for {len(table)} teams ({matched} matched to T-Rank)")


# ── 4. picks ─────────────────────────────────────────────────────────────────
def make_picks(history, league, weights):
    slate, odds = espn.scoreboard(TODAY, with_odds=True)
    print(f"{len(slate)} games today")
    existing = {p["game_id"]: p for p in history["picks"] if p["date"] == TODAY}
    for g in slate:
        league._check_season(g["season"])  # new season: ratings start from last season's, regressed
        old = existing.get(g["id"])
        if g["state"] != "pre" or g["postponed"] or not league.di_game(g):
            continue  # started, final, postponed or not two D-I teams: a pick already made stays as it was
        feats = league.features(g)
        p_home, margin = M.win_prob(weights, feats)
        r = league.ratings(TODAY)
        proj_h, proj_a, poss = league.projection(r, g)
        # Scale the projected score so it agrees with the model's margin.
        mid = (proj_h + proj_a) / 2
        proj_h, proj_a = mid + margin / 2, mid - margin / 2
        pick_home = p_home >= 0.5
        home, away = g["home_abbr"], g["away_abbr"]

        pick = {
            "date": TODAY, "game_id": g["id"], "start": g["start"], "type": g["type"], "note": g["note"],
            "home": home, "away": away, "home_id": g["home"], "away_id": g["away"],
            "home_name": g["home_name"], "away_name": g["away_name"],
            "home_rank": g["home_rank"], "away_rank": g["away_rank"],
            "home_em_rank": league_rank.get(g["home"]), "away_em_rank": league_rank.get(g["away"]),
            "home_record": league.record(g["home"]), "away_record": league.record(g["away"]),
            "neutral": g["neutral"],
            "pick": home if pick_home else away,
            "prob": round(100 * (p_home if pick_home else 1 - p_home), 1),
            "home_prob": round(100 * p_home, 1),
            "margin": round(abs(margin), 1),
            "proj": {"home": round(proj_h), "away": round(proj_a), "tempo": round(poss)},
            "model": weights.get("trained_at"),
            "correct": None,
            # Refreshed every run until the game starts, so afterwards this is
            # when the pick and price were locked in (shown on the site).
            "set_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        try:  # the moneyline pick is extra: without odds, the game pick still goes out
            o = odds.get(g["id"])
            if not o and old and old.get("ml"):  # ESPN has no price right now: keep the last one
                o = {"home": old["ml"]["home_ml"], "away": old["ml"]["away_ml"], "book": old["ml"].get("book")}
            ml = moneyline.pick(home, away, pick["home_prob"], o, pick["pick"])
        except Exception as e:
            print(f"  moneyline for {away} @ {home} failed: {e}")
            ml = None
        if ml:
            pick["ml"] = ml
        if old:
            old.clear()
            old.update(pick)
        else:
            history["picks"].append(pick)
        print(f"  {away} @ {home}: {pick['pick']} {pick['prob']}% by {pick['margin']}"
              + (f"; ML {moneyline.text(ml)}" if ml else ""))


league_rank = {}  # team id -> this morning's AdjEM rank


def main():
    weights = load(MODEL_FILE, None)
    history = load(HISTORY_FILE, {"picks": []})
    print(f"CBB Edge run for {TODAY}")
    if in_season(TODAY) or not os.path.exists(SYNC_FILE):
        sync()
    games, box = store.load_all()
    grade(history, {g["id"]: g for g in games})

    if not in_season(TODAY):
        store.roll_up_days()  # the season is over: its day files become one season file
        # Until the next season starts, the Ratings tab shows last season's final ratings.
        if games and load(RATINGS_FILE, {}).get("season") != games[-1]["season"]:
            league = M.League((weights or {}).get("league"), M.division_one(games), store.load_trank())
            for g in games:
                if g["home_pts"] != g["away_pts"]:
                    league.update(g, box.get(g["id"]))
            try:
                write_ratings(league, games[-1]["season"], games, final=True)
            except Exception as e:
                print(f"  ratings failed: {e}")
    elif games or weights:
        league = M.League((weights or {}).get("league"), M.division_one(games), store.load_trank())
        for g in games:
            if g["date"] < TODAY and g["home_pts"] != g["away_pts"]:
                league.update(g, box.get(g["id"]))
        season = season_of(TODAY)
        league._check_season(season)
        # This season's D-I teams so far, plus last season's: a team is D-I from its first game.
        league.di[season] = league.di.get(season, set()) | league.di.get(season - 1, set())
        try:
            write_ratings(league, season, games)
            league_rank.update({row["id"]: row["rank"] for row in M.team_table(league, TODAY)})
        except Exception as e:  # the ratings page is extra: picks still go out
            print(f"  ratings failed: {e}")
        if weights:
            make_picks(history, league, weights)

    history["picks"].sort(key=lambda p: (p["date"], p["start"], p["game_id"]))
    save(HISTORY_FILE, history, indent=1)


if __name__ == "__main__":
    main()
