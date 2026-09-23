"""
pull_season_data.py - builds the training set for research/train_model.py.

One row per hitter per game he batted in, for every hitter with a regular's
share of at-bats (not just the league leaders, which would bias the model
toward players already known to have had good seasons).

Every opposing-pitcher stat is as of the morning of that game, computed from
the pitcher's own game log, so the model never trains on numbers it couldn't
have known at the time. Batter to-date stats are computed in train_model.py
from these same rows.

Pulls the last SEASONS_BACK seasons (the season in progress, or the latest
finished one in the offseason, plus the ones before it) into
research/data/<season>.json.gz. A finished season is pulled once and reused;
the season in progress is pulled fresh every run. The weekly "Research Data
Pull + Retrain" workflow runs this, then train_model.py.

Env vars: SEASONS (comma-separated years, overrides the default list) and
CUTOFF_DATE (YYYY-MM-DD, last date to include).
"""

import gzip
import json
import os
import sys
import time
from datetime import date, timedelta

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from features import innings_to_float  # noqa: E402

API = "https://statsapi.mlb.com/api/v1"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SEASONS_BACK = 3
FULL_SEASON_MIN_AB = 150  # a regular over a full season; scaled down for a season in progress
TODAY = date.fromisoformat(os.environ.get("CUTOFF_DATE") or (date.today() - timedelta(days=1)).isoformat())

session = requests.Session()


def get(path, **params):
    for attempt in range(4):
        try:
            r = session.get(f"{API}/{path}", params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 3:
                raise
            print(f"  retrying {path}: {e}")
            time.sleep(2 ** attempt)


def season_dates(season):
    """(regular season start, end) from the MLB schedule, with a fallback for
    a season whose dates aren't published yet."""
    try:
        s = get(f"seasons/{season}", sportId=1)["seasons"][0]
        return date.fromisoformat(s["regularSeasonStartDate"]), date.fromisoformat(s["regularSeasonEndDate"])
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return date(season, 3, 25), date(season, 10, 1)


def default_seasons():
    start, _ = season_dates(TODAY.year)
    latest = TODAY.year if TODAY >= start else TODAY.year - 1
    return list(range(latest - SEASONS_BACK + 1, latest + 1))


def season_path(season):
    return os.path.join(DATA_DIR, f"{season}.json.gz")


def load_season(season):
    path = season_path(season)
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt") as f:
        return json.load(f)


def season_games(season, cutoff):
    """gamePk -> venue + both starters, for every completed regular-season game."""
    data = get("schedule", sportId=1, season=season, gameType="R",
               startDate=f"{season}-01-01", endDate=cutoff.isoformat(), hydrate="probablePitcher,venue")
    games = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            teams = g["teams"]
            games[g["gamePk"]] = {
                "venue": g.get("venue", {}).get("name", ""),
                "starter": {
                    teams["home"]["team"]["id"]: teams["home"].get("probablePitcher", {}).get("id"),
                    teams["away"]["team"]["id"]: teams["away"].get("probablePitcher", {}).get("id"),
                },
            }
    return games


def people_hands(ids):
    hands = {}
    ids = list(ids)
    for i in range(0, len(ids), 100):
        chunk = ",".join(str(x) for x in ids[i:i + 100])
        for p in get("people", personIds=chunk).get("people", []):
            hands[p["id"]] = {
                "bat": p.get("batSide", {}).get("code"),
                "pitch": p.get("pitchHand", {}).get("code"),
            }
    return hands


def game_log(player_id, group, season):
    data = get(f"people/{player_id}/stats", stats="gameLog", group=group, season=season)
    stats = data.get("stats", [])
    return stats[0].get("splits", []) if stats else []


def pitcher_to_date(splits):
    """date -> season-to-date line (before that date) for one pitcher."""
    splits = sorted(splits, key=lambda s: s["date"])
    by_date = {}
    er = ip = baserunners = k = 0.0
    for s in splits:
        if s["date"] not in by_date:
            by_date[s["date"]] = (
                {"era": er * 9 / ip, "whip": baserunners / ip, "k9": k * 9 / ip, "ip": ip}
                if ip > 0 else {"era": None, "whip": None, "k9": None, "ip": 0.0}
            )
        st = s.get("stat", {})
        er += int(st.get("earnedRuns", 0))
        ip += innings_to_float(st.get("inningsPitched"))
        baserunners += int(st.get("hits", 0)) + int(st.get("baseOnBalls", 0))
        k += int(st.get("strikeOuts", 0))
    return by_date


def pull_season(season):
    start, end = season_dates(season)
    cutoff = min(TODAY, end)
    complete = TODAY >= end
    print(f"Pulling {season} through {cutoff}{' (complete)' if complete else ' (in progress)'}...")
    games = season_games(season, cutoff)
    print(f"  {len(games)} completed games")

    # A regular's share of at-bats so far: 150 over a full season, fewer early on.
    share = max(0.0, min(1.0, (cutoff - start).days / max((end - start).days, 1)))
    min_ab = max(10, round(FULL_SEASON_MIN_AB * share))
    splits = get("stats", stats="season", group="hitting", season=season,
                 playerPool="ALL", limit=3000)["stats"][0]["splits"]
    hitters = {s["player"]["id"]: s["player"]["fullName"] for s in splits
               if int(s["stat"].get("atBats", 0)) >= min_ab}
    print(f"  {len(hitters)} hitters with {min_ab}+ AB")

    starters = {pid for g in games.values() for pid in g["starter"].values() if pid}
    hands = people_hands(set(hitters) | starters)

    print(f"  pulling game logs for {len(starters)} starting pitchers...")
    pitcher_lines = {}
    for i, pid in enumerate(starters, 1):
        pitcher_lines[pid] = pitcher_to_date(game_log(pid, "pitching", season))
        if i % 100 == 0:
            print(f"    {i}/{len(starters)}")
        time.sleep(0.05)

    print(f"  pulling game logs for {len(hitters)} hitters...")
    rows = []
    missing_starter = 0
    for i, (pid, name) in enumerate(hitters.items(), 1):
        for s in game_log(pid, "hitting", season):
            st = s.get("stat", {})
            ab = int(st.get("atBats", 0))
            game = games.get(s.get("game", {}).get("gamePk"))
            if ab == 0 or game is None:
                continue
            opp_id = s.get("opponent", {}).get("id")
            starter = game["starter"].get(opp_id)
            if not starter:
                missing_starter += 1
            line = pitcher_lines.get(starter, {}).get(s["date"], {})
            hits = int(st.get("hits", 0))
            rows.append({
                "player_id": pid,
                "player_name": name,
                "date": s["date"],
                "game_pk": s["game"]["gamePk"],
                "got_hit": hits > 0,
                "hits": hits,
                "at_bats": ab,
                "batter_hand": hands.get(pid, {}).get("bat"),
                "is_home": s.get("isHome"),
                "venue": game["venue"],
                "opp_pitcher_id": starter,
                "opp_pitcher_hand": hands.get(starter, {}).get("pitch"),
                "opp_pitcher_era": line.get("era"),
                "opp_pitcher_whip": line.get("whip"),
                "opp_pitcher_k9": line.get("k9"),
                "opp_pitcher_ip": line.get("ip", 0.0),
            })
        if i % 100 == 0:
            print(f"    {i}/{len(hitters)}")
        time.sleep(0.05)

    os.makedirs(DATA_DIR, exist_ok=True)
    with gzip.open(season_path(season), "wt") as f:
        json.dump({"season": season, "through": cutoff.isoformat(), "complete": complete, "rows": rows}, f)
    print(f"  saved {len(rows)} rows ({missing_starter} with no listed starter)")


def main():
    seasons = [int(s) for s in os.environ.get("SEASONS", "").split(",") if s.strip()] or default_seasons()
    print(f"Seasons: {seasons}")
    for season in seasons:
        have = load_season(season)
        if have and have.get("complete"):
            print(f"{season}: already have the full season ({len(have['rows'])} rows)")
            continue
        pull_season(season)


if __name__ == "__main__":
    main()
