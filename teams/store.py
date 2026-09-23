"""
store.py - where the MLB team model keeps final games (with starting
pitchers from the box score, see statsapi.stored).

  teams/data/seasons/<year>.json  finished seasons, written by the research
                                  pull and by roll_up_days()
  teams/data/days/<date>.json     each day of the current season, written by
                                  the daily run as its games go final, and
                                  rolled up into its season file once the
                                  season is over

Both hold {"games": [...]}. One small file per day keeps the daily commits small.
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEASONS_DIR = os.path.join(DATA, "seasons")
DAYS_DIR = os.path.join(DATA, "days")


def _read(path):
    with open(path) as f:
        return json.load(f)


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


def load_all():
    """Every stored final game in first-pitch order."""
    games = {}
    for path in sorted(glob.glob(os.path.join(SEASONS_DIR, "*.json"))) + sorted(glob.glob(os.path.join(DAYS_DIR, "*.json"))):
        for g in _read(path)["games"]:
            games[g["id"]] = g
    return sorted(games.values(), key=lambda g: (g["start"], g["id"]))


def save_season(season, games):
    _write(os.path.join(SEASONS_DIR, f"{season}.json"), {"games": games})


def save_day(day, games):
    _write(os.path.join(DAYS_DIR, f"{day}.json"), {"games": games})


def roll_up_days():
    """Merges every day file into its season's file and deletes it. Run in the
    offseason, so a finished season lives in one file and the next season's
    days start from an empty folder."""
    paths = sorted(glob.glob(os.path.join(DAYS_DIR, "*.json")))
    if not paths:
        return
    by_season = {}
    for path in paths:
        for g in _read(path)["games"]:
            by_season.setdefault(g["season"], {})[g["id"]] = g
    for season, games in by_season.items():
        path = os.path.join(SEASONS_DIR, f"{season}.json")
        if os.path.exists(path):
            games = {**{g["id"]: g for g in _read(path)["games"]}, **games}
        save_season(season, sorted(games.values(), key=lambda g: (g["start"], g["id"])))
        print(f"  rolled {len(games)} games into seasons/{season}.json")
    for path in paths:
        os.remove(path)
