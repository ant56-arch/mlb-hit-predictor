"""
store.py - where NBA Edge keeps game results and box scores.

  nba/data/seasons/<year>.json  finished seasons, written by the research pull
  nba/data/days/<date>.json     each day of the current season, written by the
                                daily run as its games go final

Both hold {"games": [...], "box": {game_id: box score}}. One small file per day
keeps the daily commits small.
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
    """Every stored final game in tip-off order, and box scores by game id."""
    games, box = {}, {}
    for path in sorted(glob.glob(os.path.join(SEASONS_DIR, "*.json"))) + sorted(glob.glob(os.path.join(DAYS_DIR, "*.json"))):
        d = _read(path)
        for g in d["games"]:
            games[g["id"]] = g
        box.update(d.get("box", {}))
    return sorted(games.values(), key=lambda g: (g["start"], g["id"])), box


def stored_days():
    return {os.path.basename(p)[:-5] for p in glob.glob(os.path.join(DAYS_DIR, "*.json"))}


def save_season(season, games, box):
    _write(os.path.join(SEASONS_DIR, f"{season}.json"), {"games": games, "box": box})


def save_day(day, games, box):
    _write(os.path.join(DAYS_DIR, f"{day}.json"), {"games": games, "box": box})
