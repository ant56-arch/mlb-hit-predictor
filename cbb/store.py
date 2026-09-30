"""
store.py - where CBB Edge keeps game results and team box scores.

  cbb/data/seasons/<year>.json  finished seasons, written by the research pull
  cbb/data/days/<date>.json     each day of the current season, written by the
                                daily run as its games go final, and rolled up
                                into its season file once the season is over

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


def roll_up_days():
    """Merges every day file into its season's file and deletes it. Run in the
    offseason, so a finished season lives in one file and the next season's
    days start from an empty folder."""
    paths = sorted(glob.glob(os.path.join(DAYS_DIR, "*.json")))
    if not paths:
        return
    by_season = {}
    for path in paths:
        d = _read(path)
        for g in d["games"]:
            games, box = by_season.setdefault(g["season"], ({}, {}))
            games[g["id"]] = g
            if g["id"] in d.get("box", {}):
                box[g["id"]] = d["box"][g["id"]]
    for season, (games, box) in by_season.items():
        path = os.path.join(SEASONS_DIR, f"{season}.json")
        if os.path.exists(path):
            old = _read(path)
            games = {**{g["id"]: g for g in old["games"]}, **games}
            box = {**old.get("box", {}), **box}
        ordered = sorted(games.values(), key=lambda g: (g["start"], g["id"]))
        save_season(season, ordered, {gid: box[gid] for gid in games if gid in box})
        print(f"  rolled {len(games)} games into seasons/{season}.json")
    for path in paths:
        os.remove(path)


# ── T-Rank snapshots ─────────────────────────────────────────────────────────
# cbb/data/trank/<season>.json: {date: {team_id: [adj_o, adj_d, barthag]}},
# T-Rank as it stood each morning (see torvik.py).
TRANK_DIR = os.path.join(DATA, "trank")


def load_trank():
    out = {}
    for path in sorted(glob.glob(os.path.join(TRANK_DIR, "*.json"))):
        out.update(_read(path))
    return out


def save_trank(season, day, snapshot):
    path = os.path.join(TRANK_DIR, f"{season}.json")
    data = _read(path) if os.path.exists(path) else {}
    data[day] = snapshot
    _write(path, dict(sorted(data.items())))


def name_to_id(games, normalize):
    """{normalized ESPN school name: team id} from stored games."""
    out = {}
    for g in games:
        out[normalize(g["home_name"])] = g["home"]
        out[normalize(g["away_name"])] = g["away"]
    return out
