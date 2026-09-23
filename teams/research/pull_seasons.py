"""
pull_seasons.py - pulls every regular-season and postseason game of the given
MLB seasons from the MLB Stats API, with each team's starting pitcher line
from the box score, for training and backtesting the team model.

SEASONS env var: comma-separated years. Default: the last three finished
seasons plus the one in progress. A season in progress is pulled through
yesterday into teams/data/seasons/<year>.json; the daily run then adds each
new day under teams/data/days/ (see store.py) and writes teams/data/sync.json.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import statsapi  # noqa: E402
import store  # noqa: E402


def default_seasons():
    y = date.today().year
    return [y - 3, y - 2, y - 1, y]


def months(season, last):
    d = date(season, 3, 1)
    end = min(date(season, 11, 30), last)
    while d <= end:
        nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield d.isoformat(), min(nxt - timedelta(days=1), end).isoformat()
        d = nxt


def main():
    seasons = [int(s) for s in os.environ.get("SEASONS", "").split(",") if s.strip()] or default_seasons()
    yesterday = date.today() - timedelta(days=1)
    print(f"Pulling seasons {seasons} through {yesterday}")
    ranges = [r for s in seasons for r in months(s, yesterday)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        per_month = list(pool.map(lambda r: statsapi.schedule(*r), ranges))
    games = {}
    for month in per_month:
        for g in month:
            if g["final"] and g["home_runs"] is not None and g["season"] in seasons:
                games[g["id"]] = g
    games = sorted(games.values(), key=lambda g: (g["start"], g["id"]))
    print(f"{len(games)} final games")

    def box(g):
        try:
            return g["id"], statsapi.boxscore(g["id"])
        except Exception as e:  # the game still counts without its starters
            print(f"  box score {g['id']} failed: {e}")
            return g["id"], {}

    with ThreadPoolExecutor(max_workers=16) as pool:
        boxes = dict(pool.map(box, games))
    print(f"{sum(1 for b in boxes.values() if b)} box scores")

    for s in seasons:
        sg = [statsapi.stored(g, boxes.get(g["id"])) for g in games if g["season"] == s]
        reg = sum(1 for g in sg if g["type"] == "regular")
        print(f"  {s}: {reg} regular-season, {len(sg) - reg} postseason games")
        store.save_season(s, sg)
    # The daily run picks up from here.
    os.makedirs(store.DATA, exist_ok=True)
    with open(os.path.join(store.DATA, "sync.json"), "w") as f:
        json.dump({"through": yesterday.isoformat()}, f, indent=1)


if __name__ == "__main__":
    main()
