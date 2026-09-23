"""
pull_seasons.py - pulls every regular-season, play-in and playoff game of the
given NBA seasons from ESPN, with box scores, for training and backtesting.

SEASONS env var: comma-separated season end years (2026 = the 2025-26
season). Default: the three seasons before the current one. Writes
nba/research/games.json and nba/research/boxscores.json.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import espn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")


def default_seasons():
    today = date.today()
    latest_done = today.year if today.month >= 7 else today.year - 1
    return [latest_done - 2, latest_done - 1, latest_done]


def season_days(end_year):
    d, last = date(end_year - 1, 10, 1), date(end_year, 6, 30)
    while d <= last:
        yield d.isoformat()
        d += timedelta(days=1)


def main():
    seasons = [int(s) for s in os.environ.get("SEASONS", "").split(",") if s.strip()] or default_seasons()
    print(f"Pulling seasons {seasons}")
    os.makedirs(SAMPLES, exist_ok=True)

    # Raw samples, so parsing can be checked against what ESPN actually sends.
    sample_day = f"{seasons[-1] - 1}-12-01"
    with open(os.path.join(SAMPLES, "scoreboard.json"), "w") as f:
        json.dump(espn.get("scoreboard", dates=sample_day.replace("-", "")), f)
    try:
        with open(os.path.join(SAMPLES, "injuries.json"), "w") as f:
            json.dump(espn.get("injuries"), f)
    except Exception as e:  # the injury feed is only a sample here
        print(f"  injuries sample failed: {e}")

    days = [d for s in seasons for d in season_days(s)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        per_day = list(pool.map(espn.scoreboard, days))
    games = {}
    for day_games in per_day:
        for g in day_games:
            if g["final"] and g["home_pts"] is not None:
                games[g["id"]] = g
    games = sorted(games.values(), key=lambda g: (g["start"], g["id"]))
    print(f"{len(games)} final games")
    for s in seasons:
        n = sum(1 for g in games if g["season"] == s and g["type"] == "regular")
        print(f"  {s}: {n} regular-season games")

    first = games[0]["id"] if games else None
    if first:
        with open(os.path.join(SAMPLES, "summary.json"), "w") as f:
            json.dump(espn.get("summary", event=first), f)

    def box(g):
        try:
            return g["id"], espn.boxscore(g["id"])
        except Exception as e:
            print(f"  box score {g['id']} failed: {e}")
            return g["id"], None

    with ThreadPoolExecutor(max_workers=8) as pool:
        boxes = {gid: b for gid, b in pool.map(box, games) if b}
    print(f"{len(boxes)} box scores")

    with open(os.path.join(HERE, "games.json"), "w") as f:
        json.dump(games, f, separators=(",", ":"))
    with open(os.path.join(HERE, "boxscores.json"), "w") as f:
        json.dump(boxes, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
