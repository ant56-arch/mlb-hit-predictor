"""
statsapi.py - MLB schedules, scores and starting pitchers from the MLB Stats
API (statsapi.mlb.com), for the team game-winner model. Shared by the season
pull (research/pull_seasons.py) and the daily run (predict.py).

A stored game is a flat dict:
  id, date (official date), start (ET), season, type ("regular" or
  "postseason"), home/away (abbreviation), home_id/away_id, home_name/away_name,
  home_runs/away_runs, and for final games home_sp/away_sp and home_pen/away_pen
  from the box score:
    sp  = [pitcher id, name, outs, earned runs, hits, walks, strikeouts, home runs]
    pen = [outs, earned runs] for everyone after the starter
"""

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

API = "https://statsapi.mlb.com/api/v1"
ET = ZoneInfo("America/New_York")
# R regular season, F wild card, D division series, L league championship, W World Series.
GAME_TYPES = {"R": "regular", "F": "postseason", "D": "postseason", "L": "postseason", "W": "postseason"}
ROUND_NAMES = {"F": "Wild Card", "D": "Division Series", "L": "LCS", "W": "World Series"}
DONE_STATES = {"Postponed", "Cancelled", "Suspended"}  # never going to be final on this date

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (MLB Edge; github.com/ant56-arch/mlb-hit-predictor)"


def get(path, **params):
    for attempt in range(5):
        try:
            r = session.get(f"{API}/{path}", params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 4:
                raise
            print(f"  retrying {path}: {e}")
            time.sleep(2 ** attempt)


def parse_game(g):
    """One schedule game -> a flat game dict, or None if it isn't a regular-season
    or postseason MLB game."""
    gtype = GAME_TYPES.get(g.get("gameType"))
    if not gtype:
        return None
    status = g.get("status", {})
    state = status.get("detailedState", "")
    abstract = status.get("abstractGameState", "")
    teams = g["teams"]
    final = abstract == "Final" and not any(state.startswith(s) for s in DONE_STATES)
    start = datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00")).astimezone(ET)
    out = {
        "id": str(g["gamePk"]),
        "date": g.get("officialDate") or start.date().isoformat(),
        "start": start.isoformat(),
        "season": int(g.get("season") or start.year),
        "type": gtype,
        "round": ROUND_NAMES.get(g.get("gameType")),
        "final": final,
        "postponed": any(state.startswith(s) for s in DONE_STATES),
        "state": "pre" if abstract == "Preview" else ("post" if abstract == "Final" else "in"),
        "status": state,
        "doubleheader": g.get("gameNumber", 1) if g.get("doubleHeader") in ("Y", "S") else 0,
    }
    for side in ("home", "away"):
        t = teams[side]
        team = t.get("team", {})
        pp = t.get("probablePitcher") or {}
        rec = t.get("leagueRecord") or {}
        out[side] = team.get("abbreviation") or team.get("name", "")[:3].upper()
        out[f"{side}_id"] = team.get("id")
        out[f"{side}_name"] = team.get("name", "")
        out[f"{side}_runs"] = t.get("score") if final else None
        out[f"{side}_probable"] = [pp["id"], pp.get("fullName", "")] if pp.get("id") else None
        out[f"{side}_record"] = f"{rec.get('wins', 0)}-{rec.get('losses', 0)}" if rec else None
    return out


def schedule(start, end=None):
    """Every regular-season and postseason game from start to end (YYYY-MM-DD)."""
    data = get("schedule", sportId=1, startDate=start, endDate=end or start, gameType="R,F,D,L,W",
               hydrate="probablePitcher,team")
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            p = parse_game(g)
            if p:
                games.append(p)
    return games


def _outs(ip):
    whole, _, frac = str(ip or "0").partition(".")
    return int(whole or 0) * 3 + int(frac or 0)


def boxscore(game_id):
    """{"home_sp", "away_sp", "home_pen", "away_pen"} from a final game's box score."""
    box = get(f"game/{game_id}/boxscore")
    out = {}
    for side in ("home", "away"):
        t = box["teams"][side]
        pitchers = t.get("pitchers") or []
        if not pitchers:
            continue
        lines = []
        for pid in pitchers:
            p = t["players"].get(f"ID{pid}", {})
            s = p.get("stats", {}).get("pitching", {})
            lines.append([pid, p.get("person", {}).get("fullName", ""), _outs(s.get("inningsPitched")),
                          s.get("earnedRuns", 0), s.get("hits", 0), s.get("baseOnBalls", 0),
                          s.get("strikeOuts", 0), s.get("homeRuns", 0)])
        out[f"{side}_sp"] = lines[0]
        out[f"{side}_pen"] = [sum(x[2] for x in lines[1:]), sum(x[3] for x in lines[1:])]
    return out


STORED_KEYS = ["id", "date", "start", "season", "type", "round", "doubleheader", "home", "away", "home_id",
               "away_id", "home_name", "away_name", "home_runs", "away_runs", "home_probable", "away_probable"]


def stored(game, box):
    """The compact record kept in teams/data for a final game."""
    out = {k: game.get(k) for k in STORED_KEYS}
    out.update(box or {})
    return out
