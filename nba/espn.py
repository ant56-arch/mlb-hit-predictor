"""
espn.py - NBA schedule, scores, box scores and injury reports from ESPN's
public site API. Shared by the season pull (research/pull_seasons.py) and the
daily pipeline (predict.py).
"""

import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

API = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ET = ZoneInfo("America/New_York")
# ESPN season types: 1 preseason, 2 regular season, 3 postseason, 5 play-in.
GAME_TYPES = {2: "regular", 3: "playoffs", 5: "play-in"}

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (NBA Edge; github.com/ant56-arch/mlb-hit-predictor)"


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


def is_franchise(team):
    """True for the 30 NBA teams (ESPN ids 1-30), which leaves out All-Star and
    exhibition opponents."""
    try:
        return 1 <= int(team.get("id", 0)) <= 30
    except ValueError:
        return False


def parse_event(ev, day):
    """One scoreboard event -> a flat game dict, or None if it isn't a real
    regular-season or postseason NBA game."""
    season = ev.get("season", {})
    gtype = GAME_TYPES.get(season.get("type"))
    comp = (ev.get("competitions") or [{}])[0]
    teams = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    if not gtype or set(teams) != {"home", "away"}:
        return None
    home, away = teams["home"], teams["away"]
    if not (is_franchise(home.get("team", {})) and is_franchise(away.get("team", {}))):
        return None
    status = comp.get("status") or ev.get("status") or {}
    stype = status.get("type", {})
    start = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone(ET)
    final = bool(stype.get("completed"))

    def score(c):
        try:
            return int(float(c.get("score")))
        except (TypeError, ValueError):
            return None

    return {
        "id": ev["id"],
        "date": day,
        "start": start.isoformat(),
        "season": season.get("year"),
        "type": gtype,
        "home": home["team"]["abbreviation"],
        "away": away["team"]["abbreviation"],
        "home_id": home["team"]["id"],
        "away_id": away["team"]["id"],
        "home_name": home["team"].get("displayName", ""),
        "away_name": away["team"].get("displayName", ""),
        "home_pts": score(home) if final else None,
        "away_pts": score(away) if final else None,
        "neutral": bool(comp.get("neutralSite")),
        "state": stype.get("state", ""),  # pre / in / post
        "final": final,
        "postponed": stype.get("name") in ("STATUS_POSTPONED", "STATUS_CANCELED"),
    }


def scoreboard(day):
    """Every NBA game on an ET calendar day (YYYY-MM-DD)."""
    data = get("scoreboard", dates=day.replace("-", ""), limit=100)
    games = []
    for ev in data.get("events", []):
        g = parse_event(ev, day)
        if g:
            games.append(g)
    return games


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def boxscore(game_id):
    """Who played in a game and what they did.

    Returns {team_abbr: {"players": [[athlete_id, name, min, pts, reb, ast, stl,
    blk, tov, fgm, fga, ftm, fta, plus_minus], ...], "dnp": [[athlete_id, name,
    reason], ...]}}.
    """
    data = get("summary", event=game_id)
    out = {}
    for team_block in data.get("boxscore", {}).get("players", []):
        abbr = team_block.get("team", {}).get("abbreviation")
        if not abbr:
            continue
        players, dnp = [], []
        for stat_group in team_block.get("statistics", []):
            keys = stat_group.get("keys") or stat_group.get("names") or []
            for a in stat_group.get("athletes", []):
                ath = a.get("athlete", {})
                aid, name = ath.get("id"), ath.get("displayName", "")
                stats = a.get("stats") or []
                if a.get("didNotPlay") or not stats:
                    dnp.append([aid, name, a.get("reason", "")])
                    continue
                row = dict(zip(keys, stats))

                def made_att(key):
                    parts = str(row.get(key, "0-0")).split("-")
                    return (_num(parts[0]), _num(parts[1])) if len(parts) == 2 else (0.0, 0.0)

                fgm, fga = made_att("fieldGoalsMade-fieldGoalsAttempted")
                ftm, fta = made_att("freeThrowsMade-freeThrowsAttempted")
                players.append([
                    aid, name, _num(row.get("minutes")), _num(row.get("points")), _num(row.get("rebounds")),
                    _num(row.get("assists")), _num(row.get("steals")), _num(row.get("blocks")),
                    _num(row.get("turnovers")), fgm, fga, ftm, fta, _num(row.get("plusMinus")),
                ])
        out[abbr] = {"players": players, "dnp": dnp}
    return out


def injuries():
    """Current injury report: {team_id: {athlete_id: status}}, where status is
    ESPN's label (Out, Doubtful, Questionable, Day-To-Day)."""
    data = get("injuries")
    out = {}
    for team in data.get("injuries", []):
        report = {}
        for inj in team.get("injuries", []):
            for link in inj.get("athlete", {}).get("links", []):
                m = re.search(r"/id/(\d+)", link.get("href", ""))
                if m:
                    report[m.group(1)] = inj.get("status", "")
                    break
        out[str(team.get("id"))] = report
    return out
