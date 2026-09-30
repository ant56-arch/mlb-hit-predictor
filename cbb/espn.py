"""
espn.py - men's college basketball schedules, scores and team box scores from
ESPN's public site API. Shared by the season pull (research/pull_seasons.py)
and the daily pipeline (predict.py).

Teams are keyed by ESPN team id everywhere (abbreviations aren't unique across
~360 Division I schools); names and abbreviations are kept for display.
"""

import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import moneyline  # noqa: E402

API = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ET = ZoneInfo("America/New_York")
# ESPN season types: 1 preseason (exhibitions), 2 regular season (conference
# tournaments included), 3 postseason (NCAA tournament, NIT, other events).
GAME_TYPES = {2: "regular", 3: "postseason"}
DIVISION_I = "50"  # ESPN's group id for every Division I conference

# Team box score columns stored per game (see boxscore()).
BOX = ["fgm", "fga", "tpm", "tpa", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "pts"]

session = requests.Session()
session.headers["User-Agent"] = "Mozilla/5.0 (CBB Edge; github.com/ant56-arch/mlb-hit-predictor)"


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


def _team(c):
    t = c.get("team") or {}
    rank = (c.get("curatedRank") or {}).get("current")
    return {
        "id": str(t.get("id", "")),
        "abbr": t.get("abbreviation") or t.get("shortDisplayName") or "",
        "name": t.get("location") or t.get("shortDisplayName") or t.get("displayName") or "",
        "full": t.get("displayName") or "",
        "rank": rank if isinstance(rank, int) and 1 <= rank <= 25 else None,
        "conf": str(t.get("conferenceId") or ""),
    }


def parse_event(ev, day):
    """One scoreboard event -> (a flat game dict, its moneyline odds or None),
    or None if it isn't a regular-season or postseason game."""
    season = ev.get("season", {})
    gtype = GAME_TYPES.get(season.get("type"))
    comp = (ev.get("competitions") or [{}])[0]
    sides = {c.get("homeAway"): c for c in comp.get("competitors", [])}
    if not gtype or set(sides) != {"home", "away"}:
        return None
    home, away = _team(sides["home"]), _team(sides["away"])
    if not home["id"] or not away["id"]:
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

    note = ""
    for n in comp.get("notes") or []:
        if n.get("headline"):
            note = n["headline"]
            break
    g = {
        "id": str(ev["id"]),
        "date": day,
        "start": start.isoformat(),
        "season": season.get("year"),
        "type": gtype,
        "note": note,  # e.g. "Men's Basketball Championship - South Region - 1st Round"
        "home": home["id"], "away": away["id"],
        "home_abbr": home["abbr"], "away_abbr": away["abbr"],
        "home_name": home["name"], "away_name": away["name"],
        "home_rank": home["rank"], "away_rank": away["rank"],
        "home_pts": score(sides["home"]) if final else None,
        "away_pts": score(sides["away"]) if final else None,
        "neutral": bool(comp.get("neutralSite")),
        "state": stype.get("state", ""),  # pre / in / post
        "final": final,
        "postponed": stype.get("name") in ("STATUS_POSTPONED", "STATUS_CANCELED"),
    }
    try:
        odds = moneyline.parse_odds(comp)
    except Exception:
        odds = None
    return g, odds


def scoreboard(day, with_odds=False):
    """Every Division I game on an ET calendar day (YYYY-MM-DD). With
    with_odds, also {game_id: moneyline odds} from the same payload."""
    data = get("scoreboard", dates=day.replace("-", ""), groups=DIVISION_I, limit=500)
    games, odds = [], {}
    for ev in data.get("events", []):
        try:
            parsed = parse_event(ev, day)
        except Exception as e:  # one odd event never costs the rest
            print(f"  skipping event {ev.get('id')}: {e}")
            continue
        if parsed:
            games.append(parsed[0])
            if parsed[1]:
                odds[parsed[0]["id"]] = parsed[1]
    return (games, odds) if with_odds else games


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _made_att(v):
    parts = str(v or "").split("-")
    if len(parts) != 2:
        return None, None
    return _num(parts[0]), _num(parts[1])


# ESPN stat names -> our columns. Made-attempted pairs come as "25-60".
_PAIRS = {
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("tpm", "tpa"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
}
_SINGLES = {
    "offensiveRebounds": "oreb", "defensiveRebounds": "dreb", "assists": "ast", "steals": "stl",
    "blocks": "blk", "turnovers": "tov", "totalTurnovers": "tov", "fouls": "pf", "points": "pts",
}


def _team_totals(stats):
    """A team's box score line from ESPN's boxscore.teams[].statistics."""
    out = {}
    for s in stats or []:
        name, val = s.get("name", ""), s.get("displayValue")
        if name in _PAIRS:
            m, a = _made_att(val)
            if m is not None:
                out[_PAIRS[name][0]], out[_PAIRS[name][1]] = m, a
        elif name in _SINGLES and _SINGLES[name] not in out:
            v = _num(val)
            if v is not None:
                out[_SINGLES[name]] = v
    return out


def _player_totals(team_block):
    """The same line summed over players, for when team totals are missing."""
    out = {k: 0.0 for k in BOX}
    seen = False
    for group in team_block.get("statistics", []):
        keys = group.get("keys") or group.get("names") or []
        for a in group.get("athletes", []):
            row = dict(zip(keys, a.get("stats") or []))
            if not row:
                continue
            seen = True
            for key, (m_col, a_col) in _PAIRS.items():
                m, att = _made_att(row.get(key))
                if m is not None:
                    out[m_col] += m
                    out[a_col] += att
            for key, col in _SINGLES.items():
                if key == "totalTurnovers":
                    continue
                v = _num(row.get(key))
                if v is not None:
                    out[col] += v
    return out if seen else {}


def boxscore(game_id):
    """{team_id: [fgm, fga, tpm, tpa, ftm, fta, oreb, dreb, ast, stl, blk, tov,
    pf, pts]} for both teams, or {} when ESPN has no usable box score."""
    data = get("summary", event=game_id)
    box = data.get("boxscore", {})
    players = {str((b.get("team") or {}).get("id")): b for b in box.get("players", [])}
    out = {}
    for t in box.get("teams", []):
        tid = str((t.get("team") or {}).get("id"))
        line = _team_totals(t.get("statistics"))
        if not all(k in line for k in ("fga", "fta", "oreb", "tov")) and tid in players:
            line = {**_player_totals(players[tid]), **line}
        if all(k in line for k in ("fgm", "fga", "fta", "oreb", "tov")):
            if "pts" not in line:
                line["pts"] = 2 * line["fgm"] + line.get("tpm", 0) + line.get("ftm", 0)
            out[tid] = [round(line.get(k, 0.0)) for k in BOX]
    return out if len(out) == 2 else {}
