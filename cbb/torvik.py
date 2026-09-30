"""
torvik.py - Bart Torvik's T-Rank team ratings (barttorvik.com), shown next to
our own ratings on the Ratings tab.

T-Rank is free and built the same way as KenPom: adjusted offense, defense
and tempo from every game, weighting recent games more, plus Barthag (the
chance of beating an average team on a neutral court). It's matched to ESPN's
teams by name, and used two ways:

  - fetch(season): today's ratings, for the Ratings tab and a daily snapshot
    (cbb/data/trank/<season>.json) the model can learn from
  - fetch_day(day): T-Rank as it stood on a past day, from Torvik's time
    machine, to backfill those snapshots for past seasons

Every step fails soft: no T-Rank means a dash on the site, and the model
falls back on our own ratings alone.
"""

import csv
import gzip
import io
import json
import re

import requests

URL = "https://barttorvik.com/{year}_team_results.csv"
TIME_MACHINE = "https://barttorvik.com/timemachine/team_results/{ymd}_team_results.json.gz"
HEADERS = {"User-Agent": "Mozilla/5.0 (CBB Edge; github.com/ant56-arch/mlb-hit-predictor)"}

# Torvik's names for schools ESPN names differently (after normalize()).
ALIASES = {
    "connecticut": "uconn", "miami fl": "miami", "miami oh": "miami oh", "nc st": "nc st",
    "mississippi": "ole miss", "illinois chicago": "uic",
    "louisiana lafayette": "louisiana", "texas a&m corpus chris": "texas a&m corpus christi",
    "southeast missouri st": "se missouri st", "arkansas pine bluff": "arkansas pine bluff",
    "cal st bakersfield": "cal st bakersfield", "umass lowell": "umass lowell",
    "fiu": "florida international", "liu": "long island university", "usc upstate": "south carolina upstate",
    "nebraska omaha": "omaha", "tennessee martin": "ut martin", "texas rio grande valley": "ut rio grande valley",
    "st thomas": "st thomas mn", "queens": "queens university", "detroit": "detroit mercy",
    "loyola md": "loyola maryland", "saint francis": "st francis pa", "mount st mary's": "mount st marys",
}


def normalize(name):
    s = name.lower().replace("&amp;", "&")
    s = re.sub(r"[.'()]", "", s)
    s = s.replace("-", " ")
    s = re.sub(r"\bsaint\b", "st", s)
    s = re.sub(r"\bstate\b", "st", s)
    s = re.sub(r"\bn\.?c\.?\b", "nc", s)
    s = re.sub(r"\s+", " ", s).strip()
    return ALIASES.get(s, s)


def _col(header, *names):
    low = [h.strip().lower() for h in header]
    for n in names:
        if n in low:
            return low.index(n)
    return None


def fetch(season):
    """{normalized team name: {"rank", "adj_o", "adj_d", "barthag", "adj_t"}}
    for a season (2027 = 2026-27), or {} when T-Rank can't be reached."""
    try:
        r = requests.get(URL.format(year=season), timeout=30, headers=HEADERS)
        r.raise_for_status()
        rows = list(csv.reader(io.StringIO(r.text)))
    except Exception as e:
        print(f"  T-Rank unavailable: {e}")
        return {}
    if len(rows) < 2:
        return {}
    header = rows[0]
    cols = {k: _col(header, *names) for k, names in {
        "team": ("team",), "rank": ("rank", "rk"), "adj_o": ("adjoe",), "adj_d": ("adjde",),
        "barthag": ("barthag",), "adj_t": ("adjt", "adj t.", "adj. t"),
    }.items()}
    if cols["team"] is None or cols["adj_o"] is None or cols["adj_d"] is None:
        print(f"  T-Rank columns not recognized: {header[:10]}")
        return {}
    out = {}
    for i, row in enumerate(rows[1:], 1):
        def num(k):
            c = cols[k]
            try:
                return float(row[c]) if c is not None else None
            except (ValueError, IndexError):
                return None
        try:
            name = row[cols["team"]]
        except IndexError:
            continue
        rank = num("rank")
        out[normalize(name)] = {"rank": int(rank) if rank else i, "adj_o": num("adj_o"), "adj_d": num("adj_d"),
                                "barthag": num("barthag"), "adj_t": num("adj_t")}
    print(f"  T-Rank: {len(out)} teams")
    return out


def fetch_day(day):
    """T-Rank as of a past day (YYYY-MM-DD) from Torvik's time machine, in
    fetch()'s shape, or {}. Rows there have no header; they follow the
    season file's column order (rank, team, conf, record, adjoe, adjoe rank,
    adjde, adjde rank, barthag, ...), so each row is sanity-checked."""
    try:
        r = requests.get(TIME_MACHINE.format(ymd=day.replace("-", "")), timeout=30, headers=HEADERS)
        r.raise_for_status()
        try:
            rows = json.loads(gzip.decompress(r.content))
        except OSError:  # already decompressed in transit
            rows = r.json()
    except Exception as e:
        print(f"  T-Rank time machine {day} unavailable: {e}")
        return {}
    out = {}
    for i, row in enumerate(rows if isinstance(rows, list) else [], 1):
        try:
            name, adj_o, adj_d, barthag = str(row[1]), float(row[4]), float(row[6]), float(row[8])
        except (TypeError, ValueError, IndexError):
            continue
        if not (70 <= adj_o <= 140 and 70 <= adj_d <= 140 and 0 <= barthag <= 1):
            continue
        out[normalize(name)] = {"rank": i, "adj_o": adj_o, "adj_d": adj_d, "barthag": barthag, "adj_t": None}
    return out


def by_team_id(trank, name_to_id):
    """{ESPN team id: [adj_o, adj_d, barthag]} for the teams we can match."""
    out = {}
    for name, t in (trank or {}).items():
        tid = name_to_id.get(name)
        if tid and t.get("adj_o") is not None and t.get("adj_d") is not None:
            out[tid] = [round(t["adj_o"], 2), round(t["adj_d"], 2), t.get("barthag")]
    return out


def attach(table, trank):
    """Adds each team's T-Rank line to the ratings table (in place); returns how many matched."""
    matched = 0
    for row in table:
        t = trank.get(normalize(row["name"])) if trank else None
        row["trank"] = t
        matched += bool(t)
    return matched
