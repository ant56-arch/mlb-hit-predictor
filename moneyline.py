"""
moneyline.py - moneyline picks for the MLB team model (teams/) and NBA Edge
(nba/), from the book prices on ESPN's public scoreboard.

The same definition runs on every Edge site (NFL, CFB, MLB, NBA):
  - Both sides' American odds become implied probabilities, and the vig is
    removed by normalizing the two to sum to 1.
  - The pick is the side where our model's win probability beats that no-vig
    book probability by the most (the edge). It's flagged Value when the edge
    is at least 6 percentage points; otherwise it's still the pick.
  - Graded at 1 unit risked at the book price: a win at +135 is +1.35u, a win
    at -150 is +0.667u, a loss is -1u. A postponed or cancelled game is no
    decision.
  - The record counts this season's live picks only, never a backtest.

The odds and pick are refreshed with the model's pick and lock with it at
first pitch or tip-off. Every step fails soft: no odds means no moneyline
pick, and never touches the model's own pick.
"""

from datetime import datetime

import requests

ESPN = "https://site.api.espn.com/apis/site/v2/sports"
VALUE_EDGE = 6.0  # percentage points of edge that make a pick Value

# ESPN and the MLB Stats API abbreviate a few MLB teams differently.
MLB_ALIASES = {"ARI": "AZ", "CHW": "CWS", "WAS": "WSH", "OAK": "ATH", "KCR": "KC", "SDP": "SD",
               "SFG": "SF", "TBR": "TB"}


# ── Odds ─────────────────────────────────────────────────────────────────────
def american(x):
    """American odds from ESPN (130, "-150", "+130", "EVEN") as an int, or None
    when there's no usable price ("OFF", "", None, nonsense)."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, str):
        s = x.strip().upper()
        if s in ("EVEN", "EV", "PK"):
            return 100
        try:
            x = float(s.replace("+", ""))
        except ValueError:
            return None
    try:
        v = int(round(float(x)))
    except (TypeError, ValueError):
        return None
    return v if abs(v) >= 100 else None


def implied(odds):
    """Implied win probability (0-1) of American odds, vig included."""
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def no_vig(home_ml, away_ml):
    """(home, away) implied probabilities with the vig removed, summing to 1."""
    h, a = implied(home_ml), implied(away_ml)
    return h / (h + a), a / (h + a)


def _line(block):
    """A price out of a newer-payload moneyline side, {"close": {"odds": "-155"},
    "open": {...}} (sometimes "current" too). The latest line wins."""
    if not isinstance(block, dict):
        return None
    for key in ("close", "current", "open"):
        line = block.get(key)
        v = american(line.get("odds")) if isinstance(line, dict) else None
        if v is not None:
            return v
    return american(block.get("odds"))


def parse_odds(comp):
    """Moneyline odds for a scoreboard competition, {"home": -155, "away": 130,
    "book": "ESPN BET"}, or None. Handles both payload shapes:
      odds[].homeTeamOdds.moneyLine / awayTeamOdds.moneyLine   (older)
      odds[].moneyline.home.close.odds / .away.close.odds      (newer)
    and takes the highest-priority provider that has both sides."""
    entries = [o for o in (comp or {}).get("odds") or [] if isinstance(o, dict)]
    entries.sort(key=lambda o: (o.get("provider") or {}).get("priority", 99) or 99)
    for o in entries:
        ml = o.get("moneyline") if isinstance(o.get("moneyline"), dict) else {}
        home = _line(ml.get("home")) if ml else None
        away = _line(ml.get("away")) if ml else None
        if home is None or away is None:
            home = american((o.get("homeTeamOdds") or {}).get("moneyLine"))
            away = american((o.get("awayTeamOdds") or {}).get("moneyLine"))
        if home is not None and away is not None:
            return {"home": home, "away": away, "book": (o.get("provider") or {}).get("name") or "ESPN"}
    return None


def _team(c):
    t = c.get("team") or {}
    return {"abbr": t.get("abbreviation", ""), "name": t.get("displayName") or t.get("name", "")}


def events(payload):
    """Scoreboard payload -> [{"id", "start" (aware datetime), "home", "away",
    "odds"}], home/away as {"abbr", "name"}. Games without odds are kept, with
    odds None."""
    out = []
    for ev in (payload or {}).get("events") or []:
        try:
            comp = (ev.get("competitions") or [{}])[0]
            sides = {c.get("homeAway"): c for c in comp.get("competitors") or []}
            if set(sides) != {"home", "away"}:
                continue
            out.append({
                "id": str(ev.get("id")),
                "start": datetime.fromisoformat((comp.get("date") or ev["date"]).replace("Z", "+00:00")),
                "home": _team(sides["home"]), "away": _team(sides["away"]),
                "odds": parse_odds(comp),
            })
        except Exception as e:  # one odd event never costs the rest
            print(f"  skipping scoreboard event {ev.get('id')}: {e}")
    return out


def fetch(sport, day):
    """Every game on an ESPN scoreboard for a day (YYYY-MM-DD), with its
    moneyline odds, sport like "baseball/mlb". [] when ESPN can't be reached."""
    try:
        r = requests.get(f"{ESPN}/{sport}/scoreboard", params={"dates": day.replace("-", ""), "limit": 100},
                         headers={"User-Agent": "Mozilla/5.0 (Sports Edge; github.com/ant56-arch)"}, timeout=30)
        r.raise_for_status()
        return events(r.json())
    except Exception as e:
        print(f"  moneyline odds unavailable for {day}: {e}")
        return []


def match(evs, away, home, away_name, home_name, start, aliases=None):
    """The scoreboard event for a game, by team names or abbreviations, taking
    the closest start time (doubleheaders). start is an aware datetime."""
    aliases = aliases or {}

    def key(t):
        return t["name"], aliases.get(t["abbr"], t["abbr"])

    cands = [e for e in evs
             if (key(e["away"])[0], key(e["home"])[0]) == (away_name, home_name)
             or (key(e["away"])[1], key(e["home"])[1]) == (away, home)]
    if not cands:
        return None
    return min(cands, key=lambda e: abs((e["start"] - start).total_seconds()))


# ── Pick ─────────────────────────────────────────────────────────────────────
def pick(home, away, home_prob, odds, model_pick=None):
    """The moneyline pick for a game. home_prob is our model's home win chance in
    percent; odds is {"home", "away", "book"}. None without odds."""
    if not odds or odds.get("home") is None or odds.get("away") is None or home_prob is None:
        return None
    book_home, book_away = no_vig(odds["home"], odds["away"])
    ours = {"home": home_prob / 100, "away": 1 - home_prob / 100}
    book = {"home": book_home, "away": book_away}
    edge = {s: ours[s] - book[s] for s in ("home", "away")}
    if abs(edge["home"] - edge["away"]) < 1e-9:  # dead even: go with the model's pick
        side = "home" if model_pick == home else "away"
    else:
        side = max(edge, key=edge.get)
    team = home if side == "home" else away
    price = odds[side]
    e = round(100 * edge[side], 1)
    return {
        "team": team, "side": side, "price": price,
        "prob": round(100 * ours[side], 1), "book_prob": round(100 * book[side], 1), "edge": e,
        "value": e >= VALUE_EDGE,
        "home_ml": odds["home"], "away_ml": odds["away"], "book": odds.get("book") or "ESPN",
        "won": None, "units": None,
    }


def price_text(price):
    return f"+{price}" if price > 0 else str(price)


def text(ml):
    """ "BUF +135" """
    return f"{ml['team']} {price_text(ml['price'])}"


def detail(ml):
    """ "our 48% vs book 42%, +6.2" """
    return f"our {ml['prob']:.0f}% vs book {ml['book_prob']:.0f}%, {ml['edge']:+.1f}"


# ── Grading and record ───────────────────────────────────────────────────────
def payout(price):
    """Units won on a 1-unit bet at American odds."""
    return round(price / 100 if price > 0 else 100 / -price, 3)


def grade(p):
    """Settles a pick's moneyline from its graded result. p is a pick with
    "winner" (and "void" for a postponed game). Returns True if it changed."""
    ml = p.get("ml")
    if not ml or ml.get("won") is not None or ml.get("void"):
        return False
    if p.get("void"):
        ml["void"] = True
        ml["units"] = 0.0
        return True
    if not p.get("winner"):
        return False
    ml["won"] = p["winner"] == ml["team"]
    ml["units"] = payout(ml["price"]) if ml["won"] else -1.0
    return True


def graded(picks):
    """Picks whose moneyline won or lost (no decisions left out)."""
    return [p for p in picks if p.get("ml") and p["ml"].get("won") is not None and not p["ml"].get("void")
            and not p.get("void")]


def record(picks):
    """{"wins", "losses", "units", "roi"} over graded moneyline picks, or None."""
    g = graded(picks)
    if not g:
        return None
    w = sum(1 for p in g if p["ml"]["won"])
    units = round(sum(p["ml"]["units"] for p in g), 2)
    return {"wins": w, "losses": len(g) - w, "units": units, "roi": units / len(g), "picks": len(g)}


def units_text(u):
    return f"{u:+.2f}u"


def result(ml, void=False):
    """For the History day picker: {"text", "value", "detail", "won", "void", "units"}."""
    if not ml:
        return None
    return {"text": text(ml), "value": bool(ml.get("value")), "detail": detail(ml),
            "won": ml.get("won"), "void": bool(void or ml.get("void")), "units": ml.get("units")}
