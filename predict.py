"""
predict.py - daily MLB Edge pipeline. Two modes (MODE env var):

  picks   - score every hitter in today's games with the trained model
            (model_weights.json), pick the day's top 10, and write
            data/slate.json + picks_history.json for build_site.py.
            Safe to run several times a day: once a game starts its picks
            are locked, and later runs only refill picks from games that
            haven't started - by then with posted lineups, so a player
            who isn't starting is never picked.
  results - grade every pending pick from past days against box scores.
            A pick with no decision (didn't bat, 0 at-bats, postponed)
            is voided rather than counted as a miss.
"""

import json
import math
import os
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

import features as F

API = "https://statsapi.mlb.com/api/v1"
ET = ZoneInfo("America/New_York")
NOW = datetime.now(ET)
TODAY = NOW.date()
SEASON = TODAY.year
MODE = os.environ.get("MODE", "picks")

HISTORY_FILE = "picks_history.json"
SLATE_FILE = os.path.join("data", "slate.json")
MODEL_FILE = "model_weights.json"

PICKS_PER_DAY = 10
MAX_PICKS_PER_GAME = 2
GAME_TYPES = {"R", "F", "D", "L", "W"}  # regular season + postseason, never spring training
REGULAR_SHARE = 0.6  # without a posted lineup, only consider hitters who've played 60%+ of games

with open(MODEL_FILE) as _f:
    MODEL = json.load(_f)

session = requests.Session()


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


def total_line(splits):
    """One stat line per player: a traded player comes back as one split per
    team, sometimes plus a combined split (the one without a team)."""
    if not splits:
        return {}
    combined = [s for s in splits if not s.get("team")]
    if combined:
        return combined[0]["stat"]
    if len(splits) == 1:
        return splits[0]["stat"]
    out = {}
    for s in splits:
        for k, v in s["stat"].items():
            if isinstance(v, int):
                out[k] = out.get(k, 0) + v
    return out


def stats_by_player(**params):
    splits = get("stats", playerPool="ALL", limit=5000, **params)["stats"]
    grouped = {}
    for s in splits[0]["splits"] if splits else []:
        grouped.setdefault(s["player"]["id"], []).append(s)
    return {pid: total_line(ss) for pid, ss in grouped.items()}


def people(ids, **params):
    out = {}
    ids = [i for i in ids if i]
    for i in range(0, len(ids), 100):
        chunk = ",".join(str(x) for x in ids[i:i + 100])
        for p in get("people", personIds=chunk, **params).get("people", []):
            out[p["id"]] = p
    return out


def load_history():
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"picks": []}


def save_json(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Today's games ────────────────────────────────────────────────────────────
def todays_games():
    data = get("schedule", sportId=1, date=TODAY.isoformat(),
               hydrate="probablePitcher,lineups,venue,team")
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {})
            if g.get("gameType") not in GAME_TYPES or status.get("detailedState") in ("Postponed", "Cancelled"):
                continue
            sides = {}
            for side in ("home", "away"):
                t = g["teams"][side]
                lineup = g.get("lineups", {}).get(f"{side}Players", [])
                sides[side] = {
                    "team_id": t["team"]["id"],
                    "team": t["team"]["name"],
                    "abbr": t["team"].get("abbreviation", t["team"]["name"][:3].upper()),
                    "pitcher_id": t.get("probablePitcher", {}).get("id"),
                    "pitcher": t.get("probablePitcher", {}).get("fullName", "TBD"),
                    "lineup": [p["id"] for p in lineup],
                }
            start = datetime.fromisoformat(g["gameDate"].replace("Z", "+00:00")).astimezone(ET)
            games.append({
                "game_id": g["gamePk"],
                "start": start.isoformat(),
                "time": start.strftime("%-I:%M %p"),
                "venue": g.get("venue", {}).get("name", ""),
                "started": status.get("abstractGameState") in ("Live", "Final"),
                "status": status.get("detailedState", ""),
                "home": sides["home"],
                "away": sides["away"],
            })
    games.sort(key=lambda g: g["start"])
    return games


# ── Scoring ──────────────────────────────────────────────────────────────────
def predict(raw):
    z = MODEL["bias"]
    contrib = {}
    for i, f in enumerate(MODEL["features"]):
        c = MODEL["weights"][i] * (raw[f] - MODEL["means"][i]) / MODEL["stds"][i]
        contrib[f] = c
        z += c
    factors = {g: round(sum(contrib.get(f, 0.0) for f in fs), 4) for g, fs in F.FACTOR_GROUPS.items()}
    return 1 / (1 + math.exp(-z)), factors


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def avg_str(x):
    return f"{x:.3f}".lstrip("0") if x is not None else "-"


def build_reason(p):
    """The two factors pushing this player's probability up the most, in words."""
    phrases = {
        "pitcher_matchup": f"Faces {p['opp_pitcher']} ({p['opp_era']:.2f} ERA, {p['opp_whip']:.2f} WHIP)"
                           if p["opp_era"] is not None else "Facing an unannounced starter",
        "playing_time": f"Batting {ordinal(p['batting_order'])}" if p["batting_order"]
                        else f"{p['ab_per_game']:.1f} at-bats per game",
        "recent_form": f"Hitting {avg_str(p['recent_avg'])} over the last 2 weeks",
        "season_avg": f"{avg_str(p['season_avg'])} season average",
        "platoon": f"Platoon edge ({p['bats']}HB vs. {p['opp_hand']}HP)",
        "home_away": "Playing at home",
        "park_factor": f"Hitter-friendly park ({p['venue']})",
    }
    top = [g for g, v in sorted(p["factors"].items(), key=lambda kv: -kv[1]) if v > 0][:2]
    return " · ".join(phrases[g] for g in top) or "Solid across the board"


def score_slate(games):
    playing = {}
    for g in games:
        for side, opp in (("home", "away"), ("away", "home")):
            team_id = g[side]["team_id"]
            # Doubleheader: score each team for its first game that hasn't started.
            if team_id not in playing or playing[team_id][0]["started"]:
                playing[team_id] = (g, side, opp)

    roster = {}
    for team_id in playing:
        for r in get(f"teams/{team_id}/roster", rosterType="active", season=SEASON).get("roster", []):
            if r.get("position", {}).get("type") != "Pitcher":
                roster[r["person"]["id"]] = team_id

    season = stats_by_player(stats="season", group="hitting", season=SEASON)
    prev = stats_by_player(stats="season", group="hitting", season=SEASON - 1)
    recent = stats_by_player(stats="byDateRange", group="hitting", season=SEASON,
                             startDate=(TODAY - timedelta(days=F.RECENT_WINDOW_DAYS)).isoformat(),
                             endDate=(TODAY - timedelta(days=1)).isoformat())
    hitters = people(roster)
    pitcher_ids = [g[s]["pitcher_id"] for g in games for s in ("home", "away")]
    pitchers = people(pitcher_ids, hydrate=f"stats(group=[pitching],type=[season],season={SEASON})")

    team_games = {}
    for pid, team_id in roster.items():
        team_games[team_id] = max(team_games.get(team_id, 0), int(season.get(pid, {}).get("gamesPlayed", 0)))

    scored = []
    for pid, team_id in roster.items():
        g, side, opp = playing[team_id]
        cur, pr, rec = season.get(pid, {}), prev.get(pid, {}), recent.get(pid, {})
        hits, ab, n_games = int(cur.get("hits", 0)), int(cur.get("atBats", 0)), int(cur.get("gamesPlayed", 0))
        prev_ab = int(pr.get("atBats", 0))

        lineup = g[side]["lineup"]
        if lineup:
            if pid not in lineup:
                continue
            batting_order = lineup.index(pid) + 1
        else:
            batting_order = None
            if team_games[team_id] >= 10:
                if n_games < REGULAR_SHARE * team_games[team_id]:
                    continue
            elif prev_ab < 250 and ab < 20:
                continue

        prior = (int(pr["hits"]) + F.LEAGUE_AVG * 100) / (prev_ab + 100) if prev_ab else F.LEAGUE_AVG
        person = hitters.get(pid, {})
        bats = person.get("batSide", {}).get("code", "R")

        pitcher_id = g[opp]["pitcher_id"]
        pitcher = pitchers.get(pitcher_id, {})
        pl = total_line(pitcher.get("stats", [{}])[0].get("splits", [])) if pitcher.get("stats") else {}
        ip = F.innings_to_float(pl.get("inningsPitched"))
        era = float(pl["era"]) if ip and pl.get("era") not in (None, "-.--") else None
        whip = float(pl["whip"]) if ip and pl.get("whip") not in (None, "-.--") else None
        k9 = int(pl.get("strikeOuts", 0)) * 9 / ip if ip else None
        opp_hand = pitcher.get("pitchHand", {}).get("code", "R")

        raw = F.batter_features(hits, ab, n_games, int(rec.get("hits", 0)), int(rec.get("atBats", 0)), prior)
        raw.update(F.pitcher_features(era, whip, k9, ip))
        raw.update(F.matchup_features(side == "home", bats, opp_hand, g["venue"], MODEL.get("park_factors")))
        prob, factors = predict(raw)

        rec_ab = int(rec.get("atBats", 0))
        player = {
            "player_id": pid,
            "player_name": person.get("fullName", "Unknown"),
            "team": g[side]["team"],
            "team_id": team_id,
            "team_abbr": g[side]["abbr"],
            "opponent_abbr": g[opp]["abbr"],
            "is_home": side == "home",
            "game_id": g["game_id"],
            "game_start": g["start"],
            "game_time": g["time"],
            "game_started": g["started"],
            "venue": g["venue"],
            "bats": bats,
            "opp_pitcher": g[opp]["pitcher"],
            "opp_hand": opp_hand,
            "opp_era": era,
            "opp_whip": whip,
            "batting_order": batting_order,
            "lineup_confirmed": bool(lineup),
            "season_avg": hits / ab if ab else None,
            "recent_avg": int(rec.get("hits", 0)) / rec_ab if rec_ab else None,
            "ab_per_game": ab / n_games if n_games else F.LEAGUE_AB_PER_GAME,
            "confidence": round(prob * 100, 1),
            "factors": factors,
            "raw_features": {k: round(v, 4) for k, v in raw.items()},
        }
        player["reason"] = build_reason(player)
        scored.append(player)

    scored.sort(key=lambda p: -p["confidence"])
    return scored


def hit_streak(player_id):
    splits = get(f"people/{player_id}/stats", stats="gameLog", group="hitting", season=SEASON)["stats"]
    streak = 0
    for g in reversed(splits[0]["splits"] if splits else []):
        st = g.get("stat", {})
        if int(st.get("atBats", 0)) == 0:
            continue
        if int(st.get("hits", 0)) == 0:
            break
        streak += 1
    return streak


PICK_FIELDS = ["player_id", "player_name", "team", "team_abbr", "opponent_abbr", "is_home", "game_id",
               "game_start", "game_time", "venue", "opp_pitcher", "opp_hand", "opp_era", "batting_order",
               "lineup_confirmed", "season_avg", "recent_avg", "confidence", "factors", "raw_features", "reason"]


def choose_picks(scored, games, history):
    """Keep today's picks whose games have started; refill the rest from
    games that haven't, best probability first, at most 2 per game."""
    started = {g["game_id"] for g in games if g["started"]}
    today = TODAY.isoformat()
    locked = [p for p in history["picks"] if p["date"] == today and p["game_id"] in started]
    per_game = {}
    for p in locked:
        per_game[p["game_id"]] = per_game.get(p["game_id"], 0) + 1
    taken = {p["player_id"] for p in locked}

    new = []
    for p in scored:
        if len(locked) + len(new) >= PICKS_PER_DAY:
            break
        if p["game_started"] or p["player_id"] in taken or per_game.get(p["game_id"], 0) >= MAX_PICKS_PER_GAME:
            continue
        per_game[p["game_id"]] = per_game.get(p["game_id"], 0) + 1
        taken.add(p["player_id"])
        new.append(p)

    for p in new:
        try:
            p["hit_streak"] = hit_streak(p["player_id"])
        except requests.RequestException:
            p["hit_streak"] = None

    fresh = [{"date": today, **{k: p[k] for k in PICK_FIELDS}, "hit_streak": p["hit_streak"],
              "model": MODEL.get("trained_at"), "got_hit": None, "hits": None, "at_bats": None, "void": False} for p in new]
    history["picks"] = [p for p in history["picks"] if p["date"] != today] + locked + fresh
    return locked + fresh


def run_picks():
    history = load_history()
    games = todays_games()
    print(f"{len(games)} games on {TODAY}")
    scored = score_slate(games) if games else []
    print(f"Scored {len(scored)} hitters")
    picks = choose_picks(scored, games, history) if games else []
    for p in sorted(picks, key=lambda p: -p["confidence"]):
        print(f"  {p['confidence']:5.1f}%  {p['player_name']} ({p['team_abbr']})"
              f"{'' if p['lineup_confirmed'] else '  [lineup not posted]'}")

    save_json(HISTORY_FILE, history)
    save_json(SLATE_FILE, {
        "date": TODAY.isoformat(),
        "generated_at": NOW.isoformat(timespec="minutes"),
        "games": games,
        "players": scored,
        "pick_ids": [p["player_id"] for p in picks],
    })


# ── Results ──────────────────────────────────────────────────────────────────
def batting_line(game_id, player_id):
    box = get(f"game/{game_id}/boxscore")
    for side in ("home", "away"):
        p = box.get("teams", {}).get(side, {}).get("players", {}).get(f"ID{player_id}")
        if p:
            return p.get("stats", {}).get("batting", {})
    return None


def run_results():
    history = load_history()
    today = TODAY.isoformat()
    for p in history["picks"]:
        # 0 at-bats (walked every time, or never got in) is no decision, not a miss.
        if p.get("got_hit") is False and p.get("at_bats") == 0:
            p.update(got_hit=None, void=True)

    pending = [p for p in history["picks"] if p["got_hit"] is None and not p.get("void") and p["date"] < today]
    if not pending:
        print("No pending picks.")
        return
    game_ids = sorted({p["game_id"] for p in pending})
    status = {}
    for i in range(0, len(game_ids), 50):
        ids = ",".join(str(g) for g in game_ids[i:i + 50])
        for d in get("schedule", sportId=1, gamePks=ids).get("dates", []):
            for g in d["games"]:
                status[g["gamePk"]] = g["status"]

    graded = voided = 0
    for p in pending:
        st = status.get(p["game_id"], {})
        stale = date.fromisoformat(p["date"]) < TODAY - timedelta(days=3)
        if st.get("abstractGameState") == "Final":
            line = batting_line(p["game_id"], p["player_id"])
            ab = int(line.get("atBats", 0)) if line else 0
            if ab == 0:
                p["void"] = True
                voided += 1
            else:
                p.update(hits=int(line.get("hits", 0)), at_bats=ab, got_hit=int(line.get("hits", 0)) > 0)
                graded += 1
                print(f"  {p['player_name']}: {p['hits']}-for-{ab}")
        elif st.get("detailedState") in ("Postponed", "Cancelled") or stale:
            p["void"] = True
            voided += 1
    save_json(HISTORY_FILE, history)
    print(f"Graded {graded}, voided {voided}.")


if __name__ == "__main__":
    run_results() if MODE == "results" else run_picks()
