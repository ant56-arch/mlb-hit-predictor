import os
import json
import math
import re
import requests
import resend
from datetime import datetime, date, timedelta, timezone

# ── Config ────────────────────────────────────────────────────────────────────
resend.api_key = os.environ["RESEND_API_KEY"]
TO_EMAIL = os.environ["TO_EMAIL"]
TODAY = date.today().strftime("%B %d, %Y")
TODAY_ISO = date.today().strftime("%Y-%m-%d")
YESTERDAY_ISO = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
START_14 = (date.today() - timedelta(days=14)).strftime("%Y-%m-%d")
YEAR = datetime.now().year
MAX_PER_PARK = 2
RUN_HOUR_ET = int(os.environ.get("RUN_HOUR_ET", "8"))
IS_AFTERNOON = RUN_HOUR_ET == 16
IS_RESULTS = RUN_HOUR_ET == 2
HISTORY_FILE = "picks_history.json"
MODEL_FILE = "model_weights.json"

# ── Hit-probability model ────────────────────────────────────────────────────
# Trained by research/train_model.py — a logistic regression fit on real
# game-by-game outcomes (research_gamelogs.json + resolved picks), not a
# hand-picked weighting. See MODEL["features"] for what it uses.
with open(MODEL_FILE) as _f:
    MODEL = json.load(_f)

FACTOR_GROUPS = {
    "recent_form": ["recent_form_avg"],
    "season_stats": ["season_avg", "season_ops"],
    "platoon": ["platoon"],
    "park_factor": ["park_factor"],
    "home_away": ["is_home"],
    "pitcher_matchup": ["opp_pitcher_era", "opp_pitcher_whip", "opp_pitcher_k9"],
}

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))

def predict_hit_probability(raw_features):
    """Returns (probability 0-1, per-raw-feature log-odds contributions)."""
    z = MODEL["bias"]
    contributions = {}
    for i, feat in enumerate(MODEL["features"]):
        x_std = (raw_features[feat] - MODEL["means"][i]) / MODEL["stds"][i]
        contrib = MODEL["weights"][i] * x_std
        contributions[feat] = contrib
        z += contrib
    return sigmoid(z), contributions

# ── Park factors ──────────────────────────────────────────────────────────────
PARK_FACTORS = {
    "Coors Field": 1.15,
    "Great American Ball Park": 1.08,
    "Fenway Park": 1.07,
    "Globe Life Field": 1.05,
    "Daikin Park": 1.04,
    "American Family Field": 1.04,
    "Wrigley Field": 1.03,
    "Oriole Park at Camden Yards": 1.03,
    "Camden Yards": 1.03,
    "Truist Park": 1.02,
    "Chase Field": 1.02,
    "Citizens Bank Park": 1.02,
    "Yankee Stadium": 1.01,
    "Rogers Centre": 1.00,
    "Kauffman Stadium": 1.00,
    "Angel Stadium": 1.00,
    "Target Field": 1.00,
    "Sutter Health Park": 1.00,
    "Busch Stadium": 0.99,
    "Dodger Stadium": 0.99,
    "UNIQLO Field at Dodger Stadium": 0.99,
    "Progressive Field": 0.99,
    "PNC Park": 0.98,
    "Nationals Park": 0.97,
    "T-Mobile Park": 0.97,
    "Tropicana Field": 0.97,
    "Comerica Park": 0.96,
    "Rate Field": 0.96,
    "Petco Park": 0.96,
    "Oracle Park": 0.95,
    "loanDepot park": 0.95,
    "Citi Field": 0.95,
}

# Domed / retractable-roof parks — weather rarely a factor
DOME_PARKS = {
    "Tropicana Field", "Rogers Centre", "Chase Field", "T-Mobile Park",
    "American Family Field", "Daikin Park", "loanDepot park", "Globe Life Field",
    "Truist Park",
}

# ── Load / save picks history ─────────────────────────────────────────────────
def load_history():
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except:
        return {"season": YEAR, "picks": []}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ── Weather parsing helper ────────────────────────────────────────────────────
def parse_weather(weather_block, venue):
    """
    Returns {'wind_score': 0-1 float, 'wind_desc': str, 'temp': int or None}
    wind_score: 1.0 = strong wind blowing out, 0.5 = neutral/crosswind/calm, 0.0 = wind blowing in
    """
    if venue in DOME_PARKS:
        return {"wind_score": 0.5, "wind_desc": "Indoors", "temp": None}

    if not weather_block:
        return {"wind_score": 0.5, "wind_desc": "No data", "temp": None}

    wind_str = weather_block.get("wind", "") or ""
    temp_str = weather_block.get("temp", "")
    temp = None
    try:
        temp = int(temp_str)
    except:
        pass

    wind_lower = wind_str.lower()
    speed_match = re.search(r"(\d+)\s*mph", wind_lower)
    speed = int(speed_match.group(1)) if speed_match else 0

    if "out to" in wind_lower or "out, l to r" in wind_lower or "out, r to l" in wind_lower:
        if speed >= 10:
            return {"wind_score": 1.0, "wind_desc": wind_str, "temp": temp}
        elif speed >= 5:
            return {"wind_score": 0.75, "wind_desc": wind_str, "temp": temp}
        else:
            return {"wind_score": 0.6, "wind_desc": wind_str, "temp": temp}
    elif "in from" in wind_lower:
        if speed >= 10:
            return {"wind_score": 0.0, "wind_desc": wind_str, "temp": temp}
        elif speed >= 5:
            return {"wind_score": 0.25, "wind_desc": wind_str, "temp": temp}
        else:
            return {"wind_score": 0.4, "wind_desc": wind_str, "temp": temp}
    else:
        # crosswind (L to R / R to L), calm, or unrecognized format
        return {"wind_score": 0.5, "wind_desc": wind_str or "Calm", "temp": temp}

# ── Step 1: Today's games ─────────────────────────────────────────────────────
def get_todays_games(target_date=TODAY_ISO):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={target_date}&hydrate=probablePitcher,lineups,weather"
    r = requests.get(url, timeout=15)
    games = []
    for date_entry in r.json().get("dates", []):
        for game in date_entry.get("games", []):
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
            away_pitcher = away.get("probablePitcher", {})
            home_pitcher = home.get("probablePitcher", {})
            venue = game.get("venue", {}).get("name", "")
            game_time_utc_str = game.get("gameDate", "")
            game_time_et = None
            game_hour_et = 0
            if game_time_utc_str:
                try:
                    game_time_utc = datetime.strptime(game_time_utc_str, "%Y-%m-%dT%H:%M:%SZ")
                    game_time_utc = game_time_utc.replace(tzinfo=timezone.utc)
                    et_offset = timezone(timedelta(hours=-4))
                    game_time_et = game_time_utc.astimezone(et_offset)
                    game_hour_et = game_time_et.hour
                except:
                    pass

            weather = parse_weather(game.get("weather"), venue)

            games.append({
                "game_id": game["gamePk"],
                "away_team": away.get("team", {}).get("name", ""),
                "away_team_id": away.get("team", {}).get("id"),
                "home_team": home.get("team", {}).get("name", ""),
                "home_team_id": home.get("team", {}).get("id"),
                "away_pitcher_id": away_pitcher.get("id"),
                "away_pitcher_name": away_pitcher.get("fullName", "TBD"),
                "home_pitcher_id": home_pitcher.get("id"),
                "home_pitcher_name": home_pitcher.get("fullName", "TBD"),
                "venue": venue,
                "park_factor": PARK_FACTORS.get(venue, 1.0),
                "game_time_et": game_time_et,
                "game_hour_et": game_hour_et,
                "game_time_str": game_time_et.strftime("%-I:%M %p ET") if game_time_et else "TBD",
                "status": game.get("status", {}).get("abstractGameState", "Preview"),
                "wind_score": weather["wind_score"],
                "wind_desc": weather["wind_desc"],
                "temp": weather["temp"],
            })
    games.sort(key=lambda x: x["game_time_et"] or datetime.max.replace(tzinfo=timezone.utc))
    if IS_AFTERNOON:
        games = [g for g in games if g["game_hour_et"] >= 16]
    return games

# ── Step 2: Pitcher stats ─────────────────────────────────────────────────────
def get_pitcher_stats(pitcher_id):
    if not pitcher_id:
        return {"era": 4.50, "whip": 1.30, "k_per9": 8.0, "hand": "R"}
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}?hydrate=stats(group=pitching,type=season),currentTeam"
    r = requests.get(url, timeout=15)
    data = r.json()
    person = data.get("people", [{}])[0]
    hand = person.get("pitchHand", {}).get("code", "R")
    for s in person.get("stats", []):
        splits = s.get("splits", [])
        if splits:
            st = splits[0].get("stat", {})
            ip = float(st.get("inningsPitched") or 1)
            k_per9 = (float(st.get("strikeOuts") or 0) / ip) * 9
            return {
                "era": float(st.get("era") or 4.50),
                "whip": float(st.get("whip") or 1.30),
                "k_per9": round(k_per9, 2),
                "hand": hand,
            }
    return {"era": 4.50, "whip": 1.30, "k_per9": 8.0, "hand": hand}

# ── Step 3: Season batter stats ───────────────────────────────────────────────
def get_batter_stats():
    url = (
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&season={YEAR}"
        f"&limit=100&sortStat=ops&order=desc"
    )
    r = requests.get(url, timeout=15)
    players = []
    for entry in r.json().get("stats", [])[0].get("splits", []):
        s = entry.get("stat", {})
        ab = int(s.get("atBats", 0))
        if ab < 100:
            continue
        avg = float(s.get("avg") or 0)
        obp = float(s.get("obp") or 0)
        slg = float(s.get("slg") or 0)
        ops = float(s.get("ops") or 0) or (obp + slg)
        k_pct = int(s.get("strikeOuts", 0)) / ab if ab > 0 else 0.25
        players.append({
            "id": entry.get("player", {}).get("id"),
            "name": entry.get("player", {}).get("fullName", "Unknown"),
            "team_id": entry.get("team", {}).get("id"),
            "team_name": entry.get("team", {}).get("name", ""),
            "avg": avg,
            "obp": obp,
            "slg": slg,
            "ops": ops,
            "ab": ab,
            "k_pct": k_pct,
            "hand": "R",
        })
    for p in players[:80]:
        try:
            pr = requests.get(f"https://statsapi.mlb.com/api/v1/people/{p['id']}", timeout=10)
            p["hand"] = pr.json().get("people", [{}])[0].get("batSide", {}).get("code", "R")
        except:
            pass
    return players

# ── Step 5: Recent 14-day form ────────────────────────────────────────────────
def get_recent_avg(player_id):
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=byDateRange&group=hitting&startDate={START_14}&endDate={TODAY_ISO}&season={YEAR}"
    )
    try:
        r = requests.get(url, timeout=10)
        splits = r.json().get("stats", [])[0].get("splits", [])
        if splits:
            st = splits[0].get("stat", {})
            if int(st.get("atBats", 0)) >= 15:
                return float(st.get("avg") or 0)
    except:
        pass
    return None

# ── Step 5b: Current hitting streak ──────────────────────────────────────────
def get_hit_streak(player_id):
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=gameLog&group=hitting&season={YEAR}"
    )
    try:
        r = requests.get(url, timeout=10)
        splits = r.json().get("stats", [])[0].get("splits", [])
        if not splits:
            return 0
        splits = list(reversed(splits))
        streak = 0
        for g in splits:
            st = g.get("stat", {})
            hits = int(st.get("hits", 0))
            ab = int(st.get("atBats", 0))
            if ab == 0:
                continue
            if hits > 0:
                streak += 1
            else:
                break
        return streak
    except:
        return 0

# ── Step 6: Score each player ─────────────────────────────────────────────────
def score_player(p, pitcher, park_factor, is_home):
    recent = get_recent_avg(p["id"])
    raw_features = {
        "recent_form_avg": recent if recent is not None else p["avg"],
        "season_avg": p["avg"],
        "season_ops": p["ops"],
        "is_home": 1.0 if is_home else 0.0,
        "platoon": 1.0 if p["hand"] != pitcher["hand"] else 0.0,
        "park_factor": park_factor,
        "opp_pitcher_era": pitcher["era"],
        "opp_pitcher_whip": pitcher["whip"],
        "opp_pitcher_k9": pitcher["k_per9"],
    }
    probability, contributions = predict_hit_probability(raw_features)

    factors = {
        group: round(sum(contributions[f] for f in feats), 4)
        for group, feats in FACTOR_GROUPS.items()
    }
    return round(probability * 100, 1), recent, factors, raw_features

# ── Step 7: Build ranked picks ────────────────────────────────────────────────
def get_top_picks(players, games):
    team_game = {}
    for g in games:
        team_game[g["away_team_id"]] = {"game": g, "is_home": False, "opp_pitcher_id": g["home_pitcher_id"], "opp_pitcher_name": g["home_pitcher_name"]}
        team_game[g["home_team_id"]] = {"game": g, "is_home": True, "opp_pitcher_id": g["away_pitcher_id"], "opp_pitcher_name": g["away_pitcher_name"]}

    pitcher_cache = {}
    scored = []
    for p in players[:80]:
        info = team_game.get(p["team_id"])
        if not info:
            continue
        pid = info["opp_pitcher_id"]
        if pid not in pitcher_cache:
            pitcher_cache[pid] = get_pitcher_stats(pid)
        pitcher = pitcher_cache[pid]
        probability, recent, factors, raw_features = score_player(
            p, pitcher, info["game"]["park_factor"], info["is_home"]
        )
        scored.append({
            **p,
            "confidence": probability,
            "recent_avg": recent,
            "factors": factors,
            "raw_features": raw_features,
            "opp_pitcher": info["opp_pitcher_name"],
            "venue": info["game"]["venue"],
            "park_factor": info["game"]["park_factor"],
            "is_home": info["is_home"],
            "pitcher_era": pitcher["era"],
            "pitcher_whip": pitcher["whip"],
            "pitcher_k9": pitcher["k_per9"],
            "pitcher_hand": pitcher["hand"],
            "game_time_str": info["game"]["game_time_str"],
            "game_id": info["game"]["game_id"],
            "game_status": info["game"]["status"],
            "wind_score": info["game"]["wind_score"],
            "wind_desc": info["game"]["wind_desc"],
            "temp": info["game"]["temp"],
        })

    scored.sort(key=lambda x: x["confidence"], reverse=True)
    park_counts = {}
    top = []
    for p in scored:
        venue = p["venue"]
        count = park_counts.get(venue, 0)
        if count < MAX_PER_PARK:
            top.append(p)
            park_counts[venue] = count + 1
        if len(top) == 10:
            break

    for p in top:
        p["hit_streak"] = get_hit_streak(p["id"])

    return top

# ── Step 8: Save picks to history ─────────────────────────────────────────────
def save_picks(picks, history):
    history["picks"] = [p for p in history["picks"] if p["date"] != TODAY_ISO]
    for p in picks:
        history["picks"].append({
            "date": TODAY_ISO,
            "player_id": p["id"],
            "player_name": p["name"],
            "team": p["team_name"],
            "game_id": p["game_id"],
            "confidence": p["confidence"],
            "factors": p["factors"],
            "raw_features": p["raw_features"],
            "got_hit": None,
            "hits": None,
            "at_bats": None,
        })
    save_history(history)
    print(f"Saved {len(picks)} picks to history.")

# ── Step 9: Check box scores for hit results ──────────────────────────────────
def check_results(history, target_date):
    updated = 0
    for pick in history["picks"]:
        if pick["date"] != target_date or pick["got_hit"] is not None:
            continue
        try:
            url = f"https://statsapi.mlb.com/api/v1/game/{pick['game_id']}/boxscore"
            r = requests.get(url, timeout=15)
            data = r.json()
            for side in ["away", "home"]:
                players = data.get("teams", {}).get(side, {}).get("players", {})
                for key, player_data in players.items():
                    if player_data.get("person", {}).get("id") == pick["player_id"]:
                        stats = player_data.get("stats", {}).get("batting", {})
                        hits = int(stats.get("hits", 0))
                        ab = int(stats.get("atBats", 0))
                        pick["got_hit"] = hits > 0
                        pick["hits"] = hits
                        pick["at_bats"] = ab
                        updated += 1
                        print(f"{pick['player_name']}: {hits}/{ab}")
                        break
        except Exception as e:
            print(f"Error checking {pick['player_name']}: {e}")
    save_history(history)
    print(f"Updated results for {updated} players.")
    return history

# ── Step 10: Get yesterday's results for email ────────────────────────────────
def get_yesterday_results(history):
    yesterday_picks = [p for p in history["picks"] if p["date"] == YESTERDAY_ISO]
    if not yesterday_picks:
        return None
    completed = [p for p in yesterday_picks if p["got_hit"] is not None]
    hits = sum(1 for p in completed if p["got_hit"])
    return {
        "picks": yesterday_picks,
        "completed": len(completed),
        "hits": hits,
        "total": len(yesterday_picks),
    }

# ── Step 11: Get season stats for email ───────────────────────────────────────
def get_season_stats(history):
    all_picks = [p for p in history["picks"] if p["got_hit"] is not None]
    if not all_picks:
        return None

    total = len(all_picks)
    hits = sum(1 for p in all_picks if p["got_hit"])
    hit_rate = round((hits / total) * 100, 1) if total > 0 else 0

    today = date.today()
    weeks = []
    for w in range(2):
        week_end = today - timedelta(days=w * 7)
        week_start = week_end - timedelta(days=6)
        week_picks = [
            p for p in all_picks
            if week_start.strftime("%Y-%m-%d") <= p["date"] <= week_end.strftime("%Y-%m-%d")
        ]
        if week_picks:
            w_hits = sum(1 for p in week_picks if p["got_hit"])
            weeks.append({
                "label": "This week" if w == 0 else "Last week",
                "hits": w_hits,
                "total": len(week_picks),
                "rate": round((w_hits / len(week_picks)) * 100, 1),
            })

    factor_names = ["recent_form", "season_stats", "platoon", "park_factor", "home_away", "pitcher_matchup"]
    factor_hits = {f: {"hits": 0, "total": 0} for f in factor_names}
    for pick in all_picks:
        if not pick.get("factors"):
            continue
        top_factor = max(pick["factors"], key=lambda k: pick["factors"][k])
        if top_factor not in factor_hits:
            continue
        factor_hits[top_factor]["total"] += 1
        if pick["got_hit"]:
            factor_hits[top_factor]["hits"] += 1

    factor_rates = []
    for f, data in factor_hits.items():
        if data["total"] >= 3:
            rate = round((data["hits"] / data["total"]) * 100, 1)
            factor_rates.append({"factor": f.replace("_", " ").title(), "rate": rate, "total": data["total"]})
    factor_rates.sort(key=lambda x: x["rate"], reverse=True)

    return {
        "total": total,
        "hits": hits,
        "hit_rate": hit_rate,
        "weeks": weeks,
        "best_factor": factor_rates[0] if factor_rates else None,
        "worst_factor": factor_rates[-1] if factor_rates else None,
    }

# ── Step 12: Check morning pick statuses for 4pm email ───────────────────────
def get_morning_pick_statuses(history, games):
    today_picks = [p for p in history["picks"] if p["date"] == TODAY_ISO]
    if not today_picks:
        return []

    game_status = {g["game_id"]: g["status"] for g in get_todays_games()}

    statuses = []
    for pick in today_picks:
        status = "pending"
        hits = None
        ab = None
        g_status = game_status.get(pick["game_id"], "Preview")

        if g_status == "Final":
            try:
                url = f"https://statsapi.mlb.com/api/v1/game/{pick['game_id']}/boxscore"
                r = requests.get(url, timeout=15)
                data = r.json()
                for side in ["away", "home"]:
                    players = data.get("teams", {}).get(side, {}).get("players", {})
                    for key, player_data in players.items():
                        if player_data.get("person", {}).get("id") == pick["player_id"]:
                            s = player_data.get("stats", {}).get("batting", {})
                            hits = int(s.get("hits", 0))
                            ab = int(s.get("atBats", 0))
                            status = "hit" if hits > 0 else "no_hit"
                            break
            except:
                status = "pending"
        elif g_status == "Live":
            status = "in_progress"
        else:
            status = "not_started"

        statuses.append({
            "player_name": pick["player_name"],
            "team": pick["team"],
            "confidence": pick["confidence"],
            "status": status,
            "hits": hits,
            "ab": ab,
        })
    return statuses

# ── Helper: build a plain-language reason for the pick ───────────────────────
def build_reason(p):
    reasons = []
    recent = p["recent_avg"]
    streak = p.get("hit_streak", 0)

    if streak >= 3:
        reasons.append(f"on a {streak}-game hit streak")
    elif recent and recent >= 0.350:
        reasons.append(f"hitting .{int(recent*1000):03d} over his last 14 games")
    elif recent and recent >= 0.300:
        reasons.append(f"solid recent form (.{int(recent*1000):03d} L14)")

    if p["pitcher_era"] >= 5.0:
        reasons.append(f"opposing pitcher has a {p['pitcher_era']:.2f} ERA")
    elif p["pitcher_era"] <= 3.00:
        reasons.append(f"still gets the nod despite a tough {p['pitcher_era']:.2f} ERA arm")

    if p["hand"] != p["pitcher_hand"]:
        reasons.append("favorable opposite-handed matchup")

    if p.get("wind_score", 0.5) >= 0.75:
        reasons.append("wind blowing out today")

    if p["park_factor"] >= 1.03:
        reasons.append("hitter-friendly park")

    if p["avg"] >= 0.300:
        reasons.append(f"{p['avg']:.3f} season hitter")

    if not reasons:
        reasons.append("strong all-around profile")

    return " · ".join(reasons[:2])

# ── Helper: build one player row (shared by both emails) ─────────────────────
def build_player_row(p, i):
    conf = p["confidence"]
    color = "#0F6E56" if conf >= 75 else "#BA7517" if conf >= 65 else "#555"
    medal = ["🥇", "🥈", "🥉"][i - 1] if i <= 3 else f"#{i}"
    home_away = "Home" if p["is_home"] else "Away"
    reason_str = build_reason(p)
    streak = p.get("hit_streak", 0)
    streak_str = f"{streak}-game hit streak" if streak > 0 else "No active streak"

    weather_bit = ""
    if p.get("wind_desc") and p["wind_desc"] not in ("Indoors", "No data", "Calm"):
        weather_bit = f" · {p['wind_desc']}"
    elif p.get("wind_desc") == "Indoors":
        weather_bit = " · Indoors"

    return f"""
    <tr style="border-bottom:1px solid #f0f0f0;">
      <td style="padding:10px 8px;font-weight:600;font-size:15px;vertical-align:top;">{medal}</td>
      <td style="padding:10px 8px;">
        <div style="font-weight:600;font-size:14px;color:#111;">
          {p['name']} <span style="font-size:11px;font-weight:400;color:#888;">· {p['team_name']}</span>
        </div>
        <div style="font-size:12px;color:#333;margin-top:4px;">
          <strong>Facing:</strong> {p['opp_pitcher']} ({p['pitcher_hand']}HP, {p['pitcher_era']:.2f} ERA) · {home_away} · {p['game_time_str']}{weather_bit}
        </div>
        <div style="font-size:12px;color:#0F6E56;margin-top:4px;">
          <strong>Why:</strong> {reason_str}
        </div>
        <div style="font-size:11px;color:#888;margin-top:4px;">
          Season AVG {p['avg']:.3f} · {streak_str}
        </div>
      </td>
      <td style="padding:10px 8px;text-align:right;font-weight:700;font-size:16px;color:{color};vertical-align:top;">{conf:.0f}%</td>
    </tr>"""

# ── Step 13: Build morning email ──────────────────────────────────────────────
def build_morning_email(picks, games, yesterday_results, season_stats):
    rows = "".join(build_player_row(p, i) for i, p in enumerate(picks, 1))

    game_rows = ""
    for g in games:
        pf = g["park_factor"]
        pf_str = f"🟢 +{int((pf-1)*100)}%" if pf > 1.02 else f"🔴 {int((pf-1)*100)}%" if pf < 0.98 else "⚪ Neutral"
        game_rows += f"""
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:8px;font-size:12px;color:#333;">{g['away_team']} @ {g['home_team']}</td>
          <td style="padding:8px;font-size:11px;color:#888;">{g['away_pitcher_name']} vs {g['home_pitcher_name']}</td>
          <td style="padding:8px;font-size:11px;color:#666;white-space:nowrap;">{g['game_time_str']}</td>
          <td style="padding:8px;font-size:11px;color:#888;white-space:nowrap;">{pf_str}</td>
        </tr>"""

    results_html = ""
    if yesterday_results and yesterday_results["completed"] > 0:
        r = yesterday_results
        pct = round((r["hits"] / r["completed"]) * 100) if r["completed"] > 0 else 0
        result_rows = ""
        for pick in r["picks"]:
            if pick["got_hit"] is None:
                icon = "⏳"
                color = "#888"
                result_str = "Pending"
            elif pick["got_hit"]:
                icon = "🟢"
                color = "#0F6E56"
                result_str = f"Hit ({pick['hits']}-{pick['at_bats']})"
            else:
                icon = "🔴"
                color = "#A32D2D"
                result_str = f"No Hit (0-{pick['at_bats']})"
            result_rows += f"""
            <tr style="border-bottom:1px solid #f0f0f0;">
              <td style="padding:8px;font-size:13px;color:#111;">{icon} {pick['player_name']}</td>
              <td style="padding:8px;font-size:11px;color:#888;">{pick['team']}</td>
              <td style="padding:8px;font-size:12px;font-weight:500;color:{color};">{result_str}</td>
              <td style="padding:8px;font-size:11px;color:#888;">{pick['confidence']}% predicted</td>
            </tr>"""

        results_html = f"""
        <div style="padding:0 24px 24px;">
          <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">
            Yesterday's Results
            <span style="font-size:12px;font-weight:400;color:#888;">· {r['hits']}/{r['completed']} hits ({pct}%)</span>
          </h2>
          <table style="width:100%;border-collapse:collapse;">
            <tr style="background:#f8f8f8;">
              <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PLAYER</th>
              <th style="padding:8px;text-align:left;font-size:11px;color:#888;">TEAM</th>
              <th style="padding:8px;text-align:left;font-size:11px;color:#888;">RESULT</th>
              <th style="padding:8px;text-align:left;font-size:11px;color:#888;">HIT PROBABILITY</th>
            </tr>
            {result_rows}
          </table>
        </div>"""

    season_html = ""
    if season_stats:
        s = season_stats
        week_rows = ""
        for w in s["weeks"]:
            week_rows += f"""
            <tr style="border-bottom:1px solid #f0f0f0;">
              <td style="padding:8px;font-size:12px;color:#333;">{w['label']}</td>
              <td style="padding:8px;font-size:12px;color:#111;font-weight:500;">{w['hits']}/{w['total']} hits</td>
              <td style="padding:8px;font-size:12px;font-weight:600;color:{'#0F6E56' if w['rate'] >= 65 else '#BA7517' if w['rate'] >= 55 else '#A32D2D'};">{w['rate']}%</td>
            </tr>"""

        best = f"<span style='color:#0F6E56;font-weight:500;'>{s['best_factor']['factor']} ({s['best_factor']['rate']}%)</span>" if s.get("best_factor") else "—"
        worst = f"<span style='color:#A32D2D;font-weight:500;'>{s['worst_factor']['factor']} ({s['worst_factor']['rate']}%)</span>" if s.get("worst_factor") else "—"

        season_html = f"""
        <div style="padding:0 24px 24px;">
          <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">
            Season Tracker
            <span style="font-size:12px;font-weight:400;color:#888;">· {s['hits']}/{s['total']} picks · {s['hit_rate']}% overall</span>
          </h2>
          <table style="width:100%;border-collapse:collapse;">
            {week_rows}
            <tr style="border-bottom:1px solid #f0f0f0;">
              <td style="padding:8px;font-size:12px;color:#333;">Best factor</td>
              <td colspan="2" style="padding:8px;font-size:12px;">{best}</td>
            </tr>
            <tr>
              <td style="padding:8px;font-size:12px;color:#333;">Worst factor</td>
              <td colspan="2" style="padding:8px;font-size:12px;">{worst}</td>
            </tr>
          </table>
        </div>"""

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:640px;margin:0 auto;background:#fff;">
      <div style="background:#0a0a0a;padding:24px;text-align:center;">
        <h1 style="color:#fff;margin:0;font-size:22px;letter-spacing:2px;">⚾ BTS EDGE</h1>
        <p style="color:#888;margin:4px 0 0;font-size:12px;">8am Morning Edition · {TODAY}</p>
      </div>
      <div style="padding:12px 24px;background:#f8f8f8;font-size:11px;color:#888;text-align:center;line-height:1.8;">
        <strong style="color:#555;">Hit probability model:</strong>
        Logistic regression trained on real game outcomes — recent form, season AVG/OPS, opposing pitcher (ERA/WHIP/K9), platoon split, home/away, and park factor.
      </div>
      <div style="padding:24px;">
        <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">Today's Top {len(picks)} Picks</h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#f8f8f8;">
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;width:32px;">#</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PLAYER &amp; MATCHUP</th>
            <th style="padding:8px;text-align:right;font-size:11px;color:#888;">HIT PROBABILITY</th>
          </tr>
          {rows}
        </table>
      </div>
      <div style="padding:0 24px 24px;">
        <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">
          Today's Games <span style="font-size:11px;font-weight:400;color:#888;">· All games · earliest to latest</span>
        </h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#f8f8f8;">
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">MATCHUP</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PITCHERS</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">TIME</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PARK</th>
          </tr>
          {game_rows}
        </table>
      </div>
      {results_html}
      {season_html}
      <div style="background:#f8f8f8;padding:16px 24px;text-align:center;">
        <p style="font-size:11px;color:#aaa;margin:0;">
          BTS Edge · 8am Morning Edition · MLB Stats API
        </p>
      </div>
    </div>"""
    return html

# ── Step 14: Build afternoon email ────────────────────────────────────────────
def build_afternoon_email(picks, games, morning_statuses):
    rows = "".join(build_player_row(p, i) for i, p in enumerate(picks, 1))

    game_rows = ""
    for g in games:
        pf = g["park_factor"]
        pf_str = f"🟢 +{int((pf-1)*100)}%" if pf > 1.02 else f"🔴 {int((pf-1)*100)}%" if pf < 0.98 else "⚪ Neutral"
        game_rows += f"""
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:8px;font-size:12px;color:#333;">{g['away_team']} @ {g['home_team']}</td>
          <td style="padding:8px;font-size:11px;color:#888;">{g['away_pitcher_name']} vs {g['home_pitcher_name']}</td>
          <td style="padding:8px;font-size:11px;color:#666;white-space:nowrap;">{g['game_time_str']}</td>
          <td style="padding:8px;font-size:11px;color:#888;white-space:nowrap;">{pf_str}</td>
        </tr>"""

    status_rows = ""
    if morning_statuses:
        for s in morning_statuses:
            if s["status"] == "hit":
                icon = "🟢"
                badge_bg = "#EAF3DE"
                badge_color = "#27500A"
                result = f"Hit ({s['hits']}-{s['ab']})"
            elif s["status"] == "no_hit":
                icon = "🔴"
                badge_bg = "#FDECEA"
                badge_color = "#A32D2D"
                result = f"No Hit (0-{s['ab']})"
            elif s["status"] == "in_progress":
                icon = "🕐"
                badge_bg = "#FFF8E6"
                badge_color = "#854F0B"
                result = "In Progress"
            else:
                icon = "⏳"
                badge_bg = "#F5F5F5"
                badge_color = "#666"
                result = "Not Started"

            status_rows += f"""
            <tr style="border-bottom:1px solid #f0f0f0;">
              <td style="padding:8px;">
                <span style="background:{badge_bg};color:{badge_color};padding:2px 8px;border-radius:12px;font-size:11px;font-weight:500;">
                  {icon} {result}
                </span>
              </td>
              <td style="padding:8px;font-size:13px;color:#111;font-weight:500;">{s['player_name']}</td>
              <td style="padding:8px;font-size:11px;color:#888;">{s['team']}</td>
              <td style="padding:8px;font-size:11px;color:#888;">{s['confidence']}% predicted</td>
            </tr>"""

    status_html = ""
    if status_rows:
        status_html = f"""
        <div style="padding:0 24px 24px;">
          <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">
            Morning Pick Status
          </h2>
          <table style="width:100%;border-collapse:collapse;">
            <tr style="background:#f8f8f8;">
              <th style="padding:8px;text-align:left;font-size:11px;color:#888;">RESULT</th>
              <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PLAYER</th>
              <th style="padding:8px;text-align:left;font-size:11px;color:#888;">TEAM</th>
              <th style="padding:8px;text-align:left;font-size:11px;color:#888;">HIT PROBABILITY</th>
            </tr>
            {status_rows}
          </table>
        </div>"""

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:640px;margin:0 auto;background:#fff;">
      <div style="background:#0a0a0a;padding:24px;text-align:center;">
        <h1 style="color:#fff;margin:0;font-size:22px;letter-spacing:2px;">⚾ BTS EDGE</h1>
        <p style="color:#888;margin:4px 0 0;font-size:12px;">4pm Evening Edition · {TODAY}</p>
      </div>
      <div style="padding:12px 24px;background:#f8f8f8;font-size:11px;color:#888;text-align:center;line-height:1.8;">
        <strong style="color:#555;">Hit probability model:</strong>
        Logistic regression trained on real game outcomes — recent form, season AVG/OPS, opposing pitcher (ERA/WHIP/K9), platoon split, home/away, and park factor.
      </div>
      <div style="padding:24px;">
        <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">Tonight's Top {len(picks)} Picks</h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#f8f8f8;">
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;width:32px;">#</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PLAYER &amp; MATCHUP</th>
            <th style="padding:8px;text-align:right;font-size:11px;color:#888;">HIT PROBABILITY</th>
          </tr>
          {rows}
        </table>
      </div>
      <div style="padding:0 24px 24px;">
        <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">
          Tonight's Games <span style="font-size:11px;font-weight:400;color:#888;">· 4pm ET or later</span>
        </h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#f8f8f8;">
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">MATCHUP</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PITCHERS</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">TIME</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PARK</th>
          </tr>
          {game_rows}
        </table>
      </div>
      {status_html}
      <div style="background:#f8f8f8;padding:16px 24px;text-align:center;">
        <p style="font-size:11px;color:#aaa;margin:0;">
          BTS Edge · 4pm Evening Edition · MLB Stats API
        </p>
      </div>
    </div>"""
    return html

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    history = load_history()

    if IS_RESULTS:
        print("Running 2am results check...")
        history = check_results(history, YESTERDAY_ISO)
        print("Results check complete.")
        return

    print(f"Fetching today's games...")
    all_games = get_todays_games()
    print(f"Found {len(all_games)} games.")

    if not all_games:
        print("No games found — skipping.")
        return

    players = get_batter_stats()
    print(f"Found {len(players)} qualified batters.")

    picks = get_top_picks(players, all_games)
    if not picks:
        print("No picks found.")
        return
    print(f"Top pick: {picks[0]['name']} ({picks[0]['confidence']}%)")

    if IS_AFTERNOON:
        morning_statuses = get_morning_pick_statuses(history, all_games)
        html = build_afternoon_email(picks, all_games, morning_statuses)
        subject = f"⚾ BTS Edge 4pm Evening Edition — {TODAY}"
    else:
        save_picks(picks, history)
        yesterday_results = get_yesterday_results(history)
        season_stats = get_season_stats(history)
        html = build_morning_email(picks, all_games, yesterday_results, season_stats)
        subject = f"⚾ BTS Edge 8am Morning Edition — {TODAY}"

    resend.Emails.send({
        "from": "onboarding@resend.dev",
        "to": TO_EMAIL,
        "subject": subject,
        "html": html,
    })
    print("Email sent successfully!")

if __name__ == "__main__":
    main()
