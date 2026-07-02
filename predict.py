import os
import requests
import resend
from datetime import datetime, date, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
resend.api_key = os.environ["RESEND_API_KEY"]
TO_EMAIL = os.environ["TO_EMAIL"]
TODAY = date.today().strftime("%B %d, %Y")
TODAY_ISO = date.today().strftime("%Y-%m-%d")
YESTERDAY_ISO = (date.today() - timedelta(days=14)).strftime("%Y-%m-%d")

# ── Park factors (all 30 MLB stadiums, league avg = 1.0) ─────────────────────
PARK_FACTORS = {
    # Hitter friendly
    "Coors Field": 1.15,              # Colorado Rockies - altitude inflates everything
    "Great American Ball Park": 1.08, # Cincinnati Reds - short porches
    "Fenway Park": 1.07,              # Boston Red Sox - Green Monster
    "Globe Life Field": 1.05,         # Texas Rangers - warm air, cozy dimensions
    "Minute Maid Park": 1.04,         # Houston Astros - short left field
    "American Family Field": 1.04,    # Milwaukee Brewers
    "Camden Yards": 1.03,             # Baltimore Orioles - short right field
    "Wrigley Field": 1.03,            # Chicago Cubs - wind dependent but averages hitter friendly
    "Truist Park": 1.02,              # Atlanta Braves
    "Chase Field": 1.02,              # Arizona Diamondbacks - heat and altitude
    "Citizens Bank Park": 1.02,       # Philadelphia Phillies - notorious hitter's park
    # Neutral
    "Yankee Stadium": 1.01,           # New York Yankees
    "Kauffman Stadium": 1.00,         # Kansas City Royals
    "Angel Stadium": 1.00,            # Los Angeles Angels
    "Target Field": 1.00,             # Minnesota Twins
    "Busch Stadium": 0.99,            # St. Louis Cardinals
    "Dodger Stadium": 0.99,           # Los Angeles Dodgers
    "Progressive Field": 0.99,        # Cleveland Guardians
    # Pitcher friendly
    "PNC Park": 0.98,                 # Pittsburgh Pirates - deep gaps
    "Nationals Park": 0.97,           # Washington Nationals - large outfield
    "T-Mobile Park": 0.97,            # Seattle Mariners - marine air, deep gaps
    "Tropicana Field": 0.97,          # Tampa Bay Rays - dome turf, odd angles
    "Comerica Park": 0.96,            # Detroit Tigers - very deep outfield
    "Guaranteed Rate Field": 0.96,    # Chicago White Sox
    "Petco Park": 0.96,               # San Diego Padres - marine air suppresses offense
    "Oracle Park": 0.95,              # San Francisco Giants - cold marine air
    "loanDepot park": 0.95,           # Miami Marlins - humid but very deep
    "Citi Field": 0.95,               # New York Mets - large dimensions
    "Oakland Coliseum": 0.94,         # Oakland Athletics - cold, massive foul territory
    "Sahlen Field": 1.00,             # Buffalo emergency use
}

# ── Step 1: Today's games ─────────────────────────────────────────────────────
def get_todays_games():
    url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&hydrate=probablePitcher,lineups"
    r = requests.get(url, timeout=15)
    games = []
    for date_entry in r.json().get("dates", []):
        for game in date_entry.get("games", []):
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
            away_pitcher = away.get("probablePitcher", {})
            home_pitcher = home.get("probablePitcher", {})
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
                "venue": game.get("venue", {}).get("name", ""),
                "park_factor": PARK_FACTORS.get(game.get("venue", {}).get("name", ""), 1.0),
            })
    return games

# ── Step 2: Pitcher stats (ERA, WHIP, handedness) ────────────────────────────
def get_pitcher_stats(pitcher_id):
    if not pitcher_id:
        return {"era": 4.50, "whip": 1.30, "hand": "R"}
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}?hydrate=stats(group=pitching,type=season),currentTeam"
    r = requests.get(url, timeout=15)
    data = r.json()
    person = data.get("people", [{}])[0]
    hand = person.get("pitchHand", {}).get("code", "R")
    stats_list = person.get("stats", [])
    for s in stats_list:
        splits = s.get("splits", [])
        if splits:
            st = splits[0].get("stat", {})
            return {
                "era": float(st.get("era") or 4.50),
                "whip": float(st.get("whip") or 1.30),
                "hand": hand,
            }
    return {"era": 4.50, "whip": 1.30, "hand": hand}

# ── Step 3: Season batter stats + handedness ─────────────────────────────────
def get_batter_stats():
    year = datetime.now().year
    url = (
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&season={year}"
        f"&limit=300&sortStat=avg&order=desc"
    )
    r = requests.get(url, timeout=15)
    players = []
    for entry in r.json().get("stats", [])[0].get("splits", []):
        s = entry.get("stat", {})
        ab = int(s.get("atBats", 0))
        if ab < 100:
            continue
        player_id = entry.get("player", {}).get("id")
        name = entry.get("player", {}).get("fullName", "Unknown")
        team_id = entry.get("team", {}).get("id")
        avg = float(s.get("avg") or 0)
        obp = float(s.get("obp") or 0)
        slg = float(s.get("slg") or 0)
        k_pct = int(s.get("strikeOuts", 0)) / ab if ab > 0 else 0.25
        players.append({
            "id": player_id,
            "name": name,
            "team_id": team_id,
            "avg": avg,
            "obp": obp,
            "slg": slg,
            "ab": ab,
            "k_pct": k_pct,
            "hand": "R",
        })
    for p in players[:60]:
        try:
            pr = requests.get(f"https://statsapi.mlb.com/api/v1/people/{p['id']}", timeout=10)
            p["hand"] = pr.json().get("people", [{}])[0].get("batSide", {}).get("code", "R")
        except:
            pass
    return players

# ── Step 4: Recent 14-day form ────────────────────────────────────────────────
def get_recent_avg(player_id):
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=byDateRange&group=hitting&startDate={YESTERDAY_ISO}&endDate={TODAY_ISO}&season={datetime.now().year}"
    )
    try:
        r = requests.get(url, timeout=10)
        splits = r.json().get("stats", [])[0].get("splits", [])
        if splits:
            st = splits[0].get("stat", {})
            ab = int(st.get("atBats", 0))
            if ab >= 15:
                return float(st.get("avg") or 0)
    except:
        pass
    return None

# ── Step 5: Score each player ─────────────────────────────────────────────────
def score_player(p, pitcher, park_factor, is_home):
    recent = get_recent_avg(p["id"])
    recent_score = (recent / 0.300) * 25 if recent else 12.5
    season_score = (p["avg"] / 0.300) * 20
    era_norm = max(0, min(1, (6.00 - pitcher["era"]) / 4.00))
    whip_norm = max(0, min(1, (1.80 - pitcher["whip"]) / 0.80))
    pitcher_score = ((1 - (era_norm * 0.6 + whip_norm * 0.4))) * 20
    platoon_score = 15 if p["hand"] != pitcher["hand"] else 8
    contact_score = (p["obp"] / 0.350) * 6 + ((1 - p["k_pct"]) / 0.80) * 4
    park_score = ((park_factor - 0.90) / 0.25) * 7
    home_score = 3 if is_home else 1.5
    total = recent_score + season_score + pitcher_score + platoon_score + contact_score + park_score + home_score
    return round(total, 2), recent

# ── Step 6: Build ranked picks ────────────────────────────────────────────────
def get_top_picks(players, games):
    team_game = {}
    for g in games:
        team_game[g["away_team_id"]] = {"game": g, "is_home": False, "opp_pitcher_id": g["home_pitcher_id"], "opp_pitcher_name": g["home_pitcher_name"]}
        team_game[g["home_team_id"]] = {"game": g, "is_home": True, "opp_pitcher_id": g["away_pitcher_id"], "opp_pitcher_name": g["away_pitcher_name"]}

    pitcher_cache = {}
    scored = []
    for p in players[:60]:
        info = team_game.get(p["team_id"])
        if not info:
            continue
        pid = info["opp_pitcher_id"]
        if pid not in pitcher_cache:
            pitcher_cache[pid] = get_pitcher_stats(pid)
        pitcher = pitcher_cache[pid]
        park_factor = info["game"]["park_factor"]
        total, recent = score_player(p, pitcher, park_factor, info["is_home"])
        scored.append({**p, "score": total, "recent_avg": recent,
                        "opp_pitcher": info["opp_pitcher_name"],
                        "venue": info["game"]["venue"],
                        "park_factor": park_factor,
                        "is_home": info["is_home"],
                        "pitcher_era": pitcher["era"],
                        "pitcher_hand": pitcher["hand"]})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:10]
    max_score = top[0]["score"] if top else 1
    for p in top:
        p["confidence"] = round((p["score"] / max_score) * 100, 1)
    return top

# ── Step 7: Build email ───────────────────────────────────────────────────────
def build_email(picks, games):
    rows = ""
    for i, p in enumerate(picks, 1):
        conf = p["confidence"]
        color = "#0F6E56" if conf >= 90 else "#BA7517" if conf >= 80 else "#666"
        medal = ["🥇","🥈","🥉"][i-1] if i <= 3 else f"#{i}"
        recent_str = f".{int(p['recent_avg']*1000):03d} last 14d" if p["recent_avg"] else "—"
        home_away = "Home" if p["is_home"] else "Away"
        platoon = "✅ Platoon adv." if p["hand"] != p["pitcher_hand"] else ""
        rows += f"""
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:10px 8px;font-weight:600;font-size:15px;">{medal}</td>
          <td style="padding:10px 8px;">
            <div style="font-weight:600;font-size:14px;color:#111;">{p['name']} {platoon}</div>
            <div style="font-size:11px;color:#555;margin-top:3px;">
              vs {p['opp_pitcher']} ({p['pitcher_hand']}HP) · {home_away} · {p['venue']}
            </div>
            <div style="font-size:11px;color:#888;margin-top:2px;">
              AVG {p['avg']:.3f} · {recent_str} · OBP {p['obp']:.3f} · Opp ERA {p['pitcher_era']:.2f}
            </div>
          </td>
          <td style="padding:10px 8px;text-align:right;font-weight:700;font-size:16px;color:{color};">{conf:.0f}%</td>
        </tr>"""

    game_rows = ""
    for g in games[:8]:
        pf = g["park_factor"]
        pf_str = f"🟢 +{int((pf-1)*100)}% hits" if pf > 1.02 else f"🔴 {int((pf-1)*100)}% hits" if pf < 0.98 else "⚪ Neutral"
        game_rows += f"""
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:8px;font-size:12px;color:#333;">{g['away_team']} @ {g['home_team']}</td>
          <td style="padding:8px;font-size:11px;color:#888;">{g['away_pitcher_name']} vs {g['home_pitcher_name']}</td>
          <td style="padding:8px;font-size:11px;color:#888;white-space:nowrap;">{pf_str}</td>
        </tr>"""

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:620px;margin:0 auto;background:#fff;">
      <div style="background:#0a0a0a;padding:24px;text-align:center;">
        <h1 style="color:#fff;margin:0;font-size:22px;letter-spacing:2px;">⚾ BTS EDGE</h1>
        <p style="color:#888;margin:4px 0 0;font-size:12px;">Daily Hit Predictions · {TODAY}</p>
      </div>
      <div style="padding:16px 24px 4px;background:#f8f8f8;font-size:11px;color:#888;text-align:center;">
        Scored on: Recent form 25% · Season AVG 20% · Pitcher matchup 20% · Platoon 15% · Contact 10% · Park 7% · Home/Away 3%
      </div>
      <div style="padding:24px;">
        <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">Today's Top {len(picks)} Picks</h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#f8f8f8;">
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">#</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PLAYER · MATCHUP</th>
            <th style="padding:8px;text-align:right;font-size:11px;color:#888;">SCORE</th>
          </tr>
          {rows}
        </table>
      </div>
      <div style="padding:0 24px 24px;">
        <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">Today's Games</h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#f8f8f8;">
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">MATCHUP</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PITCHERS</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PARK</th>
          </tr>
          {game_rows}
        </table>
      </div>
      <div style="background:#f8f8f8;padding:16px 24px;text-align:center;">
        <p style="font-size:11px;color:#aaa;margin:0;">BTS Edge · Automated daily at 8am ET · MLB Stats API</p>
      </div>
    </div>"""
    return html

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Fetching today's games...")
    games = get_todays_games()
    print(f"Found {len(games)} games.")

    print("Fetching batter stats...")
    players = get_batter_stats()
    print(f"Found {len(players)} qualified batters.")

    print("Scoring players with full model...")
    picks = get_top_picks(players, games)
    print(f"Top pick: {picks[0]['name']} ({picks[0]['confidence']}%)" if picks else "No picks found.")

    print("Sending email...")
    html = build_email(picks, games)
    resend.Emails.send({
        "from": "onboarding@resend.dev",
        "to": TO_EMAIL,
        "subject": f"⚾ BTS Edge Picks — {TODAY}",
        "html": html,
    })
    print("Email sent successfully!")

if __name__ == "__main__":
    main()
