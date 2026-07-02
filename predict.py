import os
import requests
import resend
from datetime import datetime, date, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
resend.api_key = os.environ["RESEND_API_KEY"]
TO_EMAIL = os.environ["TO_EMAIL"]
TODAY = date.today().strftime("%B %d, %Y")
TODAY_ISO = date.today().strftime("%Y-%m-%d")
START_14 = (date.today() - timedelta(days=14)).strftime("%Y-%m-%d")
YEAR = datetime.now().year

# ── Park factors (all 30 MLB stadiums, league avg = 1.0) ─────────────────────
PARK_FACTORS = {
    "Coors Field": 1.15,
    "Great American Ball Park": 1.08,
    "Fenway Park": 1.07,
    "Globe Life Field": 1.05,
    "Minute Maid Park": 1.04,
    "American Family Field": 1.04,
    "Camden Yards": 1.03,
    "Wrigley Field": 1.03,
    "Truist Park": 1.02,
    "Chase Field": 1.02,
    "Citizens Bank Park": 1.02,
    "Yankee Stadium": 1.01,
    "Kauffman Stadium": 1.00,
    "Angel Stadium": 1.00,
    "Target Field": 1.00,
    "Busch Stadium": 0.99,
    "Dodger Stadium": 0.99,
    "Progressive Field": 0.99,
    "PNC Park": 0.98,
    "Nationals Park": 0.97,
    "T-Mobile Park": 0.97,
    "Tropicana Field": 0.97,
    "Comerica Park": 0.96,
    "Guaranteed Rate Field": 0.96,
    "Petco Park": 0.96,
    "Oracle Park": 0.95,
    "loanDepot park": 0.95,
    "Citi Field": 0.95,
    "Oakland Coliseum": 0.94,
    "Sahlen Field": 1.00,
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

# ── Step 3: Season batter stats from MLB Stats API ───────────────────────────
def get_batter_stats():
    url = (
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&season={YEAR}"
        f"&limit=300&sortStat=avg&order=desc"
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
            "avg": avg,
            "obp": obp,
            "slg": slg,
            "ops": ops,
            "ab": ab,
            "k_pct": k_pct,
            "hand": "R",
            "exit_velo": None,
            "barrel_pct": None,
            "hard_hit_pct": None,
        })
    for p in players[:80]:
        try:
            pr = requests.get(f"https://statsapi.mlb.com/api/v1/people/{p['id']}", timeout=10)
            p["hand"] = pr.json().get("people", [{}])[0].get("batSide", {}).get("code", "R")
        except:
            pass
    return players

# ── Step 4: Statcast advanced metrics from Baseball Savant ───────────────────
def get_statcast_metrics():
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={YEAR}&type=batter&filter=&sort=4&sortDir=desc"
        f"&min=100&selections=player_id,player_name,exit_velocity_avg,barrel_batted_rate,hard_hit_percent"
        f"&chart=false&x=exit_velocity_avg&y=exit_velocity_avg&r=no&chartType=beeswarm&csv=true"
    )
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return {}
        headers = [h.strip().strip('"') for h in lines[0].split(",")]
        metrics = {}
        for line in lines[1:]:
            vals = [v.strip().strip('"') for v in line.split(",")]
            if len(vals) < len(headers):
                continue
            row = dict(zip(headers, vals))
            pid = row.get("player_id", "")
            if not pid:
                continue
            try:
                metrics[int(pid)] = {
                    "exit_velo": float(row.get("exit_velocity_avg") or 0) or None,
                    "barrel_pct": float(row.get("barrel_batted_rate") or 0) or None,
                    "hard_hit_pct": float(row.get("hard_hit_percent") or 0) or None,
                }
            except:
                continue
        print(f"Loaded Statcast metrics for {len(metrics)} players.")
        return metrics
    except Exception as e:
        print(f"Statcast fetch failed: {e} — skipping advanced metrics.")
        return {}

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

# ── Step 6: Score each player ─────────────────────────────────────────────────
def score_player(p, pitcher, park_factor, is_home):
    # 1. Recent form — 22 pts
    recent = get_recent_avg(p["id"])
    recent_score = (recent / 0.300) * 22 if recent else 11.0

    # 2. Advanced metrics group — 20 pts
    ev = p.get("exit_velo")
    ev_score = max(0, min(1, (ev - 84) / 10)) * 7 if ev else 3.5
    barrel = p.get("barrel_pct")
    barrel_score = max(0, min(1, barrel / 15)) * 7 if barrel else 3.5
    hard_hit = p.get("hard_hit_pct")
    hh_score = max(0, min(1, (hard_hit - 25) / 30)) * 4 if hard_hit else 2.0
    ops_score = max(0, min(1, (p["ops"] - 0.600) / 0.400)) * 2
    advanced_score = ev_score + barrel_score + hh_score + ops_score

    # 3. Pitcher matchup — 18 pts
    era_norm = max(0, min(1, (6.00 - pitcher["era"]) / 4.00))
    whip_norm = max(0, min(1, (1.80 - pitcher["whip"]) / 0.80))
    k_norm = max(0, min(1, (14.0 - pitcher["k_per9"]) / 10.0))
    pitcher_score = (1 - (era_norm * 0.4 + whip_norm * 0.35 + k_norm * 0.25)) * 18

    # 4. Season batting average — 15 pts
    season_score = max(0, min(1, p["avg"] / 0.350)) * 15

    # 5. Platoon advantage — 12 pts
    platoon_score = 12 if p["hand"] != pitcher["hand"] else 6

    # 6. Park factor — 8 pts
    park_score = max(0, min(1, (park_factor - 0.90) / 0.25)) * 8

    # 7. Home vs away — 5 pts
    home_score = 5 if is_home else 2.5

    total = recent_score + advanced_score + pitcher_score + season_score + platoon_score + park_score + home_score
    return round(total, 2), recent

# ── Step 7: Build ranked picks ────────────────────────────────────────────────
def get_top_picks(players, games, statcast):
    for p in players:
        sc = statcast.get(p["id"], {})
        p["exit_velo"] = sc.get("exit_velo")
        p["barrel_pct"] = sc.get("barrel_pct")
        p["hard_hit_pct"] = sc.get("hard_hit_pct")

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
        total, recent = score_player(p, pitcher, info["game"]["park_factor"], info["is_home"])
        scored.append({
            **p,
            "score": total,
            "recent_avg": recent,
            "opp_pitcher": info["opp_pitcher_name"],
            "venue": info["game"]["venue"],
            "park_factor": info["game"]["park_factor"],
            "is_home": info["is_home"],
            "pitcher_era": pitcher["era"],
            "pitcher_whip": pitcher["whip"],
            "pitcher_k9": pitcher["k_per9"],
            "pitcher_hand": pitcher["hand"],
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:10]
    max_score = top[0]["score"] if top else 1
    for p in top:
        p["confidence"] = round((p["score"] / max_score) * 100, 1)
    return top

# ── Step 8: Build email ───────────────────────────────────────────────────────
def build_email(picks, games):
    rows = ""
    for i, p in enumerate(picks, 1):
        conf = p["confidence"]
        color = "#0F6E56" if conf >= 90 else "#BA7517" if conf >= 80 else "#555"
        medal = ["🥇","🥈","🥉"][i-1] if i <= 3 else f"#{i}"
        recent_str = f".{int(p['recent_avg']*1000):03d} L14" if p["recent_avg"] else "—"
        home_away = "Home" if p["is_home"] else "Away"
        platoon = "✅ Platoon" if p["hand"] != p["pitcher_hand"] else ""
        ev_str = f"{p['exit_velo']:.1f} mph" if p["exit_velo"] else "—"
        barrel_str = f"{p['barrel_pct']:.1f}%" if p["barrel_pct"] else "—"
        hh_str = f"{p['hard_hit_pct']:.1f}%" if p["hard_hit_pct"] else "—"
        rows += f"""
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:10px 8px;font-weight:600;font-size:15px;">{medal}</td>
          <td style="padding:10px 8px;">
            <div style="font-weight:600;font-size:14px;color:#111;">{p['name']} {platoon}</div>
            <div style="font-size:11px;color:#444;margin-top:3px;">
              vs {p['opp_pitcher']} ({p['pitcher_hand']}HP) · ERA {p['pitcher_era']:.2f} · WHIP {p['pitcher_whip']:.2f} · K/9 {p['pitcher_k9']:.1f}
            </div>
            <div style="font-size:11px;color:#666;margin-top:2px;">
              AVG {p['avg']:.3f} · OPS {p['ops']:.3f} · {recent_str} · {home_away} · {p['venue']}
            </div>
            <div style="font-size:11px;color:#888;margin-top:2px;">
              Exit Velo {ev_str} · Barrel% {barrel_str} · Hard Hit% {hh_str}
            </div>
          </td>
          <td style="padding:10px 8px;text-align:right;font-weight:700;font-size:16px;color:{color};vertical-align:top;">{conf:.0f}%</td>
        </tr>"""

    game_rows = ""
    for g in games[:8]:
        pf = g["park_factor"]
        pf_str = f"🟢 +{int((pf-1)*100)}%" if pf > 1.02 else f"🔴 {int((pf-1)*100)}%" if pf < 0.98 else "⚪ Neutral"
        game_rows += f"""
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:8px;font-size:12px;color:#333;">{g['away_team']} @ {g['home_team']}</td>
          <td style="padding:8px;font-size:11px;color:#888;">{g['away_pitcher_name']} vs {g['home_pitcher_name']}</td>
          <td style="padding:8px;font-size:11px;color:#888;white-space:nowrap;">{pf_str}</td>
        </tr>"""

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:640px;margin:0 auto;background:#fff;">
      <div style="background:#0a0a0a;padding:24px;text-align:center;">
        <h1 style="color:#fff;margin:0;font-size:22px;letter-spacing:2px;">⚾ BTS EDGE</h1>
        <p style="color:#888;margin:4px 0 0;font-size:12px;">Daily Hit Predictions · {TODAY}</p>
      </div>
      <div style="padding:12px 24px;background:#f8f8f8;font-size:11px;color:#888;text-align:center;line-height:1.8;">
        <strong style="color:#555;">Scoring model:</strong>
        Recent form 22% · Advanced metrics 20% · Pitcher matchup 18% · Season AVG 15% · Platoon 12% · Park factor 8% · Home/Away 5%
      </div>
      <div style="padding:24px;">
        <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">Today's Top {len(picks)} Picks</h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#f8f8f8;">
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;width:32px;">#</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PLAYER · MATCHUP · STATCAST</th>
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
        <p style="font-size:11px;color:#aaa;margin:0;">
          BTS Edge · Automated daily at 8am ET · MLB Stats API + Baseball Savant Statcast
        </p>
      </div>
    </div>"""
    return html

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Fetching today's games...")
    games = get_todays_games()
    print(f"Found {len(games)} games.")

    print("Fetching season batter stats...")
    players = get_batter_stats()
    print(f"Found {len(players)} qualified batters.")

    print("Fetching Statcast advanced metrics from Baseball Savant...")
    statcast = get_statcast_metrics()

    print("Scoring players with full model...")
    picks = get_top_picks(players, games, statcast)
    if picks:
        print(f"Top pick: {picks[0]['name']} ({picks[0]['confidence']}%)")
    else:
        print("No picks found.")

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
