import os
import requests
import resend
from datetime import datetime, date

# ── Config ────────────────────────────────────────────────────────────────────
resend.api_key = os.environ["RESEND_API_KEY"]
TO_EMAIL = os.environ["TO_EMAIL"]
TODAY = date.today().strftime("%B %d, %Y")

# ── Step 1: Fetch today's games and probable starters ─────────────────────────
def get_todays_games():
    url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&hydrate=probablePitcher,lineups"
    r = requests.get(url, timeout=15)
    games = []
    for date_entry in r.json().get("dates", []):
        for game in date_entry.get("games", []):
            away = game.get("teams", {}).get("away", {})
            home = game.get("teams", {}).get("home", {})
            games.append({
                "game_id": game["gamePk"],
                "away_team": away.get("team", {}).get("name", ""),
                "home_team": home.get("team", {}).get("name", ""),
                "away_pitcher": away.get("probablePitcher", {}).get("fullName", "TBD"),
                "home_pitcher": home.get("probablePitcher", {}).get("fullName", "TBD"),
                "venue": game.get("venue", {}).get("name", ""),
            })
    return games

# ── Step 2: Fetch batter stats from MLB Stats API ─────────────────────────────
def get_batter_stats():
    year = datetime.now().year
    url = (
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&season={year}"
        f"&limit=200&offset=0"
        f"&sortStat=avg&order=desc"
    )
    r = requests.get(url, timeout=15)
    data = r.json()
    players = []
    for entry in data.get("stats", [])[0].get("splits", []):
        s = entry.get("stat", {})
        name = entry.get("player", {}).get("fullName", "Unknown")
        ab = int(s.get("atBats", 0))
        if ab < 100:
            continue
        avg = float(s.get("avg", 0) or 0)
        obp = float(s.get("obp", 0) or 0)
        slg = float(s.get("slg", 0) or 0)
        hits = int(s.get("hits", 0))
        k_pct = int(s.get("strikeOuts", 0)) / ab if ab > 0 else 0.25
        score = avg * 40 + obp * 30 + slg * 20 + (1 - k_pct) * 10
        players.append({
            "name": name,
            "avg": avg,
            "obp": obp,
            "slg": slg,
            "hits": hits,
            "ab": ab,
            "score": score,
        })
    players.sort(key=lambda x: x["score"], reverse=True)
    return players

# ── Step 3: Get top picks ──────────────────────────────────────────────────────
def get_top_picks(players, top_n=10):
    top = players[:top_n]
    max_score = top[0]["score"] if top else 1
    for p in top:
        p["confidence"] = round((p["score"] / max_score) * 100, 1)
    return top

# ── Step 4: Build and send email ──────────────────────────────────────────────
def build_email(picks, games):
    rows = ""
    for i, p in enumerate(picks, 1):
        conf = p["confidence"]
        color = "#0F6E56" if conf >= 90 else "#BA7517" if conf >= 80 else "#666"
        medal = ["🥇","🥈","🥉"][i-1] if i <= 3 else f"#{i}"
        rows += f"""
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:10px 8px;font-weight:600;font-size:15px;">{medal}</td>
          <td style="padding:10px 8px;">
            <div style="font-weight:600;font-size:14px;color:#111;">{p['name']}</div>
            <div style="font-size:11px;color:#888;margin-top:2px;">
              AVG {p['avg']:.3f} · OBP {p['obp']:.3f} · SLG {p['slg']:.3f} · {p['ab']} AB
            </div>
          </td>
          <td style="padding:10px 8px;text-align:right;font-weight:700;color:{color};">{conf:.0f}%</td>
        </tr>"""

    game_rows = ""
    for g in games[:8]:
        game_rows += f"""
        <tr style="border-bottom:1px solid #f0f0f0;">
          <td style="padding:8px;font-size:12px;color:#333;">{g['away_team']} @ {g['home_team']}</td>
          <td style="padding:8px;font-size:11px;color:#888;">{g['away_pitcher']} vs {g['home_pitcher']}</td>
        </tr>"""

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#fff;">
      <div style="background:#0a0a0a;padding:24px;text-align:center;">
        <h1 style="color:#fff;margin:0;font-size:22px;letter-spacing:2px;">⚾ BTS EDGE</h1>
        <p style="color:#888;margin:4px 0 0;font-size:12px;">Daily Hit Predictions · {TODAY}</p>
      </div>
      <div style="padding:24px;">
        <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">
          Today's Top {len(picks)} Picks
        </h2>
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#f8f8f8;">
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">#</th>
            <th style="padding:8px;text-align:left;font-size:11px;color:#888;">PLAYER</th>
            <th style="padding:8px;text-align:right;font-size:11px;color:#888;">CONFIDENCE</th>
          </tr>
          {rows}
        </table>
      </div>
      <div style="padding:0 24px 24px;">
        <h2 style="font-size:15px;color:#111;margin:0 0 12px;text-transform:uppercase;letter-spacing:1px;">
          Today's Matchups
        </h2>
        <table style="width:100%;border-collapse:collapse;">
          {game_rows}
        </table>
      </div>
      <div style="background:#f8f8f8;padding:16px 24px;text-align:center;">
        <p style="font-size:11px;color:#aaa;margin:0;">
          BTS Edge · Automated daily at 8am ET · Powered by MLB Stats API
        </p>
      </div>
    </div>"""
    return html

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Fetching today's games...")
    games = get_todays_games()
    print(f"Found {len(games)} games today.")

    print("Fetching batter stats from MLB Stats API...")
    players = get_batter_stats()
    print(f"Found {len(players)} qualified batters.")

    print("Scoring players...")
    picks = get_top_picks(players)

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
