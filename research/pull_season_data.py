"""
pull_season_data.py — ONE-TIME research script (not part of daily automation)

Pulls full 2026 season game logs (through the All-Star break) for the top 100
hitters by OPS, attaching every factor our live model (predict.py) weights:
recent form, season AVG, platoon, home/away, park factor, pitcher matchup
(ERA/WHIP/K9/hand of the opposing starter), and season-level Statcast.

Output: research_gamelogs.json — one row per player per game played.
This file is SEPARATE from picks_history.json. It does not feed the daily
emails. Run this once. It will take a while (pitcher matchup lookups require
one boxscore call per unique game across all 100 players).
"""

import requests
import json
import time

YEAR = 2026
ALL_STAR_CUTOFF = "2026-07-07"

PARK_FACTORS = {
    "Coors Field": 1.15, "Great American Ball Park": 1.08, "Fenway Park": 1.07,
    "Globe Life Field": 1.05, "Daikin Park": 1.04, "American Family Field": 1.04,
    "Wrigley Field": 1.03, "Oriole Park at Camden Yards": 1.03, "Camden Yards": 1.03,
    "Truist Park": 1.02, "Chase Field": 1.02, "Citizens Bank Park": 1.02,
    "Yankee Stadium": 1.01, "Rogers Centre": 1.00, "Kauffman Stadium": 1.00,
    "Angel Stadium": 1.00, "Target Field": 1.00, "Sutter Health Park": 1.00,
    "Busch Stadium": 0.99, "Dodger Stadium": 0.99, "UNIQLO Field at Dodger Stadium": 0.99,
    "Progressive Field": 0.99, "PNC Park": 0.98, "Nationals Park": 0.97,
    "T-Mobile Park": 0.97, "Tropicana Field": 0.97, "Comerica Park": 0.96,
    "Rate Field": 0.96, "Petco Park": 0.96, "Oracle Park": 0.95,
    "loanDepot park": 0.95, "Citi Field": 0.95,
}

session = requests.Session()
pitcher_cache = {}
boxscore_cache = {}


def get_top_100_by_ops():
    url = (
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=hitting&season={YEAR}&limit=100&sortStat=ops&order=desc"
    )
    r = session.get(url, timeout=15)
    players = []
    for entry in r.json().get("stats", [])[0].get("splits", []):
        s = entry.get("stat", {})
        players.append({
            "id": entry.get("player", {}).get("id"),
            "name": entry.get("player", {}).get("fullName", "Unknown"),
            "team_id": entry.get("team", {}).get("id"),
            "season_avg": float(s.get("avg") or 0),
            "season_ops": float(s.get("ops") or 0),
        })
    print(f"Pulled {len(players)} players (top 100 by OPS).")
    return players


def get_player_hand(player_id):
    try:
        r = session.get(f"https://statsapi.mlb.com/api/v1/people/{player_id}", timeout=10)
        return r.json().get("people", [{}])[0].get("batSide", {}).get("code", "R")
    except:
        return "R"


def load_statcast_cache():
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/custom"
        f"?year={YEAR}&type=batter&filter=&sort=4&sortDir=desc"
        f"&min=50&selections=player_id,player_name,exit_velocity_avg,barrel_batted_rate,hard_hit_percent"
        f"&chart=false&x=exit_velocity_avg&y=exit_velocity_avg&r=no&chartType=beeswarm&csv=true"
    )
    req_headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv,*/*"}
    try:
        r = session.get(url, timeout=25, headers=req_headers)
        if r.status_code != 200:
            return {}
        lines = r.text.strip().split("\n")
        if len(lines) < 2:
            return {}
        cols = [h.strip().strip('"').lower() for h in lines[0].split(",")]
        cache = {}
        for line in lines[1:]:
            vals = [v.strip().strip('"') for v in line.split(",")]
            if len(vals) < len(cols):
                continue
            row = dict(zip(cols, vals))
            pid = row.get("player_id", "")
            if not pid:
                continue
            try:
                cache[int(float(pid))] = {
                    "exit_velo": float(row.get("exit_velocity_avg") or 0) or None,
                    "barrel_pct": float(row.get("barrel_batted_rate") or 0) or None,
                    "hard_hit_pct": float(row.get("hard_hit_percent") or 0) or None,
                }
            except:
                continue
        print(f"Loaded Statcast cache for {len(cache)} players.")
        return cache
    except Exception as e:
        print(f"Statcast pull failed: {e}")
        return {}


def get_game_log(player_id):
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=gameLog&group=hitting&season={YEAR}"
    )
    try:
        r = session.get(url, timeout=15)
        return r.json().get("stats", [])[0].get("splits", [])
    except:
        return []


def get_opposing_pitcher_for_game(game_pk, batter_team_id):
    if game_pk in boxscore_cache:
        box = boxscore_cache[game_pk]
    else:
        try:
            url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
            r = session.get(url, timeout=15)
            box = r.json()
            boxscore_cache[game_pk] = box
        except:
            return None

    teams = box.get("teams", {})
    for side in ["away", "home"]:
        team_info = teams.get(side, {}).get("team", {})
        if team_info.get("id") == batter_team_id:
            other_side = "home" if side == "away" else "away"
            pitchers = teams.get(other_side, {}).get("pitchers", [])
            if pitchers:
                return pitchers[0]
    return None


def get_pitcher_season_stats(pitcher_id):
    if pitcher_id in pitcher_cache:
        return pitcher_cache[pitcher_id]
    if not pitcher_id:
        result = {"era": None, "whip": None, "k_per9": None, "hand": None}
        pitcher_cache[pitcher_id] = result
        return result
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}?hydrate=stats(group=pitching,type=season),currentTeam"
        r = session.get(url, timeout=15)
        data = r.json()
        person = data.get("people", [{}])[0]
        hand = person.get("pitchHand", {}).get("code", "R")
        for s in person.get("stats", []):
            splits = s.get("splits", [])
            if splits:
                st = splits[0].get("stat", {})
                ip = float(st.get("inningsPitched") or 1)
                k_per9 = (float(st.get("strikeOuts") or 0) / ip) * 9
                result = {
                    "era": float(st.get("era") or 4.50),
                    "whip": float(st.get("whip") or 1.30),
                    "k_per9": round(k_per9, 2),
                    "hand": hand,
                }
                pitcher_cache[pitcher_id] = result
                return result
    except:
        pass
    result = {"era": 4.50, "whip": 1.30, "k_per9": 8.0, "hand": "R"}
    pitcher_cache[pitcher_id] = result
    return result


def compute_rolling_avg(log_sorted_oldest_first, index, window=14):
    from datetime import datetime as dt, timedelta as td
    game_date = dt.strptime(log_sorted_oldest_first[index]["date"], "%Y-%m-%d")
    window_start = game_date - td(days=window)
    hits, ab = 0, 0
    for g in log_sorted_oldest_first[:index]:
        gdate = dt.strptime(g["date"], "%Y-%m-%d")
        if window_start <= gdate < game_date:
            hits += int(g["stat"].get("hits", 0))
            ab += int(g["stat"].get("atBats", 0))
    return round(hits / ab, 3) if ab >= 5 else None


def main():
    print("Step 1: Getting top 100 players by OPS...")
    players = get_top_100_by_ops()

    print("Step 2: Loading Statcast cache...")
    statcast_cache = load_statcast_cache()

    all_rows = []
    total_games_processed = 0

    print("Step 3: Pulling game logs + opposing pitchers (this WILL take a while)...")
    for i, p in enumerate(players):
        hand = get_player_hand(p["id"])
        sc = statcast_cache.get(p["id"], {"exit_velo": None, "barrel_pct": None, "hard_hit_pct": None})
        log = get_game_log(p["id"])

        log = [g for g in log if g["date"] <= ALL_STAR_CUTOFF]
        log.sort(key=lambda g: g["date"])

        for idx, game in enumerate(log):
            stat = game.get("stat", {})
            ab = int(stat.get("atBats", 0))
            if ab == 0:
                continue

            venue_name = game.get("game", {}).get("venue", {}).get("name") or ""
            is_home = game.get("isHome", None)
            game_pk = game.get("game", {}).get("gamePk")

            rolling_avg = compute_rolling_avg(log, idx)

            pitcher_id = None
            pitcher_stats = {"era": None, "whip": None, "k_per9": None, "hand": None}
            if game_pk:
                pitcher_id = get_opposing_pitcher_for_game(game_pk, p["team_id"])
                if pitcher_id:
                    pitcher_stats = get_pitcher_season_stats(pitcher_id)
                total_games_processed += 1
                if total_games_processed % 100 == 0:
                    print(f"    ...{total_games_processed} game-boxscore lookups done "
                          f"({len(boxscore_cache)} unique games cached)")
                time.sleep(0.1)

            all_rows.append({
                "player_id": p["id"],
                "player_name": p["name"],
                "date": game["date"],
                "got_hit": int(stat.get("hits", 0)) > 0,
                "hits": int(stat.get("hits", 0)),
                "at_bats": ab,
                "recent_form_avg": rolling_avg,
                "season_avg": p["season_avg"],
                "season_ops": p["season_ops"],
                "batter_hand": hand,
                "is_home": is_home,
                "venue": venue_name,
                "park_factor": PARK_FACTORS.get(venue_name, 1.0),
                "exit_velo": sc["exit_velo"],
                "barrel_pct": sc["barrel_pct"],
                "hard_hit_pct": sc["hard_hit_pct"],
                "opp_pitcher_era": pitcher_stats["era"],
                "opp_pitcher_whip": pitcher_stats["whip"],
                "opp_pitcher_k9": pitcher_stats["k_per9"],
                "opp_pitcher_hand": pitcher_stats["hand"],
                "source": "research",
            })

        print(f"  Player {i + 1}/{len(players)} done: {p['name']} ({len(log)} games)")
        time.sleep(0.15)

    print(f"Step 4: Saving {len(all_rows)} game rows to research_gamelogs.json...")
    with open("research_gamelogs.json", "w") as f:
        json.dump({"cutoff_date": ALL_STAR_CUTOFF, "rows": all_rows}, f, indent=2)

    print(f"Done! {len(all_rows)} rows from {len(players)} players, "
          f"{len(boxscore_cache)} unique games, {len(pitcher_cache)} unique pitchers.")


if __name__ == "__main__":
    main()
