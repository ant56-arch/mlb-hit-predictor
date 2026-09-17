"""
merge_and_analyze.py — ONE-TIME script, run AFTER pull_season_data.py

Combines:
  1. research_gamelogs.json  — historical games through July 7 (no bias,
     every game the top 100 OPS players played, hit or not)
  2. picks_history.json      — our live picks from July 8 onward, with
     factor scores already attached and results already known

...into one unified file: combined_gamelogs.json

Then answers the actual question: for every game, which factor scored
highest, and how often did THAT factor's top game actually result in a hit?
This tells us, across the whole season, which of our weighted factors is
the real driver of hits — not a guess, a tally.

NOTE: superseded by research/train_model.py, which fits real logistic
regression coefficients on this same underlying data instead of eyeballing
hand-picked factor weights. This script still works for a rough sanity check,
but its factor definitions predate the trained model and won't line up with
predict.py's current factor groupings exactly.
"""

import json

RESEARCH_FILE = "research_gamelogs.json"
PICKS_FILE = "picks_history.json"
OUTPUT_FILE = "combined_gamelogs.json"


def normalize_research_row(row):
    """Reshape a research_gamelogs.json row into factor-score form so it
    can be compared apples-to-apples with picks_history.json rows."""

    # Recent form score (0-22 scale, same formula as predict.py)
    recent = row.get("recent_form_avg")
    recent_score = (recent / 0.300) * 22 if recent else 11.0

    # Advanced metrics (0-20 scale) — will default to neutral since
    # the Statcast pull returned 0 players this run (known gap)
    ev = row.get("exit_velo")
    ev_score = max(0, min(1, (ev - 84) / 10)) * 7 if ev else 3.5
    barrel = row.get("barrel_pct")
    barrel_score = max(0, min(1, barrel / 15)) * 7 if barrel else 3.5
    hh = row.get("hard_hit_pct")
    hh_score = max(0, min(1, (hh - 25) / 30)) * 4 if hh else 2.0
    ops_score = max(0, min(1, (row.get("season_ops", 0.7) - 0.600) / 0.400)) * 2
    advanced_score = ev_score + barrel_score + hh_score + ops_score

    # Pitcher matchup (0-18 scale) — real per-game data this time
    era = row.get("opp_pitcher_era")
    whip = row.get("opp_pitcher_whip")
    k9 = row.get("opp_pitcher_k9")
    if era is not None and whip is not None and k9 is not None:
        era_norm = max(0, min(1, (6.00 - era) / 4.00))
        whip_norm = max(0, min(1, (1.80 - whip) / 0.80))
        k_norm = max(0, min(1, (14.0 - k9) / 10.0))
        pitcher_score = (1 - (era_norm * 0.4 + whip_norm * 0.35 + k_norm * 0.25)) * 18
    else:
        pitcher_score = None  # genuinely unknown, not fabricated

    # Season average (0-15 scale)
    season_score = max(0, min(1, row.get("season_avg", 0) / 0.350)) * 15

    # Platoon (12 or 6)
    b_hand = row.get("batter_hand")
    p_hand = row.get("opp_pitcher_hand")
    if b_hand and p_hand:
        platoon_score = 12 if b_hand != p_hand else 6
    else:
        platoon_score = None

    # Park factor (0-6 scale, matches current live weighting)
    pf = row.get("park_factor", 1.0)
    park_score = max(0, min(1, (pf - 0.90) / 0.25)) * 6

    # Home/away (3 or 1.5, matches current live weighting)
    home_score = 3 if row.get("is_home") else 1.5

    factors = {
        "recent_form": round(recent_score, 2),
        "advanced_metrics": round(advanced_score, 2),
        "season_avg": round(season_score, 2),
        "park_factor": round(park_score, 2),
        "home_away": round(home_score, 2),
    }
    if pitcher_score is not None:
        factors["pitcher_matchup"] = round(pitcher_score, 2)
    if platoon_score is not None:
        factors["platoon"] = round(platoon_score, 2)

    return {
        "date": row["date"],
        "player_id": row["player_id"],
        "player_name": row["player_name"],
        "got_hit": row["got_hit"],
        "hits": row["hits"],
        "at_bats": row["at_bats"],
        "factors": factors,
        "source": "research",
    }


def normalize_picks_row(row):
    """picks_history.json rows already have factor scores computed —
    just tag the source and keep only resolved (non-null) results."""
    if row.get("got_hit") is None:
        return None
    return {
        "date": row["date"],
        "player_id": row["player_id"],
        "player_name": row["player_name"],
        "got_hit": row["got_hit"],
        "hits": row.get("hits"),
        "at_bats": row.get("at_bats"),
        "factors": row.get("factors", {}),
        "source": "live_pick",
    }


def combine():
    combined = []

    try:
        with open(RESEARCH_FILE, "r") as f:
            research = json.load(f)
        for row in research.get("rows", []):
            combined.append(normalize_research_row(row))
        print(f"Loaded {len(research.get('rows', []))} research rows.")
    except FileNotFoundError:
        print(f"WARNING: {RESEARCH_FILE} not found — run pull_season_data.py first.")

    try:
        with open(PICKS_FILE, "r") as f:
            picks = json.load(f)
        added = 0
        for row in picks.get("picks", []):
            norm = normalize_picks_row(row)
            if norm:
                combined.append(norm)
                added += 1
        print(f"Loaded {added} resolved live-pick rows.")
    except FileNotFoundError:
        print(f"WARNING: {PICKS_FILE} not found.")

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"rows": combined}, f, indent=2)
    print(f"Saved {len(combined)} total rows to {OUTPUT_FILE}.")
    return combined


def analyze(rows):
    print(f"\n{'='*60}")
    print(f"ANALYSIS — {len(rows)} total game rows")
    print(f"{'='*60}\n")

    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    for src, srows in by_source.items():
        hits = sum(1 for r in srows if r["got_hit"])
        print(f"{src}: {len(srows)} rows, {hits} hits ({hits/len(srows)*100:.1f}%)")

    overall_hits = sum(1 for r in rows if r["got_hit"])
    print(f"\nCOMBINED: {len(rows)} rows, {overall_hits} hits ({overall_hits/len(rows)*100:.1f}%)\n")

    # For each row, find which factor scored highest -> that's the "top reason"
    factor_hits = {}
    for r in rows:
        factors = r.get("factors", {})
        if not factors:
            continue
        top_factor = max(factors, key=lambda k: factors[k])
        factor_hits.setdefault(top_factor, {"hits": 0, "total": 0})
        factor_hits[top_factor]["total"] += 1
        if r["got_hit"]:
            factor_hits[top_factor]["hits"] += 1

    print(f"{'FACTOR':<20}{'TIMES TOP REASON':<20}{'HIT RATE WHEN TOP'}")
    print("-" * 60)
    ranked = []
    for factor, d in factor_hits.items():
        rate = d["hits"] / d["total"] * 100 if d["total"] else 0
        ranked.append((factor, d["total"], rate))
    ranked.sort(key=lambda x: x[2], reverse=True)
    for factor, total, rate in ranked:
        print(f"{factor:<20}{total:<20}{rate:.1f}%")

    print(f"\n{'='*60}")
    print("Read this as: when FACTOR X was the single biggest contributor")
    print("to a player's score that game, how often did they actually get")
    print("a hit? Higher = that factor is a more reliable real predictor.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    rows = combine()
    if rows:
        analyze(rows)
