"""
build_pages.py - generates the NBA Edge pages into dist/nba/.

Run after the MLB build (build_site.py wipes dist/). Reads
nba/picks_history.json and nba/model_weights.json and writes:
  index.html    - today's games with a pick and win chance for each, and the record
  history.html  - any past day's picks and how they did
  accuracy.html - predicted vs. actual over the season, and last season's backtest
  terms.html, privacy.html
  summary.json  - today's three most confident picks, read by the home page

Shares MLB Edge's stylesheet, scripts and page helpers; nba.js adds the
History day picker for games.
"""

import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from build_site import (DASH, ET, NOW, card, pct, pill, script_json, statline)  # noqa: E402
import games as games_mod  # noqa: E402
import model_page  # noqa: E402

WEB_DIR = os.path.join(ROOT, "web")
OUT_DIR = os.path.join(ROOT, "dist", "nba")
ASSETS = ("style.css", "site.js", "nba.js")

HOME_URL = "https://ant56-arch.github.io/"
NFL_EDGE = "https://ant56-arch.github.io/nfl-edge"
MLB_EDGE = "https://ant56-arch.github.io/mlb-hit-predictor"
SPORT_LINKS = [("All", HOME_URL), ("NFL", f"{NFL_EDGE}/nfl/index.html"), ("CFB", f"{NFL_EDGE}/cfb/index.html"),
               ("MLB", f"{MLB_EDGE}/"), ("NBA", None), ("Schedule", "https://ant56-arch.github.io/schedule.html")]
TAGLINE = "Who wins every NBA game tonight and how likely it is, from a model graded against every final score."
TOP_N = 3
STRONG = 70  # win chance, in percent, that counts as a strong pick

FAVICON = ('data:image/svg+xml,'
           '%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22%3E'
           '%3Crect width=%2232%22 height=%2232%22 fill=%22%23121314%22/%3E'
           '%3Cpath d=%22M8 23V9h3.3l6.9 8.6V9H24v14h-3.3l-6.9-8.6V23z%22 fill=%22%23e5793b%22/%3E'
           '%3C/svg%3E')


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def asset_version():
    import hashlib
    h = hashlib.md5()
    for name in ASSETS:
        with open(os.path.join(WEB_DIR, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:10]


def day_label(iso):
    return date.fromisoformat(iso).strftime("%a, %b %-d")


def tip_time(p):
    return datetime.fromisoformat(p["start"]).astimezone(ET).strftime("%-I:%M %p")


def logo(abbr):
    return (f'<img class="team-logo" src="https://a.espncdn.com/i/teamlogos/nba/500/scoreboard/{abbr.lower()}.png" '
            f'alt="" loading="lazy" onerror="this.style.display=\'none\'">')


def graded(picks):
    return [p for p in picks if p.get("correct") is not None and not p.get("void")]


def season_of(iso):
    """NBA seasons run October to June; 2026-10-21 and 2027-04-01 are both in 2026-27."""
    d = date.fromisoformat(iso)
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}-{str(start + 1)[2:]}"


def top_per_day(picks, n=TOP_N):
    by_day = defaultdict(list)
    for p in picks:
        by_day[p["date"]].append(p)
    return [p for day in by_day.values() for p in sorted(day, key=lambda p: -p["prob"])[:n]]


def wl(picks):
    w = sum(p["correct"] for p in picks)
    return w, len(picks) - w

# The Sports Edge brand mark in the top bar, same on every Edge site.
BRAND_MARK = ('<svg class="brand-mark" viewBox="0 0 32 32" aria-hidden="true"><path d="M9 3h22l-8 26H1z" fill="#e5793b"/>'
              '<path transform="translate(4.3 0) skewX(-15)" d="M10 9h12v3.2h-8.4v2.3h7.4v3h-7.4v2.3H22V23H10z" '
              'fill="#121314"/></svg>')


# ── Page chrome ──────────────────────────────────────────────────────────────
def page_shell(title, active, body_html, charts=False):
    tabs = [("index.html", "Home"), ("history.html", "History"), ("accuracy.html", "Accuracy"),
            ("model.html", "Model")]
    nav = "".join(
        f'<a href="{href}" class="active" aria-current="page">{label}</a>' if href == active
        else f'<a href="{href}">{label}</a>' for href, label in tabs)
    switcher = "".join(
        f'<a href="#" class="sport-tab active" aria-current="page">{name}</a>' if url is None
        else f'<a href="{url}" class="sport-tab">{name}</a>' for name, url in SPORT_LINKS)
    chart_js = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>' if charts else ""
    ver = asset_version()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} | NBA Edge</title>
<meta name="description" content="{TAGLINE}">
<link rel="icon" href="{FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700&family=Barlow+Condensed:ital,wght@0,600;0,700;0,800;1,700;1,800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css?v={ver}">
{chart_js}
</head>
<body>
<a class="skip-link" href="#main-content">Skip to main content</a>
<header class="topbar">
  <div class="topbar-inner">
    <a class="brand" href="{HOME_URL}">{BRAND_MARK}<span class="brand-name">Sports <span>Edge</span></span></a>
    <nav class="sport-switcher" aria-label="Sport">{switcher}</nav>
  </div>
</header>
<aside class="scoreboard" aria-label="Latest top picks" hidden></aside>
<header class="masthead" data-sport="NBA">
  <div class="masthead-inner">
    <h1 class="wordmark">NBA <span>EDGE</span></h1>
    <div class="tagline">{TAGLINE}</div>
    <div class="updated-chip">Updated {NOW.strftime("%b %-d, %Y %-I:%M %p")} ET</div>
  </div>
</header>
<nav class="tabs" aria-label="Sections"><div class="tabs-inner">{nav}</div></nav>
<div class="wrap">
  <main id="main-content">
  {body_html}
  </main>
  {footer()}
</div>
<script src="site.js?v={ver}"></script>
<script src="nba.js?v={ver}"></script>
</body>
</html>"""


def footer():
    return f"""<footer class="site-footer">
    <div class="footer-brand">NBA <span>Edge</span></div>
    <p class="footer-text">Win chances come from a model fit on past NBA seasons: each team's Elo rating, point
      differential this season and over its last 10 games, rest and back-to-backs, home court, and how much of its
      regular rotation is listed out on the injury report. Scores, box scores and injury reports via ESPN. No
      betting odds are used.</p>
    <p class="footer-text">For entertainment and research only. This is not betting advice, and past results
      don't predict future ones. If gambling is a problem for you or someone you know, call 1-800-GAMBLER.</p>
    <nav class="footer-links" aria-label="Site">
      <a href="terms.html">Terms of Use</a>
      <a href="privacy.html">Privacy Policy</a>
      <a href="{HOME_URL}">All sites</a>
      <a href="{MLB_EDGE}/">MLB Edge</a>
      <span>&copy; {NOW.year} NBA Edge. Updated from final scores every night.</span>
    </nav>
  </footer>"""


# ── Home ─────────────────────────────────────────────────────────────────────
def result_html(p):
    if p.get("void"):
        return pill("NO DECISION", "void")
    if p.get("correct") is None:
        return f'<span class="faint">{DASH}</span>'
    score = f"{p['away']} {p['away_pts']}, {p['home']} {p['home_pts']} "
    return score + (pill("WIN", "positive") if p["correct"] else pill("LOSS", "danger"))


def game_notes(p):
    notes = []
    for side in ("away", "home"):
        if p.get(f"{side}_out"):
            notes.append(f"{p[side]} without {', '.join(p[f'{side}_out'])}")
    rest = p.get("rest", {})
    b2b = [p[s] for s in ("away", "home") if rest.get(s) == 0]
    if b2b:
        notes.append(f"{' and '.join(b2b)} on a back-to-back")
    return "; ".join(notes)


def games_table(picks):
    rows = ""
    for p in sorted(picks, key=lambda p: (p["start"], p["game_id"])):
        at = "vs" if p.get("neutral") else "@"
        notes = game_notes(p)
        rows += f"""<tr>
          <td><div class="player-name">{escape(p['away'])} {at} {escape(p['home'])}</div>
            <div class="player-meta">{tip_time(p)} ET · {escape(p['away'])} {p.get('away_record', '')}, {escape(p['home'])} {p.get('home_record', '')}</div></td>
          <td data-label="Pick"><span class="matchup-team">{logo(p['pick'])}{escape(p['pick'])}</span></td>
          <td data-label="Win chance" class="num prob">{p['prob']:.0f}%</td>
          <td data-label="Projected" class="num">by {p['margin']:.1f}</td>
          <td{' data-label="Notes"' if notes else ''} class="why">{escape(notes)}</td>
          <td data-label="Result" class="num"><span>{result_html(p)}</span></td>
        </tr>"""
    return f"""<table class="data responsive-stack">
      <thead><tr><th>Game</th><th>Pick</th><th class="num">Win chance</th><th class="num">Projected</th><th>Notes</th><th class="num">Result</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="table-footnote">Win chance is the model's estimate that its pick wins the game. Picks are refreshed
      with each injury report until tip-off, then locked. Players listed are regulars ruled out or doubtful.</div>"""


def backtest_stats(model):
    bt = model.get("backtest") or {}
    if not bt:
        return []
    strong = [b for b in bt["bands"] if b["range"][0] >= STRONG / 100]
    stats = [(f"{bt['correct']}-{bt['games'] - bt['correct']}", f"{model['backtest_season']} backtest",
              f"{pct(bt['accuracy'], 1)} of every game"),
             (pct(bt["top3"]["accuracy"], 1), f"Top {TOP_N} picks each day",
              f"{bt['top3']['correct']}-{bt['top3']['picks'] - bt['top3']['correct']}")]
    if strong:
        n = sum(b["n"] for b in strong)
        right = sum(round(b["actual"] * b["n"]) for b in strong)
        stats.append((pct(right / n, 1), f"Picks at {STRONG}%+", f"{right}-{n - right}"))
    return stats


def backtest_note(model):
    bt = model.get("backtest") or {}
    b = model.get("baselines", {})
    if not bt:
        return ""
    return (f'<div class="table-footnote">Backtest: fit only on seasons before {model["backtest_season"]}, then used '
            f'to pick all {bt["games"]} regular-season games of {model["backtest_season"]}, each with only what was '
            f'known that morning. The home team won {pct(b.get("home_team_accuracy"), 1)} of those games, and team '
            f'ratings alone picked {pct(b.get("elo_only_accuracy"), 1)}. See the Accuracy tab for more.</div>')


def track_record(history, model):
    picks = history["picks"]
    body = ""
    if picks:
        season = season_of(max(p["date"] for p in picks))
        g = graded([p for p in picks if season_of(p["date"]) == season])
        if g:
            w, l = wl(g)
            tw, tl = wl(top_per_day(g))
            strong = [p for p in g if p["prob"] >= STRONG]
            sw, sl = wl(strong)
            body = statline([
                (f"{w}-{l}", f"{season} record", f"{pct(w / len(g), 1)} of every game"),
                (f"{tw}-{tl}", f"Top {TOP_N} picks each day", pct(tw / (tw + tl), 1) if tw + tl else ""),
                (f"{sw}-{sl}", f"Picks at {STRONG}%+", pct(sw / len(strong), 1) if strong else DASH),
            ])
    if not body:
        body = '<div class="empty-state">No picks graded yet this season.</div>'
    bt = backtest_stats(model)
    if bt:
        body += '<div class="section-label">Backtest</div>' + statline(bt) + backtest_note(model)
    return card("Track Record", "Every pick graded against the final score", body)


def build_index(history, model):
    picks = history["picks"]
    today = NOW.date().isoformat()
    if picks:
        latest = max(p["date"] for p in picks)
        heading = "Today's Games" if latest == today else "Latest Games"
        top = sorted((p for p in picks if p["date"] == latest), key=lambda p: -p["prob"])
        picks_html = card(f"{heading}: {day_label(latest)}",
                          f"A pick for every game. The {TOP_N} most confident today: "
                          + ", ".join(f"{p['pick']} ({p['prob']:.0f}%)" for p in top[:TOP_N]) + ".",
                          games_table([p for p in picks if p["date"] == latest]))
        if latest != today:
            picks_html = card("Today's Games", "", '<div class="empty-state">No NBA games today. Picks resume on '
                              'the next game day.</div>') + picks_html
    else:
        picks_html = card("Today's Games", "", '<div class="empty-state">The season tips off in late October, '
                          'and picks start on opening night. Until then, the Track Record below shows how the model '
                          'did on every game of last season.</div>')
    return page_shell("Home", "index.html", picks_html + track_record(history, model))


# ── History ──────────────────────────────────────────────────────────────────
def build_history(history):
    by_day = defaultdict(list)
    for p in history["picks"]:
        by_day[p["date"]].append(p)
    if not by_day:
        body = card("History", "Every day's picks", '<div class="empty-state">No picks yet. The first ones go up '
                    'on opening night.</div>')
        return page_shell("History", "history.html", body)
    days = {}
    for d, picks in by_day.items():
        g = graded(picks)
        w, l = wl(g)
        days[d] = {
            "label": day_label(d) + f", {d[:4]}",
            "summary": {"wins": w, "losses": l, "voided": sum(1 for p in picks if p.get("void"))},
            "games": [{
                "matchup": f"{p['away']} {'vs' if p.get('neutral') else '@'} {p['home']}", "pick": p["pick"],
                "prob": p["prob"], "correct": p.get("correct"), "void": bool(p.get("void")),
                "score": (f"{p['away']} {p['away_pts']}, {p['home']} {p['home_pts']}"
                          if p.get("home_pts") is not None else ""),
            } for p in sorted(picks, key=lambda p: -p["prob"])],
        }
    data = {"order": sorted(days, reverse=True), "days": days}
    body = card("History", "Every day's picks and how they did. Choose a day.",
                '<select id="day-select" class="week-picker" aria-label="Day"></select>'
                '<div id="day-content" style="margin-top:16px;"></div>'
                f'<script>const NBA_HISTORY = {script_json(data)};</script>')
    return page_shell("History", "history.html", body)


# ── Accuracy ─────────────────────────────────────────────────────────────────
def band_rows(bands):
    def label(lo, hi):
        return f"{lo:.0%} and up" if hi >= 1 else f"{lo:.0%}-{hi:.0%}"
    return "".join(f"""<tr><td class="row-label">{label(*b['range'])}</td>
      <td data-label="Games" class="num">{b['n']}</td>
      <td data-label="Model said" class="num">{pct(b['predicted'], 1)}</td>
      <td data-label="Actually won" class="num accent">{pct(b['actual'], 1)}</td></tr>""" for b in bands)


def band_table(bands, note):
    return f"""<table class="data record-table responsive-stack">
      <thead><tr><th>Win chance</th><th class="num">Games</th><th class="num">Model said</th><th class="num">Actually won</th></tr></thead>
      <tbody>{band_rows(bands)}</tbody></table><div class="table-footnote">{note}</div>"""


def live_bands(picks):
    out = []
    for lo, hi in ((50, 60), (60, 70), (70, 80), (80, 101)):
        b = [p for p in picks if lo <= p["prob"] < hi]
        if b:
            out.append({"range": [lo / 100, min(hi, 100) / 100], "n": len(b),
                        "predicted": sum(p["prob"] for p in b) / len(b) / 100,
                        "actual": sum(p["correct"] for p in b) / len(b)})
    return out


def chart_data(groups, titles):
    """groups: [(label, [(predicted 0-1, correct bool), ...]), ...] in order."""
    labels, actual, predicted, cum_a, cum_p = [], [], [], [], []
    seen = right = conf = 0
    for label, items in groups:
        labels.append(label)
        actual.append(round(sum(c for _, c in items) / len(items), 3))
        predicted.append(round(sum(p for p, _ in items) / len(items), 3))
        seen += len(items)
        right += sum(c for _, c in items)
        conf += sum(p for p, _ in items)
        cum_a.append(round(right / seen, 3))
        cum_p.append(round(conf / seen, 3))
    return {"labels": labels, "actual": actual, "predicted": predicted,
            "cumulative_actual": cum_a, "cumulative_predicted": cum_p, "titles": titles}


CHART_TITLES = {
    "weekly": "Picks Won by Week: Actual vs. What the Model Predicted",
    "weekly_actual": "Actual win rate", "weekly_predicted": "Model's predicted win rate",
    "cumulative": "Season-to-Date Win Rate",
}


def charts_html(data):
    return ("".join(f'<div class="chart-card" data-state="loading"><canvas id="{c}" height="90"></canvas></div>'
                    for c in ("chart-weekly", "chart-cumulative"))
            + f'<script>const ACCURACY_DATA = {script_json(data)};</script>')


def build_accuracy(history, model):
    picks = sorted(graded(history["picks"]), key=lambda p: p["date"])
    parts = []
    charts = False
    if picks:
        weeks = defaultdict(list)
        for p in picks:
            d = date.fromisoformat(p["date"])
            weeks[d - timedelta(days=d.weekday())].append((p["prob"] / 100, p["correct"]))
        data = chart_data([("Wk of " + wk.strftime("%b %-d"), weeks[wk]) for wk in sorted(weeks)], CHART_TITLES)
        parts.append(card("Accuracy Over Time",
                          f"{len(picks)} graded picks: what the model predicted vs. what happened",
                          charts_html(data)))
        parts.append(card("Calibration", "Picks grouped by the win chance the model gave them",
                          band_table(live_bands(picks), "If the model is honest, each row's two percentages should "
                                     "be close. Small rows swing a lot.")))
        charts = True

    bt = model.get("backtest")
    if bt:
        if not charts:
            # Before the season, chart last season's backtest month by month instead.
            groups = []
            for m in bt["months"]:
                n, c = m["n"], m["correct"]
                groups.append((date.fromisoformat(m["month"] + "-01").strftime("%b %Y"),
                               [(m["predicted"], True)] * c + [(m["predicted"], False)] * (n - c)))
            titles = dict(CHART_TITLES, weekly=f"{model['backtest_season']} Backtest by Month: Actual vs. Predicted",
                          cumulative=f"{model['backtest_season']} Backtest, Season to Date")
            parts.append(card("Backtest Over Time", f"Every regular-season game of {model['backtest_season']}, "
                              "picked by a model that never saw that season", charts_html(chart_data(groups, titles))))
            charts = True
        note = (f"Fit on {model['trained_on'].split(' to ')[0]} through the season before {model['backtest_season']}, "
                f"then tested on all {bt['games']} regular-season games of {model['backtest_season']}. It picked "
                f"{pct(bt['accuracy'], 1)} of them right (it expected {pct(bt['predicted_accuracy'], 1)}), and its "
                f"projected margins were off by {bt['margin_mae']:.1f} points on average.")
        post = model.get("backtest_postseason")
        if post:
            note += f" In the play-in and playoffs it went {post['correct']}-{post['games'] - post['correct']}."
        parts.append(card("Backtest", "How the model did on a season it was never trained on",
                          statline(backtest_stats(model)) + band_table(bt["bands"], note)))
    if not parts:
        parts.append(card("Accuracy", "", '<div class="empty-state">No graded picks yet.</div>'))
    return page_shell("Accuracy", "accuracy.html", "".join(parts), charts=charts)


# ── Schedule tab and scoreboard strip ────────────────────────────────────────
def attach_picks(slate, history):
    """Each game gets the model's pick and win chance, if it made one."""
    picks = {(p["date"], p["away"], p["home"]): p for p in history["picks"]}
    for g in slate["games"]:
        day = games_mod.start_et(g).date().isoformat()
        p = picks.get((day, g["away"]["abbr"], g["home"]["abbr"]))
        if p:
            g["pick"] = {"text": f"{p['pick']} {p['prob']:.0f}%",
                         "result": None if p.get("void") or p.get("correct") is None else bool(p["correct"])}
    return slate


# ── Model tab ────────────────────────────────────────────────────────────────
FACTOR_LABELS = {
    "home_court": ("Home court", "points for the home team"),
    "elo": ("Team rating (Elo) gap", "points per 100 rating points"),
    "net": ("Season point differential gap", "points per point of differential"),
    "recent": ("Last 10 games form gap", "points per point of differential"),
    "rest": ("Extra rest", "points per extra day off vs. the opponent"),
    "b2b_home": ("Home team on a back-to-back", ""),
    "b2b_away": ("Away team on a back-to-back", ""),
    "missing_home": ("Home team's missing players", "per 10 points of missing player value"),
    "missing_away": ("Away team's missing players", "per 10 points of missing player value"),
}


def recipe_setup(recipe):
    k, carry, years = recipe.get("elo_k", 20), recipe.get("elo_carry", 0.75), recipe.get("years", 3)
    speed = "steady" if k <= 15 else "medium" if k <= 20 else "fast"
    return [
        ("Learns from:", f"the last {years} seasons of games"),
        ("Team ratings:", f"{speed} - each result moves a team's Elo rating by up to {k} points"),
        ("Over the summer:", f"keeps {carry:.0%} of each team's rating; the rest resets toward average"),
    ]


def build_model(model, runs):
    runs = list(reversed(runs))

    def skill(ll, baseline):
        return None if ll is None or not baseline else 1 - ll / baseline

    rows = []
    for r in runs:
        label, tone = model_page.decision(r)
        base = r["holdout"].get("home_rate_log_loss")
        chosen_ll = r["best_recipe"]["log_loss"] if r.get("switched_recipe") else r.get("current_recipe_log_loss")
        rows.append({
            "date": model_page.short_date(r["run_at"]), "data_through": model_page.short_date(r.get("data_through")),
            "tested": len(r.get("candidates", [])), "decision": label, "tone": tone, "reason": r["reason"].capitalize() + ".",
            "before": skill(r["live_model"].get("holdout_log_loss"), base),
            "after": skill(chosen_ll, base) if r.get("deployed") else None,
        })
    last = runs[0] if runs else {}
    now_w, before_w = model.get("coef", {}), last.get("weights_before") or {}
    factors = [{"label": FACTOR_LABELS.get(f, (f, ""))[0], "note": FACTOR_LABELS.get(f, (f, ""))[1],
                "now": now_w[f], "before": before_w.get(f), "fmt": lambda v: f"{v:+.2f} pts"}
               for f in model.get("features", []) if f in now_w]
    trained = model.get("trained_at")
    spec = {
        "intro": "Once a week during the season it checks itself against the newest games and only changes when a "
                 "new version clearly predicts better.",
        "tiles": [
            (model_page.short_date(trained)[:-6] if trained else "-", "Last retrained",
             f"games through {model_page.short_date(model.get('trained_through'))}" if model.get("trained_through") else ""),
            (f"{model.get('training_games', 0):,}", "Games learned from",
             " to ".join(model_page.short_date(d) for d in model.get("training_dates", []))),
            ("Weekly", "Next check", "once 50+ new games are in; waits in the offseason"),
            (rows[0]["decision"].split(" ")[0] if rows else "-", "Last decision",
             f"{rows[0]['tested']} versions tested" if rows else "no retrains yet"),
        ],
        "setup": recipe_setup(model.get("recipe", {})) + [
            ("Looks at:", f"{len(now_w)} factors for every game, listed below, including injuries on the day"),
        ],
        "runs": rows,
        "score_name": "Better than guessing",
        "score_fmt": lambda v: pct(v, 1),
        "higher_better": True,
        "factors": factors,
        "factors_note": "Each number is how many points of margin that factor adds for the home team (negative "
                        "helps the away team). Before is the model that was live until the last retrain.",
        "empty": "No retrains logged yet. The first one runs about a week into the season.",
    }
    return page_shell("Model", "model.html", model_page.render(spec))


# ── Home page summary ────────────────────────────────────────────────────────
def build_summary(history, model):
    """summary.json for the NBA card on the home page (github.com/ant56-arch/ant56-arch.github.io)."""
    picks = history["picks"]
    summary = {"updated": NOW.isoformat(), "heading": None, "picks": [], "record": None,
               "empty": "No NBA picks yet. They start on opening night in late October.", "result_labels": ["WIN", "LOSS"],
               "retrained": model.get("trained_at"), "model_url": "model.html"}
    if picks:
        latest = max(p["date"] for p in picks)
        prefix = "Today" if latest == NOW.date().isoformat() else "Latest"
        summary["heading"] = f"{prefix}: {day_label(latest)}"
        top = sorted((p for p in picks if p["date"] == latest), key=lambda p: -p["prob"])[:TOP_N]
        summary["picks"] = [{
            "label": f"{p['pick']} over {p['away'] if p['pick'] == p['home'] else p['home']}",
            "sub": f"{p['away']} {'vs' if p.get('neutral') else '@'} {p['home']} · {tip_time(p)} ET",
            "value": f"{p['prob']:.0f}%",
            "result": None if p.get("void") or p.get("correct") is None else bool(p["correct"]),
        } for p in top]
        season = season_of(latest)
        g = graded([p for p in picks if season_of(p["date"]) == season])
        if g:
            w, l = wl(g)
            summary["record"] = {"value": f"{w}-{l}", "label": f"{season} record",
                                 "sub": f"{pct(w / len(g), 1)} of games picked right"}
    # Only picks actually made count here, never last season's backtest.
    return summary


# ── Legal pages ──────────────────────────────────────────────────────────────
LEGAL_EFFECTIVE_DATE = "September 23, 2026"


def build_terms():
    body = f"""<article class="prose">
      <h1 class="page-title">Terms of Use</h1>
      <p>Effective {LEGAL_EFFECTIVE_DATE}. By using NBA Edge ("this site") you agree to these terms. If you don't
        agree, please don't use the site.</p>
      <h2>What this site is</h2>
      <p>This site publishes computer-generated estimates of which team wins each NBA game and how likely that is,
        along with a record of how past estimates turned out. It is a free, non-commercial project provided for
        <strong>entertainment and research only</strong>.</p>
      <h2>Not betting or financial advice</h2>
      <p>Nothing on this site is a recommendation to place any bet. Estimates are often wrong, and past results
        don't predict future ones. You are solely responsible for any decision you make, including any money you
        wager or lose.</p>
      <h2>Legal age and location</h2>
      <p>Sports betting is illegal in some places and restricted to adults everywhere it is legal. It is your
        responsibility to know and follow the laws where you live. If gambling is causing problems for you or
        someone you know, call or text <strong>1-800-GAMBLER</strong> (US).</p>
      <h2>No warranty</h2>
      <p>The site and its data are provided "as is", without warranties of any kind. Scores, box scores and injury
        reports come from ESPN and may be late, incomplete or incorrect. The site may change or go offline at any
        time without notice.</p>
      <h2>Limitation of liability</h2>
      <p>To the fullest extent allowed by law, the operator of this site is not liable for any loss or damage
        arising from your use of, or reliance on, the site or its content.</p>
      <h2>Trademarks and affiliation</h2>
      <p>This site is independent. It is not affiliated with, endorsed by or sponsored by the National Basketball
        Association, any team, ESPN or any sportsbook. Team names and logos are trademarks of their owners and are
        shown only to identify teams.</p>
      <h2>Changes</h2>
      <p>These terms may be updated. The effective date above shows when they last changed, and continued use of
        the site means you accept the current version.</p>
    </article>"""
    return page_shell("Terms of Use", None, body)


def build_privacy():
    body = f"""<article class="prose">
      <h1 class="page-title">Privacy Policy</h1>
      <p>Effective {LEGAL_EFFECTIVE_DATE}. This site is a static website with no accounts, sign-ups, forms,
        comments or payments.</p>
      <h2>What we collect</h2>
      <p><strong>Nothing.</strong> This site sets no cookies, runs no analytics or advertising trackers, and does
        not ask for or store any personal information.</p>
      <h2>Third parties your browser contacts</h2>
      <p>Loading a page makes your browser request files from these services, which can see your IP address,
        browser type and the page that made the request, as any web server does:</p>
      <ul>
        <li><strong>GitHub Pages</strong> hosts the site and may keep server logs, including IP addresses, for
          security and operations. See the <a href="https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement">GitHub Privacy Statement</a>.</li>
        <li><strong>Google Fonts</strong> serves the site's typefaces. See the
          <a href="https://developers.google.com/fonts/faq/privacy">Google Fonts privacy FAQ</a>.</li>
        <li><strong>jsDelivr</strong> serves the charting library on the Accuracy page. See the
          <a href="https://www.jsdelivr.com/terms/privacy-policy-jsdelivr-net">jsDelivr privacy policy</a>.</li>
        <li><strong>ESPN's image servers</strong> (a.espncdn.com) serve the team logos.</li>
      </ul>
      <p>We don't receive or control the data those services log.</p>
      <h2>Children</h2>
      <p>This site isn't directed at children and doesn't knowingly collect information from anyone.</p>
      <h2>Changes</h2>
      <p>If this policy changes, the effective date above will be updated.</p>
    </article>"""
    return page_shell("Privacy Policy", None, body)


def main():
    history = load_json(os.path.join(HERE, "picks_history.json"), {"picks": []})
    model = load_json(os.path.join(HERE, "model_weights.json"), {})
    games_slate = attach_picks(games_mod.load("nba"), history)
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    pages = {
        "index.html": build_index(history, model),
        "history.html": build_history(history),
        "accuracy.html": build_accuracy(history, model),
        "model.html": build_model(model, load_json(os.path.join(HERE, "model_history.json"), {"runs": []})["runs"]),
        "terms.html": build_terms(),
        "privacy.html": build_privacy(),
    }
    for name, html in pages.items():
        with open(os.path.join(OUT_DIR, name), "w") as f:
            f.write(html)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(build_summary(history, model), f, indent=1)
    games_mod.write_json(os.path.join(OUT_DIR, "games.json"), "nba", games_slate, NOW.isoformat())
    for asset in ASSETS:
        shutil.copy(os.path.join(WEB_DIR, asset), os.path.join(OUT_DIR, asset))
    print(f"Built {len(pages)} NBA pages in {OUT_DIR}")


if __name__ == "__main__":
    main()
