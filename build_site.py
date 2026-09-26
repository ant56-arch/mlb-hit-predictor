"""
build_site.py - generates the static MLB Edge website into dist/.

Reads picks_history.json (every pick and its result), data/slate.json (today's
full slate, written by predict.py) and model_weights.json (backtest numbers),
and writes:
  index.html    - the day's picks and the season track record
  games.html    - the team model's pick and win chance for every game today,
                  with a moneyline pick against the book price, this season's
                  live record (game picks and moneyline) and every past day's
                  results
                  (teams/picks_history.json, written by teams/predict.py)
  players.html  - every hitter in today's games, sortable
  history.html  - any past day's picks and how they did
  accuracy.html - predicted vs. actual hit rate over the season
  model.html    - how the hit model and the team model retrain themselves
  terms.html, privacy.html, 404.html
  summary.json  - today's top picks and the record, read by the home page
  games.json    - today's slate for the scoreboard strip, each game carrying
                  the team model's pick (or the top hitter when there's none)

Same look and page structure as NFL Edge (github.com/ant56-arch/nfl-edge),
with an NFL / CFB / MLB switcher linking the sites together. Published to
GitHub Pages by .github/workflows/daily.yml.
"""

import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import date, datetime, timedelta
from html import escape
from zoneinfo import ZoneInfo

import games as games_mod
import model_page
import moneyline

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
ASSETS = ("style.css", "site.js", "nba.js")  # nba.js also renders the Games tab's day picker
DIST_DIR = os.path.join(ROOT, "dist")
ET = ZoneInfo("America/New_York")
NOW = datetime.now(ET)

HOME_URL = "https://ant56-arch.github.io/"
NFL_EDGE = "https://ant56-arch.github.io/nfl-edge"
SPORT_LINKS = [("All", HOME_URL), ("NFL", f"{NFL_EDGE}/nfl/index.html"), ("CFB", f"{NFL_EDGE}/cfb/index.html"), ("MLB", None),
               ("NBA", "https://ant56-arch.github.io/mlb-hit-predictor/nba/index.html"),
               ("Schedule", "https://ant56-arch.github.io/schedule.html")]
TAGLINE = ("Who wins every MLB game and which hitters get a hit today, from models graded against every "
           "box score.")
TEAMS_DIR = os.path.join(ROOT, "teams")
TOP_GAMES = 3  # the team model's most confident picks each day
STRONG_GAME = 60  # win chance, in percent, that counts as a strong game pick
DASH = "-"

FAVICON = ('data:image/svg+xml,'
           '%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 32 32%22%3E'
           '%3Crect width=%2232%22 height=%2232%22 fill=%22%23121314%22/%3E'
           '%3Cpath d=%22M7 23V9h3.4l5.6 7.4L21.6 9H25v14h-3.3v-8.8L16 21.4l-5.7-7.2V23z%22 fill=%22%23e5793b%22/%3E'
           '%3C/svg%3E')


def load_json(name, default):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def asset_version():
    h = hashlib.md5()
    for name in ASSETS:
        with open(os.path.join(WEB_DIR, name), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:10]


def script_json(data):
    return json.dumps(data).replace("</", "<\\/")


def pill(text, style):
    return f'<span class="pill pill-{style}">{text}</span>'


def pct(x, digits=0):
    return f"{x:.{digits}%}" if x is not None else DASH


def avg(x):
    return f"{x:.3f}".lstrip("0") if x is not None else DASH


def day_label(iso):
    return date.fromisoformat(iso).strftime("%a, %b %-d")


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def logo(team_id):
    if not team_id:
        return ""
    return (f'<img class="team-logo" src="https://www.mlbstatic.com/team-logos/{team_id}.svg" alt="" '
            f'loading="lazy" onerror="this.style.display=\'none\'">')


def matchup(p):
    if p.get("team_abbr"):
        return f"{p['team_abbr']} {'vs' if p.get('is_home') else '@'} {p.get('opponent_abbr', '')}"
    return p.get("team", "")


def is_model_pick(p):
    """Picks from before the probability model have a relative score, not a hit chance."""
    return "raw_features" in p


def graded(picks):
    return [p for p in picks if p.get("got_hit") is not None and not p.get("void")]

# The Sports Edge brand mark in the top bar, same on every Edge site.
BRAND_MARK = ('<svg class="brand-mark" viewBox="0 0 32 32" aria-hidden="true"><path d="M9 3h22l-8 26H1z" fill="#e5793b"/>'
              '<path transform="translate(4.3 0) skewX(-15)" d="M10 9h12v3.2h-8.4v2.3h7.4v3h-7.4v2.3H22V23H10z" '
              'fill="#121314"/></svg>')


# ── Page chrome ──────────────────────────────────────────────────────────────
def page_shell(title, active, body_html, charts=False):
    tabs = [("index.html", "Home"), ("games.html", "Games"), ("players.html", "Players"),
            ("history.html", "History"), ("accuracy.html", "Accuracy"), ("model.html", "Model")]
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
<title>{title} | MLB Edge</title>
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
<header class="masthead" data-sport="MLB">
  <div class="masthead-inner">
    <h1 class="wordmark">MLB <span>EDGE</span></h1>
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
{'<script src="nba.js?v=' + ver + '"></script>' if active == "games.html" else ""}
</body>
</html>"""


def footer():
    return f"""<footer class="site-footer">
    <div class="footer-brand">MLB <span>Edge</span></div>
    <p class="footer-text">Hit chances come from a logistic regression fit on every game of past seasons: the
      hitter's season and recent average, at-bats per game, the opposing starter, platoon split, park and
      home/away. Stats, lineups and box scores via the MLB Stats API. A pick with no at-bats counts as no
      decision, not a miss.</p>
    <p class="footer-text">Game picks come from a second model fit on past seasons: each team's Elo rating, both
      probable starting pitchers' ERA and FIP to date, and home field. The model uses no
      betting odds; each game's moneyline pick compares its win chance with the book price on ESPN's scoreboard,
      vig removed. A postponed game counts as no decision.</p>
    <p class="footer-text">For entertainment and research only. This is not betting advice, and past results
      don't predict future ones. If gambling is a problem for you or someone you know, call 1-800-GAMBLER.</p>
    <nav class="footer-links" aria-label="Site">
      <a href="https://ant56-arch.github.io/terms.html">Terms of Use</a>
      <a href="https://ant56-arch.github.io/privacy.html">Privacy Policy</a>
      <a href="{HOME_URL}">All sites</a>
      <a href="{NFL_EDGE}/nfl/index.html">NFL Edge</a>
      <span>&copy; {NOW.year} MLB Edge. Updated from box scores every night.</span>
    </nav>
  </footer>"""


def card(title, subtitle, body_html):
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    return f"""<section class="card">
    <div class="card-header"><h2>{title}</h2>{sub}</div>
    <div class="card-body">{body_html}</div>
  </section>"""


def statline(stats):
    return '<div class="statline">' + "".join(
        f'<div class="stat"><div class="stat-value">{v}</div><div class="stat-label">{label}</div>'
        f'<div class="stat-sub">{sub}</div></div>' for v, label, sub in stats) + "</div>"


# ── Home ─────────────────────────────────────────────────────────────────────
def result_html(p):
    if p.get("void"):
        return pill("NO DECISION", "void")
    if p.get("got_hit") is None:
        return f'<span class="faint">{DASH}</span>'
    line = f"{p['hits']}-for-{p['at_bats']} "
    return line + (pill("HIT", "positive") if p["got_hit"] else pill("MISS", "danger"))


def pick_meta(p):
    parts = [matchup(p)]
    if p.get("game_time"):
        parts.append(f"{p['game_time']} ET")
    if p.get("batting_order"):
        parts.append(f"Batting {ordinal(p['batting_order'])}")
    elif "lineup_confirmed" in p:
        parts.append("Lineup not posted yet")
    if p.get("hit_streak"):
        parts.append(f"{p['hit_streak']}-game hit streak")
    return " · ".join(escape(x) for x in parts)


def picks_table(picks):
    rows = ""
    for p in sorted(picks, key=lambda p: -p["confidence"]):
        prob = f"{p['confidence']:.0f}%" if is_model_pick(p) else DASH
        rows += f"""<tr>
          <td><div class="player-cell">{logo(p.get('team_id'))}<div>
            <div class="player-name">{escape(p['player_name'])}</div>
            <div class="player-meta">{pick_meta(p)}</div></div></div></td>
          <td data-label="Hit chance" class="num prob">{prob}</td>
          <td data-label="Why" class="why">{escape(p.get('reason', ''))}</td>
          <td data-label="Result" class="num"><span>{result_html(p)}</span></td>
        </tr>"""
    return f"""<table class="data responsive-stack">
      <thead><tr><th>Player</th><th class="num">Hit chance</th><th>Why</th><th class="num">Result</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="table-footnote">Hit chance is the model's estimate that he gets at least one hit. Once a lineup is
      posted, only confirmed starters can be picked, and a pick is locked in once its game starts.</div>"""


def track_record(history, model):
    picks = history["picks"]
    body = ""
    if picks:
        season = max(p["date"] for p in picks)[:4]
        season_picks = [p for p in picks if p["date"].startswith(season)]
        g = graded(season_picks)
        hits = sum(p["got_hit"] for p in g)
        voided = sum(1 for p in season_picks if p.get("void"))
        model_g = [p for p in g if is_model_pick(p)]
        stats = [(f"{hits}-{len(g) - hits}", f"{season} record", f"{pct(hits / len(g), 1)} got a hit" if g else "")]
        if model_g:
            actual = sum(p["got_hit"] for p in model_g) / len(model_g)
            predicted = sum(p["confidence"] for p in model_g) / len(model_g) / 100
            stats.append((pct(actual, 1), "Hit rate, model picks", f"model said {pct(predicted, 1)}"))
        stats.append((str(voided), "No decision", "didn't bat or postponed"))
        body = statline(stats)
        if len(model_g) < len(g):
            body += (f'<div class="table-footnote">The record includes {len(g) - len(model_g)} picks made before '
                     f'the current model, which ranked hitters by a weighted score instead of a hit chance.</div>')
    else:
        body = '<div class="empty-state">No picks graded yet.</div>'

    bt = model.get("metrics", {}).get("backtest", {})
    if bt.get("top_n_hit_rate") is not None:
        body += f"""<div class="section-label">Backtest</div>
        <div class="table-footnote" style="margin-top:0;">On {bt['days']} days the model was never trained on, its top
          {bt['top_n']} hitters each day got a hit {pct(bt['top_n_hit_rate'], 1)} of the time (it predicted
          {pct(bt['top_n_predicted'], 1)}), against {pct(bt['all_hitters_hit_rate'], 1)} for every hitter in the
          data. See the Accuracy tab for how live picks compare.</div>"""
    return card("Track Record", "Every pick graded against the box score", body)


def home_games_card(team_history):
    """Today's game picks on Home, under the hitters; the Games tab keeps the
    full record and past results."""
    today = NOW.date().isoformat()
    todays = [p for p in (team_history or {}).get("picks", []) if p["date"] == today]
    if not todays:
        return ""
    return card(f"Today's Games: {day_label(today)}",
                'A winner and win chance for every game. Record and past results are on the '
                '<a href="games.html">Games tab</a>.',
                games_table(todays))


def build_index(history, slate, model, team_history=None):
    picks = history["picks"]
    if picks:
        latest = max(p["date"] for p in picks)
        day_picks = [p for p in picks if p["date"] == latest]
        heading = "Today's Picks" if latest == NOW.date().isoformat() else "Latest Picks"
        picks_html = card(f"{heading}: {day_label(latest)}",
                          "The model's 10 most likely hitters to get a hit, at most two per game.",
                          picks_table(day_picks))
    else:
        picks_html = card("Today's Picks", "", '<div class="empty-state">No picks yet.</div>')
    if slate and not slate.get("games") and slate.get("date") == NOW.date().isoformat():
        picks_html = card("Today's Picks", "", '<div class="empty-state">No MLB games today. Picks resume on the '
                          'next game day.</div>') + picks_html.replace("Today's Picks", "Latest Picks", 1)
    return page_shell("Home", "index.html", picks_html + home_games_card(team_history) + track_record(history, model))


# ── Players ──────────────────────────────────────────────────────────────────
def build_players(slate):
    if not slate or not slate.get("players"):
        body = card("Players", "Every hitter in today's games",
                    '<div class="empty-state">No slate yet today. It fills in once the day\'s games are scored.</div>')
        return page_shell("Players", "players.html", body)

    pick_ids = set(slate.get("pick_ids", []))
    rows = ""
    for p in slate["players"]:
        order = p.get("batting_order")
        order_html = ordinal(order) if order else '<span class="faint">TBD</span>'
        pick_tag = " " + pill("PICK", "primary") if p["player_id"] in pick_ids else ""
        era = f"{p['opp_era']:.2f}" if p.get("opp_era") is not None else DASH
        rows += f"""<tr>
          <td data-key="player" data-value="{escape(p['player_name'])}"><div class="player-cell">{logo(p['team_id'])}<div>
            <div class="player-name">{escape(p['player_name'])}{pick_tag}</div>
            <div class="player-meta">{escape(matchup(p))} · {escape(p['game_time'])} ET</div></div></div></td>
          <td data-key="prob" data-value="{p['confidence']}" data-label="Hit chance" class="num prob">{p['confidence']:.0f}%</td>
          <td data-key="avg" data-value="{p['season_avg'] if p['season_avg'] is not None else ''}" data-label="Season AVG" class="num">{avg(p['season_avg'])}</td>
          <td data-key="recent" data-value="{p['recent_avg'] if p['recent_avg'] is not None else ''}" data-label="Last 2 wks" class="num">{avg(p['recent_avg'])}</td>
          <td data-key="abpg" data-value="{p['ab_per_game']:.2f}" data-label="AB/game" class="num">{p['ab_per_game']:.1f}</td>
          <td data-key="pitcher" data-value="{escape(p['opp_pitcher'])}" data-label="Opp. starter"><span>{escape(p['opp_pitcher'])} <span class="faint">({escape(p['opp_hand'])}, {era})</span></span></td>
          <td data-key="order" data-value="{order or ''}" data-label="Batting" class="num">{order_html}</td>
        </tr>"""
    table = f"""<table class="data responsive-stack" data-sortable>
      <thead><tr>
        <th data-sort-key="player">Player</th><th data-sort-key="prob" class="num">Hit chance</th>
        <th data-sort-key="avg" class="num">AVG</th><th data-sort-key="recent" class="num">Last 2 wks</th>
        <th data-sort-key="abpg" class="num">AB/G</th><th data-sort-key="pitcher">Opp. starter (ERA)</th>
        <th data-sort-key="order" class="num">Batting</th>
      </tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="table-footnote">Until a team posts its lineup, only its regulars are listed. After that, only the
      nine starters are. Select a column header to sort.</div>"""
    subtitle = (f"All {len(slate['players'])} hitters in {len(slate['games'])} games on {day_label(slate['date'])}, "
                f"most likely to get a hit first.")
    return page_shell("Players", "players.html", card("Players", subtitle, table))


# ── History ──────────────────────────────────────────────────────────────────
def build_history(history):
    by_day = defaultdict(list)
    for p in history["picks"]:
        by_day[p["date"]].append(p)
    if not by_day:
        body = card("History", "Every day's picks", '<div class="empty-state">No picks yet.</div>')
        return page_shell("History", "history.html", body)

    days = {}
    for d, picks in by_day.items():
        g = graded(picks)
        days[d] = {
            "label": day_label(d) + f", {d[:4]}",
            "legacy": not any(is_model_pick(p) for p in picks),
            "summary": {"hits": sum(p["got_hit"] for p in g), "graded": len(g),
                        "voided": sum(1 for p in picks if p.get("void"))},
            "picks": [{
                "player_name": p["player_name"], "matchup": matchup(p),
                "confidence": p["confidence"] if is_model_pick(p) else None,
                "got_hit": p.get("got_hit"), "hits": p.get("hits"), "at_bats": p.get("at_bats"),
                "void": bool(p.get("void")),
            } for p in sorted(picks, key=lambda p: -p["confidence"])],
        }
    data = {"order": sorted(days, reverse=True), "days": days}
    body = card("History", "Every day's picks and how they did. Choose a day.",
                '<select id="day-select" class="week-picker" aria-label="Day"></select>'
                '<div id="day-content" style="margin-top:16px;"></div>'
                f'<script>const HISTORY_DATA = {script_json(data)};</script>')
    return page_shell("History", "history.html", body)


# ── Accuracy ─────────────────────────────────────────────────────────────────
def calibration_table(picks):
    bins = [(0, 65), (65, 70), (70, 75), (75, 101)]
    rows = ""
    for lo, hi in bins:
        b = [p for p in picks if lo <= p["confidence"] < hi]
        if not b:
            continue
        label = f"Under {hi}%" if lo == 0 else (f"{lo}% and up" if hi > 100 else f"{lo}-{hi}%")
        actual = sum(p["got_hit"] for p in b) / len(b)
        predicted = sum(p["confidence"] for p in b) / len(b) / 100
        rows += f"""<tr><td class="row-label">{label}</td>
          <td data-label="Picks" class="num">{len(b)}</td>
          <td data-label="Model said" class="num">{pct(predicted, 1)}</td>
          <td data-label="Actually hit" class="num accent">{pct(actual, 1)}</td></tr>"""
    return f"""<table class="data record-table responsive-stack">
      <thead><tr><th>Hit chance</th><th class="num">Picks</th><th class="num">Model said</th><th class="num">Actually hit</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="table-footnote">If the model is honest, each row's two percentages should be close. Small rows swing a lot.</div>"""


def build_accuracy(history, model):
    picks = sorted(graded([p for p in history["picks"] if is_model_pick(p)]), key=lambda p: p["date"])
    parts = []
    if picks:
        weeks = defaultdict(list)
        for p in picks:
            d = date.fromisoformat(p["date"])
            weeks[d - timedelta(days=d.weekday())].append(p)
        labels, actual, predicted, cum_a, cum_p = [], [], [], [], []
        seen = hits = conf = 0
        for wk in sorted(weeks):
            w = weeks[wk]
            labels.append("Wk of " + wk.strftime("%b %-d"))
            actual.append(round(sum(p["got_hit"] for p in w) / len(w), 3))
            predicted.append(round(sum(p["confidence"] for p in w) / len(w) / 100, 3))
            seen += len(w)
            hits += sum(p["got_hit"] for p in w)
            conf += sum(p["confidence"] for p in w) / 100
            cum_a.append(round(hits / seen, 3))
            cum_p.append(round(conf / seen, 3))
        data = {"labels": labels, "actual": actual, "predicted": predicted,
                "cumulative_actual": cum_a, "cumulative_predicted": cum_p}
        charts = "".join(f'<div class="chart-card" data-state="loading"><canvas id="{c}" height="90"></canvas></div>'
                         for c in ("chart-weekly", "chart-cumulative"))
        parts.append(card("Accuracy Over Time",
                          f"{len(picks)} graded picks since the model went live: what it predicted vs. what happened",
                          charts + f'<script>const ACCURACY_DATA = {script_json(data)};</script>'))
        parts.append(card("Calibration", "Picks grouped by the hit chance the model gave them",
                          calibration_table(picks)))
    else:
        parts.append(card("Accuracy Over Time", "Live picks vs. what the model predicted",
                          '<div class="empty-state">No live picks graded yet. Charts appear after the first '
                          'night of results.</div>'))

    m = model.get("metrics", {})
    if m.get("calibration"):
        def band(lo, hi):
            return f"{lo:.0%} and up" if hi >= 1 else f"{lo:.0%}-{hi:.0%}"
        rows = "".join(f"""<tr><td class="row-label">{band(*c['range'])}</td>
          <td data-label="Games" class="num">{c['n']}</td>
          <td data-label="Model said" class="num">{pct(c['predicted'], 1)}</td>
          <td data-label="Actually hit" class="num accent">{pct(c['actual'], 1)}</td></tr>""" for c in m["calibration"])
        bt = m.get("backtest", {})
        note = (f"Fit on games from {model.get('training_dates', ['', ''])[0]} up to {m.get('test_from')}, then "
                f"tested on the {m.get('test_rows')} hitter-games from {m.get('test_from')} on, which it never saw.")
        if bt.get("top_n_hit_rate") is not None:
            note += (f" Its top {bt['top_n']} per day got a hit {pct(bt['top_n_hit_rate'], 1)} of the time, "
                     f"vs. {pct(bt['all_hitters_hit_rate'], 1)} for all hitters.")
        parts.append(card("Backtest", "How the current model did on games held out of training",
                          f"""<table class="data record-table responsive-stack">
          <thead><tr><th>Hit chance</th><th class="num">Games</th><th class="num">Model said</th><th class="num">Actually hit</th></tr></thead>
          <tbody>{rows}</tbody></table><div class="table-footnote">{note}</div>"""))
    return page_shell("Accuracy", "accuracy.html", "".join(parts), charts=bool(picks))


# ── Games (the team model) ───────────────────────────────────────────────────
def game_graded(picks):
    return [p for p in picks if p.get("correct") is not None and not p.get("void")]


def game_wl(picks):
    w = sum(p["correct"] for p in picks)
    return w, len(picks) - w


def game_top_per_day(picks, n=TOP_GAMES):
    by_day = defaultdict(list)
    for p in picks:
        by_day[p["date"]].append(p)
    return [p for day in by_day.values() for p in sorted(day, key=lambda p: -p["prob"])[:n]]


def first_pitch(p):
    return datetime.fromisoformat(p["start"]).astimezone(ET).strftime("%-I:%M %p")


def starter_text(sp):
    if not sp or not sp.get("id"):
        return "TBD"
    parts = sp["name"].split(" ", 1)
    name = f"{parts[0][0]}. {parts[1]}" if len(parts) == 2 else sp["name"]
    return f"{name} ({sp['era']:.2f})" if sp.get("era") is not None else f"{name} (1st start)"


def game_meta(p):
    parts = []
    if p.get("round"):
        parts.append(p["round"])
    if p.get("doubleheader"):
        parts.append(f"Game {p['doubleheader']}")
    parts.append(f"{first_pitch(p)} ET")
    if p.get("type") == "regular":
        parts.append(f"{p['away']} {p.get('away_record', '')}, {p['home']} {p.get('home_record', '')}")
    return " · ".join(parts)


def game_result_html(p):
    if p.get("void"):
        return pill("NO DECISION", "void")
    if p.get("correct") is None:
        return f'<span class="faint">{DASH}</span>'
    score = f"{p['away']} {p['away_runs']}, {p['home']} {p['home_runs']} "
    return escape(score) + (pill("WIN", "positive") if p["correct"] else pill("LOSS", "danger"))


def games_table(picks):
    rows = ""
    for p in sorted(picks, key=lambda p: (p["start"], p["game_id"])):
        pick_id = p["home_id"] if p["pick"] == p["home"] else p["away_id"]
        starters = f"{starter_text(p.get('away_sp'))} vs. {starter_text(p.get('home_sp'))}"
        rows += f"""<tr>
          <td><div class="player-name">{escape(p['away'])} @ {escape(p['home'])}</div>
            <div class="player-meta">{escape(game_meta(p))}</div>
            <div class="player-meta">{escape(starters)}</div></td>
          <td data-label="Pick"><span class="matchup-team">{logo(pick_id)}{escape(p['pick'])}</span></td>
          <td data-label="Win chance" class="num prob">{p['prob']:.0f}%</td>
          {ml_cell(p)}
          <td data-label="Result" class="num"><span>{game_result_html(p)}</span></td>
        </tr>"""
    return f"""<table class="data responsive-stack">
      <thead><tr><th>Game and starters (ERA)</th><th>Pick</th><th class="num">Win chance</th><th>Moneyline bet</th><th class="num">Result</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="table-footnote">Win chance is the model's estimate that its pick wins the game. Starters are the
      announced probables, away team first, with their ERA this season. Picks refresh until first pitch as starters
      are named, then lock. {ML_NOTE}</div>"""


# ── Moneyline picks (moneyline.py), shared with NBA Edge ─────────────────────
ML_NOTE = ("Moneyline bet is the team to take on the moneyline and its price (from ESPN's scoreboard): the side "
           "where the model's win chance beats the chance the price implies (vig removed) by the most. It can be an "
           "underdog the model still expects to lose, when the payout is worth the risk. Value means an edge of 6 "
           "points or more; Lean is a smaller edge. Graded at 1 unit a pick.")


def ml_result_html(ml, void=False):
    if void or ml.get("void"):
        return pill("NO DECISION", "void")
    if ml.get("won") is None:
        return ""
    return pill(moneyline.units_text(ml["units"]), "positive" if ml["won"] else "danger")


def ml_cell(p):
    """The Moneyline bet column, spelled out: "BUF to win +135" (VALUE at a 6+
    point edge, otherwise LEAN), what the price pays, our chance vs. the
    price's, a note when it's a long shot on the team we expect to lose, and
    once graded the units won or lost."""
    ml = p.get("ml")
    if not ml:
        return '<td data-label="Moneyline bet" class="ml-cell"><div class="ml"><span class="faint">No odds</span></div></td>'
    value = " " + (pill("VALUE", "primary") if ml.get("value") else pill("LEAN", "market"))
    res = ml_result_html(ml, p.get("void"))
    other = p.get("home") if ml["team"] == p.get("away") else p.get("away")
    subs = "".join(f'<div class="ml-sub">{escape(line)}</div>' for line in moneyline.detail_lines(ml, other))
    return f"""<td data-label="Moneyline bet" class="ml-cell"><div class="ml">
            <div class="ml-pick">{escape(moneyline.text(ml))}{value}{' ' + res if res else ''}</div>
            {subs}</div></td>"""


def ml_record_html(picks, empty="No moneyline picks graded yet this season."):
    """The Moneyline section of a record card. picks: this season's live picks only."""
    rec = moneyline.record(picks)
    body = '<div class="section-label">Moneyline</div>'
    if not rec:
        return body + f'<div class="empty-state">{empty}</div>'
    value = moneyline.record([p for p in picks if p.get("ml", {}).get("value")])
    body += statline([
        (f"{rec['wins']}-{rec['losses']}", "Moneyline record",
         f"Value picks {value['wins']}-{value['losses']}" if value else "No value picks graded yet"),
        (moneyline.units_text(rec["units"]), "Units", "1 unit a pick at the book price"),
        (f"{rec['roi']:+.1%}", "ROI", f"on {rec['picks']} {'pick' if rec['picks'] == 1 else 'picks'}"),
    ])
    return body + ('<div class="table-footnote">Live moneyline picks only, at the price when the game started. '
                   'A postponed game is no decision.</div>')


def ml_day(picks):
    """A History day's moneyline summary and per-game fields for nba.js."""
    rec = moneyline.record(picks)
    return {"wins": rec["wins"], "losses": rec["losses"], "units": moneyline.units_text(rec["units"])} if rec else None


def game_bands(picks):
    out = []
    for lo, hi in ((50, 55), (55, 60), (60, 65), (65, 101)):
        b = [p for p in picks if lo <= p["prob"] < hi]
        if b:
            label = f"{lo}% and up" if hi > 100 else f"{lo}-{hi}%"
            out.append((label, len(b), sum(p["prob"] for p in b) / len(b) / 100, sum(p["correct"] for p in b) / len(b)))
    rows = "".join(f"""<tr><td class="row-label">{label}</td>
      <td data-label="Picks" class="num">{n}</td>
      <td data-label="Model said" class="num">{pct(said, 1)}</td>
      <td data-label="Actually won" class="num accent">{pct(won, 1)}</td></tr>""" for label, n, said, won in out)
    return f"""<table class="data record-table responsive-stack">
      <thead><tr><th>Win chance</th><th class="num">Picks</th><th class="num">Model said</th><th class="num">Actually won</th></tr></thead>
      <tbody>{rows}</tbody></table>
    <div class="table-footnote">If the model is honest, each row's two percentages should be close. In baseball
      even strong favorites lose often, and small rows swing a lot.</div>"""


def game_record(picks):
    """This season's live record. Only picks actually made count, never a backtest."""
    if not picks:
        return card("This Season", "Every pick graded against the final score",
                    '<div class="empty-state">No game picks graded yet. The record starts after the first night '
                    'of results.</div>' + ml_record_html([])), False
    season = max(p["date"] for p in picks)[:4]
    season_picks = [p for p in picks if p["date"].startswith(season)]
    g = sorted(game_graded(season_picks), key=lambda p: p["date"])
    voided = sum(1 for p in season_picks if p.get("void"))
    if not g:
        return card(f"{season} Record", "Every pick graded against the final score",
                    '<div class="empty-state">No game picks graded yet. The record starts after the first night '
                    'of results.</div>' + ml_record_html(season_picks)), False
    w, l = game_wl(g)
    tw, tl = game_wl(game_top_per_day(g))
    strong = [p for p in g if p["prob"] >= STRONG_GAME]
    sw, sl = game_wl(strong)
    said = sum(p["prob"] for p in g) / len(g) / 100
    body = statline([
        (f"{w}-{l}", f"{season} record", f"{pct(w / len(g), 1)} right; model said {pct(said, 1)}"),
        (f"{tw}-{tl}", f"Top {TOP_GAMES} picks each day", pct(tw / (tw + tl), 1) if tw + tl else ""),
        (f"{sw}-{sl}", f"Picks at {STRONG_GAME}%+", pct(sw / len(strong), 1) if strong else DASH),
    ])
    note = (f"Live picks only, made before first pitch this season. {voided} postponed "
            f"{'game counts' if voided == 1 else 'games count'} as no decision.") if voided else \
        "Live picks only, made before first pitch this season."
    body += f'<div class="table-footnote">{note}</div>' + ml_record_html(season_picks)
    charts = False
    weeks = defaultdict(list)
    for p in g:
        d = date.fromisoformat(p["date"])
        weeks[d - timedelta(days=d.weekday())].append(p)
    if len(weeks) >= 2:
        labels, actual, predicted, cum_a, cum_p = [], [], [], [], []
        seen = right = conf = 0
        for wk in sorted(weeks):
            ps = weeks[wk]
            labels.append("Wk of " + wk.strftime("%b %-d"))
            actual.append(round(sum(p["correct"] for p in ps) / len(ps), 3))
            predicted.append(round(sum(p["prob"] for p in ps) / len(ps) / 100, 3))
            seen += len(ps)
            right += sum(p["correct"] for p in ps)
            conf += sum(p["prob"] for p in ps) / 100
            cum_a.append(round(right / seen, 3))
            cum_p.append(round(conf / seen, 3))
        data = {"labels": labels, "actual": actual, "predicted": predicted, "cumulative_actual": cum_a,
                "cumulative_predicted": cum_p,
                "titles": {"weekly": "Picks Won by Week: Actual vs. What the Model Predicted",
                           "weekly_actual": "Actual win rate", "weekly_predicted": "Model's predicted win rate",
                           "cumulative": "Season-to-Date Win Rate"}}
        body += '<div class="section-label">Accuracy</div>' + "".join(
            f'<div class="chart-card" data-state="loading"><canvas id="{c}" height="90"></canvas></div>'
            for c in ("chart-weekly", "chart-cumulative")) + f'<script>const ACCURACY_DATA = {script_json(data)};</script>'
        charts = True
    body += '<div class="section-label">Calibration</div>' + game_bands(g)
    return card(f"{season} Record", "Every pick graded against the final score", body), charts


def game_history(picks):
    by_day = defaultdict(list)
    for p in picks:
        if p["date"] < NOW.date().isoformat() or p.get("correct") is not None or p.get("void"):
            by_day[p["date"]].append(p)
    if not by_day:
        return ""
    days = {}
    for d, ps in by_day.items():
        w, l = game_wl(game_graded(ps))
        days[d] = {
            "label": day_label(d) + f", {d[:4]}",
            "summary": {"wins": w, "losses": l, "voided": sum(1 for p in ps if p.get("void")), "ml": ml_day(ps)},
            "games": [{
                "matchup": f"{p['away']} @ {p['home']}",
                "meta": f"{starter_text(p.get('away_sp'))} vs. {starter_text(p.get('home_sp'))}",
                "pick": p["pick"], "prob": p["prob"], "correct": p.get("correct"), "void": bool(p.get("void")),
                "ml": moneyline.result(p.get("ml"), p.get("void")),
                "score": (f"{p['away']} {p['away_runs']}, {p['home']} {p['home_runs']}"
                          if p.get("home_runs") is not None else ""),
            } for p in sorted(ps, key=lambda p: -p["prob"])],
        }
    data = {"order": sorted(days, reverse=True), "days": days}
    return card("Results", "Every past day's game picks and how they did. Choose a day.",
                '<select id="day-select" class="week-picker" aria-label="Day"></select>'
                '<div id="day-content" style="margin-top:16px;"></div>'
                f'<script>const GAME_HISTORY = {script_json(data)};</script>')


def build_games(team_history):
    picks = team_history["picks"]
    today = NOW.date().isoformat()
    todays = [p for p in picks if p["date"] == today]
    if todays:
        top = sorted(todays, key=lambda p: -p["prob"])[:TOP_GAMES]
        head = card(f"Today's Games: {day_label(today)}",
                    f"A pick for every game. The {len(top)} most confident: "
                    + ", ".join(f"{p['pick']} ({p['prob']:.0f}%)" for p in top) + ".",
                    games_table(todays))
    elif picks:
        head = card("Today's Games", "", '<div class="empty-state">No MLB games today, or the slate isn\'t up '
                    'yet. Picks go up on the next game day; past days are under Results below.</div>')
    else:
        head = card("Today's Games", "", '<div class="empty-state">Game picks go up with the next daily '
                    'update: a winner and a win chance for every game.</div>')
    record, charts = game_record(picks)
    return page_shell("Games", "games.html", head + record + game_history(picks), charts=charts)


# ── Schedule tab and scoreboard strip ────────────────────────────────────────
# ESPN and the MLB Stats API abbreviate a few teams differently.
ABBR_ALIASES = moneyline.MLB_ALIASES


def _team_key(team):
    """(name, abbreviation) forms of an ESPN scoreboard team for matching a pick."""
    abbr = team.get("abbr", "")
    return team.get("name", ""), ABBR_ALIASES.get(abbr, abbr)


def team_pick_for(g, day_picks):
    """The team model's pick for an ESPN scoreboard game, matched by team names
    (or abbreviations), taking the closest first pitch for a doubleheader."""
    (an, aa), (hn, ha) = _team_key(g["away"]), _team_key(g["home"])
    cands = [p for p in day_picks
             if (p["away_name"], p["home_name"]) == (an, hn) or (p["away"], p["home"]) == (aa, ha)]
    if not cands:
        return None
    start = games_mod.start_et(g)
    return min(cands, key=lambda p: abs((datetime.fromisoformat(p["start"]) - start).total_seconds()))


def attach_picks(slate, history, team_history):
    """Each game gets the team model's pick and win chance ("NYY 58%"). A game
    without one falls back to the model's most likely hitter in it."""
    by_day, team_by_day = defaultdict(list), defaultdict(list)
    for p in history["picks"]:
        by_day[p["date"]].append(p)
    for p in team_history["picks"]:
        team_by_day[p["date"]].append(p)
    for g in slate["games"]:
        day = games_mod.start_et(g).date().isoformat()
        tp = team_pick_for(g, team_by_day.get(day, []))
        if tp:
            side = "home" if tp["pick"] == tp["home"] else "away"
            g["pick"] = {"text": f"{g[side]['abbr'] or tp['pick']} {tp['prob']:.0f}%",
                         "result": None if tp.get("void") or tp.get("correct") is None else bool(tp["correct"])}
            continue
        teams = {g["away"]["name"], g["home"]["name"]}
        cands = [p for p in by_day.get(day, []) if p.get("team") in teams and is_model_pick(p)]
        if not cands:
            continue
        p = max(cands, key=lambda p: p["confidence"])
        name = p["player_name"].split(" ", 1)
        short = f"{name[0][0]}. {name[1]}" if len(name) == 2 else p["player_name"]
        g["pick"] = {"text": f"{short} {p['confidence']:.0f}%",
                     "result": None if p.get("void") or p.get("got_hit") is None else bool(p["got_hit"])}
    return slate


# ── Model tab ────────────────────────────────────────────────────────────────
FACTOR_LABELS = {
    "season_avg": ("Season batting average", ""),
    "recent_form_avg": ("Recent form", "batting average over his last games"),
    "ab_per_game": ("At-bats per game", "more trips to the plate, more chances"),
    "is_home": ("Home vs. away", ""),
    "platoon": ("Platoon edge", "batting against the opposite hand"),
    "park_factor": ("Ballpark", "how hitter-friendly the park is"),
    "opp_pitcher_era": ("Opposing starter's ERA", ""),
    "opp_pitcher_whip": ("Opposing starter's WHIP", "walks + hits per inning"),
    "opp_pitcher_k9": ("Opposing starter's strikeouts", "per nine innings"),
}


def next_weekday(weekday, hour_utc):
    """The next time a weekly UTC cron (weekday: Monday=0) fires, in ET."""
    now = datetime.now(ZoneInfo("UTC"))
    d = now.replace(hour=hour_utc, minute=0, second=0, microsecond=0)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    if d <= now:
        d += timedelta(days=7)
    return d.astimezone(ET)


def recipe_setup(recipe):
    seasons = recipe.get("seasons", 1)
    l2 = recipe.get("l2", 0.001)
    return [
        ("Learns from:", "the last season of games" if seasons == 1 else f"the last {seasons} seasons of games"),
        ("Ballparks:", "learns each park's effect from the games" if recipe.get("park") == "learned"
         else "uses a fixed park-factor table"),
        ("Smoothing:", "light, so strong patterns can show" if l2 <= 0.001 else "medium, to avoid chasing noise"),
    ]


def shares(weights):
    total = sum(abs(w) for w in weights.values()) or 1
    return {k: abs(w) / total for k, w in weights.items()}


def hit_model_html(model, runs):
    runs = list(reversed(runs))
    nxt = next_weekday(0, 8)
    rows = []
    for r in runs:
        label, tone = model_page.decision(r)
        chosen = r["best_recipe"] if r.get("switched_recipe") else (r.get("current_recipe") or r["best_recipe"])
        rows.append({
            "date": model_page.short_date(r["run_at"]), "data_through": model_page.short_date(r.get("data_through")),
            "tested": len(r.get("candidates", [])), "decision": label, "tone": tone, "reason": r["reason"].capitalize() + ".",
            "before": (r.get("live_model", {}).get("holdout") or {}).get("top10_hit_rate"),
            "after": chosen.get("top10_hit_rate") if r.get("deployed") else None,
        })
    last = runs[0] if runs else {}
    now_w = dict(zip(model.get("features", []), model.get("weights", [])))
    before_w = last.get("weights_before")
    now_s, before_s = shares(now_w), (shares(before_w) if before_w else {})
    factors = [{"label": FACTOR_LABELS.get(f, (f, ""))[0], "note": FACTOR_LABELS.get(f, (f, ""))[1],
                "now": now_s[f], "before": before_s.get(f), "fmt": lambda v: pct(v, 1)}
               for f in sorted(now_s, key=lambda f: -now_s[f])]
    trained = model.get("trained_at")
    spec = {
        "intro": "Every Monday it checks itself against the newest games and only changes when a new version "
                 "clearly predicts better.",
        "tiles": [
            (model_page.short_date(trained)[:-6] if trained else "-", "Last retrained",
             f"games through {model_page.short_date(model.get('trained_through'))}" if model.get("trained_through") else ""),
            (f"{model.get('training_rows', 0):,}", "Hitter-games learned from",
             " to ".join(model_page.short_date(d) for d in model.get("training_dates", []))),
            (nxt.strftime("%b %-d"), "Next check", nxt.strftime("Monday, %-I %p ET")),
            (rows[0]["decision"].split(" ")[0] if rows else "-", "Last decision",
             f"{rows[0]['tested']} versions tested" if rows else "no retrains yet"),
        ],
        "setup": recipe_setup(model.get("recipe", {})) + [
            ("Looks at:", f"{len(now_w)} factors for every hitter, listed below"),
            ("Retrains:", "Mondays, only when new games have been played; in the offseason it waits"),
        ],
        "runs": rows,
        "score_name": "Top 10 hit rate",
        "score_fmt": lambda v: pct(v, 1),
        "higher_better": True,
        "factors": factors,
        "factors_note": "Share of influence: how much each factor moves a hit chance, relative to the others, "
                        "for a typical swing in that factor. Before is the model that was live until the last retrain.",
        "empty": "No retrains logged yet. The first one runs on the next Monday after new games.",
    }
    return model_page.render(spec)


TEAM_FACTOR_LABELS = {
    "home_field": ("Home field", "for the home team, before anything else"),
    "elo": ("Team rating (Elo) gap", "per 100 rating points"),
    "starters": ("Starting pitcher gap", "per run per 9 innings better than the other starter"),
    "bullpen": ("Bullpen ERA gap", "per run of bullpen ERA"),
}


def team_recipe_setup(recipe):
    k, carry = recipe.get("elo_k", 4), recipe.get("elo_carry", 0.67)
    years, shrink = recipe.get("years", 2), recipe.get("sp_shrink", 40)
    speed = "steady" if k <= 3 else "medium" if k <= 4 else "fast"
    return [
        ("Learns from:", f"the last {years} seasons of games"),
        ("Team ratings:", f"{speed} - one game moves a team's Elo rating only a few points, a bit more for "
                          f"a blowout"),
        ("Over the winter:", f"keeps {carry:.0%} of each team's rating; the rest resets toward average"),
        ("Starters:", f"ERA and FIP this season and past seasons, blended with {shrink} innings of a "
                      f"league-average starter so a few starts don't swing it"),
    ]


def team_model_html(model, runs):
    """The team model's section of the Model tab, from teams/model_weights.json
    and teams/model_history.json."""
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
            "tested": len(r.get("candidates", [])), "decision": label, "tone": tone,
            "reason": r["reason"].capitalize() + ".",
            "before": skill(r["live_model"].get("holdout_log_loss"), base),
            "after": skill(chosen_ll, base) if r.get("deployed") else None,
        })
    last = runs[0] if runs else {}
    now_w, before_w = model.get("coef", {}), last.get("weights_before") or {}
    factors = [{"label": TEAM_FACTOR_LABELS.get(f, (f, ""))[0], "note": TEAM_FACTOR_LABELS.get(f, (f, ""))[1],
                "now": now_w[f], "before": before_w.get(f), "fmt": lambda v: f"{v:+.3f}"}
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
            ("Weekly", "Next check", "once 60+ new games are in; waits in the offseason"),
            (rows[0]["decision"].split(" ")[0] if rows else "-", "Last decision",
             f"{rows[0]['tested']} versions tested" if rows else "no retrains yet"),
        ],
        "setup": team_recipe_setup(model.get("recipe", {})) + [
            ("Looks at:", f"{len(now_w)} factors for every game, listed below, with the day's probable starters"),
        ],
        "runs": rows,
        "score_name": "Better than guessing",
        "score_fmt": lambda v: pct(v, 1),
        "higher_better": True,
        "factors": factors,
        "factors_note": "Each number is how much that factor moves the home team's log-odds of winning (negative "
                        "helps the away team); 0.1 is about 2.5 points of win chance near a coin flip. Better than "
                        "guessing is how much less wrong (log loss) the model is than always giving the home team "
                        "its usual win rate. Before is the model that was live until the last retrain.",
        "empty": "No retrains logged yet. The first one runs about a week into the season.",
    }
    # model_page is shared across sites; point its live-results line at this model's tab.
    html = model_page.render(spec).replace("Live results are on the Accuracy tab.", "Live results are on the Games tab.")
    bt = model.get("backtest")
    if bt:
        b = model.get("baselines", {})
        post = model.get("backtest_postseason")
        tw = bt["top3"]
        body = statline([
            (f"{bt['correct']}-{bt['games'] - bt['correct']}", f"{model['backtest_season']} backtest",
             f"{pct(bt['accuracy'], 1)} of games; model said {pct(bt['predicted_accuracy'], 1)}"),
            (pct(tw["accuracy"], 1), f"Top {TOP_GAMES} picks each day", f"{tw['correct']}-{tw['picks'] - tw['correct']}"),
            (f"{bt['log_loss']:.3f}", "Log loss", f"{b.get('home_rate_log_loss', 0):.3f} for home-team rate"),
        ])
        note = (f"Fit only on {model['trained_on'].replace(' to ', '-')} and never shown {model['backtest_season']}, "
                f"then used to pick all {bt['games']} regular-season games of {model['backtest_season']} with only "
                f"what was known that morning. The home team won {pct(b.get('home_team_accuracy'), 1)} of those "
                f"games, and team ratings alone picked {pct(b.get('elo_only_accuracy'), 1)}. Lower log loss is "
                f"better.")
        if post:
            note += f" In the postseason it went {post['correct']}-{post['games'] - post['correct']}."
        note += " This is a test on an old season; the Games tab counts only live picks."
        rows_html = "".join(f"""<tr><td class="row-label">{f"{b_['range'][0]:.0%} and up" if b_['range'][1] >= 1 else f"{b_['range'][0]:.0%}-{b_['range'][1]:.0%}"}</td>
          <td data-label="Games" class="num">{b_['n']}</td>
          <td data-label="Model said" class="num">{pct(b_['predicted'], 1)}</td>
          <td data-label="Actually won" class="num accent">{pct(b_['actual'], 1)}</td></tr>""" for b_ in bt["bands"])
        html += card("Backtest", "How the team model did on a season it was never trained on", body + f"""
          <table class="data record-table responsive-stack">
          <thead><tr><th>Win chance</th><th class="num">Games</th><th class="num">Model said</th><th class="num">Actually won</th></tr></thead>
          <tbody>{rows_html}</tbody></table><div class="table-footnote">{note}</div>""")
    return html


def build_model(model, runs, team_model, team_runs):
    jump = ('<nav class="subtabs model-jump" aria-label="Models">'
            '<a class="subtab" href="#hit-model">Hit model</a><a class="subtab" href="#team-model">Team model</a></nav>')
    body = (jump
            + '<h2 class="model-group" id="hit-model">Hit model <span>who gets a hit</span></h2>'
            + hit_model_html(model, runs)
            + '<h2 class="model-group" id="team-model">Team model <span>who wins each game</span></h2>'
            + (team_model_html(team_model, team_runs) if team_model.get("coef") else
               card("Team Model", "", '<div class="empty-state">The team model hasn\'t been trained yet.</div>')))
    return page_shell("Model", "model.html", body)


# ── Home page summary ────────────────────────────────────────────────────────
def build_summary(history, model):
    """summary.json - the latest day's top three picks and the season record, for
    the card on the home page (ant56-arch.github.io, github.com/ant56-arch/ant56-arch.github.io)."""
    picks = history["picks"]
    summary = {"updated": NOW.isoformat(), "heading": None, "picks": [], "record": None,
               "empty": "No picks yet.", "retrained": model.get("trained_at"), "model_url": "model.html"}
    if not picks:
        return summary
    latest = max(p["date"] for p in picks)
    prefix = "Today" if latest == NOW.date().isoformat() else "Latest"
    summary["heading"] = f"{prefix}: {day_label(latest)}"
    top = sorted((p for p in picks if p["date"] == latest), key=lambda p: -p["confidence"])[:3]
    summary["picks"] = [{
        "label": p["player_name"], "sub": matchup(p),
        "value": f"{p['confidence']:.0f}%" if is_model_pick(p) else DASH,
        "result": None if p.get("void") or p.get("got_hit") is None else bool(p["got_hit"]),
    } for p in top]
    season = latest[:4]
    g = graded([p for p in picks if p["date"].startswith(season)])
    if g:
        hits = sum(p["got_hit"] for p in g)
        summary["record"] = {"value": f"{hits}-{len(g) - hits}", "label": f"{season} record",
                             "sub": f"{pct(hits / len(g), 1)} got a hit"}
    return summary


# ── Root pages ───────────────────────────────────────────────────────────────
def build_404():
    body = """<div class="error-body">
        <div class="error-code">404</div>
        <h1 class="error-title">Page not found</h1>
        <p>This page doesn't exist. It may have been moved or renamed.</p>
        <a class="btn-primary" href="index.html">Go to today's picks</a>
      </div>"""
    return page_shell("Page Not Found", None, body)


def main():
    history = load_json("picks_history.json", {"picks": []})
    slate = load_json(os.path.join("data", "slate.json"), None)
    model = load_json("model_weights.json", {})
    team_history = load_json(os.path.join("teams", "picks_history.json"), {"picks": []})
    team_model = load_json(os.path.join("teams", "model_weights.json"), {})
    games_slate = attach_picks(games_mod.load("mlb"), history, team_history)

    if os.path.exists(DIST_DIR):
        shutil.rmtree(DIST_DIR)
    os.makedirs(DIST_DIR)
    pages = {
        "index.html": build_index(history, slate, model, team_history),
        "games.html": build_games(team_history),
        "players.html": build_players(slate),
        "history.html": build_history(history),
        "accuracy.html": build_accuracy(history, model),
        "schedule.html": games_mod.schedule_redirect("mlb"),
        "model.html": build_model(model, load_json("model_history.json", {"runs": []})["runs"], team_model,
                                  load_json(os.path.join("teams", "model_history.json"), {"runs": []})["runs"]),
        "terms.html": games_mod.legal_redirect("terms"),
        "privacy.html": games_mod.legal_redirect("privacy"),
        "404.html": build_404(),
    }
    for name, html in pages.items():
        with open(os.path.join(DIST_DIR, name), "w") as f:
            f.write(html)
    with open(os.path.join(DIST_DIR, "summary.json"), "w") as f:
        json.dump(build_summary(history, model), f, indent=1)
    games_mod.write_json(os.path.join(DIST_DIR, "games.json"), "mlb", games_slate, NOW.isoformat())
    for asset in ASSETS:
        shutil.copy(os.path.join(WEB_DIR, asset), os.path.join(DIST_DIR, asset))
    print(f"Built {len(pages)} pages in {DIST_DIR}")


if __name__ == "__main__":
    main()
