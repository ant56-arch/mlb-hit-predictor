"""
build_pages.py - generates the CBB Edge (men's college basketball) pages into
dist/cbb/.

Run after the MLB build (build_site.py wipes dist/). Reads
cbb/picks_history.json, cbb/ratings.json and cbb/model_weights.json and writes:
  index.html    - today's games between Division I teams with a pick, win
                  chance, projected score and moneyline pick for each, and the
                  record (game picks and moneyline)
  ratings.html  - every Division I team's KenPom-style ratings, four factors,
                  strength of schedule and T-Rank, sortable
  history.html  - any past day's picks and how they did
  accuracy.html - predicted vs. actual over the season, and last season's backtest
  model.html    - how the model retrains itself
  terms.html, privacy.html
  summary.json  - today's five most confident picks, for the home page

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
from build_site import (DASH, ET, ML_NOTE, NOW, card, ml_cell, ml_day, ml_history, ml_record_html, pct, pill,  # noqa: E402
                        script_json, statline)
import games as games_mod  # noqa: E402
import model_page  # noqa: E402
import moneyline  # noqa: E402

WEB_DIR = os.path.join(ROOT, "web")
OUT_DIR = os.path.join(ROOT, "dist", "cbb")
ASSETS = ("style.css", "site.js", "nba.js")

HOME_URL = "https://ant56-arch.github.io/"
NFL_EDGE = "https://ant56-arch.github.io/nfl-edge"
MLB_EDGE = "https://ant56-arch.github.io/mlb-hit-predictor"
SPORT_LINKS = [("All", HOME_URL), ("NFL", f"{NFL_EDGE}/nfl/index.html"), ("CFB", f"{NFL_EDGE}/cfb/index.html"),
               ("MLB", f"{MLB_EDGE}/"), ("NBA", f"{MLB_EDGE}/nba/index.html"), ("CBB", None),
               ("Schedule", "https://ant56-arch.github.io/schedule.html")]
TAGLINE = ("Who wins every Division I men's basketball game and by how much, from KenPom-style team ratings "
           "graded against every final score.")
TOP_N = 5
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


def logo(team_id):
    return (f'<img class="team-logo" src="https://a.espncdn.com/i/teamlogos/ncaa/500/{escape(str(team_id))}.png" '
            f'alt="" loading="lazy" onerror="this.style.display=\'none\'">')


def graded(picks):
    return [p for p in picks if p.get("correct") is not None and not p.get("void")]


def season_of(iso):
    """Seasons run November to April; 2026-11-03 and 2027-04-05 are both in 2026-27."""
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
    tabs = [("index.html", "Home"), ("ratings.html", "Ratings"), ("history.html", "History"),
            ("accuracy.html", "Accuracy"), ("model.html", "Model")]
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
<title>{title} | CBB Edge</title>
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
<header class="masthead" data-sport="CBB">
  <div class="masthead-inner">
    <h1 class="wordmark">CBB <span>EDGE</span></h1>
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
    <div class="footer-brand">CBB <span>Edge</span></div>
    <p class="footer-text">Win chances come from a model fit on past college basketball seasons: KenPom-style
      adjusted offense, defense and tempo computed from every Division I box score, the four factors (shooting,
      turnovers, offensive rebounding, free throws), Elo, recent form, rest, home court, and Bart Torvik's T-Rank.
      Scores, box scores and moneyline prices via ESPN; T-Rank via barttorvik.com. The model uses no betting odds;
      each game's moneyline pick compares its win chance with the book price, vig removed. Not affiliated with
      KenPom.</p>
    <p class="footer-text">For entertainment and research only. This is not betting advice, and past results
      don't predict future ones. If gambling is a problem for you or someone you know, call 1-800-GAMBLER.</p>
    <nav class="footer-links" aria-label="Site">
      <a href="https://ant56-arch.github.io/terms.html">Terms of Use</a>
      <a href="https://ant56-arch.github.io/privacy.html">Privacy Policy</a>
      <a href="{HOME_URL}">All sites</a>
      <a href="{MLB_EDGE}/">MLB Edge</a>
      <span>&copy; {NOW.year} CBB Edge. Updated from final scores every night.</span>
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


def team_label(p, side):
    """'#4 Duke' style: AP rank when ranked."""
    rank = p.get(f"{side}_rank")
    return (f"#{rank} " if rank else "") + p.get(f"{side}_name", p[side])


def game_notes(p):
    notes = []
    if p.get("note"):
        notes.append(p["note"])
    ranks = [f"{p[s]} No. {p[f'{s}_em_rank']}" for s in ("away", "home") if p.get(f"{s}_em_rank")]
    if ranks:
        notes.append("AdjEM rank: " + ", ".join(ranks))
    return "; ".join(notes)


def games_table(picks):
    rows = ""
    for p in sorted(picks, key=lambda p: (p["start"], p["game_id"])):
        at = "vs" if p.get("neutral") else "@"
        notes = game_notes(p)
        pick_id = p["home_id"] if p["pick"] == p["home"] else p["away_id"]
        proj = p.get("proj") or {}
        projected = (f"{escape(p['away'])} {proj['away']}, {escape(p['home'])} {proj['home']}"
                     if proj else f"by {p['margin']:.1f}")
        tempo = f", {proj['tempo']} possessions" if proj.get("tempo") else ""
        rows += f"""<tr>
          <td><div class="player-name">{escape(team_label(p, 'away'))} {at} {escape(team_label(p, 'home'))}</div>
            <div class="player-meta">{tip_time(p)} ET · {escape(p['away'])} {p.get('away_record', '')}, {escape(p['home'])} {p.get('home_record', '')}</div></td>
          <td data-label="Pick"><span class="matchup-team">{logo(pick_id)}{escape(p['pick'])}</span></td>
          <td data-label="Win chance" class="num prob">{p['prob']:.0f}%</td>
          <td data-label="Projected" class="num">{projected}<div class="player-meta">by {p['margin']:.1f}{tempo}</div></td>
          {ml_cell(p)}
          <td{' data-label="Notes"' if notes else ''} class="why">{escape(notes)}</td>
          <td data-label="Result" class="num"><span>{result_html(p)}</span></td>
        </tr>"""
    return f"""<table class="data responsive-stack">
      <thead><tr><th>Game</th><th>Pick</th><th class="num">Win chance</th><th class="num">Projected</th><th>Moneyline bet</th><th>Notes</th><th class="num">Result</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    <div class="table-footnote">Win chance is the model's estimate that its pick wins the game. The projected
      score comes from each team's adjusted offense, defense and tempo, set to the model's margin. Picks are made
      only for games between two Division I teams, refreshed until tip-off, then locked. {ML_NOTE}</div>"""


def backtest_stats(model):
    bt = model.get("backtest") or {}
    if not bt:
        return []
    strong = [b for b in bt["bands"] if b["range"][0] >= STRONG / 100]
    top = bt["top"]
    stats = [(f"{bt['correct']}-{bt['games'] - bt['correct']}", f"{model['backtest_season']} backtest",
              f"{pct(bt['accuracy'], 1)} of every game"),
             (pct(top["accuracy"], 1), f"Top {top['n']} picks each day", f"{top['correct']}-{top['picks'] - top['correct']}")]
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
            f'to pick all {bt["games"]} regular-season Division I games of {model["backtest_season"]}, each with only '
            f'what was known that morning. The home team won {pct(b.get("home_team_accuracy"), 1)} of those games, '
            f'and the adjusted efficiency ratings alone picked {pct(b.get("ratings_only_accuracy"), 1)}. See the '
            f'Accuracy tab for more.</div>')


def track_record(history, model):
    picks = history["picks"]
    body = ""
    season_picks = []
    if picks:
        season = season_of(max(p["date"] for p in picks))
        season_picks = [p for p in picks if season_of(p["date"]) == season]
        g = graded(season_picks)
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
    body += ml_record_html(season_picks)  # live picks this season only, never the backtest
    bt = backtest_stats(model)
    if bt:
        body += '<div class="section-label">Backtest</div>' + statline(bt) + backtest_note(model)
    return card("Track Record", "Every pick graded against the final score", body)


def ratings_preview(ratings, n=25):
    teams = (ratings or {}).get("teams", [])[:n]
    if not teams:
        return ""
    return card(f"Top {len(teams)} by Adjusted Efficiency",
                f"As of {day_label(ratings['date'])}. Every team, the four factors and T-Rank are on the Ratings tab.",
                ratings_table(teams, compact=True))


def build_index(history, model, ratings):
    picks = history["picks"]
    today = NOW.date().isoformat()
    if picks:
        latest = max(p["date"] for p in picks)
        heading = "Today's Games" if latest == today else "Latest Games"
        top = sorted((p for p in picks if p["date"] == latest), key=lambda p: -p["prob"])
        picks_html = card(f"{heading}: {day_label(latest)}",
                          f"A pick for every game. The {TOP_N} most confident: "
                          + ", ".join(f"{p['pick']} ({p['prob']:.0f}%)" for p in top[:TOP_N]) + ".",
                          games_table([p for p in picks if p["date"] == latest]))
        if latest != today:
            picks_html = card("Today's Games", "", '<div class="empty-state">No Division I games today. Picks '
                              'resume on the next game day.</div>') + picks_html
    else:
        picks_html = card("Today's Games", "", '<div class="empty-state">The season tips off in early November, '
                          'and picks start on opening night. Until then, the Track Record below shows how the model '
                          'did on every game of last season.</div>')
    return page_shell("Home", "index.html", picks_html + track_record(history, model) + ratings_preview(ratings))


# ── Ratings ──────────────────────────────────────────────────────────────────
def ff(v):
    return f"{100 * v:.1f}"


def ratings_table(teams, compact=False):
    def cell(key, value, text):
        return f'<td class="num" data-key="{key}" data-value="{value}">{text}</td>'

    rows = ""
    for t in teams:
        tr = t.get("trank") or {}
        o, d = t["off"], t["def"]
        row = (cell("rank", t["rank"], t["rank"])
               + f'<td data-key="team" data-value="{escape(t["name"])}"><span class="matchup-team">{logo(t["id"])}'
                 f'{escape(t["name"])}</span> <span class="faint">{t["record"]}</span></td>'
               + cell("em", t["adj_em"], f"{t['adj_em']:+.1f}")
               + cell("o", t["adj_o"], f"{t['adj_o']:.1f}") + cell("d", t["adj_d"], f"{t['adj_d']:.1f}")
               + cell("t", t["adj_t"], f"{t['adj_t']:.1f}"))
        if not compact:
            row += (cell("sos", t["sos"], f"{t['sos']:+.1f}")
                    + cell("efg_o", o["efg"], ff(o["efg"])) + cell("efg_d", d["efg"], ff(d["efg"]))
                    + cell("tov_o", o["tov"], ff(o["tov"])) + cell("tov_d", d["tov"], ff(d["tov"]))
                    + cell("orb_o", o["orb"], ff(o["orb"])) + cell("orb_d", d["orb"], ff(d["orb"]))
                    + cell("ftr_o", o["ftr"], ff(o["ftr"])) + cell("ftr_d", d["ftr"], ff(d["ftr"])))
        row += cell("trank", tr.get("rank") or "", tr.get("rank") or DASH)
        if not compact:
            row += cell("barthag", tr.get("barthag") or "", f"{tr['barthag']:.3f}" if tr.get("barthag") else DASH)
        rows += f"<tr>{row}</tr>"

    def th(key, label, title="", cls="num"):
        t = f' title="{title}"' if title else ""
        return f'<th data-sort-key="{key}" class="{cls}"{t}>{label}</th>'

    head = (th("rank", "Rk") + th("team", "Team", cls="")
            + th("em", "AdjEM", "Points per 100 possessions better than an average team")
            + th("o", "AdjO", "Points scored per 100 possessions, adjusted for opponents and venue")
            + th("d", "AdjD", "Points allowed per 100 possessions, adjusted (lower is better)")
            + th("t", "AdjT", "Possessions per game against an average opponent"))
    if not compact:
        head += (th("sos", "SOS", "Average AdjEM of opponents played")
                 + th("efg_o", "eFG% O") + th("efg_d", "eFG% D") + th("tov_o", "TO% O") + th("tov_d", "TO% D")
                 + th("orb_o", "OR% O") + th("orb_d", "OR% D") + th("ftr_o", "FTR O") + th("ftr_d", "FTR D"))
    head += th("trank", "T-Rank", "Bart Torvik's T-Rank ranking")
    if not compact:
        head += th("barthag", "Barthag", "T-Rank's chance of beating an average team on a neutral court")
    return f"""<table class="data ratings-table" data-sortable>
      <thead><tr>{head}</tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def build_ratings(ratings):
    teams = (ratings or {}).get("teams", [])
    if not teams:
        body = card("Ratings", "", '<div class="empty-state">Ratings appear once the season has games. Every team '
                    'starts from part of last season\'s rating.</div>')
        return page_shell("Ratings", "ratings.html", body)
    avg = ratings.get("average", {})
    note = f"""<div class="table-footnote">Computed from every Division I box score this season, the way KenPom
      does it. Possessions = FGA - offensive rebounds + turnovers + 0.475 x FTA. AdjO and AdjD are points scored and
      allowed per 100 possessions, adjusted for each opponent's strength and for home court. AdjEM is the gap
      between them: how many points per 100 possessions a team would beat an average team by. AdjT is possessions
      against an average opponent. An average offense scores {avg.get('eff', DASH)} per 100 possessions at a tempo
      of {avg.get('tempo', DASH)}, and home court is worth about {avg.get('home_court', DASH)} points. The four
      factors are effective FG%, turnover rate, offensive rebound rate and free throw rate (FTA per FGA), for the
      team's offense (O) and what its defense allows (D), not adjusted for opponents. SOS is the average AdjEM of
      the opponents played. T-Rank and Barthag are Bart Torvik's (barttorvik.com). Early in the season every team
      starts from part of last season's rating. Select a column header to sort.</div>"""
    body = card("Ratings", f"All {len(teams)} Division I teams as of {day_label(ratings['date'])}, best first.",
                ratings_table(teams) + note)
    return page_shell("Ratings", "ratings.html", body)


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
            "summary": {"wins": w, "losses": l, "voided": sum(1 for p in picks if p.get("void")), "ml": ml_day(picks)},
            "games": [{
                "matchup": f"{p['away']} {'vs' if p.get('neutral') else '@'} {p['home']}", "pick": p["pick"],
                "prob": p["prob"], "correct": p.get("correct"), "void": bool(p.get("void")),
                "ml": ml_history(p),
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
            note += f" In the postseason it went {post['correct']}-{post['games'] - post['correct']}."
        parts.append(card("Backtest", "How the model did on a season it was never trained on",
                          statline(backtest_stats(model)) + band_table(bt["bands"], note)))
    if not parts:
        parts.append(card("Accuracy", "", '<div class="empty-state">No graded picks yet.</div>'))
    return page_shell("Accuracy", "accuracy.html", "".join(parts), charts=charts)



# ── Model tab ────────────────────────────────────────────────────────────────
FACTOR_LABELS = {
    "home_court": ("Home court", "points for the home team"),
    "em": ("Adjusted efficiency margin", "per point of margin projected from AdjEM and tempo"),
    "elo": ("Team rating (Elo) gap", "points per 100 rating points"),
    "efg": ("Shooting matchup (eFG%)", "points per percentage point of edge"),
    "tov": ("Turnover matchup", "points per percentage point of edge"),
    "orb": ("Offensive rebounding matchup", "points per percentage point of edge"),
    "ftr": ("Free throw rate matchup", "points per percentage point of edge"),
    "recent": ("Last 5 games form gap", "points per point of differential"),
    "rest": ("Extra rest", "points per extra day off vs. the opponent"),
    "trank": ("T-Rank vs. our ratings", "per point T-Rank's projected margin differs from ours"),
}


def recipe_setup(recipe):
    hl, carry, years = recipe.get("half_life"), recipe.get("carry", 0.6), recipe.get("years", 3)
    return [
        ("Learns from:", f"the last {years} seasons of Division I games"),
        ("Team ratings:", f"recent games count more; a game's weight halves every {hl} days" if hl
         else "every game this season counts the same"),
        ("Over the summer:", f"each team starts from {carry:.0%} of last season's rating, which fades as games come in"),
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
            ("Weekly", "Next check", "once 300+ new games are in; waits in the offseason"),
            (rows[0]["decision"].split(" ")[0] if rows else "-", "Last decision",
             f"{rows[0]['tested']} versions tested" if rows else "no retrains yet"),
        ],
        "setup": recipe_setup(model.get("recipe", {})) + [
            ("Looks at:", f"{len(now_w)} factors for every game, listed below"),
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
    """summary.json for a CBB card on the home page (github.com/ant56-arch/ant56-arch.github.io)."""
    picks = history["picks"]
    summary = {"updated": NOW.isoformat(), "heading": None, "picks": [], "record": None,
               "empty": "No college basketball picks yet. They start on opening night in early November.",
               "result_labels": ["WIN", "LOSS"],
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
        ml = moneyline.record([p for p in picks if season_of(p["date"]) == season])
        if ml:  # optional: the home page can show it next to the record
            summary["ml_record"] = {"value": f"{ml['wins']}-{ml['losses']}", "label": f"{season} moneyline",
                                    "sub": f"{moneyline.units_text(ml['units'])}, {ml['roi']:+.1%} ROI"}
    # Only picks actually made count here, never last season's backtest.
    return summary


def main():
    history = load_json(os.path.join(HERE, "picks_history.json"), {"picks": []})
    model = load_json(os.path.join(HERE, "model_weights.json"), {})
    ratings = load_json(os.path.join(HERE, "ratings.json"), {})
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    pages = {
        "index.html": build_index(history, model, ratings),
        "ratings.html": build_ratings(ratings),
        "history.html": build_history(history),
        "accuracy.html": build_accuracy(history, model),
        "model.html": build_model(model, load_json(os.path.join(HERE, "model_history.json"), {"runs": []})["runs"]),
        "terms.html": games_mod.legal_redirect("terms"),
        "privacy.html": games_mod.legal_redirect("privacy"),
    }
    for name, html in pages.items():
        with open(os.path.join(OUT_DIR, name), "w") as f:
            f.write(html)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(build_summary(history, model), f, indent=1)
    for asset in ASSETS:
        shutil.copy(os.path.join(WEB_DIR, asset), os.path.join(OUT_DIR, asset))
    print(f"Built {len(pages)} CBB pages in {OUT_DIR}")


if __name__ == "__main__":
    main()
