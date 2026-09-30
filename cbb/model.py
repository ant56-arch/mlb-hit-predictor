"""
model.py - CBB Edge's game model: KenPom-style team ratings, pre-game features
and the win probability.

League walks every game in date order. Before each game it can describe both
teams as they stood that morning (features), then it learns from the result
(update). Training, the backtest and the daily picks all use this one walk, so
the model is never fed anything it couldn't have known before tip-off.

Team ratings work the way KenPom's do. From each game's box score:
  possessions = FGA - offensive rebounds + turnovers + 0.475 x FTA
  efficiency  = points per 100 possessions
Adjusted offense (AdjO) and defense (AdjD) come from one weighted least-squares
fit over every Division I game this season: each team's offensive efficiency
in a game = league average + its offense + the opponent's defense + home court.
Recent games can count more (half_life), and every team starts the season at
part of last year's rating (carry), which fades as its games come in (prior).
Adjusted tempo (AdjT) is the same fit on possessions per game. AdjEM = AdjO -
AdjD, points per 100 possessions better than an average team.

Features are home-minus-away and favor the home team when positive:
  home_court  1 at a home court, 0 at a neutral site
  em          projected neutral-court margin from AdjEM at the expected tempo
  elo         Elo rating (margin-of-victory adjusted, carried across seasons)
  efg, tov, orb, ftr
              Dean Oliver's four factors (shooting, turnovers, offensive
              rebounding, free throw rate): each offense against the other
              team's defense, season to date, shrunk toward average early on
  recent      point differential over the last 5 games, shrunk the same way
  rest        days since the last game (capped at 4)
  trank       how much Bart Torvik's T-Rank disagrees with our ratings: its
              projected neutral-court margin minus em, from the latest T-Rank
              snapshot before the game day; 0 when there isn't one, so the
              model falls back on our ratings
The model predicts the home team's margin; the win chance is the normal CDF of
margin / sigma.
"""

import bisect
import math
from collections import defaultdict
from datetime import date

import numpy as np

ELO_START = 1500
ELO_K = 24
ELO_HOME = 90
ELO_CARRY = 0.7
RECENT_GAMES = 5
RECENT_SHRINK = 3
FACTOR_SHRINK = 5  # games of league-average four factors blended into each team's numbers
REST_CAP = 4
DI_MIN_GAMES = 5  # games in a season that make a team Division I (ESPN lists D-II and NAIA opponents too)

# Tunable settings of the walk. Retraining (research/train.py) tries other
# values and saves the winners with the model as "league".
DEFAULT_PARAMS = {"half_life": 60, "carry": 0.6, "prior": 4, "elo_k": ELO_K, "elo_carry": ELO_CARRY}

FEATURES = ["home_court", "em", "elo", "efg", "tov", "orb", "ftr", "recent", "rest", "trank"]
TRANK_MAX_AGE = 7  # days a T-Rank snapshot stays usable

# League-average four factors, used to shrink small samples.
AVG = {"efg": 0.505, "tov": 0.175, "orb": 0.29, "ftr": 0.32}
HOME_EFF = 1.5  # starting guess for home court, efficiency points per side


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def possessions(line):
    """Possessions from a team box score line (espn.BOX order)."""
    fgm, fga, tpm, tpa, ftm, fta, oreb, dreb, ast, stl, blk, tov, pf, pts = line
    return fga - oreb + tov + 0.475 * fta


def division_one(games):
    """{season: set of Division I team ids}: teams with DI_MIN_GAMES+ stored
    games that season, plus last season's set (so returning teams count from
    their first game)."""
    counts = defaultdict(lambda: defaultdict(int))
    for g in games:
        counts[g["season"]][g["home"]] += 1
        counts[g["season"]][g["away"]] += 1
    out, prev = {}, set()
    for s in sorted(counts):
        cur = {t for t, n in counts[s].items() if n >= DI_MIN_GAMES}
        out[s] = cur | prev
        prev = cur
    return out


def _solve(n, cols, vals, y, w, prior, lam, fixed_prior, fixed_lam):
    """Weighted ridge least squares over n unknowns. Row r of the design has
    vals[r] in columns cols[r] (both (R, k)). Unknowns 0..len(fixed_prior)-1 are
    shared terms pulled toward fixed_prior with weights fixed_lam; the rest are
    team terms pulled toward prior with weight lam."""
    wv = vals * w[:, None]
    flat = (cols[:, :, None] * n + cols[:, None, :]).ravel().astype(np.int64)
    M = np.bincount(flat, weights=(wv[:, :, None] * vals[:, None, :]).ravel(), minlength=n * n).reshape(n, n).astype(float)
    b = np.bincount(cols.ravel(), weights=(wv * y[:, None]).ravel(), minlength=n).astype(float)
    f = len(fixed_prior)
    M[np.arange(f), np.arange(f)] += fixed_lam
    b[:f] += np.asarray(fixed_lam) * np.asarray(fixed_prior)
    idx = np.arange(f, n)
    M[idx, idx] += lam
    b[f:] += lam * prior
    return np.linalg.solve(M, b)


class League:
    def __init__(self, params=None, di=None, trank=None):
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.di = di or {}  # season -> Division I team ids (division_one)
        self.trank = trank or {}  # date -> {team id: [adj_o, adj_d, barthag]} (store.load_trank)
        self.trank_days = sorted(self.trank)
        self.season = None
        self.elo = defaultdict(lambda: ELO_START)
        self.names = {}  # team id -> (abbr, name)
        self.reset_season()
        # Last season's final ratings, the starting point for this one.
        self.prior = {"o": {}, "d": {}, "t": {}, "mu": 104.0, "mu_t": 68.0, "h": HOME_EFF}

    def reset_season(self):
        # This season's D-I games with box scores: (day, home, away, neutral, home line, away line,
        # ordinal day, possessions, home efficiency, away efficiency).
        self.games = []
        self.margins = defaultdict(list)  # Division I games only, for recent form
        self.wins = defaultdict(lambda: [0, 0])  # every game, for the record
        self.opponents = defaultdict(list)
        self.last_date = {}
        self.factor_sums = defaultdict(lambda: defaultdict(float))
        self.factor_games = defaultdict(int)
        self._ratings = None
        self._ratings_key = None  # how many of self.games the cached ratings used

    # ── season rollover ──
    def _start_season(self, season):
        if self.season is not None:
            r = self.ratings(None)
            c = self.p["carry"]
            self.prior = {
                "o": {t: c * v for t, v in r["o"].items()}, "d": {t: c * v for t, v in r["d"].items()},
                "t": {t: c * v for t, v in r["t"].items()},
                "mu": r["mu"], "mu_t": r["mu_t"], "h": r["h"],
            }
            mean = sum(self.elo.values()) / max(len(self.elo), 1)
            for t in list(self.elo):
                self.elo[t] = mean + self.p["elo_carry"] * (self.elo[t] - mean)
            self.reset_season()
        self.season = season

    def _check_season(self, season):
        if season != self.season:
            self._start_season(season)

    def is_di(self, team, season=None):
        s = self.di.get(season or self.season)
        return True if s is None else team in s

    def di_game(self, g):
        return self.is_di(g["home"], g["season"]) and self.is_di(g["away"], g["season"])

    # ── adjusted efficiency ratings ──
    def ratings(self, day):
        """Adjusted offense, defense and tempo for every team from this
        season's games before `day` (all of them when day is None). Offense and
        defense are points per 100 possessions above average (for defense,
        above average means allowing more); tempo is possessions above average."""
        cut = len(self.games) if day is None else sum(1 for x in self.games if x[0] < day)
        if self._ratings is not None and self._ratings_key == cut:
            return self._ratings
        rows = self.games[:cut]
        pr = self.prior
        teams = sorted({t for x in rows for t in (x[1], x[2])} | set(pr["o"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        g = np.array([x[6:] for x in rows]).reshape(-1, 4)  # ordinal day, possessions, home eff, away eff
        hi = np.array([idx[x[1]] for x in rows], dtype=int)
        ai = np.array([idx[x[2]] for x in rows], dtype=int)
        loc = np.array([0.0 if x[3] else 1.0 for x in rows])
        hl = self.p["half_life"]
        w = 0.5 ** ((g[:, 0].max() - g[:, 0]) / hl) if hl and len(g) else np.ones(len(g))
        poss = g[:, 1]

        # Efficiency: home offense vs away defense, and away offense vs home defense.
        ones, R = np.ones(len(g)), len(g)
        eff_cols = np.concatenate([np.stack([0 * hi, 0 * hi + 1, 2 + hi, 2 + n + ai], 1),
                                   np.stack([0 * ai, 0 * ai + 1, 2 + ai, 2 + n + hi], 1)])
        eff_vals = np.concatenate([np.stack([ones, loc, ones, ones], 1), np.stack([ones, -loc, ones, ones], 1)])
        prior_od = np.array([pr["o"].get(t, 0.0) for t in teams] + [pr["d"].get(t, 0.0) for t in teams])
        x = _solve(2 + 2 * n, eff_cols.reshape(-1, 4), eff_vals.reshape(-1, 4),
                   np.concatenate([g[:, 2], g[:, 3]]) if R else np.zeros(0), np.concatenate([w, w]),
                   prior_od, self.p["prior"], [pr["mu"], pr["h"]], [1.0, 20.0])
        # Tempo: possessions = average + both teams' tempo terms.
        prior_t = np.array([pr["t"].get(t, 0.0) for t in teams])
        xt = _solve(1 + n, np.stack([0 * hi, 1 + hi, 1 + ai], 1).reshape(-1, 3), np.ones((R, 3)), poss, w,
                    prior_t, self.p["prior"], [pr["mu_t"]], [1.0])

        o, dd, t = x[2:2 + n], x[2 + n:], xt[1:]
        # Center the team terms (only the sums o_i + d_j and mu matter to predictions).
        rated = [idx[tm] for tm in teams if self.is_di(tm)] or list(range(n))
        mo, md, mt = (float(np.mean(v[rated])) if n else 0.0 for v in (o, dd, t))
        out = {
            "mu": float(x[0]) + mo + md, "h": float(x[1]), "mu_t": float(xt[0]) + 2 * mt,  # mu_t: an average matchup's possessions
            "o": {tm: float(o[i]) - mo for tm, i in idx.items()},
            "d": {tm: float(dd[i]) - md for tm, i in idx.items()},
            "t": {tm: float(t[i]) - mt for tm, i in idx.items()},
        }
        self._ratings, self._ratings_key = out, cut
        return out

    def team_rating(self, r, team):
        """(AdjO, AdjD, AdjEM, AdjT) on KenPom's scale."""
        o, d, t = r["o"].get(team, 0.0), r["d"].get(team, 0.0), r["t"].get(team, 0.0)
        return r["mu"] + o, r["mu"] + d, o - d, r["mu_t"] + t

    def projection(self, r, g):
        """(projected home points, away points, possessions) for a game."""
        h, a = g["home"], g["away"]
        loc = 0.0 if g.get("neutral") else 1.0
        poss = r["mu_t"] + r["t"].get(h, 0.0) + r["t"].get(a, 0.0)
        eff_h = r["mu"] + r["o"].get(h, 0.0) + r["d"].get(a, 0.0) + r["h"] * loc
        eff_a = r["mu"] + r["o"].get(a, 0.0) + r["d"].get(h, 0.0) - r["h"] * loc
        return eff_h * poss / 100, eff_a * poss / 100, poss

    # ── four factors ──
    def four_factors(self, team):
        """Season-to-date offense and defense four factors, shrunk toward average."""
        s, n = self.factor_sums[team], self.factor_games[team]
        k = FACTOR_SHRINK

        def rate(num, den, avg):
            if s[den] <= 0:
                return avg
            per_game = s[den] / n
            return (s[num] + k * per_game * avg) / (s[den] + k * per_game)

        return {side: {
            "efg": rate(f"{side}_efg_num", f"{side}_fga", AVG["efg"]),
            "tov": rate(f"{side}_tov", f"{side}_poss", AVG["tov"]),
            "orb": rate(f"{side}_oreb", f"{side}_reb_chances", AVG["orb"]),
            "ftr": rate(f"{side}_fta", f"{side}_fga", AVG["ftr"]),
        } for side in ("off", "def")}

    def _add_factors(self, team, own, opp):
        s = self.factor_sums[team]
        for side, line, other in (("off", own, opp), ("def", opp, own)):
            fgm, fga, tpm, tpa, ftm, fta, oreb, dreb, ast, stl, blk, tov, pf, pts = line
            s[f"{side}_efg_num"] += fgm + 0.5 * tpm
            s[f"{side}_fga"] += fga
            s[f"{side}_tov"] += tov
            s[f"{side}_poss"] += possessions(line)
            s[f"{side}_oreb"] += oreb
            s[f"{side}_reb_chances"] += oreb + other[7]  # + the other team's defensive rebounds
            s[f"{side}_fta"] += fta
        self.factor_games[team] += 1

    # ── T-Rank ──
    def trank_snapshot(self, day):
        """The latest T-Rank snapshot from before `day`, if it's recent enough."""
        i = bisect.bisect_left(self.trank_days, day) - 1
        if i < 0:
            return None
        d = self.trank_days[i]
        if (date.fromisoformat(day) - date.fromisoformat(d)).days > TRANK_MAX_AGE:
            return None
        return self.trank[d]

    # ── other form ──
    def recent(self, team):
        last = self.margins[team][-RECENT_GAMES:]
        return sum(last) / (len(last) + RECENT_SHRINK)

    def rest_days(self, team, day):
        last = self.last_date.get(team)
        if last is None:
            return REST_CAP
        return max(0, min(REST_CAP, (date.fromisoformat(day) - date.fromisoformat(last)).days - 1))

    def record(self, team):
        w, l = self.wins[team]
        return f"{w}-{l}"

    # ── features ──
    def features(self, g):
        self._check_season(g["season"])
        h, a = g["home"], g["away"]
        r = self.ratings(g["date"])
        _, _, em_h, _ = self.team_rating(r, h)
        _, _, em_a, _ = self.team_rating(r, a)
        poss = r["mu_t"] + r["t"].get(h, 0.0) + r["t"].get(a, 0.0)
        fh, fa = self.four_factors(h), self.four_factors(a)

        def matchup(k, sign=1):
            return 100 * sign * ((fh["off"][k] + fa["def"][k]) - (fa["off"][k] + fh["def"][k]))

        em = (em_h - em_a) * poss / 100
        trank = 0.0
        snap = self.trank_snapshot(g["date"])
        if snap and h in snap and a in snap:
            trank = ((snap[h][0] - snap[h][1]) - (snap[a][0] - snap[a][1])) * poss / 100 - em
        return {
            "home_court": 0.0 if g.get("neutral") else 1.0,
            "em": em,
            "elo": (self.elo[h] - self.elo[a]) / 100,
            "efg": matchup("efg"),
            "tov": matchup("tov", -1),  # turnovers hurt the offense
            "orb": matchup("orb"),
            "ftr": matchup("ftr"),
            "recent": self.recent(h) - self.recent(a),
            "rest": float(self.rest_days(h, g["date"]) - self.rest_days(a, g["date"])),
            "trank": trank,
        }

    # ── learning from a result ──
    def update(self, g, box=None):
        self._check_season(g["season"])
        h, a = g["home"], g["away"]
        self.names[h] = (g.get("home_abbr", ""), g.get("home_name", ""))
        self.names[a] = (g.get("away_abbr", ""), g.get("away_name", ""))
        margin = g["home_pts"] - g["away_pts"]
        self.wins[h][0 if margin > 0 else 1] += 1
        self.wins[a][1 if margin > 0 else 0] += 1
        self.last_date[h] = self.last_date[a] = g["date"]
        if not self.di_game(g):
            return  # non-Division I opponents don't move ratings, like KenPom
        self.margins[h].append(margin)
        self.margins[a].append(-margin)

        self.opponents[h].append(a)
        self.opponents[a].append(h)
        home_adv = 0 if g.get("neutral") else ELO_HOME
        diff = self.elo[h] + home_adv - self.elo[a]
        expected = 1 / (1 + 10 ** (-diff / 400))
        won = 1.0 if margin > 0 else 0.0
        winner_diff = diff if margin > 0 else -diff
        mult = ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)
        shift = self.p["elo_k"] * mult * (won - expected)
        self.elo[h] += shift
        self.elo[a] -= shift

        box = box or {}
        hl, al = box.get(h), box.get(a)
        poss = (possessions(hl) + possessions(al)) / 2 if hl and al else 0
        if poss >= 40:  # below that, a bad box score
            self.games.append((g["date"], h, a, bool(g.get("neutral")), hl, al,
                               date.fromisoformat(g["date"]).toordinal(), poss, 100 * hl[-1] / poss,
                               100 * al[-1] / poss))
            self._add_factors(h, hl, al)
            self._add_factors(a, al, hl)


def predict_margin(weights, feats):
    return sum(weights["coef"][k] * feats[k] for k in weights["features"])


def win_prob(weights, feats):
    """(home win probability, projected home margin)."""
    m = predict_margin(weights, feats)
    return norm_cdf(m / weights["sigma"]), m


def team_table(league, day=None):
    """Every Division I team's ratings, four factors, record and strength of
    schedule as of `day`, best AdjEM first, for the site's Ratings tab."""
    r = league.ratings(day)
    rows = []
    for t in r["o"]:
        # Every Division I team, including (before its first game) its preseason rating.
        if not league.is_di(t) or (league.di.get(league.season) is None and not league.opponents.get(t)):
            continue
        adj_o, adj_d, em, tempo = league.team_rating(r, t)
        ff = league.four_factors(t)
        opps = league.opponents.get(t, [])
        sos = sum(league.team_rating(r, o)[2] for o in opps) / len(opps) if opps else 0.0
        abbr, name = league.names.get(t, ("", ""))
        rows.append({
            "id": t, "abbr": abbr, "name": name, "record": league.record(t),
            "adj_em": round(em, 2), "adj_o": round(adj_o, 1), "adj_d": round(adj_d, 1), "adj_t": round(tempo, 1),
            "sos": round(sos, 2), "elo": round(league.elo[t]),
            "off": {k: round(v, 4) for k, v in ff["off"].items()},
            "def": {k: round(v, 4) for k, v in ff["def"].items()},
        })
    rows.sort(key=lambda x: -x["adj_em"])
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows
