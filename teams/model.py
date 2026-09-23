"""
model.py - the MLB team model: pre-game features and the win probability.

League walks every final game in date order. Before each game it can describe
both teams and both starting pitchers as they stood that morning (features),
then it learns from the result (update). Training, the backtest and the daily
picks all use this one walk, so the model never sees anything it couldn't
have known before first pitch.

Features (home minus away unless noted):
  home_field - 1 for every game (MLB has no neutral sites to speak of); acts
               as the home-field edge
  elo        - Elo rating gap / 100 (run-margin adjusted, carried across seasons)
  starters   - the starting pitchers' gap: each starter's edge over a
               league-average starter in runs per 9 innings (positive = better),
               home minus away
Season run differential and bullpen ERA were tried too; neither added
anything once Elo and the starters are in (tested on the 2025 and 2026
seasons), so they aren't features. The stored games keep bullpen lines in
case that changes.
A starter's line blends his ERA and FIP this season and (at a discount) his
past seasons, shrunk toward league average; an unannounced or unknown starter
counts as a bit worse than average. The win chance is a logistic regression
on these features (research/train.py fits it).
"""

import math
from collections import defaultdict
from datetime import date

ELO_START = 1500
ELO_HOME = 24
ELO_K = 4
ELO_CARRY = 0.67  # share of a team's Elo gap from average kept over the offseason
SP_SHRINK_IP = 40  # innings of league average blended into a starter's line
PRIOR_WEIGHT = 0.6  # how much a past season inning counts next to one this season
PRIOR_DECAY = 0.5  # each older season counts half as much again
UNKNOWN_SP = 0.35  # runs per 9 worse than average for a starter with no track record
LEAGUE_RA = 4.3  # starting point for league ERA before any games are stored

DEFAULT_PARAMS = {"elo_k": ELO_K, "elo_carry": ELO_CARRY, "sp_shrink": SP_SHRINK_IP}

FEATURES = ["home_field", "elo", "starters"]


def sigmoid(z):
    return 1 / (1 + math.exp(-max(-30.0, min(30.0, z))))


def _add(a, b, w=1.0):
    for i, v in enumerate(b):
        a[i] += w * v


class League:
    def __init__(self, params=None):
        self.p = {**DEFAULT_PARAMS, **(params or {})}
        self.elo = defaultdict(lambda: ELO_START)
        self.season = None
        self.wins = defaultdict(lambda: [0, 0])  # regular-season wins and losses
        self.last_date = {}
        # pitcher -> [outs, er, h, bb, k, hr] as a starter
        self.sp_season = defaultdict(lambda: [0.0] * 6)
        self.sp_prior = defaultdict(lambda: [0.0] * 6)
        self.sp_name = {}
        self.lg = [0.0] * 6  # league starter totals this season
        self.lg_prior = None  # last season's league starter totals

    # ── season rollover ──
    def _start_season(self, season):
        if self.season is not None:
            mean = sum(self.elo.values()) / max(len(self.elo), 1)
            for t in list(self.elo):
                self.elo[t] = mean + self.p["elo_carry"] * (self.elo[t] - mean)
            for pid in set(self.sp_prior) | set(self.sp_season):
                prior = [PRIOR_DECAY * v for v in self.sp_prior[pid]]
                _add(prior, self.sp_season.get(pid, [0.0] * 6))
                self.sp_prior[pid] = prior
            self.lg_prior = self.lg if self.lg[0] > 3000 else self.lg_prior
            self.sp_season.clear()
            self.wins.clear()
            self.lg = [0.0] * 6
        self.season = season

    def check_season(self, season):
        if season != self.season:
            self._start_season(season)

    # ── league context ──
    def _league_totals(self):
        """This season's starter totals, topped up with last season's early on."""
        if self.lg[0] >= 3000 * 3 or not self.lg_prior:
            return self.lg
        t = list(self.lg)
        need = max(0.0, 1 - self.lg[0] / 9000)
        _add(t, self.lg_prior, need * 9000 / max(self.lg_prior[0], 1))
        return t

    def league_ra(self):
        t = self._league_totals()
        return 27 * t[1] / t[0] if t[0] > 300 else LEAGUE_RA

    def fip_const(self):
        t = self._league_totals()
        if t[0] <= 300:
            return 3.1
        ip = t[0] / 3
        return self.league_ra() - (13 * t[5] + 3 * t[3] - 2 * t[4]) / ip

    # ── starters ──
    def sp_rating(self, pid):
        """Runs per 9 better than a league-average starter (positive = better)."""
        lg = self.league_ra()
        if not pid:
            return -UNKNOWN_SP
        s = list(self.sp_season.get(pid, [0.0] * 6))
        _add(s, self.sp_prior.get(pid, [0.0] * 6), PRIOR_WEIGHT)
        ip = s[0] / 3
        if ip < 1:
            return -UNKNOWN_SP
        era = 9 * s[1] / ip
        fip = (13 * s[5] + 3 * s[3] - 2 * s[4]) / ip + self.fip_const()
        raw = 0.5 * era + 0.5 * fip
        shrink = self.p["sp_shrink"]
        base = lg + UNKNOWN_SP  # a starter with little track record is usually a bit worse than average
        est = (ip * raw + shrink * base) / (ip + shrink)
        return lg - est

    def sp_line(self, pid):
        """(innings, ERA) this season as a starter, for display."""
        s = self.sp_season.get(pid)
        if not s or s[0] == 0:
            return 0, None
        return round(s[0] / 3, 1), round(27 * s[1] / s[0], 2)

    # ── teams ──
    def record(self, team):
        w, l = self.wins[team]
        return f"{w}-{l}"

    def rest_days(self, team, day):
        last = self.last_date.get(team)
        if last is None:
            return 3
        return max(0, min(3, (date.fromisoformat(day) - date.fromisoformat(last)).days - 1))

    def features(self, g, home_sp=None, away_sp=None):
        """Features for game g with the given starter ids (None = not announced)."""
        self.check_season(g["season"])
        h, a = g["home"], g["away"]
        return {
            "home_field": 1.0,
            "elo": (self.elo[h] - self.elo[a]) / 100,
            "starters": self.sp_rating(home_sp) - self.sp_rating(away_sp),
        }

    def features_for_past(self, g):
        """Features for a stored final game. Uses the starters who actually
        started, which were almost always the announced probables."""
        def sp(side):
            line = g.get(f"{side}_sp") or g.get(f"{side}_probable")
            return line[0] if line else None
        return self.features(g, sp("home"), sp("away"))

    # ── learning from a result ──
    def update(self, g):
        self.check_season(g["season"])
        h, a = g["home"], g["away"]
        margin = g["home_runs"] - g["away_runs"]
        diff = self.elo[h] + ELO_HOME - self.elo[a]
        expected = 1 / (1 + 10 ** (-diff / 400))
        won = 1.0 if margin > 0 else 0.0
        winner_diff = diff if margin > 0 else -diff
        mult = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
        shift = self.p["elo_k"] * mult * (won - expected)
        self.elo[h] += shift
        self.elo[a] -= shift

        if g["type"] == "regular":
            self.wins[h][0 if margin > 0 else 1] += 1
            self.wins[a][1 if margin > 0 else 0] += 1
        self.last_date[h] = self.last_date[a] = g["date"]

        for side in ("home", "away"):
            sp = g.get(f"{side}_sp")
            if sp:
                pid, name, stats = sp[0], sp[1], sp[2:8]
                self.sp_name[pid] = name
                _add(self.sp_season[pid], stats)
                _add(self.lg, stats)


def win_prob(weights, feats):
    """Home team's win probability."""
    return sigmoid(sum(weights["coef"][k] * feats[k] for k in weights["features"]))
