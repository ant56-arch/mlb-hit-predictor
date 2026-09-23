"""
model.py - NBA Edge's game model: pre-game features and the win probability.

League walks every game in date order. Before each game it can describe both
teams as they stood that morning (features), then it learns from the result
(update). Training, the backtest and the daily picks all use this one walk, so
the model is never fed anything it couldn't have known before tip-off.

Features are home-minus-away differences plus a home-court flag:
  elo        - Elo rating (margin-of-victory adjusted, carried across seasons)
  net        - season point differential per game, shrunk toward 0 early on
  recent     - point differential over the last 10 games, shrunk the same way
  rest, b2b  - days since the last game (capped at 3) and back-to-back flags
  missing    - value of regular rotation players who aren't playing, from
               each player's average game score this season
The model predicts the home team's margin; the win chance is the normal CDF of
margin / sigma.
"""

import math
from collections import defaultdict, deque
from datetime import date

ELO_START = 1500
ELO_K = 20
ELO_HOME = 100
ELO_CARRY = 0.75  # share of a team's Elo gap from average kept over the offseason
RECENT_GAMES = 10
NET_SHRINK = 5  # games of "average" (0 margin) blended into season differential
RECENT_SHRINK = 3
ROTATION_GAMES = 10  # a player counts as a regular if he played in any of the team's last 10 games
PLAYER_PRIOR_GAMES = 8  # last season's average fades out over this many games this season

FEATURES = ["home_court", "elo", "net", "recent", "rest", "b2b_home", "b2b_away", "missing_home", "missing_away"]


def game_score(p):
    """Simplified Hollinger game score from an espn.boxscore row."""
    _, _, _mins, pts, reb, ast, stl, blk, tov, fgm, fga, ftm, fta, _pm = p
    return pts + 0.4 * fgm - 0.7 * fga - 0.4 * (fta - ftm) + 0.5 * reb + stl + 0.7 * ast + 0.7 * blk - tov


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


class League:
    def __init__(self):
        self.elo = defaultdict(lambda: ELO_START)
        self.season = None
        self.margins = defaultdict(list)  # team -> this season's margins
        self.last_date = {}
        self.team_rosters = defaultdict(lambda: deque(maxlen=ROTATION_GAMES))  # team -> sets of player ids
        self.player_team = {}  # player -> team he last played for
        self.player_name = {}
        self.p_season = defaultdict(lambda: [0, 0.0])  # player -> [games, game score sum] this season
        self.p_prior = {}  # player -> last season's average game score

    # ── season rollover ──
    def _start_season(self, season):
        if self.season is not None:
            mean = sum(self.elo.values()) / max(len(self.elo), 1)
            for t in list(self.elo):
                self.elo[t] = mean + ELO_CARRY * (self.elo[t] - mean)
            self.p_prior = {pid: s / g for pid, (g, s) in self.p_season.items() if g >= 10}
            self.p_season.clear()
            self.margins.clear()
            self.team_rosters.clear()
        self.season = season

    def _check_season(self, season):
        if season != self.season:
            self._start_season(season)

    # ── player values ──
    def player_value(self, pid):
        g, s = self.p_season.get(pid, (0, 0.0))
        prior = self.p_prior.get(pid)
        if prior is None:
            return s / g if g else 0.0
        w = min(g, PLAYER_PRIOR_GAMES) / PLAYER_PRIOR_GAMES
        return w * (s / g if g else 0.0) + (1 - w) * prior

    def rotation(self, team):
        """Players who played for this team in any of its last 10 games and
        haven't turned up for another team since."""
        ids = set().union(*self.team_rosters[team]) if self.team_rosters[team] else set()
        return {pid for pid in ids if self.player_team.get(pid) == team}

    def missing_value(self, team, playing_ids):
        """Summed value of rotation players who aren't in playing_ids."""
        return sum(max(self.player_value(pid), 0.0) for pid in self.rotation(team) - set(playing_ids))

    def missing_players(self, team, playing_ids, top=4):
        out = [(self.player_value(pid), pid) for pid in self.rotation(team) - set(playing_ids)]
        out.sort(reverse=True)
        return [(self.player_name.get(pid, ""), round(v, 1)) for v, pid in out[:top] if v >= 5]

    # ── features ──
    def team_form(self, team):
        m = self.margins[team]
        net = sum(m) / (len(m) + NET_SHRINK)
        last = m[-RECENT_GAMES:]
        recent = sum(last) / (len(last) + RECENT_SHRINK)
        return net, recent

    def rest_days(self, team, day):
        last = self.last_date.get(team)
        if last is None:
            return 3
        return max(0, min(3, (date.fromisoformat(day) - date.fromisoformat(last)).days - 1))

    def features(self, g, home_missing=0.0, away_missing=0.0):
        self._check_season(g["season"])
        h, a = g["home"], g["away"]
        h_net, h_recent = self.team_form(h)
        a_net, a_recent = self.team_form(a)
        h_rest, a_rest = self.rest_days(h, g["date"]), self.rest_days(a, g["date"])
        return {
            "home_court": 0.0 if g.get("neutral") else 1.0,
            "elo": (self.elo[h] - self.elo[a]) / 100,
            "net": h_net - a_net,
            "recent": h_recent - a_recent,
            "rest": float(h_rest - a_rest),
            "b2b_home": 1.0 if h_rest == 0 else 0.0,
            "b2b_away": 1.0 if a_rest == 0 else 0.0,
            "missing_home": home_missing / 10,
            "missing_away": away_missing / 10,
        }

    def features_from_box(self, g, box):
        """Features for a past game, taking who played from its box score."""
        self._check_season(g["season"])
        box = box or {}

        def played(team):
            return [p[0] for p in box.get(team, {}).get("players", [])]

        hm = self.missing_value(g["home"], played(g["home"])) if g["home"] in box else 0.0
        am = self.missing_value(g["away"], played(g["away"])) if g["away"] in box else 0.0
        return self.features(g, hm, am)

    # ── learning from a result ──
    def update(self, g, box=None):
        self._check_season(g["season"])
        h, a = g["home"], g["away"]
        margin = g["home_pts"] - g["away_pts"]
        home_adv = 0 if g.get("neutral") else ELO_HOME
        diff = self.elo[h] + home_adv - self.elo[a]
        expected = 1 / (1 + 10 ** (-diff / 400))
        won = 1.0 if margin > 0 else 0.0
        winner_diff = diff if margin > 0 else -diff
        mult = ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)
        shift = ELO_K * mult * (won - expected)
        self.elo[h] += shift
        self.elo[a] -= shift

        self.margins[h].append(margin)
        self.margins[a].append(-margin)
        self.last_date[h] = self.last_date[a] = g["date"]

        for team in (h, a):
            rows = (box or {}).get(team, {}).get("players", [])
            if not rows:
                continue
            ids = set()
            for p in rows:
                pid = p[0]
                ids.add(pid)
                self.player_team[pid] = team
                self.player_name[pid] = p[1]
                rec = self.p_season[pid]
                rec[0] += 1
                rec[1] += game_score(p)
            self.team_rosters[team].append(ids)


def predict_margin(weights, feats):
    return sum(weights["coef"][k] * feats[k] for k in weights["features"])


def win_prob(weights, feats):
    """(home win probability, projected home margin)."""
    m = predict_margin(weights, feats)
    return norm_cdf(m / weights["sigma"]), m
