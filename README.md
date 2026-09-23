# ⚾ MLB Edge

The chance each MLB hitter gets at least one hit today, published as a static
site on GitHub Pages. It's the baseball sibling of
[NFL Edge](https://ant56-arch.github.io/nfl-edge/nfl/index.html), with the same
look and cross-linked NFL / CFB / MLB tabs. No betting odds are used anywhere.

## How it works
`.github/workflows/daily.yml` runs `predict.py`, commits `picks_history.json`,
then builds and publishes the site:

| Time (ET) | Mode | What happens |
|---|---|---|
| 11:00am, 4:00pm, 6:30pm | `picks` | Scores every hitter in today's games and picks the top 10 (max 2 per game). Once a team posts its lineup, only its starters can be picked. A pick locks when its game starts; later runs only refill picks from games that haven't. |
| 2:00am | `results` | Grades pending picks from box scores. A pick with no at-bats (sat, walked every time, postponed) is **no decision**, not a miss. |

## The model
A logistic regression (`research/train_model.py`) fit on every game of recent
seasons for every regular hitter. Each feature is computed as of the morning of
that game, so the model never learns from stats it couldn't have known:

- season and last-14-day batting average (shrunk toward last season's average, or the league's, when the sample is small, so it works from Opening Day)
- at-bats per game (a proxy for lineup spot and playing time)
- opposing starter's ERA, WHIP and K/9
- platoon split, park factor, home/away

Features that never vary in the training data are dropped automatically.
`features.py` holds the feature definitions shared by training and the daily
run. The key check is how often the model's top 10 per day got a hit.

### Retraining (automatic, weekly)
The **Research Data Pull + Retrain** workflow runs every Monday morning. It
pulls the last three seasons (`research/pull_season_data.py`; finished seasons
are cached, the one in progress is pulled fresh), then `research/train_model.py`:

1. holds out the most recent ~8,000 hitter-games (about five weeks),
2. fits each candidate *recipe* (1, 2 or 3 years of games; two regularization
   strengths; the fixed park table or park factors learned from results, which
   also covers new and renamed ballparks) on the games before that, including
   the recipe the live model uses, and scores them on the held-out games,
3. switches recipe only if the best one beats the current recipe's log loss by
   a margin; otherwise the current recipe stays,
4. refits the chosen recipe on every game and saves `model_weights.json`, unless
   it fails a sanity check (it must beat guessing the average on held-out games
   and can't score clearly worse than the live model on recent games),
5. appends the run to `model_history.json`: every candidate's score, what was
   kept or switched, and the live picks' record over the last week.

It does nothing in the offseason (no games newer than the live model). Run the
workflow by hand to force a retrain. Every pick records which model made it
(`model`, the model's `trained_at`).

## Site
`build_site.py` writes `dist/` from `picks_history.json`, `data/slate.json`
(today's full slate, not committed) and `model_weights.json`. The pages are
Home, Games, Players, History, Accuracy and Model. Games (the team model below)
reads `teams/picks_history.json`. Styles and scripts live in `web/`.

To build it locally, run `python predict.py && python build_site.py`, then open `dist/index.html`.

## Game picks (MLB team model)
`teams/` picks the winner of every MLB game with a win chance, the way NBA Edge
does. Every daily run (any mode) runs `teams/predict.py`, which stores each
finished day's final scores and starting-pitcher lines from the MLB Stats API
(`teams/data/days/`), grades pending picks (a postponed game is no decision),
and picks every game today from the probable starters, refreshing until first
pitch and then locking. `teams/model.py` is a logistic regression on home
field, the Elo gap (carried across seasons) and the starters' gap (ERA and FIP
to date plus past seasons, shrunk toward average). `teams/research/train.py`
retrains at most weekly with the same promote-only-if-better guard as NBA and
logs to `teams/model_history.json`. **MLB Team Model Data Pull + Retrain**
rebuilds `teams/data/seasons/` from scratch if ever needed.

## NBA Edge
The NBA site lives in `nba/` and publishes to
https://ant56-arch.github.io/mlb-hit-predictor/nba/ from the same daily
workflow. Every run (any mode) runs `nba/predict.py`, which:

1. stores each finished day's scores and box scores from ESPN (`nba/data/days/`),
2. grades pending picks against final scores, and
3. gives every game today a pick and a win chance, refreshed with the latest
   injury report until tip-off, then locked.

`nba/build_pages.py` then writes `dist/nba/` (Home, History, Accuracy) and
`summary.json` for the home page. No betting odds are used.

The model (`nba/model.py`) predicts the home team's margin from each team's Elo
rating, season and last-10-game point differential, rest and back-to-backs,
home court, and the value of rotation players who are out (from box-score game
scores). The win chance is the normal CDF of margin / sigma. Every feature is
computed as of that morning.

Backtest (`nba/research/train.py`): fit on 2023-24 and 2024-25, then used to
pick all 1,231 regular-season games of 2025-26. It got 69.0% right (Elo alone:
69.2%, home team: 55.5%) with better calibrated probabilities than Elo alone
(log loss 0.585 vs. 0.599), and its top 3 picks each day won 78.6%.

### Retraining (automatic)
The daily run also runs `nba/research/train.py`, which retrains at most once a
week and only after 50+ new games. It holds out the last 250 games, fits each
candidate recipe (how fast Elo reacts, how much of it carries over the summer,
2 or 3 years of games) on the games before them, and switches recipe only if the
best one beats the current recipe's log loss by a margin. It then refits the
chosen recipe on every game, re-runs the backtest on the latest finished season,
and saves `nba/model_weights.json` if it passes the same sanity checks as MLB.
Every run is logged to `nba/model_history.json`.

Once a season is over (July), the daily run rolls that season's day files into
`nba/data/seasons/<year>.json`, so each new season starts clean with nothing to
do by hand. **NBA Research Data Pull + Retrain** is still there to rebuild the
stored seasons from ESPN if ever needed.

## Files
- `predict.py`: daily picks and results
- `features.py`: feature definitions shared by training and prediction
- `build_site.py`, `web/`: the static site
- `model_weights.json`: trained model
- `model_history.json`: every retrain, what it tried and what it chose
- `picks_history.json`: every pick and its result
- `research/`: season data pull and model training (pulled data goes to `research/data/`, not committed)
- `teams/`: the MLB team game-winner model (pipeline, model, data) behind the Games tab
- `nba/`: NBA Edge (pipeline, model, pages, data)
