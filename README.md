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
A logistic regression (`research/train_model.py`) fit on every game of the past
season for every regular hitter. Each feature is computed as of the morning of
that game, so the model never learns from stats it couldn't have known:

- season and last-14-day batting average (shrunk toward a prior when the sample is small, so it works from Opening Day)
- at-bats per game (a proxy for lineup spot and playing time)
- opposing starter's ERA, WHIP and K/9
- platoon split, park factor, home/away

Features that never vary in the training data are dropped automatically.
`features.py` holds the feature definitions shared by training and the daily
run. Training scores the model on the last 20% of the season before refitting
on everything. The key check is how often the model's top 10 per day got a hit.

### Retraining (do this each offseason)
Run the **Research Data Pull + Retrain** workflow by hand after the regular
season ends. It pulls the full season (`research/pull_season_data.py`),
retrains, and commits `research_gamelogs.json` and `model_weights.json`.

## Site
`build_site.py` writes `dist/` from `picks_history.json`, `data/slate.json`
(today's full slate, not committed) and `model_weights.json`. The pages are
Home, Players, History and Accuracy. Styles and scripts live in `web/`.

To build it locally, run `python predict.py && python build_site.py`, then open `dist/index.html`.

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

### Retraining (each offseason, after the Finals)
Run **NBA Research Data Pull + Retrain** by hand. It pulls the last three
finished seasons (`nba/research/pull_seasons.py`) into `nba/data/seasons/`,
retrains, re-runs the backtest on the latest season and commits the data and
`nba/model_weights.json`. Day files for that season can then be deleted.

## Files
- `predict.py`: daily picks and results
- `features.py`: feature definitions shared by training and prediction
- `build_site.py`, `web/`: the static site
- `model_weights.json`: trained model
- `picks_history.json`: every pick and its result
- `research/`: season data pull and model training
- `nba/`: NBA Edge (pipeline, model, pages, data)
