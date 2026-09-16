# ⚾ BTS Edge — MLB Daily Hit Predictor

Automated daily MLB hit prediction system built in Python, delivered via email through GitHub Actions.

## How it works
- **8am ET** — Morning email with top 10 daily picks across all games
- **4pm ET** — Evening email with picks for games starting 4pm or later, plus morning pick status
- **2am ET** — Silent results check, records whether each pick got a hit

## Scoring model
Picks are ranked by an actual **predicted probability of getting a hit**, not
a hand-weighted score. `research/train_model.py` fits a logistic regression on
real game-by-game outcomes (every game the top 100 OPS hitters played,
whether they got a hit or not — no survivorship bias) and saves the learned
coefficients to `model_weights.json`. `predict.py` loads those coefficients
and, for each player in today's games, plugs in:

- Recent form (trailing 14-day AVG)
- Season AVG and OPS
- Opposing starting pitcher (ERA, WHIP, K/9)
- Platoon split (batter hand vs. pitcher hand)
- Park factor
- Home vs. away

...and outputs a calibrated `P(hit)` for that specific matchup. No sportsbook
odds are used anywhere — this is purely a statistical estimate of hit
probability, not a betting line.

Note on model strength: whether a player gets a hit in a single game is
inherently high-variance (batting average is a game of small samples), so
don't expect huge swings in probability — the model's job is to be honestly
calibrated, not to promise certainty.

### Retraining
Run `pip install -r research/requirements.txt && python research/train_model.py`
from the repo root to refit the model. It trains on `research_gamelogs.json`
(historical backfill) plus any resolved picks in `picks_history.json` that
have `raw_features` attached (every pick saved by `predict.py` going forward
includes them), so retraining gets better as the season accumulates more
results. Commit the resulting `model_weights.json`.

## Data sources
- MLB Stats API (games, lineups, pitcher stats, box scores)

## Files
- `predict.py` — main prediction script
- `model_weights.json` — trained logistic regression coefficients used by `predict.py`
- `research/train_model.py` — (re)trains the hit-probability model
- `picks_history.json` — running log of all picks and results
- `requirements.txt` — Python dependencies for the daily automation
- `research/requirements.txt` — extra dependencies (numpy) needed only for training
- `.github/workflows/daily.yml` — automation schedule

## Future goals
See Issues tab for planned improvements.
