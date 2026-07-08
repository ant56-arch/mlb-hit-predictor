# ⚾ BTS Edge — MLB Daily Hit Predictor

Automated daily MLB hit prediction system built in Python, delivered via email through GitHub Actions.

## How it works
- **8am ET** — Morning email with top 10 daily picks across all games
- **4pm ET** — Evening email with picks for games starting 4pm or later, plus morning pick status
- **2am ET** — Silent results check, records whether each pick got a hit

## Scoring model
| Factor | Weight |
|---|---|
| Recent form (last 14 days) | 22% |
| Advanced metrics (exit velo, barrel%, hard hit%, OPS) | 20% |
| Pitcher matchup (ERA, WHIP, K/9) | 18% |
| Season batting average | 15% |
| Platoon advantage | 12% |
| Park factor | 8% |
| Home vs away | 5% |

## Data sources
- MLB Stats API (games, lineups, pitcher stats, box scores)
- Baseball Savant Statcast (exit velocity, barrel%, hard hit%)

## Files
- `predict.py` — main prediction script
- `picks_history.json` — running log of all picks and results
- `requirements.txt` — Python dependencies
- `.github/workflows/daily.yml` — automation schedule

## Future goals
See Issues tab for planned improvements.
