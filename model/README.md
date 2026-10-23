# NFL power-rating + Monte Carlo model

Projects the score of every game in the coming week, simulates each game 20,000 times, and compares the result
to the sportsbook lines so you can find where the Monday number is off. All data comes from nflverse:
`games.csv` (schedule, results, rest, roof, lines) from this repo, and play-by-play, depth charts and injury reports from
[nflverse-data releases](https://github.com/nflverse/nflverse-data/releases). These are the same files that
`nflfastR::load_pbp()` and `nfl_data_py` read.

## Monday workflow

```bash
pip install -r model/requirements.txt
python -m model.run_week                       # next week; downloads fresh nflverse data
python -m model.run_week --lines lines.csv     # compare against your book's Monday numbers
```

Writes `model/output/week_<season>_<week>.md` (report), `.csv` (all numbers) and `power_ratings_<season>_<week>.csv`.

**Lines file** (optional): `away_team,home_team,home_spread,total[,home_spread_odds,away_spread_odds,over_odds,under_odds,home_ml,away_ml]`.
`home_spread` uses sportsbook convention: -3 means the home team is favored by 3. Without `--lines`, the lines
currently in nflverse `games.csv` are used.

**Manual adjustments** go in `model/adjustments.csv`:

| column | meaning |
|---|---|
| season, week, team | which game-week and team |
| qb_id | override the starting QB (nflverse/gsis id, e.g. `00-0034857`) |
| points | add/subtract expected points for that team (non-QB injuries, weather, etc.) |
| note | free text, shown in the report |

The report marks a game "QB change" whenever the depth chart's QB1 differs from the team's last starter. Check
those games before betting them.

## How it works

The design follows the published methods behind the best-known public models:
- **ESPN FPI:** EPA-based team efficiency, opponent adjustment and a QB component.
- **nfelo / FiveThirtyEight Elo:** regression to the mean between seasons and QB adjustments.
- **Neil Paine's ridge model:** ridge-regressed offense and defense ratings with weekly exponential decay.
- **DraftKings' "ML + Monte Carlo" design:** a statistical layer sets team strength, and a simulator turns it into a distribution of final scores.

1. **Power ratings** (`ratings.py`): for each metric (points, EPA/play without turnovers, turnover rate, success rate, drives, special-teams EPA),
   a weighted ridge regression fits `metric = league avg + home + offense[team] - defense[opponent]` on all prior
   games. Weights halve every 8 weeks, with an extra 4 weeks charged for the offseason. The ridge penalty shrinks every
   team toward average, so the model regresses to the mean automatically. Garbage time (win prob outside 10-90%) is
   down-weighted. QBs get their own shrunk EPA/dropback rating. The QB adjustment is the starter's rating minus the rating of the QBs the team's numbers were earned with.
   Two more rating systems feed in:
   - **Elo** (`elo.py`): FiveThirtyEight/nfelo-style, with a margin-of-victory multiplier and 1/3 regression to the mean each offseason.
   - **Market ratings** (`market.py`): what earlier weeks' closing spreads and totals imply about each team. Only lines from games already played are used, never the line being bet, so the projection is still the model's own number.
2. **Points model** (`projection.py`): a ridge regression turns the ratings plus context into each team's expected points.
   The context features are home field, rest difference, travel distance, dome, QB change and the scoring environment.
   It is trained walk-forward, so every training row's features were computed only from games before it.
3. **Monte Carlo simulator** (`simulate.py`): each simulation plays the game possession by possession with a game clock:
   - Drive results come from nflverse 2021-2025 drive data: TD, FG, defensive TD, safety or no score.
   - Game script: leaders slow down and trailers speed up.
   - End-of-game logic: a team that needs a TD stops kicking field goals, and leaders run out the clock.
   - Regulation ties go to overtime.
   - Game-day noise is calibrated so the simulated outcomes spread exactly as much as the model's real out-of-sample error.
   - Scorelines are weighted so key-number frequencies match real 2015-2025 games: 3 in 14.7% of games, 7 in 8.5%.
4. **Edges** (`run_week.py`): the model's projection is its own opinion and never looks at the line. For pricing, the report shows two cover probabilities:
   - **raw:** treats the model as the truth. Picks and flags use this number, because it's what the backtest validated: PLAY at 56%+ and STRONG at 58%+.
   - **blended:** believes 35% of the model's disagreement with the line, which is what the backtest supports. It's the realistic win rate, so size bets on this one.

## Backtest

`python -m model.backtest --start 2016 --end 2025` (about 15 minutes). The backtest is walk-forward: ratings only use earlier games, the points model only trains on
earlier seasons, and every game is simulated 20,000 times. The full results are in [`BACKTEST.md`](BACKTEST.md). Headline numbers, 2016-2025 (2,761 games):

| | Result |
|---|---|
| Straight-up winners | 65.3% (closing-line favorite: 66.4%) |
| Average miss on the margin | 9.97 pts (closing line: 9.78) |
| ATS vs closing lines, every game | 52.9%, +61 units |
| ATS vs closing lines, raw cover ≥ 58% (STRONG) | 545-442, 55.2%, +6.4% ROI |
| ATS vs early-week lines (2016-2020), every game | 55.2% |
| ATS vs early-week lines, raw cover ≥ 58% | 256-202, 55.9% |
| Totals | about 51-52%, no reliable edge |

Breakeven at -110 is 52.4%. Results swing by season: the worst year (2023) lost 16 units at the 54% threshold.

`python -m model.calibrate` refits the simulator calibration (`model/calibration.json`). Re-run it after each season.
