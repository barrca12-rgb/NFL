# NFL projection model

Power ratings + Monte Carlo simulation (20,000 sims per game) that projects every game's score for the coming
week and flags where the Monday sportsbook lines look wrong.

```bash
pip install -r model/requirements.txt
python -m model.run_week                     # next week, lines from nflverse
python -m model.run_week --lines lines.csv   # compare against your book's Monday lines
python -m model.backtest --start 2016 --end 2025
```

See [`model/README.md`](model/README.md) for the method, inputs, and adjustments, and
[`model/BACKTEST.md`](model/BACKTEST.md) for the latest walk-forward backtest.

Data: [nflverse](https://github.com/nflverse) — play-by-play, depth charts and schedules/lines are downloaded at run
time. `data/` holds three small reference files from [nflverse/nfldata](https://github.com/nflverse/nfldata):
team airports (travel), early-week lines 2013-2020 (backtest), and a `games.csv` fallback for offline runs.
