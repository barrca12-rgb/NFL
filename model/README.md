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

Writes the **bet card** `reports/<season>_week<NN>.html` (open it in a browser), plus a Markdown copy, the full
numbers in `model/output/week_<season>_<week>.csv`, and `power_ratings_<season>_<week>.csv`.

The bet card gives every game one call:

| Call | Meaning | Stake |
|---|---|---|
| **BET** | Weeks 1-4: model cover % >= 53% *and* +EV at the realistic price (STRONG at 56%+). Week 5 on: the realistic cover % beats the break-even of the best book's price (VALUE) | Weeks 1-4: 2u STRONG, 1u PLAY. Week 5 on: 0.5u |
| **H2H** | Coach head-to-head angle: bet against the head coach whose teams have covered more in 4+ past meetings with this coach (57% since 1999). Also checks model bets: agreement is noted, disagreement halves the stake | 0.5u, or 1u when the model agrees |
| **SMALL** | Weeks 1-4 only: the model likes it but the realistic price is thin, usually a favorite laying 3 or 7 | 0.5u |
| **WATCH** | Not a bet now, but becomes one if a line moves up to 1.5 points in its favor (the card shows the number) | 0 until it gets there |
| **PASS** | No edge at any book | 0 |

Every bet also shows the **book** to bet it at and **good down to**: the worst number you can still take it at.

### Why the rule changes after Week 4 (research, walk-forward backtest 2016-2025)

* Thresholds were chosen on 2016-2020 and tested on 2021-2025. In Weeks 1-4 the edge held out of sample: model
  cover >= 60% went 62.0% in training and 61.3% in the test; a point edge >= 2 went 61.3% in both.
* From Week 5 on, no column's threshold survived: every "best" cutoff from 2016-2020 fell to 49-52% in 2021-2025.
  The share of the model's disagreement that showed up in results fell from 0.38 to -0.04 against closing lines.
* What did survive mid-season was **price**. Half a point is worth about 1.5 points of cover probability normally,
  2.6 at 7 and 4.1 at 3. Betting when the realistic cover % beats the best book's break-even went 53.0-53.3% in
  2021-2025 with half a point of shopping, against 51% without. Sharps' mid-season plays come mostly from this:
  betting the best number across books, early in the week.
* Within a picked side, no column (model cover %, point edge, realistic cover %) predicted covering much better
  than another. The value is mostly in which side to take.

**Lines from several books** (recommended): `--lines` accepts the flattened export from The Odds API or its
Google Sheets add-on (columns `game_id, commence_time, bookmaker, home_team, away_team, market, label_1, odd_1,
point_1, ...`). Download the sheet as CSV into `lines/` and run:

```bash
python -m model.run_week --lines lines/odds_<date>.csv              # FanDuel, DraftKings, BetMGM, Caesars
python -m model.run_week --lines lines/odds_<date>.csv --books all  # every book in the file
python -m model.odds --api-key $ODDS_API_KEY --out lines/today.csv  # pull live (needs api.the-odds-api.com allowed)
```

Each book's number is priced on its own and the card tells you which book to bet at. A plain CSV also works:
`away_team,home_team,home_spread[,book,home_spread_odds,away_spread_odds,total,over_odds,under_odds,home_ml,away_ml]`,
where `home_spread` uses sportsbook convention (-3 means home favored by 3). Without `--lines`, nflverse's consensus
line is used.

**Manual adjustments** go in `model/adjustments.csv`:

| column | meaning |
|---|---|
| season, week, team | which game-week and team |
| qb_id | override the starting QB (nflverse/gsis id, e.g. `00-0034857`) |
| points | add/subtract expected points for that team (non-QB injuries, weather, etc.) |
| note | free text, shown in the report |

The report marks a game "QB change" whenever the depth chart's QB1 differs from the team's last starter. Check
those games before betting them.

## Teasers

The card also lists 6-point teaser legs through the key numbers 3 and 7 (model/teasers.py) and suggests 2-team
teasers, never two legs from the same game:

| Grade | Leg | Backtest |
|---|---|---|
| **A** | underdog +1.5 to +2.5, total 47.5 or lower, model not against the side by 2+ | about 78% per leg since 2006 |
| **B** | the same underdogs with higher totals, or favorites -7.5 to -8.5 the model likes | about 75% per leg |

Break-even per leg is 73.9% on a 2-team teaser at -120 and 75.2% at -130, so shop teaser prices. AA teasers are
1u, AB 0.5u.

## Tracking real bets (closing line value and the shutoff)

```bash
python -m model.tracker log --season 2026 --week 8                 # record the card's bets
python -m model.tracker log --season 2026 --week 8 --only "TB,LAC"  # or just the ones you placed
python -m model.tracker settle                                     # after the games
python -m model.tracker report                                     # record, ROI, CLV and stake multiplier per angle
```

Bets are kept in `bets/bets.csv` (edit a line, price or stake to what you actually got). After the games, `settle`
fills in the closing line, result and profit. **Closing line value** (your number minus the closing number) is the
early warning: an angle that beats the close is real long before its win rate can prove it.

Each angle (STRONG, PLAY, VALUE, H2H, SMALL, TEASER) starts from its backtested win rate. Live results update it, and
the next card's stakes for that angle are scaled down as its updated win rate slides toward break-even (zero at
break-even), or capped at half if it averages negative CLV after 25 bets.

## Weekly schedule: Monday run, Thursday re-run

Run the same command again later in the week. It detects which nflverse injury report is available and uses it automatically:

| When | Injury report available | What the run does |
|---|---|---|
| Monday | Only last week's | Lists long-term absences carried forward from last week's report. **Doesn't adjust projections**: those players are already reflected in the ratings, and adjusting for them again made the backtest worse. |
| Wed/Thu | This week's practice report | **Adjusts projections** using practice participation. Did not practice on Thursday means a 76% chance of missing the game; limited means 22%. |
| Fri/Sat | Final report with game statuses | **Adjusts projections** using Out (100%), Doubtful (99%) and Questionable (30%). |

Every miss probability is measured from 2013-2025 nflverse injury reports and snap counts, not assumed. A player's
value is his usual offensive or defensive snap share. Only the change versus who the team was already missing
counts. QBs are handled separately by the QB rating.

How much injuries move the line (backtest, 2016-2025): each missing defensive starter-equivalent (a full-time
player's worth of snaps) is worth about +0.5 to +0.7 points to the opposing offense. Non-QB offensive absences
barely matter (about 0). It's a modest effect: it improves projected totals most and leaves ATS rates
about where they were. Its job is to keep Monday's numbers from going stale as injury news comes out. Add
`--no-injuries` to switch it off.

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
   - **raw:** treats the model as the truth. Picks and flags use this number, because it's what the backtest validated. Weeks 1-4: STRONG at 56%+ and PLAY at 53%+. Week 5 on: LEAN at 58%+.
   - **blended:** centers the game between the market consensus and the model (45% toward the model in Weeks 1-4, 20% after, as measured in the backtest), then prices it at each book's number. It's the realistic win rate; EV and the mid-season rule use it.

## Backtest

`python -m model.backtest --start 2016 --end 2025` (about 15 minutes). The backtest is walk-forward: ratings only use earlier games, the points model only trains on
earlier seasons, and every game is simulated 20,000 times. The full results are in [`BACKTEST.md`](BACKTEST.md). Headline numbers, 2016-2025 (2,761 games):

| | Result |
|---|---|
| Straight-up winners, model alone | 65.9% (closing-line favorite: 66.4%) |
| Straight-up winners, model blended with the line (report's winner picks) | 66.8% vs closing lines, 67.1% vs early-week lines |
| Average miss on the margin | 9.96 pts (closing line: 9.78) |
| ATS vs closing lines, every game | 52.3% |
| **ATS Weeks 1-4, raw cover ≥ 56% (STRONG)** | **174-119, 59.4% vs closing lines (9 of 10 seasons profitable); 61.9% vs early-week lines** |
| ATS Weeks 1-4, raw cover ≥ 53% (PLAY) | 245-181, 57.5% vs closing lines |
| ATS Week 5 on, raw cover ≥ 58% (LEAN) | 52.3% vs closing lines, 53.3% vs early-week lines |
| Totals | about 51-52%, no reliable edge |

Breakeven at -110 is 52.4%. The edge is concentrated early in the season. In Weeks 1-4 books still lean on
preseason opinions, and the model's ratings react faster. From Week 5 on the market has caught up, and only
large disagreements at the Monday number are worth a small bet.

`python -m model.calibrate` refits the simulator calibration (`model/calibration.json`). Re-run it after each season.
