# Research log: raising the all-game ATS win rate

All tests are walk-forward: each season is predicted only from earlier data, and thresholds or models are fit on
earlier seasons than the ones they are scored on. Breakeven at -110 is 52.4%. With about 2,200 games, a single
season-set measurement has a standard error of about ±1.1 points; with the 740 early-line games it is about ±1.8.

## Where things stand (2016-2025, current model)

| Measure | Result |
|---|---|
| All games vs closing lines, side chosen by simulated cover % | 52.3% |
| All games vs early-week lines (2016-2020) | 55.3% |
| Weeks 1-4, model cover >= 56%, vs closing lines | 59.4% |
| Week 5+, EV-positive at the best of several books | about 53% (2021-25, with shopping) |

## Tried and kept

| Change | Effect |
|---|---|
| Market-implied ratings from past closing lines | ATS vs close 50.9% -> 52.2%, MAE 10.02 -> 9.97 |
| Elo, turnover-free EPA, turnover rate | small MAE gain, kept. Turnovers end up nearly fully regressed: 1 SD of turnover rate is worth 0.04 pts (offense) and 0.12 (defense) vs 1.26 for the QB rating |
| 40-week QB memory over 4 seasons | MAE 9.972 -> 9.956, straight-up 65.3% -> 65.9% |
| Thursday/Friday injury report | totals MAE improved; ATS unchanged |
| Season-phase trust and rules, line shopping | turns the mid-season edge from ~51% into ~53% |

## Tried and rejected (did not hold out of sample)

| Idea | Result vs current |
|---|---|
| Model the margin directly instead of team points | worse (MAE 9.97-10.01) |
| Gradient-boosted trees, alone or blended with the linear model | worse or equal |
| Pass-EPA splits, bye/timezone/division flags | noise |
| "Regulars missing last game" from snap counts | worse |
| Monday carry-forward of last week's injury report | worse (double counts) |
| A cover classifier on top of the model (edge, season phase, Elo/stats/market rating gaps, QB, rest, line size) | 51.0-52.1% vs 52.0% on closing lines; best variant 54.6% vs 53.5% on early lines (within noise) |
| Early-down (1st/2nd down) EPA and success rate, as extra inputs or replacing success rate | MAE 9.964-9.967 vs 9.956; ATS 51.6-51.9% vs 52.2% |
| Pass rate over expected (coaching aggressiveness) | no change (9.957) |
| QB completion % over expected (CPOE) rating | worse (9.973) |
| Passing EPA offense/defense ratings, re-tested on the current model | worse (9.961; ATS 51.0%) |
| Joint tuning of rating memory, offseason gap and extra defensive shrinkage (27 settings, chosen on 2016-19, scored on 2020-25) | the best-on-2016-19 setting was worse on 2020-25 (9.975 vs 9.970); all within 0.03 |
| Turnover luck vs the market: bet against teams with better recent turnover margin / fumble-recovery luck | 48.6-49.9% in both periods; the market does not overrate lucky teams |
| Model disagrees with the line and picks the less-lucky team | 54.5% (2016-20), 47.0% (2021-25) |
| Mid-season model-cover thresholds (58%, 60%, point edge >= 3) | 2016-20 "best" cutoffs fell to 49-52% in 2021-25 |
| Situational spots mid-season (QB-change games, underdogs only, crossing 3/7, dead-zone exclusion) | 54-58% in 2016-20, 47-50% in 2021-25 |

## What this means

On every game against the closing line, the model is at about 52%. That is close to the ceiling for public
models, because the closing line already contains almost everything a box-score model knows. The remaining levers
are not more modeling of the same data:

1. **Bet earlier.** Against early-week lines the same picks win about 55%. The edge is largest on Monday and
   shrinks as the line moves toward the model (line-movement test: lines moved toward the model in 155 vs 109
   games when the disagreement was 3+ points).
2. **Shop prices.** Half a point is worth 1.3-2.0 points of cover probability normally, 2.6 at 7 and 4.1 at 3.
3. **New information the market is slow on:** confirmed starters, injury news after Monday, weather. These need
   data sources beyond nflverse (live injury/news feeds, forecasts).
4. **Concentrate stakes in Weeks 1-4,** where the model's edge is large and stable (9 of 10 seasons profitable).
