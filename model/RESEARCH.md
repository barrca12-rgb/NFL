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

| **Coach head-to-head fade** (model/coaches.py): bet against the head coach whose teams have covered more in past meetings with this coach (shrink 20, 4+ meetings, rating >= 1.0) | 58.0% (1999-2009), 58.7% (2010-15), 53.5% (2016-20), 58.3% (2021-25); 255-191 combined. 57.4% when the model agrees, 54.3% when it disagrees. Not a home/away or division effect. Loose version (2+ meetings) as a check on model bets: 54.0% when it agrees, 50.4% when it disagrees |

| **Teasers through 3 and 7** (model/teasers.py): 6-point legs on underdogs +1.5..+2.5 | 77.0% / 75.6% / 77.6% (2006-12 / 2013-18 / 2019-25) vs 73.9% break-even at -120; ~79% with totals <= 47.5 (2016-25); 78.5% when the model likes the side. Favorites -7.5..-8.5: ~71-76%, only 75% when the model likes them |

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
| Unit matchups: expected pressure (sacks+hits allowed vs generated), pressure mismatch interaction, pass/run efficiency matchups, play-mix-weighted matchups | MAE 9.957-9.968 vs 9.956; ATS no better |
| Coach career ATS rating | 51.3-51.7% in every era at a 1-pt gap; too small to bet, not added |
| Same-season division rematch fade | 52-55% before 2021, 42.6% in 2021-25 |
| Late-season motivation: teams out of contention (can't reach 9 wins) vs teams still alive, Weeks 12-18 | 51.7% / 50.2% / 50.8% / 45.2% by era; no consistent edge, the market prices it |
| Mid-season model-cover thresholds (58%, 60%, point edge >= 3) | 2016-20 "best" cutoffs fell to 49-52% in 2021-25 |
| Situational spots mid-season (QB-change games, underdogs only, crossing 3/7, dead-zone exclusion) | 54-58% in 2016-20, 47-50% in 2021-25 |

Coverage-scheme matchups (Cover 1/2/3, man/zone) were not testable live: nflverse publishes coverage and pressure
charting only after each season ends (2026 is not out), so in-season the model can only use last season's tendencies.

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

## Glide test: coach head-to-head fade (October 2026, analysis only, no rule change)

`python -m model.coaches` prints the full grid. Walk-forward 1999-2025 on games.csv, same data and same running
head-to-head sums as `coach_h2h` (only earlier meetings count; pushes and zero ratings dropped). The question: does
the fade's win rate rise steadily as the thresholds get stricter (a real effect), or does it only spike at the
cutoff in use (4 meetings, |rating| >= 1.0, shrink 20: 255-191, 57.2%)? The full grid has 264 cells (meetings 1-8 ×
|rating| 0-2.5 by 0.25 × shrink 6/10/20).

Win % by minimum |rating|, 4+ meetings (bets in brackets):

| shrink | 0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 |
|---|---|---|---|---|---|---|
| 6 | 51.0 (1190) | 52.2 (1038) | 53.4 (844) | 55.5 (677) | 56.4 (491) | 57.5 (348) |
| 10 | 51.0 (1190) | 52.9 (982) | 54.2 (736) | 56.0 (491) | 58.2 (306) | 59.7 (181) |
| 20 | 51.0 (1190) | 53.5 (824) | **57.2 (446)** | 57.4 (202) | 54.7 (75) | 45.2 (31) |

Disjoint bands (shrink 20, 4+ meetings), so each game counts once: |rating| 0-0.5 45.4% (366), 0.5-1 49.2% (378),
1-1.5 57.0% (244), 1.5-2 59.1% (127), 2+ 54.7% (75). By meetings (shrink 20, |rating| >= 1): 1 meeting 45.4% (163),
2-3 50.4% (355), 4-5 57.8% (204), 6-8 60.4% (134), 9+ 51.9% (108).

Nearby cells (shrink 10-20, 3-5 meetings, |rating| 0.75-1.25, 18 cells) average 54.7%; the current cutoff is the best
of them except two at |rating| 1.25. By era at the current cutoff: 58.0% (138) / 58.7% (92) / 53.5% (101) / 58.3%
(108). The weak 2016-20 era is weak across the neighborhood (42-55%, average 49% in those 18 cells), and with 6+
meetings 2016-20 is under 50% in most cells (average 47%). The other three eras are above 50% in every one of those cells (averages 55.6-56.2%).

**Verdict.** Mostly a glide, not a lone spike. Under shrink 6 and 10 the win rate climbs almost steadily from 51% to
56-60% as the rating threshold tightens, and with 4+ meetings it rises with every step, which is what a real effect
looks like. Fewer than 4 meetings is close to a coin flip, so the 4-meeting floor is a real step, not an arbitrary
choice. Two cautions: at shrink 20 the curve falls back above |rating| 2 (on under 120 bets, so mostly noise), and
the exact cutoff in use is one of the best cells in its neighborhood, so some of its 57.2% is selection luck. Plan
on about 55% going forward rather than 57%: still above the 52.4% break-even, and the live tracker will show if it
is lower. The 2016-20 dip across every nearby cell is the one real warning sign.
