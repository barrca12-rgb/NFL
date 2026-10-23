"""Head-coach head-to-head against the spread.

For every pair of head coaches, the running sum of how much each has covered in their past meetings (ATS margin
vs the closing line, all prior seasons and teams). Fading the coach with the better head-to-head ATS record has won
in every era since 1999 (games.csv, walk-forward: only meetings before the game count):

    shrink 20, >= 4 meetings, |rating| >= 1.0  (STRICT)
        1999-2009 58.0% | 2010-15 58.7% | 2016-20 53.5% | 2021-25 58.3%   (combined 255-191, 57.2%)
        with the model on the same side 57.4%, against it 54.3% (2016-2025)
    shrink 6, >= 2 meetings, |rating| >= 1.5   (LOOSE: used as a confidence check on model bets)
        model pick agrees 54.0%, disagrees 50.4% (2016-2025)

Not a home/away or division artifact (fading home 53.1%, away 53.8%; division 53.7%, non-division 53.2%).
Likely reasons: the market overweights "coach X owns coach Y", and the losing staff adjusts its plan.
"""
from __future__ import annotations

import pandas as pd

STRICT = {"shrink": 20.0, "min_meetings": 4, "min_rating": 1.0}
LOOSE = {"shrink": 6.0, "min_meetings": 2, "min_rating": 1.5}


def coach_h2h(games: pd.DataFrame, week_games: pd.DataFrame) -> pd.DataFrame:
    """For each game in week_games: meetings, the home coach's cumulative head-to-head ATS margin, both ratings,
    and which team the strict / loose rules say to bet (the side of the coach who has covered LESS)."""
    done = games[games["result"].notna() & games["spread_line"].notna()].sort_values(["season", "week", "gameday"])
    pair: dict[tuple, list] = {}
    for r in done.itertuples():
        pair.setdefault((r.home_coach, r.away_coach), []).append(r.result - r.spread_line)
    rows = []
    for r in week_games.itertuples():
        as_home = pair.get((r.home_coach, r.away_coach), [])
        as_away = pair.get((r.away_coach, r.home_coach), [])
        n = len(as_home) + len(as_away)
        total = sum(as_home) - sum(as_away)          # home coach's cumulative ATS margin vs this coach
        fade = r.away_team if total > 0 else r.home_team
        row = {"game_id": r.game_id, "h2h_meetings": n, "h2h_total": total, "home_coach": r.home_coach,
               "away_coach": r.away_coach}
        for name, rule in (("strict", STRICT), ("loose", LOOSE)):
            rating = total / (n + rule["shrink"]) if n else 0.0
            ok = n >= rule["min_meetings"] and abs(rating) >= rule["min_rating"]
            row[f"h2h_{name}"] = fade if ok else None
        rows.append(row)
    return pd.DataFrame(rows)
