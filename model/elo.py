"""Margin-of-victory Elo (FiveThirtyEight / nfelo family), used as an extra, results-based opinion of team strength.

Ratings are snapshotted before every week, so a week's games never see their own results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K = 20.0
HFA = 40.0              # Elo points of home field
REVERT = 1 / 3          # share of the gap to 1500 removed each offseason
ELO_PER_POINT = 25.0    # 25 Elo points ~ 1 point of spread


def elo_snapshots(games: pd.DataFrame) -> pd.DataFrame:
    """Pregame Elo for every (season, week, team) that has a game: columns season, week, team, elo."""
    g = games[games["result"].notna() | games["home_score"].isna()].sort_values(["season", "week", "gameday"])
    elo: dict[str, float] = {}
    rows = []
    last_season = None
    for (season, week), wk in g.groupby(["season", "week"], sort=True):
        if season != last_season:
            elo = {t: 1500 + (1 - REVERT) * (e - 1500) for t, e in elo.items()}
            last_season = season
        for t in pd.concat([wk["home_team"], wk["away_team"]]).unique():
            rows.append((season, week, t, elo.get(t, 1500.0)))
        for r in wk[wk["result"].notna()].itertuples():
            h, a = elo.get(r.home_team, 1500.0), elo.get(r.away_team, 1500.0)
            hfa = 0.0 if r.location == "Neutral" else HFA
            diff = h + hfa - a
            exp_home = 1 / (1 + 10 ** (-diff / 400))
            actual = 1.0 if r.result > 0 else 0.0 if r.result < 0 else 0.5
            winner_diff = diff if r.result > 0 else -diff
            mult = np.log(abs(r.result) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
            delta = K * mult * (actual - exp_home)
            elo[r.home_team] = h + delta
            elo[r.away_team] = a - delta
    return pd.DataFrame(rows, columns=["season", "week", "team", "elo"])
