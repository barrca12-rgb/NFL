"""Market-implied power ratings (the nfelo idea): what past closing lines say about each team.

Before week W, fit  spread_line = hfa * home + mkt[home] - mkt[away]  and  total_line = b0 + tot[home] + tot[away]
on closing lines of earlier games (time-decayed ridge). Only lines of games already played are used, never the
line being bet on, so the model still forms its own number for this week; the market's past opinion is one input
and the stats (EPA, success rate, QB, Elo, ...) decide where it disagrees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ratings import time_index

HALF_LIFE = 4.0      # weeks
OFFSEASON_GAP = 6.0  # extra weeks of decay across an offseason
PENALTY = 2.0        # ridge shrinkage toward average, in "games"


def market_snapshot(games: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """Team market ratings from closing lines of games before (season, week). Index = team; columns mkt, mkt_tot."""
    t = time_index(games["season"], games["week"])
    now = time_index(season, week)
    h = games[(t < now) & (games["season"] >= season - 2) & games["spread_line"].notna()]
    if h.empty:
        return pd.DataFrame(columns=["mkt", "mkt_tot"])
    teams = sorted(set(h["home_team"]) | set(h["away_team"]))
    idx = {tm: i for i, tm in enumerate(teams)}
    k, n = len(teams), len(h)
    ago = now - time_index(h["season"], h["week"]) + OFFSEASON_GAP * (season - h["season"].to_numpy())
    w = 0.5 ** (np.asarray(ago, dtype=float) / HALF_LIFE)
    hi = h["home_team"].map(idx).to_numpy()
    ai = h["away_team"].map(idx).to_numpy()
    rows = np.arange(n)

    def solve(X, y):
        lam = np.full(X.shape[1], PENALTY * w.mean())
        lam[0] = 1e-6
        return np.linalg.solve((X.T * w) @ X + np.diag(lam), (X.T * w) @ y)

    X = np.zeros((n, 1 + k))
    X[:, 0] = (h["location"] != "Neutral").to_numpy()
    X[rows, 1 + hi] = 1.0
    X[rows, 1 + ai] = -1.0
    b = solve(X, h["spread_line"].to_numpy(dtype=float))

    Xt = np.zeros((n, 1 + k))
    Xt[:, 0] = 1.0
    Xt[rows, 1 + hi] += 1.0
    Xt[rows, 1 + ai] += 1.0
    tot = h["total_line"].fillna(h["total_line"].mean()).to_numpy(dtype=float)
    bt = solve(Xt, tot)
    return pd.DataFrame({"mkt": b[1:], "mkt_tot": bt[1:]}, index=teams)
