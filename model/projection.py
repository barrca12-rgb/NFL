"""Expected points model: turns power ratings + game context into each team's expected points.

A linear model fit on walk-forward features (so the coefficients learn how much each rating is worth and how
much to trust it). Fitting is weighted toward recent seasons because scoring environment and home-field
advantage drift over time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURES


@dataclass
class PointsModel:
    coef: pd.Series
    intercept: float
    resid_sd: float

    def predict(self, f: pd.DataFrame) -> np.ndarray:
        return self.intercept + f[self.coef.index].to_numpy(dtype=float) @ self.coef.to_numpy()


def fit_points_model(train: pd.DataFrame, ridge: float = 5.0, recency: float = 0.85,
                     features: list[str] = FEATURES) -> PointsModel:
    """Weighted ridge on standardized features. recency^k down-weights a season k years before the newest."""
    tr = train.dropna(subset=["points"])
    tr = tr[tr["week"] > 1]  # week-1 ratings are pure carry-over; they still get predicted, just not trained on
    X = tr[features].to_numpy(dtype=float)
    y = tr["points"].to_numpy(dtype=float)
    w = recency ** (tr["season"].max() - tr["season"].to_numpy())
    mu = np.average(X, axis=0, weights=w)
    sd = X.std(axis=0) + 1e-9
    Z = (X - mu) / sd
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    lam = np.full(Z1.shape[1], ridge)
    lam[0] = 0
    beta = np.linalg.solve((Z1.T * w) @ Z1 + np.diag(lam), (Z1.T * w) @ y)
    coef = pd.Series(beta[1:] / sd, index=features)
    intercept = beta[0] - (beta[1:] / sd) @ mu
    resid = y - (intercept + X @ coef.to_numpy())
    return PointsModel(coef, float(intercept), float(np.sqrt(np.average(resid ** 2, weights=w))))


def game_level(f: pd.DataFrame, pred_col: str = "pred") -> pd.DataFrame:
    """Collapse team-perspective rows into one row per game with home/away expected points."""
    h = f[f["side"] == "home"].set_index("game_id")
    a = f[f["side"] == "away"].set_index("game_id")
    g = pd.DataFrame({
        "season": h["season"], "week": h["week"], "home_team": h["team"], "away_team": a["team"],
        "home_pts": h[pred_col], "away_pts": a[pred_col].reindex(h.index),
        "exp_drives": h["exp_drives"],
    })
    for c in ("game_type", "home_score", "away_score", "spread_line", "total_line", "result", "total"):
        if c in h:
            g[c] = h[c]
    g["proj_margin"] = g["home_pts"] - g["away_pts"]   # + = home wins by, same sign as spread_line/result
    g["proj_total"] = g["home_pts"] + g["away_pts"]
    return g.reset_index()
