"""Power ratings: opponent-adjusted, time-decayed ridge regression (the Massey/Sagarin/FPI family).

For a per-team-game metric y (offense `team` vs defense `opp`):

    y = intercept + hfa * is_home + off[team] - def[opp]

fit by weighted ridge, where each game's weight decays exponentially with how long ago it was played
(an extra gap is charged for the offseason, i.e. last year's games count less than last month's). The ridge
penalty shrinks every rating toward league average, which is the regression-to-the-mean that every good public
model applies (538/nfelo revert Elo between seasons, FPI shrinks to a prior, Paine uses ridge directly).
Ratings are in the units of the metric: off = how much better than average the offense is, def = how much
the defense suppresses it (positive = good defense).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEKS_PER_SEASON = 22


@dataclass
class RatingConfig:
    half_life: float = 8.0       # weeks for a game's weight to halve
    offseason_gap: float = 4.0  # extra "weeks" of decay charged across an offseason
    lookback_seasons: int = 2    # seasons of history before the current one that are used at all


METRICS = {
    # metric: ridge penalty, expressed in "games" of shrinkage toward average
    "points": 12.0,
    "epa": 12.0,
    "sr": 12.0,
    "pass_epa": 16.0,
    "drives": 20.0,
    "epa_adj": 12.0,
    "epa_nto": 12.0,
    "to_rate": 30.0,
}


def time_index(season, week):
    return np.asarray(season) * WEEKS_PER_SEASON + np.asarray(week)


def decay_weights(tg: pd.DataFrame, season: int, week: int, cfg: RatingConfig) -> np.ndarray:
    weeks_ago = time_index(season, week) - time_index(tg["season"], tg["week"])
    weeks_ago = weeks_ago + cfg.offseason_gap * (season - tg["season"].to_numpy())
    return 0.5 ** (weeks_ago / cfg.half_life)


def history_before(tg: pd.DataFrame, season: int, week: int, cfg: RatingConfig) -> pd.DataFrame:
    """Team-games strictly before (season, week): what is known on the Monday/Tuesday before that week."""
    t = time_index(tg["season"], tg["week"])
    mask = (t < time_index(season, week)) & (tg["season"] >= season - cfg.lookback_seasons)
    h = tg[mask].copy()
    h["wt"] = decay_weights(h, season, week, cfg)
    return h


def ridge_ratings(h: pd.DataFrame, metric: str, teams: list[str], penalty: float, with_def: bool = True):
    """Weighted ridge for y = b0 + hfa*is_home + off[team] - def[opp]. Returns (b0, hfa, off, def) as dict/Series."""
    h = h[h[metric].notna()]
    n, k = len(h), len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    ti = h["team"].map(idx).to_numpy()
    oi = h["opp"].map(idx).to_numpy()
    ncol = 2 + k + (k if with_def else 0)
    X = np.zeros((n, ncol))
    X[:, 0] = 1.0
    X[:, 1] = h["is_home"].to_numpy()
    X[np.arange(n), 2 + ti] = 1.0
    if with_def:
        X[np.arange(n), 2 + k + oi] = -1.0
    y = h[metric].to_numpy(dtype=float)
    w = h["wt"].to_numpy()
    # Penalty is scaled by the average game weight so it means "N average games of prior".
    lam = np.full(ncol, penalty * w.mean())
    lam[:2] = 1e-6
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X + np.diag(lam), XtW @ y)
    off = pd.Series(beta[2:2 + k], index=teams)
    dfn = pd.Series(beta[2 + k:], index=teams) if with_def else pd.Series(0.0, index=teams)
    return {"b0": beta[0], "hfa": beta[1], "off": off, "def": dfn}


def qb_ratings(h: pd.DataFrame, prior_mean: float = -0.02, prior_db: float = 250.0) -> pd.Series:
    """Shrunk, time-decayed EPA per dropback for every QB who has started a game in the window.

    The prior is slightly below league average: QBs with little track record are usually backups/rookies."""
    q = h[h["qb_id"].notna()]
    num = (q["wt"] * q["qb_epa_sum"]).groupby(q["qb_id"]).sum()
    den = (q["wt"] * q["qb_db"]).groupby(q["qb_id"]).sum()
    return (num + prior_mean * prior_db) / (den + prior_db)


def team_qb_baseline(h: pd.DataFrame, qbr: pd.Series) -> pd.Series:
    """Weighted average rating of the QBs a team's ratings were earned with."""
    q = h[h["qb_id"].notna()].copy()
    q["r"] = q["qb_id"].map(qbr)
    return (q["wt"] * q["r"]).groupby(q["team"]).sum() / q["wt"].groupby(q["team"]).sum()


def last_starter(h: pd.DataFrame) -> pd.Series:
    q = h[h["qb_id"].notna()].sort_values(["season", "week"])
    return q.groupby("team")["qb_id"].last()


def build_ratings(tg: pd.DataFrame, season: int, week: int, cfg: RatingConfig | None = None,
                  elo: pd.DataFrame | None = None, games: pd.DataFrame | None = None) -> dict:
    """All ratings as known before (season, week).

    elo: output of elo.elo_snapshots; games: nflverse games (for market ratings from past closing lines)."""
    cfg = cfg or RatingConfig()
    h = history_before(tg, season, week, cfg)
    teams = sorted(set(h["team"]) | set(h["opp"]))
    out = {"teams": teams, "history": h}
    for m, pen in METRICS.items():
        out[m] = ridge_ratings(h, m, teams, pen)
    out["st"] = ridge_ratings(h, "st_net", teams, 8.0, with_def=False)
    out["qbr"] = qb_ratings(h)
    out["qb_base"] = team_qb_baseline(h, out["qbr"]).reindex(teams).fillna(out["qbr"].mean())
    out["starter"] = last_starter(h)
    if elo is not None:
        e = elo[(elo["season"] == season) & (elo["week"] == week)]
        out["elo"] = e.set_index("team")["elo"]
    if games is not None:
        from .market import market_snapshot
        out["market"] = market_snapshot(games, season, week)
    return out


def power_table(r: dict) -> pd.DataFrame:
    """Human-readable power ratings for one point in time."""
    t = pd.DataFrame(index=r["teams"])
    t["off_pts"] = r["points"]["off"]
    t["def_pts"] = r["points"]["def"]
    t["off_epa"] = r["epa"]["off"]
    t["def_epa"] = r["epa"]["def"]
    t["st"] = r["st"]["off"]
    t["qb"] = r["starter"].map(r["qbr"])
    t["elo"] = r["elo"].reindex(t.index) if "elo" in r else np.nan
    t["market"] = r["market"]["mkt"].reindex(t.index) if "market" in r else np.nan
    return t
