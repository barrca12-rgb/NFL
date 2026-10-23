"""Pregame features for every game, from the team-perspective (one row for each side of a game).

Everything here is knowable before kickoff: ratings built from earlier games, schedule rest days, travel,
roof, and the starting QB. Weather is deliberately left out because on Monday it is not known; use a manual
adjustment instead.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import REPO
from .elo import elo_snapshots
from .ratings import RatingConfig, build_ratings

# Features used by the points model, chosen by walk-forward backtest (2016-2025):
#  - EPA excluding turnovers + turnover rate (regressed) beat raw EPA; fumble-luck-adjusted EPA did not help
#  - Elo with margin of victory and market-implied ratings (from PAST closing lines) both added accuracy
#  - pass-EPA splits, bye/timezone/division flags and "regulars missing last game" added noise, not signal
FEATURES = [
    "off_pts", "def_pts", "off_epa_nto", "def_epa_nto", "off_to_rate", "def_to_rate", "off_sr", "def_sr",
    "st_diff", "qb_delta", "elo_diff", "mkt_diff", "mkt_tot",
    "is_home", "rest_diff", "travel", "dome", "env_pts",
]


def _airports() -> pd.DataFrame:
    a = pd.read_csv(os.path.join(REPO, "data", "airports.csv")).set_index("team")
    for old, new in {"OAK": "LV", "SD": "LAC", "STL": "LA"}.items():
        if old in a.index and new not in a.index:
            a.loc[new] = a.loc[old]
    return a


AIRPORTS = _airports()


def _miles(t1, t2) -> np.ndarray:
    a, b = AIRPORTS.reindex(t1), AIRPORTS.reindex(t2)
    la1, lo1, la2, lo2 = (np.radians(x.to_numpy(dtype=float)) for x in
                          (a["latitude"], a["longitude"], b["latitude"], b["longitude"]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return np.nan_to_num(3959 * 2 * np.arcsin(np.sqrt(h)))


def game_features(week_games: pd.DataFrame, r: dict, starters: dict[str, str] | None = None) -> pd.DataFrame:
    """week_games: rows of games.csv. starters: {team: qb_id} for the QB expected to start (default: last starter)."""
    starters = starters or {}
    rows = []
    for side, opp_side, home_flag in (("home", "away", 1), ("away", "home", -1)):
        d = pd.DataFrame({
            "game_id": week_games["game_id"].to_numpy(),
            "team": week_games[f"{side}_team"].to_numpy(),
            "opp": week_games[f"{opp_side}_team"].to_numpy(),
            "side": side,
        })
        neutral = (week_games["location"] == "Neutral").to_numpy()
        d["is_home"] = np.where(neutral, 0, home_flag)
        rest = week_games[f"{side}_rest"].fillna(7).to_numpy()
        orest = week_games[f"{opp_side}_rest"].fillna(7).to_numpy()
        d["rest_diff"] = np.clip(rest - orest, -7, 7)
        d["off_bye"] = (rest >= 12).astype(float)
        d["def_bye"] = (orest >= 12).astype(float)
        miles = _miles(week_games["away_team"], week_games["home_team"]) / 1000.0
        d["travel"] = np.where(neutral, 0, np.where(home_flag == -1, miles, -miles))
        tz = (AIRPORTS.reindex(week_games["home_team"])["time_zone"].to_numpy()
              - AIRPORTS.reindex(week_games["away_team"])["time_zone"].to_numpy())
        d["tz_east"] = np.where(neutral, 0, np.nan_to_num(tz) * (-home_flag))  # + = this team travelled east
        d["dome"] = week_games["roof"].isin(["dome", "closed"]).to_numpy().astype(float)
        d["div_game"] = week_games["div_game"].fillna(0).to_numpy()
        rows.append(d)
    f = pd.concat(rows, ignore_index=True)

    t, o = f["team"], f["opp"]
    f["off_pts"] = t.map(r["points"]["off"]).fillna(0).to_numpy()
    f["def_pts"] = o.map(r["points"]["def"]).fillna(0).to_numpy()
    f["off_epa"] = t.map(r["epa"]["off"]).fillna(0).to_numpy()
    f["def_epa"] = o.map(r["epa"]["def"]).fillna(0).to_numpy()
    f["off_sr"] = t.map(r["sr"]["off"]).fillna(0).to_numpy()
    f["def_sr"] = o.map(r["sr"]["def"]).fillna(0).to_numpy()
    f["off_pass"] = t.map(r["pass_epa"]["off"]).fillna(0).to_numpy()
    f["def_pass"] = o.map(r["pass_epa"]["def"]).fillna(0).to_numpy()
    for m in ("epa_adj", "epa_nto", "to_rate", "ed_epa", "ed_sr", "proe", "press_rate", "rush_epa", "pass_share"):
        f[f"off_{m}"] = t.map(r[m]["off"]).fillna(0).to_numpy()
        f[f"def_{m}"] = o.map(r[m]["def"]).fillna(0).to_numpy()
    # Matchup terms. Ratings are additive (offense minus defense); these test whether specific unit matchups matter
    # beyond that: pressure expected from this pass rush against this line, its interaction, and pass/run
    # efficiency weighted by how much this offense actually passes.
    pr = r["press_rate"]
    f["exp_press"] = (pr["b0"] + t.map(pr["off"]).fillna(0) - o.map(pr["def"]).fillna(0)).to_numpy()
    f["press_x"] = (t.map(pr["off"]).fillna(0) * -o.map(pr["def"]).fillna(0)).to_numpy()
    share = (r["pass_share"]["b0"] + t.map(r["pass_share"]["off"]).fillna(0)).clip(0.3, 0.8).to_numpy()
    pass_mm = (t.map(r["pass_epa"]["off"]).fillna(0) - o.map(r["pass_epa"]["def"]).fillna(0)).to_numpy()
    rush_mm = (t.map(r["rush_epa"]["off"]).fillna(0) - o.map(r["rush_epa"]["def"]).fillna(0)).to_numpy()
    f["style_pass_mm"] = share * pass_mm
    f["style_rush_mm"] = (1 - share) * rush_mm
    f["pass_mm"], f["rush_mm"] = pass_mm, rush_mm
    if "elo" in r:
        f["elo_diff"] = ((t.map(r["elo"]).fillna(1500) - o.map(r["elo"]).fillna(1500)) / 25.0).to_numpy()
    if "market" in r:
        mk = r["market"]
        # Half of the market's rating gap goes to each side's points; market total = both teams' contributions.
        f["mkt_diff"] = ((t.map(mk["mkt"]).fillna(0) - o.map(mk["mkt"]).fillna(0)) / 2).to_numpy()
        f["mkt_tot"] = (t.map(mk["mkt_tot"]).fillna(0) + o.map(mk["mkt_tot"]).fillna(0)).to_numpy()
    st = r["st"]["off"]
    f["st_diff"] = (t.map(st).fillna(0) - o.map(st).fillna(0)).to_numpy()
    f["env_pts"] = r["points"]["b0"]
    # Expected drives for this team: pace of both offenses.
    dr = r["drives"]
    f["exp_drives"] = (dr["b0"] + t.map(dr["off"]).fillna(0) - o.map(dr["def"]).fillna(0)
                       + o.map(dr["off"]).fillna(0) - t.map(dr["def"]).fillna(0)
                       + dr["b0"]).to_numpy() / 2.0

    qbr, base = r["qbr"], r["qb_base"]
    replacement = qbr.quantile(0.15) if len(qbr) else -0.1  # unknown QB = backup-level

    def qb_delta(team_series):
        out = []
        for tm in team_series:
            qid = starters.get(tm, r["starter"].get(tm))
            rating = qbr.get(qid, replacement) if qid is not None else replacement
            out.append(rating - base.get(tm, rating))
        return np.array(out)

    f["qb_delta"] = qb_delta(t)
    if r.get("qb_cpoe") is not None:
        qc = r["qb_cpoe"]
        h = r["history"]
        hh = h[h["qb_id"].notna()].assign(c=lambda x: x["qb_id"].map(qc))
        cbase = (hh["wt"] * hh["c"]).groupby(hh["team"]).sum() / hh["wt"].groupby(hh["team"]).sum()
        crep = qc.quantile(0.15) if len(qc) else -3.0
        cd = []
        for tm in t:
            qid = starters.get(tm, r["starter"].get(tm))
            v = qc.get(qid, crep) if qid is not None else crep
            cd.append(v - cbase.get(tm, v))
        f["qb_cpoe_delta"] = np.array(cd)
    f["opp_qb_delta"] = qb_delta(o)
    f["starter"] = [starters.get(tm, r["starter"].get(tm)) for tm in t]
    return f


def walk_forward_features(games: pd.DataFrame, tg: pd.DataFrame, first_season: int, last_season: int,
                          cfg: RatingConfig | None = None, use_actual_starters: bool = True) -> pd.DataFrame:
    """Features for every game in [first_season, last_season], each built only from games before its week.

    use_actual_starters: in the backtest, use the QB who actually started (on Mondays the starter is almost always
    known; when it isn't, the live run takes a manual override)."""
    cfg = cfg or RatingConfig()
    elo = elo_snapshots(games[games["season"] >= first_season - 12])
    gs = games[(games["season"] >= first_season) & (games["season"] <= last_season)]
    actual_qb = tg.set_index(["game_id", "team"])["qb_id"] if use_actual_starters else None
    out = []
    for (season, week), wk in gs.groupby(["season", "week"]):
        r = build_ratings(tg, season, week, cfg, elo=elo, games=games)
        starters = {}
        if actual_qb is not None:
            for gid, h, a in wk[["game_id", "home_team", "away_team"]].itertuples(index=False):
                for tm in (h, a):
                    q = actual_qb.get((gid, tm))
                    if isinstance(q, str):
                        starters[tm] = q
        f = game_features(wk, r, starters)
        f["season"], f["week"] = season, week
        out.append(f)
    f = pd.concat(out, ignore_index=True)
    meta = games[["game_id", "game_type", "home_score", "away_score", "spread_line", "total_line", "result", "total"]]
    f = f.merge(meta, on="game_id", how="left")
    f["points"] = np.where(f["side"] == "home", f["home_score"], f["away_score"])
    return f
