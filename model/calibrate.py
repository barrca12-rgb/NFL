"""Calibrate the simulator to the model's real out-of-sample error and to real NFL key-number frequencies.

1. Walk-forward projections for past seasons (each season predicted by a points model trained only on earlier
   seasons) give the model's true error: SD of (actual - projected) margin and total.
2. The game-day noise in the simulator is sized so simulated outcomes spread exactly that much around the
   projection (swing_sd -> margin only, shared_sd -> total only, team_sd -> both).
3. Key-number weights: ratio of real to simulated frequency of each absolute final margin, so P(win by exactly 3)
   etc. match 2015+ NFL results.

Run:  python -m model.calibrate        (writes model/calibration.json)
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .data import CACHE, load_all
from .features import walk_forward_features
from .injuries import add_injury_features, historical_table
from .projection import fit_points_model, game_level
from .simulate import CALIBRATION_PATH, SimConfig, key_weight, simulate_game

FEATURES_PATH = os.path.join(CACHE, "features.parquet")


def oos_projections(f: pd.DataFrame, first_test: int, last_test: int) -> pd.DataFrame:
    out = []
    for s in range(first_test, last_test + 1):
        m = fit_points_model(f[f["season"] < s])
        te = f[f["season"] == s].copy()
        te["pred"] = m.predict(te)
        out.append(game_level(te))
    return pd.concat(out, ignore_index=True)


def _pooled(sample: pd.DataFrame, cfg: SimConfig, n: int):
    rm, rt, margins, ws = [], [], [], []
    for i, g in enumerate(sample.itertuples()):
        sim = simulate_game(g.home_pts, g.away_pts, g.exp_drives, playoff=g.game_type != "REG",
                            cfg=SimConfig(**{**cfg.__dict__, "n_sims": n}), rng=np.random.default_rng(i))
        m = sim["home"] - sim["away"]
        t = sim["home"] + sim["away"]
        rm.append(m - g.proj_margin)
        rt.append(t - g.proj_total)
        margins.append(m)
        ws.append(sim["w"])
    rm, rt, margins, ws = map(np.concatenate, (rm, rt, margins, ws))
    w = ws / ws.sum()
    sd_m = np.sqrt(w @ (rm - w @ rm) ** 2)
    sd_t = np.sqrt(w @ (rt - w @ rt) ** 2)
    return sd_m, sd_t, margins


def calibrate(n_games: int = 300, n_sims: int = 1500, iters: int = 4, verbose: bool = True) -> dict:
    f = pd.read_parquet(FEATURES_PATH)
    proj = oos_projections(f, 2016, int(f["season"].max()) - 1).dropna(subset=["result"])
    target_m = float((proj["result"] - proj["proj_margin"]).std())
    target_t = float((proj["total"] - proj["proj_total"]).std())
    real = proj["result"].abs().astype(int)
    real_freq = real.value_counts(normalize=True)

    sample = proj.sample(n_games, random_state=11)
    cfg = SimConfig(team_sd=3.0, swing_sd=3.0, shared_sd=1.0)
    for it in range(iters):
        # key weights from the unweighted margins of the current noise level
        _, _, margins = _pooled(sample, SimConfig(**{**cfg.__dict__, "key_weights": {}}), n_sims)
        sim_freq = pd.Series(np.abs(margins)).value_counts(normalize=True)
        kw = {}
        for k in range(0, 31):
            if sim_freq.get(k, 0) > 0 and real_freq.get(k, 0) > 0:
                kw[k] = float(np.clip(real_freq[k] / sim_freq[k], 0.3, 3.0))
        # renormalize so weights average to 1 over the simulated distribution
        avg = np.mean(key_weight(margins, kw))
        cfg.key_weights = {k: v / avg for k, v in kw.items()}

        sd_m, sd_t, _ = _pooled(sample, cfg, n_sims)
        if verbose:
            print(f"iter {it}: sim margin sd {sd_m:.2f} (target {target_m:.2f}), "
                  f"total sd {sd_t:.2f} (target {target_t:.2f}), swing {cfg.swing_sd:.2f}, shared {cfg.shared_sd:.2f}")
        # Noise adds variance roughly linearly (each sd enters margin/total variance with factor 4); game script
        # damps part of it, so step a bit further than the undamped solution.
        cfg.swing_sd = float(np.sqrt(max(cfg.swing_sd ** 2 + (target_m ** 2 - sd_m ** 2) / 4 * 1.6, 0.0)))
        cfg.shared_sd = float(np.sqrt(max(cfg.shared_sd ** 2 + (target_t ** 2 - sd_t ** 2) / 4 * 1.6, 0.0)))

    sd_m, sd_t, margins = _pooled(sample, cfg, n_sims)
    w = key_weight(margins, cfg.key_weights)
    am = np.abs(margins)
    check = {k: (round(float(np.average(am == k, weights=w)), 4), round(float(real_freq.get(k, 0)), 4))
             for k in (1, 2, 3, 4, 6, 7, 10, 14)}
    result = {
        "team_sd": cfg.team_sd, "swing_sd": cfg.swing_sd, "shared_sd": cfg.shared_sd,
        "key_weights": {str(k): v for k, v in cfg.key_weights.items()},
        "target_margin_sd": target_m, "target_total_sd": target_t,
        "sim_margin_sd": float(sd_m), "sim_total_sd": float(sd_t),
        "key_number_check_sim_vs_real": {str(k): v for k, v in check.items()},
    }
    with open(CALIBRATION_PATH, "w") as fh:
        json.dump(result, fh, indent=2)
    if verbose:
        print(json.dumps({k: v for k, v in result.items() if k != "key_weights"}, indent=2))
    return result


def build_features(last_season: int, refresh: bool = False, first: int = 2014,
                   refresh_injuries: bool = False) -> pd.DataFrame:
    """Walk-forward features for every game from `first` through `last_season` (cached for calibrate/backtest),
    with the historical injury features attached (used only by the injury-aware Thursday/Friday model)."""
    games, tg = load_all(2012, last_season, refresh=refresh)
    f = walk_forward_features(games, tg, first, last_season)
    f = add_injury_features(f, injury_table(last_season, refresh=refresh or refresh_injuries)[0])
    f.to_parquet(FEATURES_PATH)
    return f


def injury_table(last_season: int, refresh: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Cached historical injury features and miss probabilities (snap counts/injury reports start in 2013)."""
    tpath, ppath = os.path.join(CACHE, "injury_table.parquet"), os.path.join(CACHE, "p_miss.csv")
    if refresh or not (os.path.exists(tpath) and os.path.exists(ppath)):
        t, p = historical_table(range(2013, last_season + 1), refresh=refresh)
        t.to_parquet(tpath)
        p.to_csv(ppath)
    return pd.read_parquet(tpath), pd.read_csv(ppath, index_col=0)["p_miss"]


if __name__ == "__main__":
    calibrate()
