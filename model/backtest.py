"""Walk-forward backtest.

For every game in the test seasons: power ratings use only games before that week, the points model is trained
only on earlier seasons, and the game is simulated (default 20,000 times). Results are scored against the actual
outcome and against the market:

  * accuracy: margin/total MAE vs the closing line's MAE, straight-up winner %, Brier score
  * betting: ATS and over/under record + ROI vs closing lines at increasing edge thresholds
  * line movement (CLV): does the model's disagreement with an earlier line predict where the line closes?

Run:  python -m model.backtest --start 2018 --end 2025 [--sims 20000]
"""
from __future__ import annotations

import argparse
import os
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .calibrate import FEATURES_PATH, build_features, oos_projections
from .data import CACHE, HERE, REPO
from .simulate import SimConfig, simulate_game, summarize

OUT = os.path.join(HERE, "output")


def _sim_row(args):
    row, n = args
    cfg = SimConfig.calibrated(n_sims=n)
    sim = simulate_game(row["home_pts"], row["away_pts"], row["exp_drives"], playoff=row["game_type"] != "REG",
                        cfg=cfg, rng=np.random.default_rng(zlib.crc32(row["game_id"].encode())))
    out = {"game_id": row["game_id"], **summarize(sim, row["spread_line"], row["total_line"])}
    if not np.isnan(row.get("early_spread", np.nan)):
        e = summarize(sim, row["early_spread"])
        out.update({"p_home_cover_early": e["p_home_cover"], "p_away_cover_early": e["p_away_cover"]})
    return out


def simulate_all(proj: pd.DataFrame, n_sims: int, workers: int | None = None) -> pd.DataFrame:
    rows = proj[["game_id", "home_pts", "away_pts", "exp_drives", "game_type", "spread_line", "total_line",
                 "early_spread"]]
    jobs = [(r, n_sims) for r in rows.to_dict("records")]
    with ProcessPoolExecutor(workers or os.cpu_count()) as ex:
        res = list(ex.map(_sim_row, jobs, chunksize=8))
    return proj.merge(pd.DataFrame(res), on="game_id")


def _payout(odds):
    odds = np.where(np.isnan(odds), -110, odds)
    return np.where(odds < 0, 100 / -odds, odds / 100)


def bet_table(bt: pd.DataFrame, kind: str, thresholds=(0.50, 0.524, 0.54, 0.56, 0.58, 0.60)) -> pd.DataFrame:
    """Record and ROI (1 unit stakes at the actual closing odds) when the model's win prob clears a threshold."""
    if kind == "spread":
        pa, pb = bt["p_home_cover"], bt["p_away_cover"]
        won_a = bt["result"] > bt["spread_line"]
        won_b = bt["result"] < bt["spread_line"]
        odds_a, odds_b = bt["home_spread_odds"], bt["away_spread_odds"]
    else:
        pa, pb = bt["p_over"], bt["p_under"]
        won_a = bt["total"] > bt["total_line"]
        won_b = bt["total"] < bt["total_line"]
        odds_a, odds_b = bt["over_odds"], bt["under_odds"]
    pick_a = pa >= pb
    p = np.where(pick_a, pa, pb) / (pa + pb)            # win prob excluding pushes
    won = np.where(pick_a, won_a, won_b)
    lost = np.where(pick_a, won_b, won_a)
    pay = np.where(pick_a, _payout(odds_a.to_numpy(dtype=float)), _payout(odds_b.to_numpy(dtype=float)))
    rows = []
    for th in thresholds:
        sel = p >= th
        w, l = int(won[sel].sum()), int(lost[sel].sum())
        profit = float((pay[sel] * won[sel]).sum() - lost[sel].sum())
        rows.append({"min_model_prob": th, "bets": int(sel.sum()), "wins": w, "losses": l,
                     "pushes": int(sel.sum()) - w - l, "win_pct": w / max(w + l, 1),
                     "roi": profit / max(sel.sum(), 1), "units": profit})
    return pd.DataFrame(rows)


def calibration_table(bt: pd.DataFrame) -> pd.DataFrame:
    p = bt["p_home_cover"] / (bt["p_home_cover"] + bt["p_away_cover"])
    hit = (bt["result"] > bt["spread_line"]).astype(float)
    keep = bt["result"] != bt["spread_line"]
    d = pd.DataFrame({"p": p[keep], "hit": hit[keep]})
    d["bin"] = pd.cut(d["p"], [0, .35, .40, .45, .50, .55, .60, .65, 1])
    return d.groupby("bin", observed=True).agg(games=("hit", "size"), model_p=("p", "mean"), actual=("hit", "mean"))


def early_lines() -> pd.DataFrame:
    """sc_lines.csv (2013-2020): lines from earlier in the week than the close (nflverse spread_line), nflverse sign."""
    s = pd.read_csv(os.path.join(REPO, "data", "sc_lines.csv"))
    s = s[s["side"] == s["home_team"]][["season", "week", "home_team", "line"]]
    s["home_team"] = s["home_team"].replace({"OAK": "LV", "SD": "LAC", "STL": "LA"})
    s["early_spread"] = -s["line"]
    return s.drop(columns="line").drop_duplicates(["season", "week", "home_team"])


def early_line_bets(bt: pd.DataFrame, thresholds=(0.50, 0.524, 0.54, 0.56, 0.58, 0.60)) -> pd.DataFrame:
    """The Monday use case: bet the model's side against the EARLY line, graded on the final score at -110."""
    d = bt.dropna(subset=["early_spread", "p_home_cover_early"])
    ph, pa = d["p_home_cover_early"], d["p_away_cover_early"]
    home = ph >= pa
    p = np.where(home, ph, pa) / (ph + pa)
    won = np.where(home, d["result"] > d["early_spread"], d["result"] < d["early_spread"])
    lost = np.where(home, d["result"] < d["early_spread"], d["result"] > d["early_spread"])
    rows = []
    for th in thresholds:
        sel = p >= th
        w, l = int(won[sel].sum()), int(lost[sel].sum())
        rows.append({"min_model_prob": th, "bets": int(sel.sum()), "wins": w, "losses": l,
                     "win_pct": w / max(w + l, 1), "roi_at_-110": (w * 100 / 110 - l) / max(sel.sum(), 1)})
    return pd.DataFrame(rows)


def clv_test(bt: pd.DataFrame) -> dict | None:
    """If the model has signal the market lacks early in the week, the line should move toward it by kickoff."""
    d = bt.dropna(subset=["early_spread"])
    if d.empty:
        return None
    d["model_vs_early"] = d["proj_margin"] - d["early_spread"]
    d["move"] = d["spread_line"] - d["early_spread"]
    out = {"games": len(d), "corr(model-early, close-early)": float(np.corrcoef(d["model_vs_early"], d["move"])[0, 1])}
    for th in (1.5, 3.0):
        big = d[d["model_vs_early"].abs() >= th]
        toward = np.sign(big["move"]) == np.sign(big["model_vs_early"])
        away = (np.sign(big["move"]) == -np.sign(big["model_vs_early"])) & (big["move"] != 0)
        out[f"edge>={th}: games"] = len(big)
        out[f"edge>={th}: line moved toward model"] = int(toward.sum())
        out[f"edge>={th}: line moved away"] = int(away.sum())
        out[f"edge>={th}: avg move toward model (pts)"] = float((big["move"] * np.sign(big["model_vs_early"])).mean())
    return out


def run(start: int, end: int, n_sims: int, rebuild: bool) -> None:
    os.makedirs(OUT, exist_ok=True)
    if rebuild or not os.path.exists(FEATURES_PATH):
        build_features(last_season=end + 1 if end < 2026 else end)
    f = pd.read_parquet(FEATURES_PATH)
    proj = oos_projections(f, start, end).dropna(subset=["result", "spread_line"])
    games = pd.read_csv(os.path.join(CACHE, "games.csv"))
    proj = proj.merge(games[["game_id", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
                             "home_moneyline", "away_moneyline"]], on="game_id", how="left")
    proj = proj.merge(early_lines(), on=["season", "week", "home_team"], how="left")
    print(f"simulating {len(proj)} games x {n_sims:,} sims ...", flush=True)
    bt = simulate_all(proj, n_sims)
    bt.to_csv(os.path.join(OUT, f"backtest_games_{start}_{end}.csv"), index=False)

    bt["home_won"] = (bt["result"] > 0).astype(float) + 0.5 * (bt["result"] == 0)
    ml_imp = _implied(bt["home_moneyline"], bt["away_moneyline"])
    acc = pd.DataFrame({
        "games": bt.groupby("season").size(),
        "model_margin_MAE": (bt["proj_margin"] - bt["result"]).abs().groupby(bt["season"]).mean(),
        "close_spread_MAE": (bt["spread_line"] - bt["result"]).abs().groupby(bt["season"]).mean(),
        "model_total_MAE": (bt["proj_total"] - bt["total"]).abs().groupby(bt["season"]).mean(),
        "close_total_MAE": (bt["total_line"] - bt["total"]).abs().groupby(bt["season"]).mean(),
        "model_SU_pct": ((bt["p_home_win"] > .5) == (bt["result"] > 0)).groupby(bt["season"]).mean(),
        "market_SU_pct": ((ml_imp > .5) == (bt["result"] > 0)).groupby(bt["season"]).mean(),
        "model_brier": ((bt["p_home_win"] - bt["home_won"]) ** 2).groupby(bt["season"]).mean(),
        "market_brier": ((ml_imp - bt["home_won"]) ** 2).groupby(bt["season"]).mean(),
    })
    tot = acc.drop(columns="games").mul(acc["games"], axis=0).sum() / acc["games"].sum()
    acc.loc["ALL"] = [acc["games"].sum(), *tot.to_numpy()]

    ats = bet_table(bt, "spread")
    ou = bet_table(bt, "total")
    cal = calibration_table(bt)
    clv = clv_test(bt)
    early = early_line_bets(bt)
    ats_season = pd.concat({s: bet_table(g, "spread", (0.54,)) for s, g in bt.groupby("season")}).droplevel(1)

    lines = [f"# Backtest {start}-{end} ({len(bt)} games, {n_sims:,} sims each)\n",
             "Walk-forward: ratings use only prior games; points model trained only on prior seasons.\n",
             "## Accuracy vs closing market\n", acc.round(3).to_markdown(), "\n",
             "## ATS vs closing spread (stake 1u at actual closing odds)\n", ats.round(3).to_markdown(index=False), "\n",
             "## ATS by season (model prob >= 54%)\n", ats_season.round(3).to_markdown(), "\n",
             "## Over/Under vs closing total\n", ou.round(3).to_markdown(index=False), "\n",
             "## Cover-probability calibration (home cover, pushes excluded)\n", cal.round(3).to_markdown(), "\n"]
    if clv:
        lines += ["## Line movement test (early-week sc_lines -> close, overlapping seasons)\n",
                  pd.Series(clv).round(3).to_markdown(), "\n",
                  "## ATS vs EARLY-week lines (the Monday use case; sc_lines seasons only)\n",
                  early.round(3).to_markdown(index=False), "\n"]
    report = "\n".join(lines)
    with open(os.path.join(OUT, f"backtest_summary_{start}_{end}.md"), "w") as fh:
        fh.write(report)
    print(report)


def _implied(home_ml, away_ml):
    def p(ml):
        ml = ml.astype(float)
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    ph, pa = p(home_ml), p(away_ml)
    return pd.Series(ph / (ph + pa), index=home_ml.index)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--rebuild", action="store_true", help="rebuild walk-forward features first")
    a = ap.parse_args()
    run(a.start, a.end, a.sims, a.rebuild)
