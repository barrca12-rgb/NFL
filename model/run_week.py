"""Monday run: project every game of the coming week, simulate it, and compare to the available lines.

    python -m model.run_week                         # next week, nflverse lines, 20,000 sims per game
    python -m model.run_week --week 8 --lines my_lines.csv
    python -m model.run_week --no-refresh            # use cached data (offline)

Outputs (model/output/):
    week_<season>_<week>.csv        one row per game, every number below
    week_<season>_<week>.md         readable report: projections, edges, flagged plays
    power_ratings_<season>_<week>.csv

Lines file (optional, CSV): away_team, home_team, home_spread, total, and optionally home_spread_odds,
away_spread_odds, over_odds, under_odds, home_ml, away_ml. home_spread uses SPORTSBOOK convention
(-3 = home favored by 3). Without it, the current lines in nflverse games.csv are used.

Manual adjustments: model/adjustments.csv (season, week, team, qb_id, points, note). `qb_id` overrides the
starting QB (gsis id); `points` adds/subtracts expected points for that team (non-QB injuries, weather, etc.).
"""
from __future__ import annotations

import argparse
import os
import zlib

import numpy as np
import pandas as pd

from .calibrate import build_features
from .data import HERE, depth_chart_qb1, load_all, load_games
from .elo import elo_snapshots
from .features import game_features
from .projection import fit_points_model
from .ratings import build_ratings, power_table
from .simulate import SimConfig, american, simulate_game, summarize

OUT = os.path.join(HERE, "output")
ADJ_PATH = os.path.join(HERE, "adjustments.csv")

# How much of the model's disagreement with the line to believe when pricing bets. The model's raw projection is
# its own (independent) opinion; these weights only affect the "blended" cover probabilities and EV. Spread value
# reflects the backtest: regressing (result - line) on (model - line) gives a slope of 0.34 +/- 0.13 against
# early-week lines and 0.23 +/- 0.08 against closing lines. Totals: slope 0.02, i.e. no edge, so no blend.
BLEND_SPREAD = 0.35
BLEND_TOTAL = 0.0
# Simulated cover probability (raw, pushes excluded) at which a spread is flagged. Walk-forward backtest, 2016-2025,
# 20,000 sims per game (model/BACKTEST.md):
#   >= 56%: 702-612 (53.4%) vs closing lines, 338-275 (55.1%) vs early-week lines
#   >= 58%: 545-442 (55.2%, +6.4% ROI) vs closing lines, 256-202 (55.9%) vs early-week lines
# Breakeven at -110 is 52.4%.
PROB_FLAG = 0.56
PROB_STRONG = 0.58


def _payout(odds):
    odds = -110.0 if odds is None or pd.isna(odds) else float(odds)
    return 100 / -odds if odds < 0 else odds / 100


def default_week(games: pd.DataFrame, season: int) -> int:
    s = games[games["season"] == season]
    played = s[s["result"].notna()]
    return int(played["week"].max()) + 1 if len(played) else 1


def load_lines(path: str | None, wk: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_id", "home_spread", "total", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
            "home_ml", "away_ml"]
    if path:
        ln = pd.read_csv(path)
        ln = wk[["game_id", "away_team", "home_team"]].merge(ln, on=["away_team", "home_team"], how="left")
        for c in cols:
            if c not in ln:
                ln[c] = np.nan
        ln["line_source"] = os.path.basename(path)
        return ln[cols + ["line_source"]]
    ln = pd.DataFrame({
        "game_id": wk["game_id"], "home_spread": -wk["spread_line"], "total": wk["total_line"],
        "home_spread_odds": wk["home_spread_odds"], "away_spread_odds": wk["away_spread_odds"],
        "over_odds": wk["over_odds"], "under_odds": wk["under_odds"],
        "home_ml": wk["home_moneyline"], "away_ml": wk["away_moneyline"], "line_source": "nflverse games.csv",
    })
    return ln


def load_adjustments(season: int, week: int) -> pd.DataFrame:
    if not os.path.exists(ADJ_PATH):
        return pd.DataFrame(columns=["team", "qb_id", "points", "note"])
    a = pd.read_csv(ADJ_PATH, dtype={"qb_id": str})
    return a[(a["season"] == season) & (a["week"] == week)]


def run(season: int | None, week: int | None, n_sims: int, lines_path: str | None, refresh: bool) -> pd.DataFrame:
    os.makedirs(OUT, exist_ok=True)
    if season is None:
        g = load_games(refresh=refresh)
        season = int(g.loc[g["result"].notna(), "season"].max())
    games, tg = load_all(2012, season, refresh=refresh)
    week = week or default_week(games, season)
    wk = games[(games["season"] == season) & (games["week"] == week)].copy()
    if wk.empty:
        raise SystemExit(f"no games found for {season} week {week}")
    print(f"{season} week {week}: {len(wk)} games, {n_sims:,} sims each")

    # Points model: trained on every completed game's walk-forward features (built only from earlier games).
    f_hist = build_features(last_season=season, refresh=False)
    f_hist = f_hist[(f_hist["season"] < season) | (f_hist["week"] < week)]
    model = fit_points_model(f_hist)

    r = build_ratings(tg, season, week, elo=elo_snapshots(games[games["season"] >= season - 12]), games=games)
    adj = load_adjustments(season, week)

    # Starting QBs: manual override > current depth chart QB1 > last game's starter.
    dc = depth_chart_qb1(season, refresh=refresh).set_index("team")
    starters, qb_notes = {}, {}
    names = tg.dropna(subset=["qb_id"]).drop_duplicates("qb_id", keep="last").set_index("qb_id")["qb_name"]
    for tm in pd.concat([wk["home_team"], wk["away_team"]]):
        last = r["starter"].get(tm)
        qid = dc["qb_id"].get(tm) or last
        ov = adj[(adj["team"] == tm) & adj["qb_id"].notna()]
        if len(ov):
            qid = ov["qb_id"].iloc[-1]
        starters[tm] = qid
        nm = names.get(qid, dc["qb_name"].get(tm, qid))
        qb_notes[tm] = nm + ("" if qid == last else f" (QB change; last start: {names.get(last, last)})")

    f = game_features(wk, r, starters)
    f["pred"] = model.predict(f)
    pts_adj = adj.groupby("team")["points"].sum() if "points" in adj else pd.Series(dtype=float)
    f["pred"] = f["pred"] + f["team"].map(pts_adj).fillna(0)

    lines = load_lines(lines_path, wk)
    cfg = SimConfig.calibrated(n_sims=n_sims)
    rows = []
    for g in wk.itertuples():
        h = f[(f["game_id"] == g.game_id) & (f["side"] == "home")].iloc[0]
        a = f[(f["game_id"] == g.game_id) & (f["side"] == "away")].iloc[0]
        ln = lines[lines["game_id"] == g.game_id].iloc[0]
        spread_nv = -ln["home_spread"] if pd.notna(ln["home_spread"]) else np.nan  # nflverse sign: + = home fav
        playoff = g.game_type != "REG"
        rng = np.random.default_rng(zlib.crc32(g.game_id.encode()))
        sim = simulate_game(h["pred"], a["pred"], h["exp_drives"], playoff, cfg, rng)
        s = summarize(sim, spread_nv, ln["total"])
        row = {
            "game_id": g.game_id, "gameday": g.gameday.date(), "away_team": g.away_team, "home_team": g.home_team,
            "away_qb": qb_notes[g.away_team], "home_qb": qb_notes[g.home_team],
            "proj_away": s["sim_away_pts"], "proj_home": s["sim_home_pts"],
            "model_home_spread": -s["sim_margin"], "model_total": s["sim_total"],
            "p_home_win": s["p_home_win"], "fair_home_ml": s["fair_home_ml"], "fair_away_ml": s["fair_away_ml"],
            **{c: ln[c] for c in ("home_spread", "total", "home_ml", "away_ml", "line_source")},
        }
        if pd.notna(spread_nv):
            row["spread_edge"] = s["sim_margin"] - spread_nv          # + = model likes HOME vs the line
            # Pick the side the simulation says covers more often (the rule the backtest validated). It can differ
            # from the sign of the point edge because favorites' margins are right-skewed.
            side = "home" if s["p_home_cover"] >= s["p_away_cover"] else "away"
            # Blended view: believe BLEND_SPREAD of the disagreement, re-simulate to price the bet.
            bm = spread_nv + BLEND_SPREAD * row["spread_edge"]
            bt = (ln["total"] + BLEND_TOTAL * (s["sim_total"] - ln["total"])) if pd.notna(ln["total"]) else s["sim_total"]
            bsim = simulate_game((bt + bm) / 2, (bt - bm) / 2, h["exp_drives"], playoff, cfg, rng)
            b = summarize(bsim, spread_nv, ln["total"])
            p_win = b[f"p_{side}_cover"]
            p_lose = b[f"p_{'away' if side == 'home' else 'home'}_cover"]
            row.update({
                "spread_pick": (g.home_team if side == "home" else g.away_team),
                "p_cover_raw": s[f"p_{side}_cover"] / (s["p_home_cover"] + s["p_away_cover"]),
                "p_cover_blend": p_win / (p_win + p_lose),
                "spread_ev": p_win * _payout(ln[f"{side}_spread_odds"]) - p_lose,
                # Straight-up: the blended (35% model, 65% line) view picked 66.8% of winners in 2016-2025 vs
                # 66.4% for the closing-line favorite and 65.3% for the model alone.
                "p_home_win_blend": b["p_home_win"],
            })
            if pd.notna(ln["total"]):
                row["total_edge"] = s["sim_total"] - ln["total"]
                ou = "over" if row["total_edge"] > 0 else "under"
                pw, pl = b[f"p_{ou}"], b[f"p_{'under' if ou == 'over' else 'over'}"]
                row.update({"total_pick": ou, "p_total_raw": s[f"p_{ou}"] / (s["p_over"] + s["p_under"]),
                            "p_total_blend": pw / (pw + pl), "total_ev": pw * _payout(ln[f"{ou}_odds"]) - pl})
        rows.append(row)
    out = pd.DataFrame(rows)
    p = out.get("p_cover_raw", pd.Series(0.0, index=out.index)).fillna(0)
    out["flag"] = np.select([p >= PROB_STRONG, p >= PROB_FLAG], ["STRONG", "PLAY"], "")

    tag = f"{season}_{week:02d}"
    out.to_csv(os.path.join(OUT, f"week_{tag}.csv"), index=False)
    pt = power_table(r)
    pt["net_pts"] = pt["off_pts"] + pt["def_pts"]
    pt = pt.sort_values("net_pts", ascending=False)
    pt.to_csv(os.path.join(OUT, f"power_ratings_{tag}.csv"))
    report = render(out, pt, season, week, n_sims, adj)
    with open(os.path.join(OUT, f"week_{tag}.md"), "w") as fh:
        fh.write(report)
    print(report)
    return out


def _fmt_spread(x):
    if pd.isna(x):
        return "-"
    return "PK" if abs(x) < 0.05 else f"{x:+.1f}"


def _fmt_ml(x):
    return "-" if pd.isna(x) else f"{x:+.0f}"


def _flagged_section(out: pd.DataFrame) -> list[str]:
    """The plays to act on, at the top of the report."""
    L = ["## Flagged plays\n"]
    fl = out[out["flag"] != ""].copy() if "flag" in out else out.iloc[0:0]
    if fl.empty:
        return L + ["No spread clears the 56% threshold this week.", ""]
    fl["rank"] = (fl["flag"] == "STRONG").astype(int)
    fl = fl.sort_values(["rank", "p_cover_raw"], ascending=False)
    L += ["| Flag | Play | Matchup | Cover % raw | Cover % blended | EV/unit | Note |", "|---|---|---|---|---|---|---|"]
    for r in fl.itertuples():
        pick_line = r.home_spread if r.spread_pick == r.home_team else -r.home_spread
        note = ("best bet: also +EV on the blended price" if r.spread_ev > 0
                else "flag only: blended price is thin, consider a smaller stake")
        L.append(f"| **{r.flag}** | **{r.spread_pick} {_fmt_spread(pick_line)}** | {r.away_team} @ {r.home_team} | "
                 f"{r.p_cover_raw:.1%} | {r.p_cover_blend:.1%} | {r.spread_ev:+.3f} | {note} |")
    L += ["", f"STRONG = raw cover >= {PROB_STRONG:.0%} (backtest: 55.2% ATS vs closing lines, 55.9% vs early-week "
          f"lines). PLAY = raw cover >= {PROB_FLAG:.0%} (53.4% / 55.1%). Breakeven at -110 is 52.4%.", ""]
    return L


def render(out: pd.DataFrame, pt: pd.DataFrame, season: int, week: int, n_sims: int, adj: pd.DataFrame) -> str:
    L = [f"# {season} Week {week} projections\n",
         f"{n_sims:,} Monte Carlo simulations per game. Spreads use sportsbook convention (home -3 = home favored by 3). "
         f"Lines: {out['line_source'].iloc[0]}.\n"]
    L += _flagged_section(out)
    L += ["## Projected scores\n",
         "| Matchup | Proj score | Model spread | Line | Spread edge | Model total | Line total | Home win % | Fair ML (A/H) | Market ML (A/H) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in out.itertuples():
        L.append(f"| {r.away_team} @ {r.home_team} | {r.away_team} {r.proj_away:.1f} - {r.home_team} {r.proj_home:.1f} | "
                 f"{r.home_team} {_fmt_spread(r.model_home_spread)} | {r.home_team} {_fmt_spread(r.home_spread)} | "
                 f"{getattr(r, 'spread_edge', np.nan):+.1f} | {r.model_total:.1f} | {r.total if pd.notna(r.total) else '-'} | "
                 f"{r.p_home_win:.0%} | {_fmt_ml(r.fair_away_ml)}/{_fmt_ml(r.fair_home_ml)} | "
                 f"{_fmt_ml(r.away_ml)}/{_fmt_ml(r.home_ml)} |")
    L += ["", "Spread edge = model margin minus the line, in points (+ favors the home side).", ""]

    pw = out["p_home_win_blend"] if "p_home_win_blend" in out else out["p_home_win"]
    su = out.assign(pw=pw.fillna(out["p_home_win"]))
    su["winner"] = np.where(su["pw"] >= 0.5, su["home_team"], su["away_team"])
    su["conf"] = np.maximum(su["pw"], 1 - su["pw"])
    L += ["## Winner picks (straight up, most confident first)\n", "| Matchup | Winner | Win % |", "|---|---|---|"]
    for r in su.sort_values("conf", ascending=False).itertuples():
        L.append(f"| {r.away_team} @ {r.home_team} | **{r.winner}** | {r.conf:.0%} |")
    L += ["", "Win % blends the model (35%) with the line (65%). In the 2016-2025 backtest this picked 66.8% of "
          "winners (closing-line favorite: 66.4%).", ""]

    if "spread_edge" in out:
        sp = out[out["spread_edge"].notna()].sort_values("p_cover_raw", ascending=False)
        L += ["## Spread edges (highest cover probability first)\n",
              "| Matchup | Pick | Line | Edge (pts) | Cover % raw | Cover % blended | EV/unit | Flag |", "|---|---|---|---|---|---|---|---|"]
        for r in sp.itertuples():
            pick_line = r.home_spread if r.spread_pick == r.home_team else -r.home_spread
            L.append(f"| {r.away_team} @ {r.home_team} | **{r.spread_pick} {_fmt_spread(pick_line)}** | "
                     f"{r.home_team} {_fmt_spread(r.home_spread)} | {abs(r.spread_edge):.1f} | {r.p_cover_raw:.1%} | "
                     f"{r.p_cover_blend:.1%} | {r.spread_ev:+.3f} | {r.flag} |")
        L += ["", f"Flag: PLAY = simulated cover % (raw) >= {PROB_FLAG:.0%}, STRONG = >= {PROB_STRONG:.0%}. In the "
              "2016-2025 backtest these won 53.4% / 55.2% ATS vs closing lines and 55.1% / 55.9% vs early-week lines "
              "(breakeven 52.4%). Raw cover % is the backtested selection number but overstates the true win rate. "
              f"Blended cover % believes {BLEND_SPREAD:.0%} of the disagreement and is the realistic price for "
              "sizing; EV/unit uses it.", ""]
    if "total_edge" in out:
        tt = out[out["total_edge"].notna()].reindex(out["total_edge"].abs().sort_values(ascending=False).index)
        L += ["## Totals (unproven: no edge vs closing totals in backtest; treat as leans)\n",
              "| Matchup | Lean | Line | Model | Edge | Prob raw | Prob blended | EV/unit |", "|---|---|---|---|---|---|---|---|"]
        for r in tt.itertuples():
            L.append(f"| {r.away_team} @ {r.home_team} | {r.total_pick} | {r.total} | {r.model_total:.1f} | "
                     f"{r.total_edge:+.1f} | {r.p_total_raw:.1%} | {r.p_total_blend:.1%} | {r.total_ev:+.3f} |")
        L.append("")
    L += ["## Starting QBs used\n"]
    for r in out.itertuples():
        L.append(f"- {r.away_team}: {r.away_qb} | {r.home_team}: {r.home_qb}")
    if len(adj):
        L += ["", "## Manual adjustments applied\n", adj.to_markdown(index=False)]
    L += ["", "## Power ratings (points vs average team, neutral field)\n",
          "| Rank | Team | Net | Off | Def | Off EPA/play | Def EPA/play | ST | Elo | Market |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, (tm, r) in enumerate(pt.iterrows(), 1):
        L.append(f"| {i} | {tm} | {r.net_pts:+.1f} | {r.off_pts:+.1f} | {r.def_pts:+.1f} | {r.off_epa:+.3f} | "
                 f"{r.def_epa:+.3f} | {r.st:+.1f} | {r.elo:.0f} | {r.market:+.1f} |")
    L += ["", "Net/Off/Def: stat-based ratings in points vs an average team. Market: rating implied by past closing "
          "lines (points vs average)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--lines", help="CSV of sportsbook lines to compare against")
    ap.add_argument("--no-refresh", action="store_true", help="don't download fresh nflverse data")
    a = ap.parse_args()
    run(a.season, a.week, a.sims, a.lines, refresh=not a.no_refresh)
