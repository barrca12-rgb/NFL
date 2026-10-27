"""Monday run: project every game of the coming week, simulate it, and compare to the available lines.

    python -m model.run_week                         # next week, nflverse lines, 20,000 sims per game
    python -m model.run_week --week 8 --lines my_lines.csv
    python -m model.run_week --no-refresh            # use cached data (offline)

Outputs (model/output/):
    week_<season>_<week>.csv        one row per game, every number below
    week_<season>_<week>.md         readable report: projections, edges, flagged plays
    week_<season>_<week>.html       bet card: verdicts and stakes first, sortable tables (also saved to reports/)
    power_ratings_<season>_<week>.csv

Lines file (optional, CSV): away_team, home_team, home_spread, total, and optionally home_spread_odds,
away_spread_odds, over_odds, under_odds, home_ml, away_ml. home_spread uses SPORTSBOOK convention
(-3 = home favored by 3). Without it, the current lines in nflverse games.csv are used.

Manual adjustments: model/adjustments.csv (season, week, team, qb_id, points, note). `qb_id` overrides the
starting QB (gsis id); `points` adds/subtracts expected points for that team (non-QB injuries, weather, etc.).
"""
from __future__ import annotations

import argparse
import json
import os
import zlib

import numpy as np
import pandas as pd

from .calibrate import build_features, injury_table
from .coaches import coach_h2h
from .teasers import TEASE, grade_leg, pair_legs, pairs_table
from .tracker import logged_bets, stake_multipliers
from .data import HERE, REPO, depth_chart_qb1, load_all, load_games
from .elo import elo_snapshots
from .features import FEATURES, game_features
from .injuries import FEATURES as INJ_FEATURES, add_injury_features, live_table
from .odds import DEFAULT_BOOKS, LINK_COLS, latest_snapshot, load_book_lines, pull_all
from .projection import fit_points_model
from .ratings import build_ratings, power_table
from . import ledger
from .report_html import build_html, build_record_html
from .simulate import SimConfig, american, simulate_game, summarize

OUT = os.path.join(HERE, "output")
REPORTS = os.path.join(REPO, "reports")  # committed copies of each week's report
ADJ_PATH = os.path.join(HERE, "adjustments.csv")

# How much of the model's disagreement with a line to believe when pricing a bet, by part of the season. The raw
# projection is the model's own opinion; these weights only shape the "realistic" cover % and EV. Measured as the
# slope of (result - line) on (model - line), walk-forward 2016-2025:
#   Weeks 1-4: 0.36-0.50 vs closing lines, 0.43 vs early-week lines   -> 0.45
#   Weeks 5+:  0.38 (2016-20) but -0.04 (2021-25) vs closing lines, 0.36 vs early-week lines -> 0.20
# Totals: slope 0.02 (no edge), so no blend.
BLEND = {"early": 0.45, "late": 0.20}
BLEND_TOTAL = 0.0
EARLY_WEEKS = 4
# Betting rule (out-of-sample: thresholds picked on 2016-20, tested on 2021-25):
#   Weeks 1-4: model cover >= 53% and +EV at the realistic price. STRONG (>= 56%) went 59.4% ATS vs closing lines,
#     61.9% vs early-week lines; model cover >= 60% held at 61.3% in the 2021-25 test.
#   Week 5 on: no model-cover threshold survived the out-of-sample test. What did: betting when the realistic cover %
#     beats the break-even of the BEST price across books (53.0-53.3% ATS in 2021-25 with half a point of shopping,
#     51% without). So mid-season bets are price plays: shop the four books, bet small.
FLAG_RULES = {"early": [("STRONG", 0.56), ("PLAY", 0.53)], "late": [("VALUE", 0.0)]}
STAKES = {"STRONG": 2.0, "PLAY": 1.0, "VALUE": 0.5}
FLAG_TEXT = {
    "early": "Weeks 1-4: STRONG = model cover >= 56% and +EV (backtest 59.4% ATS vs closing lines, 61.9% vs "
             "early-week lines, profitable in 9 of 10 seasons); PLAY = model cover >= 53% and +EV (57.5%).",
    "late": "Week 5 on: VALUE = the realistic cover % beats the break-even of the best price across books. Mid-season "
            "edges come from price, not the model alone: backtest 53% ATS with line shopping, 51% without. 0.5u each.",
}
# Injury-report stages that adjust projections (see model/injuries.py). Backtest 2016-2025: Thursday/final reports
# improved accuracy (opponent's missing defensive starters ~ +0.5-0.7 pts each); the Monday carry-forward did not.
APPLIED_STAGES = ("thursday", "final")

LADDER_STEPS = 6       # half-points checked against the bet for "good down to" (3 points)
WATCH_STEPS = 3        # half-points checked in the bet's favor for the watch list (1.5 points: a realistic move)
OFFER_SIMS = 8000      # minimum sims for the realistic-price simulation


def season_phase(week: int, game_type: str = "REG") -> str:
    return "early" if game_type == "REG" and week <= EARLY_WEEKS else "late"


def flag_for(p_raw: float, ev: float, phase: str) -> str:
    """Flag name if the offer is a bet under the phase's rule, else ''."""
    if not ev > 0:
        return ""
    for name, th in FLAG_RULES[phase]:
        if p_raw >= th:
            return name
    return ""


class Pricer:
    """Prices one side of one game at any book's line.

    raw:       the model's own simulation.
    realistic: a simulation centered BLEND[phase] of the way from the market CONSENSUS line to the model. It is the
               same for every book; a book hanging a better number than consensus is then genuinely worth more
               (the line-shopping edge the backtest measured)."""

    def __init__(self, sim, model_margin, consensus, total, exp_drives, playoff, cfg, rng, phase, side):
        self.sim, self.side, self.phase = sim, side, phase
        center = consensus + BLEND[phase] * (model_margin - consensus)
        self.real = simulate_game((total + center) / 2, (total - center) / 2, exp_drives, playoff,
                                  SimConfig(**{**cfg.__dict__, "n_sims": max(OFFER_SIMS, cfg.n_sims)}), rng)

    def _p(self, s):
        other = "away" if self.side == "home" else "home"
        return s[f"p_{self.side}_cover"], s[f"p_{other}_cover"]

    def price(self, L, odds) -> dict:
        """L in nflverse sign (+ = home favored by L)."""
        pc, pl = self._p(summarize(self.sim, L))
        bw, bl = self._p(summarize(self.real, L))
        raw, ev = pc / (pc + pl), bw * _payout(odds) - bl
        return {"p_raw": raw, "p_blend": bw / (bw + bl), "ev": ev, "flag": flag_for(raw, ev, self.phase)}

    def pick_line(self, L):  # sportsbook number for the picked team
        return -L if self.side == "home" else L

    def ladder(self, L, odds) -> dict:
        """Worst line the bet still qualifies at ("good down to"), or the nearest better line where it would."""
        sign = 1 if self.side == "home" else -1          # +sign*0.5 makes the bet worse
        out = {"bet_down_to": np.nan, "bet_at": np.nan}
        if self.price(L, odds)["flag"]:
            worst = L
            for k in range(1, LADDER_STEPS + 1):
                if not self.price(L + sign * 0.5 * k, odds)["flag"]:
                    break
                worst = L + sign * 0.5 * k
            out["bet_down_to"] = self.pick_line(worst)
        else:
            for k in range(1, WATCH_STEPS + 1):
                if self.price(L - sign * 0.5 * k, odds)["flag"]:
                    out["bet_at"] = self.pick_line(L - sign * 0.5 * k)
                    break
        return out


def add_verdicts(out: pd.DataFrame, phase: str) -> pd.DataFrame:
    """One clear call per game, a stake in units, and the reason in plain words.

    BET   - the best available offer is a bet under the phase's rule (see FLAG_RULES). Stake by flag (STAKES).
            Coach head-to-head (loose rule) agreeing is noted; disagreeing halves the stake (backtest: model picks
            54.0% when it agrees, 50.4% when it disagrees).
    H2H   - strict coach head-to-head fade (model/coaches.py; 57.2% since 1999): 0.5u, or 1u when the model's pick
            is the same side. Replaces a model bet on the other side (54.3% vs ~50% in that case).
    SMALL - Weeks 1-4 only: model cover clears 53% but the realistic price is not +EV: 0.5u.
    WATCH - not a bet at the best number available, but becomes one if a line moves to `bet_at`.
    PASS  - no edge worth betting."""
    verdicts, stakes, reasons, sides = [], [], [], []
    for r in out.itertuples():
        flag, ev = getattr(r, "flag", ""), getattr(r, "spread_ev", np.nan)
        flag = flag if isinstance(flag, str) else ""
        at = f" at {r.best_book}" if getattr(r, "best_book", "") else ""
        strict = getattr(r, "h2h_strict", None)
        strict = strict if isinstance(strict, str) else None
        loose = getattr(r, "h2h_loose", None)
        loose = loose if isinstance(loose, str) else None
        pick = getattr(r, "spread_pick", None)
        if strict and not (flag and pick == strict):
            # strict head-to-head fade: on its own, or overriding a model bet on the other side
            u = 1.0 if pick == strict else 0.5
            v, side = "H2H", strict
            better = r.away_coach if strict == r.home_team else r.home_coach
            why = (f"Coach head-to-head: {better}'s teams have covered by {abs(r.h2h_total):.0f} pts total in "
                   f"{int(r.h2h_meetings)} meetings; fading that has won 57% since 1999"
                   + ("; model agrees" if pick == strict else ""))
        elif flag:
            v, u, side = "BET", STAKES[flag], pick
            why = f"{flag}{at}: model covers {r.p_cover_raw:.0%}, realistic {r.p_cover_blend:.0%}, EV {ev:+.3f}"
            if strict == pick:
                u += 0.5
                why += f"; strict coach head-to-head agrees ({int(r.h2h_meetings)} meetings)"
            elif loose == pick:
                why += "; coach head-to-head agrees"
            elif loose and loose != pick:
                u /= 2
                why += "; coach head-to-head disagrees, stake halved"
        elif phase == "early" and getattr(r, "p_cover_raw", 0) >= FLAG_RULES["early"][-1][1]:
            v, u, side = "SMALL", 0.5, pick
            why = f"Model likes it ({r.p_cover_raw:.0%}) but the realistic price is thin ({r.p_cover_blend:.0%})"
        elif pd.notna(getattr(r, "bet_at", np.nan)):
            v, u, side = "WATCH", 0.0, pick
            why = f"Becomes a bet at {r.spread_pick} {_fmt_spread(r.bet_at)}"
        else:
            v, u, side = "PASS", 0.0, pick
            why = "No edge at any book's number"
        verdicts.append(v)
        stakes.append(u)
        reasons.append(why)
        sides.append(side)
    out = out.assign(verdict=verdicts, stake_units=stakes, reason=reasons, bet_side=sides)
    # Live-results shutoff (model/tracker.py): scale each angle's stake by how its real bets are doing.
    mult = stake_multipliers()
    if mult:
        angle = np.where(out["verdict"].eq("BET"), out["flag"], out["verdict"])
        m = pd.Series(angle, index=out.index).map(mult).fillna(1.0)
        cut = m < 1
        out["stake_units"] = (out["stake_units"] * m).round(2)
        out.loc[cut, "reason"] = out.loc[cut, "reason"] + " (stake x" + m[cut].round(2).astype(str) + \
            ": live results below backtest)"
    # what to actually bet: the model's best offer, or the head-to-head side's best number
    h2h_bet = out["verdict"].eq("H2H")
    for a_, b_ in (("bet_line", "pick_line"), ("bet_odds", "pick_odds"), ("bet_book", "best_book")):
        alt = {"bet_line": "h2h_line", "bet_odds": "h2h_odds", "bet_book": "h2h_book"}[a_]
        out[a_] = np.where(h2h_bet, out.get(alt), out.get(b_))
    return out


def resist_moves(out: pd.DataFrame, logged: pd.DataFrame) -> pd.DataFrame:
    """Market resistance: never add to a side already bet this week once the line has moved against it.

    `logged` is the week's straight bets from bets/bets.csv (model/tracker.py). For a side already bet, if the best
    number now on offer for that side across books is worse than the logged line, the stake is capped at what is
    already logged on that side (so a re-run never adds to it)."""
    if not len(logged) or "bet_side" not in out:
        return out
    out = out.copy()
    for i, r in out.iterrows():
        side = r["bet_side"]
        mine = logged[(logged["game_id"] == r["game_id"]) & (logged["team"] == side)]
        if not len(mine) or not r["stake_units"] > 0:
            continue
        home = side == r["home_team"]
        menu = r.get("book_lines")
        nums = [(b["home_spread"] if home else -b["home_spread"]) for b in (menu if isinstance(menu, list) else [])
                if b.get("home_spread") is not None]
        if not nums or max(nums) >= mine["line"].max():
            continue
        cap = float(mine["stake"].sum())
        if r["stake_units"] > cap:
            out.at[i, "stake_units"] = cap
        out.at[i, "reason"] = (f"{r['reason']}; line moved against: no add (logged {side} "
                               f"{_fmt_spread(mine['line'].max())}, now {_fmt_spread(max(nums))})")
    return out


def _payout(odds):
    odds = -110.0 if odds is None or pd.isna(odds) else float(odds)
    return 100 / -odds if odds < 0 else odds / 100


def default_week(games: pd.DataFrame, season: int) -> int:
    """The current week: the earliest week that still has a game to play (so a played Thursday game doesn't skip
    the rest of its week)."""
    s = games[games["season"] == season]
    left = s[s["result"].isna()]
    if len(left):
        return int(left["week"].min())
    return int(s["week"].max()) + 1 if len(s) else 1


def game_state(g) -> str:
    """'final' when the score is in, 'live' once kickoff (US Eastern) has passed, else 'upcoming'."""
    if pd.notna(getattr(g, "result", np.nan)):
        return "final"
    try:
        ko = pd.Timestamp(f"{pd.Timestamp(g.gameday).date()} {g.gametime}", tz="America/New_York")
        return "live" if pd.Timestamp.now(tz="America/New_York") >= ko else "upcoming"
    except (TypeError, ValueError):
        return "upcoming"


def load_lines(path: str | None, wk: pd.DataFrame, books=DEFAULT_BOOKS) -> pd.DataFrame:
    """Every available offer, one row per (game, book). nflverse's consensus line is always included as a fallback
    book ("nflverse"); with a lines file, the books in it (default: FanDuel, DraftKings, BetMGM, Caesars) are added.
    Accepts the Odds API / Google Sheet export or a plain CSV (away_team, home_team, home_spread, ... [, book])."""
    base = pd.DataFrame({
        "game_id": wk["game_id"], "book": "nflverse", "home_spread": -wk["spread_line"], "total": wk["total_line"],
        "home_spread_odds": wk["home_spread_odds"], "away_spread_odds": wk["away_spread_odds"],
        "over_odds": wk["over_odds"], "under_odds": wk["under_odds"],
        "home_ml": wk["home_moneyline"], "away_ml": wk["away_moneyline"],
    })
    if not path:
        return base
    ln = load_book_lines(path, books)
    ln = wk[["game_id", "away_team", "home_team"]].merge(ln, on=["away_team", "home_team"], how="inner")
    ln = ln.drop(columns=["away_team", "home_team", "commence_time"], errors="ignore")
    have = set(ln["game_id"])
    # games the file doesn't cover keep the nflverse line; totals missing from the file come from nflverse
    out = pd.concat([ln, base[~base["game_id"].isin(have)]], ignore_index=True)
    tot = base.set_index("game_id")[["total", "over_odds", "under_odds"]]
    for c in ("total", "over_odds", "under_odds"):
        out[c] = out[c].fillna(out["game_id"].map(tot[c]))
    return out


def book_lines(gl: pd.DataFrame) -> list[dict]:
    """Every book's full menu for one game (spread, total, moneyline with prices) for the slate tickets.
    The nflverse consensus is shown only when no sportsbook has posted the game."""
    real = gl[gl["book"] != "nflverse"]
    gl = real if len(real) else gl
    cols = ["home_spread", "home_spread_odds", "away_spread_odds", "total", "over_odds", "under_odds", "home_ml",
            "away_ml"]
    out = []
    for r in gl.sort_values("book").to_dict("records"):
        d = {"book": "Consensus" if r["book"] == "nflverse" else r["book"]}
        d.update({c: None if pd.isna(r.get(c)) else float(r[c]) for c in cols})
        d.update({c: r[c] for c in LINK_COLS if isinstance(r.get(c), str) and r[c].startswith("https://")})
        out.append(d)
    return out


def write_record(season: int) -> None:
    """Grade finished plays and write the site's Record page (reports/record.html)."""
    d = ledger.settle(ledger.load(), load_games(refresh=False))
    ledger.save(d)
    with open(os.path.join(REPORTS, "record.html"), "w") as fh:
        fh.write(build_record_html(ledger.summary(d[d["season"] == season]), season))


def write_week_index(season: int, current: int) -> None:
    """reports/weeks.json: the season's report pages and which one is current, for the site's week switcher."""
    weeks = sorted(int(f[len(f"{season}_week"):-5]) for f in os.listdir(REPORTS)
                   if f.startswith(f"{season}_week") and f.endswith(".html"))
    with open(os.path.join(REPORTS, "weeks.json"), "w") as fh:
        json.dump({"season": season, "current": current, "weeks": weeks}, fh)


def load_adjustments(season: int, week: int) -> pd.DataFrame:
    if not os.path.exists(ADJ_PATH):
        return pd.DataFrame(columns=["team", "qb_id", "points", "note"])
    a = pd.read_csv(ADJ_PATH, dtype={"qb_id": str})
    return a[(a["season"] == season) & (a["week"] == week)]


def run(season: int | None, week: int | None, n_sims: int, lines_path: str | None, refresh: bool,
        use_injuries: bool = True, books=DEFAULT_BOOKS) -> pd.DataFrame:
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
    f_hist = build_features(last_season=season, refresh=False, refresh_injuries=refresh)
    f_hist = f_hist[(f_hist["season"] < season) | (f_hist["week"] < week)]
    model = fit_points_model(f_hist)

    # Injury adjustment: once this week's nflverse injury report exists (Wednesday/Thursday practice reports, then
    # Friday game statuses), retrain with the injury features for that report stage and use them.
    inj_mode, inj_table, inj_players = None, None, None
    if use_injuries:
        inj_mode, inj_table, inj_players = live_table(wk, season, week, injury_table(season)[1], refresh=refresh)
        # Monday: last week's report is shown for long-term injuries but NOT used to adjust projections. Backtest:
        # those absences are already in the ratings (the team played without them), and carrying them forward
        # made projections slightly worse. Thursday/final reports add new information and are applied.
        if inj_mode in APPLIED_STAGES:
            model = fit_points_model(f_hist, features=FEATURES + INJ_FEATURES[inj_mode])
        print(f"injury report: {inj_mode or 'none'}"
              + (" (applied)" if inj_mode in APPLIED_STAGES else " (shown only)" if inj_mode else ""))

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
    f["inj_pts"] = 0.0
    if inj_mode in APPLIED_STAGES:
        f = add_injury_features(f, inj_table)
        feats = INJ_FEATURES[inj_mode]
        f["inj_pts"] = f[feats].to_numpy() @ model.coef[feats].to_numpy()
    f["pred"] = model.predict(f)
    pts_adj = adj.groupby("team")["points"].sum() if "points" in adj else pd.Series(dtype=float)
    f["pred"] = f["pred"] + f["team"].map(pts_adj).fillna(0)

    lines = load_lines(lines_path, wk, books)
    snap = os.path.basename(lines_path or "")
    snap_note = (f" (live odds pulled {snap[10:20]} {snap[21:23]}:{snap[23:25]} UTC)"
                 if snap.startswith("odds_live_") else "")
    teasers = []
    h2h = coach_h2h(games, wk).set_index("game_id")
    phase = season_phase(week, wk["game_type"].iloc[0])
    cfg = SimConfig.calibrated(n_sims=n_sims)
    rows = []
    for g in wk.itertuples():
        h = f[(f["game_id"] == g.game_id) & (f["side"] == "home")].iloc[0]
        a = f[(f["game_id"] == g.game_id) & (f["side"] == "away")].iloc[0]
        offers = lines[(lines["game_id"] == g.game_id) & lines["home_spread"].notna()]
        cons_spread = offers["home_spread"].median() if len(offers) else np.nan   # consensus, sportsbook sign
        cons_total = lines.loc[lines["game_id"] == g.game_id, "total"].median()
        spread_nv = -cons_spread                                                   # nflverse sign: + = home fav
        playoff = g.game_type != "REG"
        rng = np.random.default_rng(zlib.crc32(g.game_id.encode()))
        sim = simulate_game(h["pred"], a["pred"], h["exp_drives"], playoff, cfg, rng)
        s = summarize(sim, spread_nv, cons_total)
        row = {
            "game_id": g.game_id, "gameday": g.gameday.date(), "away_team": g.away_team, "home_team": g.home_team,
            "away_qb": qb_notes[g.away_team], "home_qb": qb_notes[g.home_team],
            "away_inj_pts": a["inj_pts"], "home_inj_pts": h["inj_pts"],
            "proj_away": s["sim_away_pts"], "proj_home": s["sim_home_pts"],
            "model_home_spread": -s["sim_margin"], "model_total": s["sim_total"],
            "p_home_win": s["p_home_win"], "fair_home_ml": s["fair_home_ml"], "fair_away_ml": s["fair_away_ml"],
            "home_spread": cons_spread, "total": cons_total, "n_books": int(offers["book"].nunique()),
            "home_ml": lines.loc[lines["game_id"] == g.game_id, "home_ml"].median(),
            "away_ml": lines.loc[lines["game_id"] == g.game_id, "away_ml"].median(),
            "line_source": (", ".join(sorted(offers["book"].unique())) or "none") + snap_note,
            "kickoff": f"{g.weekday} {g.gametime}" if isinstance(g.gametime, str) else g.weekday,
            "state": game_state(g),
            "away_score": None if pd.isna(g.away_score) else int(g.away_score),
            "home_score": None if pd.isna(g.home_score) else int(g.home_score),
            "book_lines": book_lines(lines[lines["game_id"] == g.game_id]),
        }
        if len(offers):
            row["spread_edge"] = s["sim_margin"] - spread_nv          # + = model likes HOME vs the consensus
            # Side: the one the simulation says covers more often at the consensus number (the backtested rule).
            side = "home" if s["p_home_cover"] >= s["p_away_cover"] else "away"
            bt = (cons_total + BLEND_TOTAL * (s["sim_total"] - cons_total)) if pd.notna(cons_total) else s["sim_total"]
            pr = Pricer(sim, s["sim_margin"], spread_nv, bt, h["exp_drives"], playoff, cfg, rng, phase, side)
            # Price every book's number for that side; the best EV is the offer to bet.
            priced = []
            for o in offers.itertuples():
                odds = getattr(o, f"{side}_spread_odds")
                q = pr.price(-o.home_spread, odds)
                priced.append({"book": o.book, "line": pr.pick_line(-o.home_spread),
                               "odds": None if pd.isna(odds) else float(odds), **q})
            priced.sort(key=lambda x: x["ev"], reverse=True)
            best = priced[0]
            best_L = -best["line"] if side == "home" else best["line"]
            row.update({
                "spread_pick": g.home_team if side == "home" else g.away_team,
                "best_book": best["book"], "pick_line": best["line"], "pick_odds": best["odds"],
                "p_cover_raw": best["p_raw"], "p_cover_blend": best["p_blend"], "spread_ev": best["ev"],
                "flag": best["flag"], "offers": priced,
                **pr.ladder(best_L, best["odds"]),
            })
            # Coach head-to-head (model/coaches.py): price the fade side too, so it can be bet on its own.
            hh = h2h.loc[g.game_id]
            row.update({"home_coach": hh["home_coach"], "away_coach": hh["away_coach"],
                        "h2h_meetings": int(hh["h2h_meetings"]), "h2h_total": float(hh["h2h_total"]),
                        "h2h_loose": hh["h2h_loose"], "h2h_strict": hh["h2h_strict"]})
            if hh["h2h_strict"]:
                fside = "home" if hh["h2h_strict"] == g.home_team else "away"
                fp = Pricer(sim, s["sim_margin"], spread_nv, bt, h["exp_drives"], playoff, cfg, rng, phase, fside)
                fo = sorted(({"book": o.book, "line": fp.pick_line(-o.home_spread),
                              "odds": getattr(o, f"{fside}_spread_odds"),
                              **fp.price(-o.home_spread, getattr(o, f"{fside}_spread_odds"))}
                             for o in offers.itertuples()), key=lambda x: (x["line"], -abs(x["odds"] or 110)),
                            reverse=True)[0]   # most points, then best price
                row.update({"h2h_line": fo["line"], "h2h_odds": fo["odds"], "h2h_book": fo["book"],
                            "h2h_p_blend": fo["p_blend"]})
            # Teaser legs (model/teasers.py): best number for each side across books, teased 6 points.
            for tside in ("home", "away"):
                team = g.home_team if tside == "home" else g.away_team
                sl = offers["home_spread"] if tside == "home" else -offers["home_spread"]
                bi = sl.idxmax()                                   # most points for this side
                line = float(sl.loc[bi])
                # model margin minus the line from this side's view (home covers if m + line > 0, away if -m + line > 0)
                edge_side = (s["sim_margin"] + line) if tside == "home" else (-s["sim_margin"] + line)
                grade = grade_leg(line, cons_total, edge_side)
                if grade:
                    tp = Pricer(sim, s["sim_margin"], spread_nv, bt, h["exp_drives"], playoff, cfg, rng, phase, tside)
                    teased_nv = -(line + TEASE) if tside == "home" else (line + TEASE)
                    q = tp.price(teased_nv, -120)
                    teasers.append({"game_id": g.game_id, "matchup": f"{g.away_team} @ {g.home_team}", "team": team,
                                    "line": line, "teased_line": line + TEASE, "book": offers.loc[bi, "book"],
                                    "total": cons_total, "model_edge": edge_side, "grade": grade,
                                    "p_leg": q["p_blend"]})
            # Straight-up: model blended with the consensus line (66.8% of winners in the 2016-2025 backtest).
            bm = spread_nv + 0.35 * row["spread_edge"]
            win = summarize(simulate_game((bt + bm) / 2, (bt - bm) / 2, h["exp_drives"], playoff,
                                          SimConfig(**{**cfg.__dict__, "n_sims": OFFER_SIMS}), rng))
            row["p_home_win_blend"] = win["p_home_win"]
            if pd.notna(cons_total):
                row["total_edge"] = s["sim_total"] - cons_total
                ou = "over" if row["total_edge"] > 0 else "under"
                row.update({"total_pick": ou, "p_total_raw": s[f"p_{ou}"] / (s["p_over"] + s["p_under"])})
        rows.append(row)
    out = pd.DataFrame(rows)
    out["flag"] = out.get("flag", pd.Series("", index=out.index)).fillna("")
    out = add_verdicts(out, phase)
    out = resist_moves(out, logged_bets(season, week))
    # Games already finished or under way can't be bet pregame: show them, but never as plays.
    done = out["state"] != "upcoming"
    out.loc[done, "verdict"] = out.loc[done, "state"].str.upper()
    out.loc[done, "stake_units"] = 0.0
    teasers = [t for t in teasers if t["game_id"] in set(out.loc[~done, "game_id"])]
    tz = pd.DataFrame(teasers)
    tpairs = pairs_table(pair_legs(tz)) if len(tz) else pd.DataFrame()
    if len(tpairs):
        tpairs["stake_units"] = (tpairs["stake_units"] * stake_multipliers().get("TEASER", 1.0)).round(2)
    # the model's own record (model/ledger.py): keep this card's plays for games that haven't kicked off
    ledger.save(ledger.record_card(season, week, out.to_dict("records"),
                                   tpairs.to_dict("records") if len(tpairs) else [], ledger.kickoffs(games, season)))

    tag = f"{season}_{week:02d}"
    out.to_csv(os.path.join(OUT, f"week_{tag}.csv"), index=False)
    pt = power_table(r)
    pt["net_pts"] = pt["off_pts"] + pt["def_pts"]
    pt = pt.sort_values("net_pts", ascending=False)
    pt.to_csv(os.path.join(OUT, f"power_ratings_{tag}.csv"))
    if len(tz):
        tz.to_csv(os.path.join(OUT, f"teasers_{tag}.csv"), index=False)
    if len(tpairs):
        tpairs.to_json(os.path.join(OUT, f"teaser_pairs_{tag}.json"), orient="records", indent=1)
    report = render(out, pt, season, week, n_sims, adj, inj_mode, inj_players, tz, tpairs)
    qb_change = {t: n for t, n in qb_notes.items() if "QB change" in n}
    try:   # record for the card's KPI strip and Record panel; read only, never stops the card
        led = ledger.load()
        rec = ledger.summary(led[led["season"] == season]) if len(led) else None
    except Exception as e:
        print(f"Record for the card failed ({e}); continuing.")
        rec = None
    page = build_html(out, pt, season, week, n_sims, phase, FLAG_TEXT[phase], inj_mode, inj_players, qb_change,
                      tz, tpairs, rec)
    os.makedirs(REPORTS, exist_ok=True)
    for folder, name in ((OUT, f"week_{tag}"), (REPORTS, f"{season}_week{week:02d}")):
        with open(os.path.join(folder, f"{name}.md"), "w") as fh:
            fh.write(report)
        with open(os.path.join(folder, f"{name}.html"), "w") as fh:
            fh.write(page)
    print(report)
    return out


def _all_books(src: pd.Series) -> str:
    """Every book used anywhere on the card (games can differ when a book hasn't posted all of them yet)."""
    note = next((x[x.index(" (live"):] for x in src.dropna() if " (live" in x), "")
    books = sorted({b for x in src.dropna() for b in x.split(" (live")[0].split(", ")})
    return ", ".join(books) + note


def _fmt_spread(x):
    if pd.isna(x):
        return "-"
    return "PK" if abs(x) < 0.05 else f"{x:+.1f}"


def _fmt_ml(x):
    return "-" if pd.isna(x) else f"{x:+.0f}"


def _status_label(st: str) -> str:
    if st == "none_missed":
        return "missed last game, not on report (likely IR)"
    if st.startswith("lag_"):
        base, played = st[4:].rsplit("_", 1)
        return f"last week: {base}, {'played' if played == 'played' else 'did not play'}"
    return st.replace("prac_", "practice: ")


def _injury_section(out: pd.DataFrame, mode: str | None, players: pd.DataFrame | None) -> list[str]:
    L = ["## Injury adjustment\n"]
    if mode is None:
        return L + ["No injury report available yet.", ""]
    if mode == "monday":
        L += ["**Not applied to projections (Monday run).** This week's report isn't out, so the list below "
              "carries last week's report forward to show long-term absences. Those players are already reflected "
              "in the ratings (the team has been playing without them); in the backtest, adjusting for them again "
              "made projections worse. Re-run Thursday (practice report) or Friday/Saturday (game statuses) to "
              "apply new injuries.", ""]
    else:
        label = {"thursday": "Thursday practice report (no game statuses yet)",
                 "final": "final report with game statuses"}[mode]
        L += [f"Applied from the {label}. Each team's projected points include this many points for its own "
              "offensive absences and the opponent's defensive absences, relative to who the team has already been "
              "playing without.\n", "| Matchup | Away adj | Home adj |", "|---|---|---|"]
        for r in out.itertuples():
            L.append(f"| {r.away_team} @ {r.home_team} | {r.away_team} {r.away_inj_pts:+.1f} | "
                     f"{r.home_team} {r.home_inj_pts:+.1f} |")
        L.append("")
    if players is not None and len(players):
        L += ["Regulars likely to miss (non-QB, usual snap share >= 40%, miss probability >= 50%):\n",
              "| Team | Player | Pos | Status | Miss % | Usual snaps |", "|---|---|---|---|---|---|"]
        for p in players.itertuples():
            L.append(f"| {p.team} | {p.player} | {p.position} | {_status_label(p.status)} | "
                     f"{p.p_miss:.0%} | {p.usual:.0%} |")
    return L + [""]


def _teaser_section(tz, tpairs) -> list[str]:
    L = ["## Teasers (6 points through 3 and 7)\n"]
    if tz is None or tz.empty:
        return L + ["No qualifying teaser legs this week (underdogs +1.5 to +2.5, or favorites -7.5 to -8.5 the "
                    "model likes).", ""]
    if tpairs is not None and len(tpairs):
        L += ["| Teaser (2-team, 6 pts) | Grades | Both legs win (backtest) | Model estimate | EV at -120 | Stake |",
              "|---|---|---|---|---|---|"]
        for r in tpairs.itertuples():
            L.append(f"| **{r.teaser}** | {r.grades} | {r.p_both:.1%} | {r.p_both_model:.1%} | {r.ev:+.3f} | "
                     f"{r.stake_units:g}u |")
        L.append("")
    L += ["| Leg | Grade | Line → teased | Book | Total | Leg win % (realistic) |", "|---|---|---|---|---|---|"]
    for r in tz.sort_values(["grade", "p_leg"], ascending=[True, False]).itertuples():
        L.append(f"| {r.team} ({r.matchup}) | {r.grade} | {_fmt_spread(r.line)} → {_fmt_spread(r.teased_line)} | "
                 f"{r.book} | {r.total} | {r.p_leg:.1%} |")
    L += ["", "Grade A legs have won about 78% since 2006 (break-even 73.9% per leg on a 2-team teaser at -120, "
          "75.2% at -130). Shop teaser prices: -120 or better.", ""]
    return L


def _flagged_section(out: pd.DataFrame, phase: str) -> list[str]:
    """The bets, at the top of the report."""
    L = ["## Bets\n"]
    fl = out[out["verdict"].isin(["BET", "H2H", "SMALL"])].copy() if "verdict" in out else out.iloc[0:0]
    if fl.empty:
        return L + ["No bets at the available numbers. " + FLAG_TEXT[phase], ""]
    fl = fl.sort_values(["stake_units", "p_cover_raw"], ascending=False)
    L += ["| Call | Stake | Bet | Book | Good down to | Model cover | Realistic cover | EV/unit |",
          "|---|---|---|---|---|---|---|---|"]
    for r in fl.itertuples():
        h2h = r.verdict == "H2H"
        L.append(f"| **{r.verdict}** {'' if h2h else r.flag} | {r.stake_units:g}u | **{r.bet_side} "
                 f"{_fmt_spread(r.bet_line)} {_fmt_ml(r.bet_odds)}** ({r.away_team} @ {r.home_team}) | {r.bet_book} | "
                 f"{'-' if h2h else _fmt_spread(r.bet_down_to)} | {'-' if h2h else f'{r.p_cover_raw:.1%}'} | "
                 f"{(r.h2h_p_blend if h2h else r.p_cover_blend):.1%} | {'-' if h2h else f'{r.spread_ev:+.3f}'} |")
    L += ["", FLAG_TEXT[phase] + " Breakeven at -110 is 52.4%.", ""]
    return L


def render(out: pd.DataFrame, pt: pd.DataFrame, season: int, week: int, n_sims: int, adj: pd.DataFrame,
           inj_mode: str | None = None, inj_players: pd.DataFrame | None = None,
           tz: pd.DataFrame | None = None, tpairs: pd.DataFrame | None = None) -> str:
    L = [f"# {season} Week {week} projections\n",
         f"{n_sims:,} Monte Carlo simulations per game. Spreads use sportsbook convention (home -3 = home favored by 3). "
         f"Lines: {_all_books(out['line_source'])}.\n"]
    phase = season_phase(week)
    L += _flagged_section(out, phase)
    L += _teaser_section(tz, tpairs)
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
        L += ["## Spread edges (highest model cover first; priced at the best book)\n",
              "| Matchup | Pick (best book) | Consensus | Edge (pts) | Model cover | Realistic cover | EV/unit | Flag |",
              "|---|---|---|---|---|---|---|---|"]
        for r in sp.itertuples():
            L.append(f"| {r.away_team} @ {r.home_team} | **{r.spread_pick} {_fmt_spread(r.pick_line)}** "
                     f"{_fmt_ml(r.pick_odds)} ({r.best_book}) | {r.home_team} {_fmt_spread(r.home_spread)} | "
                     f"{abs(r.spread_edge):.1f} | {r.p_cover_raw:.1%} | {r.p_cover_blend:.1%} | {r.spread_ev:+.3f} | "
                     f"{r.flag} |")
        L += ["", FLAG_TEXT[phase] + f" Realistic cover % believes {BLEND[phase]:.0%} of the model's disagreement "
              "with each book's line; EV/unit uses it at that book's price.", ""]
        L += ["### Every book's number for the pick\n", "| Matchup | Pick | " + " | ".join(
            "Book" for _ in range(1)) + " offers (line, price, EV) |", "|---|---|---|"]
        for r in sp.itertuples():
            offers = "; ".join(f"{o['book']} {_fmt_spread(o['line'])} {_fmt_ml(o['odds'])} ({o['ev']:+.3f})"
                               for o in (r.offers if isinstance(r.offers, list) else []))
            L.append(f"| {r.away_team} @ {r.home_team} | {r.spread_pick} | {offers} |")
        L.append("")
    if "total_edge" in out:
        tt = out[out["total_edge"].notna()].reindex(out["total_edge"].abs().sort_values(ascending=False).index)
        L += ["## Totals (unproven: no edge vs closing totals in backtest; treat as leans)\n",
              "| Matchup | Lean | Line | Model | Edge | Model % |", "|---|---|---|---|---|---|"]
        for r in tt.itertuples():
            L.append(f"| {r.away_team} @ {r.home_team} | {r.total_pick} | {r.total} | {r.model_total:.1f} | "
                     f"{r.total_edge:+.1f} | {r.p_total_raw:.1%} |")
        L.append("")
    L += _injury_section(out, inj_mode, inj_players)
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
    ap.add_argument("--lines", help="CSV of sportsbook lines to compare against (default: pull live odds when "
                                     "SGO_API_KEY or ODDS_API_KEY is set)")
    ap.add_argument("--no-live", action="store_true", help="don't pull live odds; use --lines or nflverse lines")
    ap.add_argument("--no-refresh", action="store_true", help="don't download fresh nflverse data")
    ap.add_argument("--no-injuries", action="store_true", help="ignore this week's injury report")
    ap.add_argument("--books", default=",".join(DEFAULT_BOOKS),
                    help="books to shop from the lines file (comma list, or 'all'); default: " + ",".join(DEFAULT_BOOKS))
    a = ap.parse_args()
    books = None if a.books == "all" else a.books.split(",")
    lines_path = a.lines
    if not lines_path and not a.no_live:
        # every provider with a key, merged (The Odds API for DraftKings/FanDuel/BetMGM, SportsGameOdds for
        # Caesars and betslip links); if both fail, the newest saved snapshot
        keys = {"theoddsapi": os.environ.get("ODDS_API_KEY"), "sgo": os.environ.get("SGO_API_KEY")}
        sources = [p for p, k in keys.items() if k]
        log = [f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC  keys set: " + (", ".join(sources) or "none")]
        if sources:
            try:
                lines_path, plog = pull_all(keys, books)
                log += plog + [f"-> {os.path.basename(lines_path)}"]
            except Exception as e:
                log.append(f"all providers failed ({e})")
                lines_path = latest_snapshot()
                log.append(f"fell back to {lines_path or 'nflverse lines'}")
        if not sources:
            print("no odds key set (SGO_API_KEY or ODDS_API_KEY): using nflverse consensus lines")
        # which source the last run used (and why not the first choice), kept in the repo for troubleshooting
        with open(os.path.join(REPO, "lines", "last_pull.txt"), "w") as fh:
            fh.write("\n".join(log) + "\n")
    if a.week:
        out = run(a.season, a.week, a.sims, lines_path, refresh=not a.no_refresh, use_injuries=not a.no_injuries,
                  books=books)
        write_record(a.season or int(load_games(refresh=False).loc[lambda g: g["result"].notna(), "season"].max()))
    else:
        # no week given: the current week and the next one, so the site can switch between them
        g = load_games(refresh=not a.no_refresh)
        season = a.season or int(g.loc[g["result"].notna(), "season"].max())
        cur = default_week(g, season)
        weeks = [w for w in (cur, cur + 1) if ((g["season"] == season) & (g["week"] == w)).any()]
        for i, w in enumerate(weeks):
            run(season, w, a.sims, lines_path, refresh=not a.no_refresh and i == 0,
                use_injuries=not a.no_injuries, books=books)
        write_week_index(season, cur)
        write_record(season)
