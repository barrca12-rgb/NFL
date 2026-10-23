"""Bet tracking, closing line value (CLV), and an automatic shutoff per angle.

A backtest assumes the market stands still; once real bets go in, lines react and edges erode. This tracks what was
actually bet, how it did against the closing line and the result, and scales each angle's stake down as live
evidence says its edge is fading.

    python -m model.tracker log --season 2026 --week 8          # record the card's bets (BET / H2H / SMALL + teasers)
    python -m model.tracker log --season 2026 --week 8 --only "TB,LAC"   # just the ones you actually placed
    python -m model.tracker settle                              # after games: closing lines, results, profit, CLV
    python -m model.tracker report                              # record, ROI and CLV by angle, stake multipliers

Bets live in bets/bets.csv (edit it by hand to fix a line, price or stake you actually got).

Closing line value: points you got versus the closing number from the same side's view (+1 = you had a point
better than close). Beating the close is the fastest honest signal that an edge is real: it shows up after dozens of
bets, while win rates need hundreds.

Shutoff: each angle starts from its backtested win rate as a Beta prior worth PRIOR_BETS bets. Live results update
it; the stake multiplier is how far the updated win rate still sits above break-even, relative to the backtest:
    multiplier = clip((posterior - breakeven) / (backtest - breakeven), 0, 1)
An angle running at its backtest rate keeps full stakes; one sliding toward break-even is scaled down and reaches
zero at break-even. Angles averaging negative CLV after MIN_CLV_BETS bets are also capped at half stake.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .data import HERE, REPO, load_games

BETS_DIR = os.path.join(REPO, "bets")
BETS = os.path.join(BETS_DIR, "bets.csv")
STATUS = os.path.join(BETS_DIR, "angle_status.json")
OUT = os.path.join(HERE, "output")

# Backtested win rates and break-evens per angle (spread bets at -110: 52.4%; 2-team 6-pt teaser at -120: 54.5%)
ANGLES = {
    "STRONG": {"backtest": 0.594, "breakeven": 0.524},
    "PLAY": {"backtest": 0.575, "breakeven": 0.524},
    "SMALL": {"backtest": 0.545, "breakeven": 0.524},
    "VALUE": {"backtest": 0.531, "breakeven": 0.524},
    "H2H": {"backtest": 0.572, "breakeven": 0.524},
    "TEASER": {"backtest": 0.590, "breakeven": 0.545},
}
PRIOR_BETS = 100
MIN_CLV_BETS = 25

COLS = ["bet_id", "season", "week", "angle", "selection", "game_id", "team", "line", "odds", "book", "stake",
        "legs", "close_line", "clv_pts", "result", "profit", "status"]


TEXT_COLS = ["angle", "selection", "game_id", "team", "book", "legs", "result", "status"]


def _load() -> pd.DataFrame:
    b = pd.read_csv(BETS, dtype={c: str for c in TEXT_COLS}) if os.path.exists(BETS) else pd.DataFrame(columns=COLS)
    for c in TEXT_COLS:
        b[c] = b[c].astype(object)
    return b


def _save(b: pd.DataFrame) -> None:
    os.makedirs(BETS_DIR, exist_ok=True)
    b[COLS].to_csv(BETS, index=False)


def _payout(odds) -> float:
    odds = -110.0 if odds is None or pd.isna(odds) else float(odds)
    return 100 / -odds if odds < 0 else odds / 100


def log(season: int, week: int, only: list[str] | None = None) -> pd.DataFrame:
    """Append the week's recommended bets from the card. `only` keeps bets whose team is in the list."""
    tag = f"{season}_{week:02d}"
    card = pd.read_csv(os.path.join(OUT, f"week_{tag}.csv"))
    b = _load()
    new = []
    for r in card[card["verdict"].isin(["BET", "H2H", "SMALL"])].itertuples():
        angle = "H2H" if r.verdict == "H2H" else ("SMALL" if r.verdict == "SMALL" else r.flag)
        new.append({"season": season, "week": week, "angle": angle,
                    "selection": f"{r.bet_side} {r.bet_line:+g}", "game_id": r.game_id, "team": r.bet_side,
                    "line": r.bet_line, "odds": r.bet_odds, "book": r.bet_book, "stake": r.stake_units, "legs": ""})
    tp = os.path.join(OUT, f"teaser_pairs_{tag}.json")
    if os.path.exists(tp):
        for t in json.load(open(tp)):
            new.append({"season": season, "week": week, "angle": "TEASER", "selection": t["teaser"],
                        "game_id": "", "team": "/".join(l["team"] for l in t["legs"]), "line": np.nan, "odds": -120,
                        "book": "/".join(sorted({l["book"] for l in t["legs"]})), "stake": t["stake_units"],
                        "legs": json.dumps(t["legs"])})
    new = pd.DataFrame(new)
    if only:
        keep = {x.strip().upper() for x in only}
        new = new[new["team"].apply(lambda t: any(p in keep for p in str(t).split("/")))]
    # skip bets already logged
    if len(b):
        key = set(zip(b["season"], b["week"], b["selection"]))
        new = new[[k not in key for k in zip(new["season"], new["week"], new["selection"])]]
    start = int(b["bet_id"].max()) + 1 if len(b) else 1
    new = new.assign(bet_id=range(start, start + len(new)), status="open")
    b = pd.concat([b, new], ignore_index=True)
    _save(b.reindex(columns=COLS))
    print(f"logged {len(new)} bets")
    return new


def _side_close_and_margin(g: pd.Series, team: str):
    """Closing line from this team's view (sportsbook sign) and its final margin."""
    home = team == g["home_team"]
    close = -g["spread_line"] if home else g["spread_line"]
    margin = g["result"] if home else -g["result"]
    return close, margin


def settle(refresh: bool = True) -> pd.DataFrame:
    games = load_games(refresh=refresh).set_index("game_id")
    b = _load()
    for i, r in b[b["status"] == "open"].iterrows():
        if r["angle"] == "TEASER":
            legs = json.loads(r["legs"])
            outcomes, clv = [], []
            for leg in legs:
                g = games.loc[leg["game_id"]]
                if pd.isna(g["result"]):
                    break
                close, margin = _side_close_and_margin(g, leg["team"])
                cover = margin + leg["teased_line"]
                outcomes.append("W" if cover > 0 else "P" if cover == 0 else "L")
                clv.append(leg["line"] - close)
            else:
                # 2-team teaser: any loss loses; a push reduces to a straight bet on the other leg (book rules vary)
                if "L" in outcomes:
                    res, prof = "L", -r["stake"]
                elif outcomes.count("W") == 2:
                    res, prof = "W", r["stake"] * _payout(r["odds"])
                else:
                    res, prof = "P", 0.0
                b.loc[i, ["result", "profit", "clv_pts", "status"]] = [res, prof, float(np.mean(clv)), "settled"]
            continue
        g = games.loc[r["game_id"]]
        if pd.isna(g["result"]):
            continue
        close, margin = _side_close_and_margin(g, r["team"])
        cover = margin + r["line"]
        res = "W" if cover > 0 else "P" if cover == 0 else "L"
        prof = r["stake"] * _payout(r["odds"]) if res == "W" else (-r["stake"] if res == "L" else 0.0)
        b.loc[i, ["close_line", "clv_pts", "result", "profit", "status"]] = [close, r["line"] - close, res, prof,
                                                                            "settled"]
    _save(b)
    return b


def angle_status(b: pd.DataFrame | None = None) -> dict:
    """Live record and stake multiplier per angle."""
    b = _load() if b is None else b
    s = b[b["status"] == "settled"]
    out = {}
    for angle, cfg in ANGLES.items():
        a = s[s["angle"] == angle]
        w, l = int((a["result"] == "W").sum()), int((a["result"] == "L").sum())
        p0, be = cfg["backtest"], cfg["breakeven"]
        post = (p0 * PRIOR_BETS + w) / (PRIOR_BETS + w + l)
        mult = float(np.clip((post - be) / (p0 - be), 0, 1))
        clv = float(a["clv_pts"].mean()) if len(a) else None
        note = ""
        if len(a) >= MIN_CLV_BETS and clv is not None and clv < 0:
            mult = min(mult, 0.5)
            note = "negative CLV"
        out[angle] = {"bets": len(a), "wins": w, "losses": l, "units": float(a["profit"].sum()),
                      "staked": float(a["stake"].sum()), "avg_clv_pts": clv,
                      "beat_close_pct": float((a["clv_pts"] > 0).mean()) if len(a) else None,
                      "posterior_win_rate": post, "stake_multiplier": mult, "note": note}
    os.makedirs(BETS_DIR, exist_ok=True)
    with open(STATUS, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def stake_multipliers() -> dict:
    """What run_week applies to each angle's stake (1.0 until live results say otherwise)."""
    if not os.path.exists(STATUS):
        return {}
    with open(STATUS) as fh:
        return {k: v["stake_multiplier"] for k, v in json.load(fh).items()}


def report() -> None:
    st = angle_status()
    rows = []
    for angle, v in st.items():
        roi = v["units"] / v["staked"] if v["staked"] else None
        rows.append({"angle": angle, "bets": v["bets"], "record": f"{v['wins']}-{v['losses']}",
                     "units": round(v["units"], 2), "roi": None if roi is None else round(roi, 3),
                     "avg CLV (pts)": None if v["avg_clv_pts"] is None else round(v["avg_clv_pts"], 2),
                     "beat close": None if v["beat_close_pct"] is None else f"{v['beat_close_pct']:.0%}",
                     "win rate (updated)": f"{v['posterior_win_rate']:.1%}",
                     "stake x": round(v["stake_multiplier"], 2), "note": v["note"]})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log")
    lg.add_argument("--season", type=int, required=True)
    lg.add_argument("--week", type=int, required=True)
    lg.add_argument("--only", help="comma list of teams you actually bet")
    st = sub.add_parser("settle")
    st.add_argument("--no-refresh", action="store_true")
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "log":
        log(a.season, a.week, a.only.split(",") if a.only else None)
    elif a.cmd == "settle":
        settle(refresh=not a.no_refresh)
        report()
    else:
        report()
