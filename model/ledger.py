"""The model's own record: every play the site recommended, as it stood at kickoff, graded against the final score.

Each run records the current plays for games that haven't started; a later run before kickoff replaces them, so
the ledger keeps the last recommendation made before each game began (what someone following the site would have
bet). Plays that drop off the card before kickoff are removed. Teasers lock when their first leg kicks off.

    bets/model_plays.csv   one row per play (spread bets and 2-team teasers)

This is separate from bets/bets.csv, which tracks bets actually placed (model/tracker.py).
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np
import pandas as pd

from .data import REPO
from .odds import payout

LEDGER = os.path.join(REPO, "bets", "model_plays.csv")
COLS = ["season", "week", "kind", "angle", "game_id", "kickoff", "matchup", "selection", "team", "line", "odds", "book",
        "stake", "legs", "recorded_at", "result", "profit", "close_line", "clv_pts"]


def load() -> pd.DataFrame:
    if not os.path.exists(LEDGER):
        return pd.DataFrame(columns=COLS)
    d = pd.read_csv(LEDGER, dtype={"legs": str, "result": str, "game_id": str})
    return d.reindex(columns=COLS)


def save(d: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    d = d.sort_values(["season", "week", "kickoff", "kind", "selection"])
    d[COLS].to_csv(LEDGER, index=False)


def kickoffs(games: pd.DataFrame, season: int) -> dict:
    """game_id -> kickoff time (UTC) from the schedule's US Eastern date and time."""
    s = games[games["season"] == season]
    out = {}
    for g in s.itertuples():
        try:
            out[g.game_id] = pd.Timestamp(f"{pd.Timestamp(g.gameday).date()} {g.gametime}",
                                          tz="America/New_York").tz_convert("UTC")
        except (TypeError, ValueError):
            pass
    return out


def _plays(season: int, week: int, games: list[dict], teaser_pairs: list[dict], ko: dict, at: str) -> list[dict]:
    """Normalize a card's plays (game rows and teaser pairs as dicts) to ledger rows."""
    rows = []
    for g in games:
        if g.get("verdict") not in ("BET", "SMALL", "H2H") or not g.get("stake_units"):
            continue
        angle = "H2H" if g["verdict"] == "H2H" else ("SMALL" if g["verdict"] == "SMALL" else (g.get("flag") or "BET"))
        # older pages named these spread_pick / pick_line / pick_odds / best_book
        side = g.get("bet_side") or g.get("spread_pick")
        line = g.get("bet_line") if g.get("bet_line") is not None else g.get("pick_line")
        if not side or line is None:
            continue
        g = {**g, "bet_side": side, "bet_odds": g.get("bet_odds", g.get("pick_odds")),
             "bet_book": g.get("bet_book") or g.get("best_book")}
        rows.append({"season": season, "week": week, "kind": "spread", "angle": angle, "game_id": g["game_id"],
                     "kickoff": ko.get(g["game_id"]), "matchup": f"{g['away_team']} @ {g['home_team']}",
                     "selection": f"{g['bet_side']} {line:+g}", "team": g["bet_side"], "line": line,
                     "odds": g.get("bet_odds"), "book": g.get("bet_book"), "stake": g["stake_units"], "legs": "",
                     "recorded_at": at})
    for t in teaser_pairs or []:
        legs = [{k: l[k] for k in ("game_id", "team", "line", "teased_line", "book")} for l in t.get("legs") or []]
        if not legs:
            continue
        first = min((ko.get(l["game_id"]) for l in legs if ko.get(l["game_id"]) is not None), default=None)
        rows.append({"season": season, "week": week, "kind": "teaser", "angle": "TEASER " + str(t.get("grades", "")),
                     "game_id": ",".join(l["game_id"] for l in legs), "kickoff": first, "matchup": "",
                     "selection": t["teaser"], "team": "/".join(l["team"] for l in legs), "line": np.nan,
                     "odds": -120, "book": "/".join(sorted({str(l["book"]) for l in legs})), "stake": t["stake_units"],
                     "legs": json.dumps(legs), "recorded_at": at})
    return rows


def record_card(season: int, week: int, games: list[dict], teaser_pairs: list[dict], ko: dict,
                now: pd.Timestamp | None = None, d: pd.DataFrame | None = None) -> pd.DataFrame:
    """Replace the not-yet-started plays of this week with the card's current plays."""
    now = now or pd.Timestamp.now(tz="UTC")
    d = load() if d is None else d
    kick = pd.to_datetime(d["kickoff"], utc=True, errors="coerce")
    open_ = (d["season"] == season) & (d["week"] == week) & d["result"].isna() & ~(kick <= now)
    d = d[~open_]
    new = [r for r in _plays(season, week, games, teaser_pairs, ko, now.isoformat(timespec="minutes"))
           if r["kickoff"] is not None and r["kickoff"] > now]
    if new:
        n = pd.DataFrame(new)
        n["kickoff"] = n["kickoff"].map(lambda t: t.isoformat())
        d = pd.concat([d, n.reindex(columns=COLS)], ignore_index=True)
    return d


def _side(g, team: str):
    """Closing line and final margin from this team's side."""
    home = team == g["home_team"]
    return (-g["spread_line"] if home else g["spread_line"]), (g["result"] if home else -g["result"])


def settle(d: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    gi = games.set_index("game_id")
    for i, r in d[d["result"].isna()].iterrows():
        if r["kind"] == "teaser":
            legs = json.loads(r["legs"])
            if not all(l["game_id"] in gi.index and pd.notna(gi.loc[l["game_id"], "result"]) for l in legs):
                continue
            outs, clv = [], []
            for l in legs:
                close, margin = _side(gi.loc[l["game_id"]], l["team"])
                c = margin + l["teased_line"]
                outs.append("W" if c > 0 else "P" if c == 0 else "L")
                clv.append(l["line"] - close)
            res = "L" if "L" in outs else ("W" if outs.count("W") == len(outs) else "P")
            d.loc[i, ["close_line", "clv_pts"]] = [np.nan, float(np.mean(clv))]
        else:
            if r["game_id"] not in gi.index or pd.isna(gi.loc[r["game_id"], "result"]):
                continue
            close, margin = _side(gi.loc[r["game_id"]], r["team"])
            c = margin + float(r["line"])
            res = "W" if c > 0 else "P" if c == 0 else "L"
            d.loc[i, ["close_line", "clv_pts"]] = [close, float(r["line"]) - close]
        stake = float(r["stake"])
        d.loc[i, "result"] = res
        d.loc[i, "profit"] = stake * payout(r["odds"]) if res == "W" else (-stake if res == "L" else 0.0)
    return d


def summary(d: pd.DataFrame) -> dict:
    """Record overall, by bet type and by week, for the site's Record page."""
    def agg(x: pd.DataFrame) -> dict:
        s = x[x["result"].isin(["W", "L", "P"])]
        staked = float(s["stake"].sum())
        units = float(s["profit"].sum())
        return {"w": int((s["result"] == "W").sum()), "l": int((s["result"] == "L").sum()),
                "p": int((s["result"] == "P").sum()), "units": round(units, 2), "staked": round(staked, 2),
                "roi": round(units / staked, 4) if staked else None,
                "clv": round(float(s["clv_pts"].mean()), 2) if len(s) and s["clv_pts"].notna().any() else None,
                "pending": int(x["result"].isna().sum())}
    d = d.copy()
    d["type"] = np.where(d["kind"] == "teaser", "TEASER", d["angle"])
    rows = [{c: (None if pd.isna(v) else v) for c, v in r.items()} for r in d[COLS].to_dict("records")]
    return {"overall": agg(d),
            "by_type": {k: agg(g) for k, g in d.groupby("type")},
            "by_week": {f"{int(s)}-{int(w)}": agg(g) for (s, w), g in d.groupby(["season", "week"])},
            "plays": rows}


def backfill_from_git(season: int, games: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the ledger from every committed version of the season's report pages (each site update is a commit),
    so the record covers games that were played before the ledger existed."""
    ko = kickoffs(games, season)
    d = pd.DataFrame(columns=COLS)
    log = subprocess.run(["git", "-C", REPO, "log", "--reverse", "--format=%H %cI", "--", "reports"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    for line in filter(None, log):
        sha, when = line.split()
        now = pd.Timestamp(when).tz_convert("UTC")
        files = subprocess.run(["git", "-C", REPO, "ls-tree", "--name-only", sha, "reports/"],
                               capture_output=True, text=True).stdout.split()
        for f in files:
            name = os.path.basename(f)
            if not (name.startswith(f"{season}_week") and name.endswith(".html")):
                continue
            page = subprocess.run(["git", "-C", REPO, "show", f"{sha}:{f}"], capture_output=True, text=True).stdout
            a = page.find('<script id="report-data" type="application/json">')
            if a < 0:
                continue
            a = page.index(">", a) + 1
            data = json.loads(page[a:page.index("</script>", a)].replace("<\\/", "</"))
            d = record_card(season, int(data["week"]), data.get("games", []), data.get("teaser_pairs", []), ko,
                            now=now, d=d)
    return d
