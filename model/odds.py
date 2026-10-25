"""Sportsbook lines from several books.

Two sources, one output format:
  * read_odds_export(path) - the flattened export from The Odds API / its Google Sheets add-on
    (columns: game_id, commence_time, bookmaker, home_team, away_team, market, label_1, odd_1, point_1, label_2, ...)
  * fetch_odds_api(api_key) - The Odds API v4 directly (needs api.the-odds-api.com reachable and an API key)
  * fetch_sgo(api_key) - SportsGameOdds v2 (api.sportsgameodds.com; used when SGO_API_KEY is set)

Output: one row per (book, game) with nflverse team codes and sportsbook-convention numbers:
    book, away_team, home_team, commence_time, home_spread, home_spread_odds, away_spread_odds,
    total, over_odds, under_odds, home_ml, away_ml
home_spread follows sportsbook convention: -3 means the home team is favored by 3.

    python -m model.odds --export lines/odds.csv          # convert an export, print the best number per game
    python -m model.odds --api-key KEY --out lines/x.csv   # pull live odds
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

# The four books we price by default; others in the data are kept only if asked for.
DEFAULT_BOOKS = ["FanDuel", "DraftKings", "BetMGM", "Caesars"]
API_BOOK_KEYS = {"FanDuel": "fanduel", "DraftKings": "draftkings", "BetMGM": "betmgm", "Caesars": "williamhill_us",
                 "Fanatics": "fanatics"}
BOOK_ALIASES = {"William Hill (US)": "Caesars", "Caesars Sportsbook": "Caesars", "williamhill_us": "Caesars",
                "fanduel": "FanDuel", "draftkings": "DraftKings", "betmgm": "BetMGM", "caesars": "Caesars",
                "fanatics": "Fanatics", "bet365": "bet365"}

TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

COLS = ["book", "away_team", "home_team", "commence_time", "home_spread", "home_spread_odds", "away_spread_odds",
        "total", "over_odds", "under_odds", "home_ml", "away_ml"]
# betslip links for each bet (SportsGameOdds only; empty for other sources)
LINK_COLS = ["home_spread_link", "away_spread_link", "over_link", "under_link", "home_ml_link", "away_ml_link"]


def _book(name: str) -> str:
    return BOOK_ALIASES.get(name, name)


def _rows_from_outcomes(book, home, away, commence, market, outcomes) -> dict:
    """outcomes: list of (label, price, point)."""
    r = {"book": _book(book), "home_team": TEAMS.get(home, home), "away_team": TEAMS.get(away, away),
         "commence_time": commence}
    for label, price, point in outcomes:
        if market == "spreads":
            if label == home:
                r["home_spread"], r["home_spread_odds"] = point, price
            elif label == away:
                r["away_spread_odds"] = price
                r.setdefault("home_spread", -point if point == point else np.nan)
        elif market == "totals":
            if str(label).lower() == "over":
                r["total"], r["over_odds"] = point, price
            elif str(label).lower() == "under":
                r["under_odds"] = price
        elif market == "h2h":
            if label == home:
                r["home_ml"] = price
            elif label == away:
                r["away_ml"] = price
    return r


def _combine(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=COLS)
    d = pd.DataFrame(rows)
    cols = COLS + [c for c in LINK_COLS if c in d]
    for c in COLS:
        if c not in d:
            d[c] = np.nan
    # one row per book and game, merging the spread / total / moneyline markets
    return d.groupby(["book", "away_team", "home_team"], as_index=False).first()[cols]


def read_odds_export(path: str) -> pd.DataFrame:
    """Read the flattened Odds API export (CSV, e.g. downloaded from the Google Sheet)."""
    e = pd.read_csv(path)
    rows = []
    for r in e.itertuples(index=False):
        outcomes = [(r.label_1, r.odd_1, r.point_1), (r.label_2, r.odd_2, r.point_2)]
        rows.append(_rows_from_outcomes(r.bookmaker, r.home_team, r.away_team, str(r.commence_time), r.market,
                                        outcomes))
    return _combine(rows)


def fetch_odds_api(api_key: str, books=DEFAULT_BOOKS, markets=("spreads", "totals", "h2h")) -> pd.DataFrame:
    keys = ",".join(API_BOOK_KEYS.get(b, b) for b in books)
    url = ("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
           f"?apiKey={api_key}&regions=us&markets={','.join(markets)}&oddsFormat=american&bookmakers={keys}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        events = json.load(resp)
        fetch_odds_api.credits = {k: resp.headers.get(f"x-requests-{k}") for k in ("used", "remaining", "last")}
    rows = []
    for ev in events:
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                outs = [(o.get("name"), o.get("price"), o.get("point", np.nan)) for o in mk.get("outcomes", [])]
                rows.append(_rows_from_outcomes(bk.get("title", bk.get("key")), ev["home_team"], ev["away_team"],
                                                ev.get("commence_time"), mk["key"], outs))
    return _combine(rows)


# SportsGameOdds (api.sportsgameodds.com/v2): bookmaker IDs and the six full-game markets we price.
SGO_BOOK_IDS = {"FanDuel": "fanduel", "DraftKings": "draftkings", "BetMGM": "betmgm", "Caesars": "caesars",
                "ESPN BET": "espnbet", "BetRivers": "betrivers", "Fanatics": "fanatics", "Hard Rock": "hardrockbet",
                "bet365": "bet365"}
SGO_ODDS = {"points-home-game-sp-home": ("home_spread", "home_spread_odds", "home_spread_link"),
            "points-away-game-sp-away": (None, "away_spread_odds", "away_spread_link"),
            "points-all-game-ou-over": ("total", "over_odds", "over_link"),
            "points-all-game-ou-under": (None, "under_odds", "under_link"),
            "points-home-game-ml-home": (None, "home_ml", "home_ml_link"),
            "points-away-game-ml-away": (None, "away_ml", "away_ml_link")}
_TEAM_KEYS = {k.lower(): v for k, v in TEAMS.items()}
SGO_MAX_AGE_HOURS = 6   # ignore SportsGameOdds prices last updated longer ago than this


def _sgo_team(t: dict) -> str:
    """nflverse code from an SGO team object (teamID like KANSAS_CITY_CHIEFS_NFL, names.long like Kansas City Chiefs)."""
    names = t.get("names") or {}
    for cand in (names.get("long"), str(t.get("teamID", "")).removesuffix("_NFL").replace("_", " ")):
        if cand and cand.lower() in _TEAM_KEYS:
            return _TEAM_KEYS[cand.lower()]
    return names.get("short") or t.get("teamID")


def _num(x):
    try:
        return float(str(x).replace("+", ""))
    except (TypeError, ValueError):
        return np.nan


def fetch_sgo(api_key: str, books=DEFAULT_BOOKS, raw_path: str | None = None) -> pd.DataFrame:
    """Upcoming NFL games from SportsGameOdds, one row per (book, game), same columns as fetch_odds_api."""
    ids = {SGO_BOOK_IDS.get(b, b.lower()): b for b in books}
    events, cursor = [], None
    while True:
        q = {"leagueID": "NFL", "oddsAvailable": "true", "limit": 50, "oddID": ",".join(SGO_ODDS)}
        if cursor:
            q["cursor"] = cursor
        req = urllib.request.Request("https://api.sportsgameodds.com/v2/events?" + urllib.parse.urlencode(q),
                                     headers={"X-Api-Key": api_key, "User-Agent": "nfl-model"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.load(resp)
        events += body.get("data") or []
        cursor = body.get("nextCursor")
        if not cursor or len(events) > 400:
            break
    if raw_path:   # keep the raw response so field changes are easy to diagnose
        with open(raw_path, "w") as fh:
            json.dump(events, fh, indent=1)
    rows = []
    for ev in events:
        if (ev.get("status") or {}).get("started"):   # in-game lines aren't pregame prices
            continue
        teams = ev.get("teams") or {}
        home, away = _sgo_team(teams.get("home") or {}), _sgo_team(teams.get("away") or {})
        start = (ev.get("status") or {}).get("startsAt")
        per_book: dict[str, dict] = {}
        for odd_id, (line_col, price_col, link_col) in SGO_ODDS.items():
            for bk, o in ((ev.get("odds") or {}).get(odd_id, {}).get("byBookmaker") or {}).items():
                if bk not in ids or o.get("available") is False:
                    continue
                # a price the book hasn't touched in hours is probably stale (the line moved and the feed lags)
                upd = pd.to_datetime(o.get("lastUpdatedAt"), utc=True, errors="coerce")
                if pd.notna(upd) and pd.Timestamp.now(tz="UTC") - upd > pd.Timedelta(hours=SGO_MAX_AGE_HOURS):
                    continue
                r = per_book.setdefault(bk, {"book": ids[bk], "home_team": home, "away_team": away,
                                             "commence_time": start})
                r[price_col] = _num(o.get("odds"))
                if o.get("deeplink"):
                    r[link_col] = o["deeplink"]
                if line_col == "home_spread":
                    r["home_spread"] = _num(o.get("spread"))
                elif line_col == "total":
                    r["total"] = _num(o.get("overUnder"))
        rows += per_book.values()
    return _combine(rows)


def pull_live(api_key: str, books=DEFAULT_BOOKS, out_dir: str | None = None, provider: str = "theoddsapi") -> str:
    """Pull live odds and save a timestamped snapshot (lines/odds_live_<UTC time>.csv), so every run's numbers are
    on record for line-movement and CLV checks. Returns the snapshot path."""
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lines")
    os.makedirs(out_dir, exist_ok=True)
    if provider == "sgo":
        t = fetch_sgo(api_key, books or DEFAULT_BOOKS, raw_path=os.path.join(out_dir, "sgo_raw_latest.json"))
    else:
        t = fetch_odds_api(api_key, books or DEFAULT_BOOKS)
    if not len(t):
        raise RuntimeError("no lines returned for the requested books")
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d_%H%M")
    path = os.path.join(out_dir, f"odds_live_{stamp}.csv")
    t.to_csv(path, index=False)
    got = sorted(t["book"].unique()) if len(t) else []
    missing = [b for b in (books or DEFAULT_BOOKS) if b not in got]
    c = getattr(fetch_odds_api, "credits", {}) if provider != "sgo" else {}
    print(f"live odds ({'SportsGameOdds' if provider == 'sgo' else 'The Odds API'}): {len(t)} book-game rows from "
          f"{', '.join(got) or 'no books'}" + (f" (no lines from {', '.join(missing)})" if missing else "")
          + (f"; API credits used {c.get('last')}, remaining {c.get('remaining')}" if c else "") + f"  -> {path}")
    return path


def merge_sources(oapi: pd.DataFrame | None, sgo: pd.DataFrame | None) -> pd.DataFrame:
    """One row per (book, game) from both providers. The Odds API's prices win where both have the book (they're
    kept current for DraftKings, FanDuel and BetMGM); SportsGameOdds fills books and games the other lacks (Caesars)
    and supplies betslip links, kept only where its number for that bet matches the price being shown."""
    frames = [f for f in (oapi, sgo) if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=COLS)
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    key = ["book", "away_team", "home_team"]
    a, b = oapi.copy(), sgo.copy()
    a["src_rank"], b["src_rank"] = 0, 1
    m = pd.concat([a, b], ignore_index=True).sort_values("src_rank", kind="stable").drop_duplicates(key, keep="first")
    links = b.set_index(key)
    same = {"home_spread_link": ("home_spread",), "away_spread_link": ("home_spread",), "over_link": ("total",),
            "under_link": ("total",), "home_ml_link": (), "away_ml_link": ()}
    for c in LINK_COLS:
        if c not in links:
            continue
        vals = []
        for r in m.itertuples(index=False):
            k = (r.book, r.away_team, r.home_team)
            if r.src_rank == 1 or k not in links.index:
                vals.append(getattr(r, c, np.nan))
                continue
            src = links.loc[k]
            ok = all(pd.notna(getattr(r, f)) and getattr(r, f) == src[f] for f in same[c])
            vals.append(src[c] if ok else np.nan)
        m[c] = vals
    return m.drop(columns="src_rank").reset_index(drop=True)


def pull_all(keys: dict, books=DEFAULT_BOOKS, out_dir: str | None = None) -> tuple[str, list[str]]:
    """Pull every provider with a key (keys: {"theoddsapi": key, "sgo": key}), merge, save one snapshot.
    Returns (snapshot path, log lines). Raises if no provider returned lines."""
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lines")
    os.makedirs(out_dir, exist_ok=True)
    got, log = {}, []
    for provider, key in keys.items():
        if not key:
            continue
        try:
            t = (fetch_sgo(key, books or DEFAULT_BOOKS, raw_path=os.path.join(out_dir, "sgo_raw_latest.json"))
                 if provider == "sgo" else fetch_odds_api(key, books or DEFAULT_BOOKS))
            got[provider] = t
            per = ", ".join(f"{b} {n}" for b, n in t.groupby("book").size().items()) if len(t) else "no lines"
            log.append(f"{provider}: ok ({per})")
        except Exception as e:  # network, quota or format trouble: use whatever else came back
            detail = e.read().decode(errors="replace")[:300] if hasattr(e, "read") else ""
            log.append(f"{provider}: FAILED {e} {detail}".strip())
    t = merge_sources(got.get("theoddsapi"), got.get("sgo"))
    if not len(t):
        raise RuntimeError("no lines returned: " + "; ".join(log))
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d_%H%M")
    path = os.path.join(out_dir, f"odds_live_{stamp}.csv")
    t.to_csv(path, index=False)
    c = getattr(fetch_odds_api, "credits", {})
    if "theoddsapi" in got and c:
        log.append(f"theoddsapi credits used {c.get('last')}, remaining {c.get('remaining')}")
    print("live odds: " + " | ".join(log) + f"  -> {path}")
    return path, log


def latest_snapshot(out_dir: str | None = None) -> str | None:
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lines")
    if not os.path.isdir(out_dir):
        return None
    snaps = sorted(f for f in os.listdir(out_dir) if f.startswith("odds_live_") and f.endswith(".csv"))
    return os.path.join(out_dir, snaps[-1]) if snaps else None


def load_book_lines(path: str, books=DEFAULT_BOOKS) -> pd.DataFrame:
    """Lines file in any supported format -> per-book table. `books=None` keeps every book in the file."""
    head = pd.read_csv(path, nrows=1)
    d = read_odds_export(path) if "bookmaker" in head.columns else pd.read_csv(path)
    if "book" not in d:
        d["book"] = os.path.basename(path)
    if books:
        d = d[d["book"].isin(books)]
    return d.reset_index(drop=True)


def payout(odds) -> float:
    odds = -110.0 if odds is None or pd.isna(odds) else float(odds)
    return 100 / -odds if odds < 0 else odds / 100


def breakeven(odds) -> float:
    return 1 / (1 + payout(odds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", help="Odds API export CSV to convert")
    ap.add_argument("--api-key", default=os.environ.get("ODDS_API_KEY"))
    ap.add_argument("--books", default=",".join(DEFAULT_BOOKS), help="comma list, or 'all'")
    ap.add_argument("--out", help="write the per-book table here")
    a = ap.parse_args()
    books = None if a.books == "all" else a.books.split(",")
    if a.export:
        t = load_book_lines(a.export, books)
    elif a.api_key:
        t = fetch_odds_api(a.api_key, books or DEFAULT_BOOKS)
        print("API credits:", getattr(fetch_odds_api, "credits", {}))
    else:
        raise SystemExit("give --export FILE or --api-key KEY (or set ODDS_API_KEY)")
    if a.out:
        t.to_csv(a.out, index=False)
    print(t.to_string(index=False))
