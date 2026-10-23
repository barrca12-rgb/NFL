"""Sportsbook lines from several books.

Two sources, one output format:
  * read_odds_export(path) - the flattened export from The Odds API / its Google Sheets add-on
    (columns: game_id, commence_time, bookmaker, home_team, away_team, market, label_1, odd_1, point_1, label_2, ...)
  * fetch_odds_api(api_key) - The Odds API v4 directly (needs api.the-odds-api.com reachable and an API key)

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
import urllib.request

import numpy as np
import pandas as pd

# The four books we price by default; others in the data are kept only if asked for.
DEFAULT_BOOKS = ["FanDuel", "DraftKings", "BetMGM", "Caesars"]
API_BOOK_KEYS = {"FanDuel": "fanduel", "DraftKings": "draftkings", "BetMGM": "betmgm", "Caesars": "williamhill_us"}
BOOK_ALIASES = {"William Hill (US)": "Caesars", "Caesars Sportsbook": "Caesars", "williamhill_us": "Caesars",
                "fanduel": "FanDuel", "draftkings": "DraftKings", "betmgm": "BetMGM"}

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
    for c in COLS:
        if c not in d:
            d[c] = np.nan
    # one row per book and game, merging the spread / total / moneyline markets
    return d.groupby(["book", "away_team", "home_team"], as_index=False).first()[COLS]


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
    rows = []
    for ev in events:
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                outs = [(o.get("name"), o.get("price"), o.get("point", np.nan)) for o in mk.get("outcomes", [])]
                rows.append(_rows_from_outcomes(bk.get("title", bk.get("key")), ev["home_team"], ev["away_team"],
                                                ev.get("commence_time"), mk["key"], outs))
    return _combine(rows)


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
    else:
        raise SystemExit("give --export FILE or --api-key KEY (or set ODDS_API_KEY)")
    if a.out:
        t.to_csv(a.out, index=False)
    print(t.to_string(index=False))
