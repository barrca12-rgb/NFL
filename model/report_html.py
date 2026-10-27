"""Weekly bet card as a self-contained HTML page: verdicts first, then sortable tables for everything else.

The page has no external dependencies except Google Fonts (with fallbacks), so it opens from a file or publishes
as-is.
"""
from __future__ import annotations

import html
import json
import os
import math

import numpy as np
import pandas as pd


# Two-letter state for sportsbook links that are state-specific (BetMGM, Caesars). Override with BET_STATE.
BET_STATE = os.environ.get("BET_STATE", "pa").lower()
SITE_URL = "https://barrca12-rgb.github.io/NFL/"
X_HANDLE = "OffTheGlassHQ"
# Off The Glass brand, same as the CBB and NBA cards (barrca12-rgb/CBB cbb/report_html.py HEAD and BRAND);
# site/assets is copied into the published site by the workflow
HEAD = f"""<meta name="description" content="Off The Glass: an NFL betting model. Every play logged at the posted number, graded in public.">
<meta name="theme-color" content="#0A2C49">
<link rel="icon" type="image/png" sizes="32x32" href="assets/favicon-32.png">
<link rel="icon" type="image/png" sizes="64x64" href="assets/favicon-64.png">
<link rel="apple-touch-icon" href="assets/apple-touch-icon.png">
<meta property="og:site_name" content="Off The Glass">
<meta property="og:type" content="website">
<meta property="og:image" content="{SITE_URL}assets/otg-social.jpg">
<meta property="og:description" content="NFL model: every play logged at the posted number, graded in public.">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@{X_HANDLE}">
<meta name="twitter:image" content="{SITE_URL}assets/otg-social.jpg">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Graduate&family=Inter:wght@400;500;600;700;800&display=swap">"""
BRAND = ('<a class="brand" href="https://barrca12-rgb.github.io/" aria-label="Off The Glass home"><img src="assets/otg-logo-512.png" alt="" width="34" height="34">'
         '<span class="wordmark">Off The Glass</span></a>')
FOLLOW = ('<footer class="foot"><a class="follow" href="https://x.com/' + X_HANDLE + '" target="_blank" rel="noopener">'
          '<img src="assets/otg-logo-512.png" alt="">Follow @' + X_HANDLE + ' on X for every play</a></footer>')


def _clean(v):
    if v is None:
        return None
    if isinstance(v, list):
        return [{k: _clean(x) for k, x in o.items()} for o in v]
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.strftime("%Y-%m-%d")
    return v if isinstance(v, (int, str, bool)) else str(v)


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    return [{c: _clean(r.get(c)) for c in cols} for r in df.to_dict("records")]


def build_html(out: pd.DataFrame, pt: pd.DataFrame, season: int, week: int, n_sims: int, phase: str,
               flag_text: str, inj_mode: str | None, inj_players: pd.DataFrame | None, qb_change: dict,
               tz: pd.DataFrame | None = None, tpairs: pd.DataFrame | None = None) -> str:
    g = out.copy()
    g["matchup"] = g["away_team"] + " @ " + g["home_team"]
    pw = g["p_home_win_blend"].fillna(g["p_home_win"]) if "p_home_win_blend" in g else g["p_home_win"]
    g["winner"] = np.where(pw >= 0.5, g["home_team"], g["away_team"])
    g["win_pct"] = np.maximum(pw, 1 - pw)
    g["qb_note"] = [
        "; ".join(f"{t}: {qb_change[t]}" for t in (a, h) if t in qb_change) for a, h in zip(g["away_team"], g["home_team"])
    ]
    games = _records(g, [
        "game_id", "gameday", "matchup", "away_team", "home_team", "proj_away", "proj_home", "model_home_spread",
        "home_spread", "spread_edge", "spread_pick", "pick_line", "p_cover_raw", "p_cover_blend", "spread_ev",
        "verdict", "stake_units", "reason", "bet_down_to", "bet_at", "flag", "model_total", "total", "total_pick",
        "p_total_raw", "winner", "win_pct", "away_inj_pts", "home_inj_pts", "qb_note",
        "best_book", "pick_odds", "offers", "n_books", "bet_side", "bet_line", "bet_odds", "bet_book",
        "h2h_meetings", "h2h_total", "h2h_strict", "h2h_loose", "h2h_p_blend", "home_coach", "away_coach",
        "kickoff", "book_lines", "total_edge", "state", "away_score", "home_score",
    ])
    p = pt.reset_index().rename(columns={"index": "team"})
    p["rank"] = range(1, len(p) + 1)
    ratings = _records(p, ["rank", "team", "net_pts", "off_pts", "def_pts", "off_epa", "def_epa", "st", "elo", "market"])
    players = _records(inj_players, ["team", "player", "position", "status", "p_miss", "usual"]) \
        if inj_players is not None and len(inj_players) else []

    data = {
        "season": season, "week": week, "n_sims": n_sims, "phase": phase, "flag_text": flag_text,
        "bet_state": BET_STATE,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "inj_mode": inj_mode,
        "line_source": ", ".join(sorted({b for x in out["line_source"].dropna() for b in str(x).split(" (live")[0].split(", ")}
                                       - {"nflverse", "none"})) or "consensus",
        "games": games, "ratings": ratings, "injuries": players,
        "teasers": _records(tz, ["matchup", "team", "grade", "line", "teased_line", "book", "total", "p_leg"])
        if tz is not None and len(tz) else [],
        "teaser_pairs": _records(tpairs, ["teaser", "grades", "legs", "p_both", "p_both_model", "ev", "stake_units"])
        if tpairs is not None and len(tpairs) else [],
    }
    payload = json.dumps(data).replace("</", "<\\/")
    title = f"Off The Glass · NFL · Week {week}"
    return (TEMPLATE.replace("__HEAD__", HEAD).replace("__BRAND__", BRAND).replace("__FOLLOW__", FOLLOW)
            .replace("__TITLE__", html.escape(title)).replace("__DATA__", payload))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<meta property="og:title" content="__TITLE__">
<meta name="claude:viewport-fit-sides" content="cover">
__HEAD__
<style>
/* Off The Glass, same palette as the CBB and NBA cards (barrca12-rgb/CBB, cbb/report_html.py). Names kept from the
   original NFL card: ink = text, chalk = sunken, blue = accent, turf = positive, pass = faint. */
:root {
  --bg: #fbf6ea; --surface: #ffffff; --chalk: #f7f0e1; --line: #ebe1cd; --line-strong: #d9cbb0;
  --ink: #0a2c49; --muted: #56627a; --pass: #8a93a3;
  --blue: #b84a08; --turf: #0f7b4b; --turf-soft: #e6f4ed; --neg: #c2352b; --amber: #8a5a00; --amber-soft: #fbf0d9;
  --on-accent: #ffffff;
  --brand: #fd6c0f; --bar: #0a2c49; --bar-text: #fbf6ea; --bar-muted: #b7c3d3;
  --display: "Graduate", "Rockwell", Georgia, serif;
  --body: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #08090b; --surface: #12151a; --chalk: #181c22; --line: #242a33; --line-strong: #333b46;
    --ink: #f6eedd; --muted: #a8b0bc; --pass: #6f7886;
    --blue: #ff8a3d; --turf: #3ccf8e; --turf-soft: #12301f; --neg: #ff6b5e; --amber: #f0b44a; --amber-soft: #33270f;
    --on-accent: #08090b;
    --brand: #fd6c0f; --bar: #000000; --bar-text: #f6eedd; --bar-muted: #9aa2ae;
    color-scheme: dark;
  }
}
* { box-sizing: border-box }
.show-sm { display: none }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 var(--body); font-feature-settings: "tnum" 1, "cv11" 1; -webkit-font-smoothing: antialiased }
a { color: var(--blue) }

/* Top bar (same as the CBB and NBA cards) */
.top { background: var(--bar); border-bottom: 3px solid var(--brand); position: sticky; top: 0; z-index: 5 }
.topin { max-width: 1320px; margin: 0 auto; padding: 0 16px; display: flex; align-items: center; gap: 22px; height: 58px }
.brand { display: flex; align-items: center; gap: 10px; white-space: nowrap; color: var(--bar-text); text-decoration: none }
.brand img { width: 34px; height: 34px; border-radius: 7px; display: block }
.wordmark { font-family: var(--display); font-size: 19px; letter-spacing: .02em; text-transform: uppercase; color: var(--bar-text); line-height: 1 }
.tabs { display: flex; gap: 2px; height: 100% }
.tabs a { display: flex; align-items: center; padding: 0 12px; color: var(--bar-muted); font-weight: 600; font-size: 13px; letter-spacing: .04em;
  text-transform: uppercase; border-bottom: 3px solid transparent; margin-bottom: -3px; text-decoration: none }
.tabs a:hover { color: var(--bar-text) }
.tabs a[aria-current="page"] { color: var(--bar-text); border-bottom-color: var(--brand) }
.status { margin-left: auto; display: flex; align-items: center; gap: 12px; color: var(--bar-muted); font-size: 13px; min-width: 0 }
.status a { color: var(--bar-text) }
.status #upd-status { white-space: nowrap; overflow: hidden; text-overflow: ellipsis }
.status button, .filters button { font: inherit; font-weight: 600; font-size: 13px; color: var(--ink); background: var(--surface); border: 1px solid var(--line-strong);
  border-radius: 6px; padding: 6px 12px; cursor: pointer }
.filters button:hover { background: var(--chalk) }
.status button { background: transparent; color: var(--bar-text); border-color: rgba(251,246,234,.35) } .status button:hover { background: rgba(251,246,234,.1) }
.status button:disabled { opacity: .5; cursor: progress }

/* Page */
.wrap { max-width: 1320px; margin: 0 auto; padding: 24px 16px 48px; display: grid; gap: 24px }
h1, h2, h3 { margin: 0 }
h1 { font-family: var(--display); font-size: 26px; font-weight: 400; letter-spacing: .01em; line-height: 1.15 }
h2 { font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: .07em; display: flex; align-items: center; gap: 8px }
.sec-head h2::before { content: ""; width: 4px; height: 14px; background: var(--brand); border-radius: 2px }
.head { display: grid; gap: 8px }
.eyebrow { color: var(--muted); font-size: 13px; font-weight: 500 }
.lede { color: var(--muted); max-width: 80ch; margin: 0 }
.meta { display: flex; flex-wrap: wrap; gap: 6px }
.chip { font-size: 12px; font-weight: 500; padding: 3px 8px; border-radius: 4px; border: 1px solid var(--line); background: var(--surface); color: var(--muted) }
section { display: grid; gap: 10px; min-width: 0 }
.sec-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px; padding-bottom: 8px; border-bottom: 1px solid var(--line) }
.note { color: var(--muted); font-size: 12px; max-width: 90ch; margin: 0 }

/* Week switcher */
.weeks { display: flex; flex-wrap: wrap; gap: 4px }
.weeks a { padding: 5px 12px; border-radius: 6px; color: var(--ink); font-weight: 600; font-size: 13px; text-decoration: none;
  border: 1px solid var(--line-strong); background: var(--surface); white-space: nowrap }
.weeks a:hover { background: var(--chalk) }
.weeks a[aria-current="page"] { color: var(--surface); border-color: var(--ink); background: var(--ink) }
.weeks a .tag { font-size: 11px; font-weight: 700; margin-left: 6px; color: var(--brand) }

/* Model plays: same panel and table as the CBB card (barrca12-rgb/CBB cbb/report_html.py) */
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; box-shadow: 0 1px 2px rgba(10,44,73,.06) }
.panel-h { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; padding: 12px 16px; border-bottom: 1px solid var(--line) }
.panel-h h2::before { content: ""; width: 4px; height: 14px; background: var(--brand); border-radius: 2px }
.scroll { overflow-x: auto }
table.ct th, table.ct td { text-align: left; padding: 9px 12px; vertical-align: middle }
table.ct thead th { position: static; cursor: default; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; padding: 8px 12px }
table.ct tbody tr:hover { background: none }
table.ct .r { text-align: right } .m { color: var(--muted) } .f { color: var(--pass) } .b { font-weight: 600 }
.pos { color: var(--turf) } .neg { color: var(--neg) } td.pos { font-weight: 600 }
table.ct td.empty { padding: 28px 16px; color: var(--muted); text-align: center; white-space: normal }
.inj { color: var(--amber); font-size: 11px; font-weight: 500; white-space: normal; max-width: 420px; margin-top: 1px }
.tag { display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 6px; border-radius: 4px; border: 1px solid var(--line-strong); color: var(--muted); white-space: nowrap }
.tag.bet { color: #fff; border-color: var(--brand); background: var(--brand) }
.tl img { width: auto; vertical-align: -3px; margin-right: 5px; object-fit: contain }
.pick .tl img { vertical-align: -5px; margin-right: 6px }
.pagebtn { padding: 5px 12px; border-radius: 6px; color: var(--ink); font-weight: 600; font-size: 13px; text-decoration: none;
  border: 1px solid var(--line-strong); background: var(--surface); white-space: nowrap }
.pagebtn:hover { background: var(--chalk) }
.navrow { display: flex; flex-wrap: wrap; gap: 4px; align-items: center }
@media (max-width: 1100px) { .hide-md { display: none } }
@media (max-width: 860px) {
  #t-plays td:first-child { white-space: normal; min-width: 150px }
  table.stack thead { display: none }
  table.stack tbody tr { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px 12px; padding: 12px; border-bottom: 1px solid var(--line) }
  table.stack tbody tr:last-child { border-bottom: 0 }
  table.stack td { display: block; border: 0; padding: 0; white-space: normal; text-align: left; min-width: 0 }
  table.stack td.r { text-align: left }
  table.stack td:first-child { grid-column: 1 / -1; min-width: 0 }
  table.stack td[data-l]::before { content: attr(data-l); display: block; font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .05em; color: var(--pass); margin-bottom: 1px }
  table.stack td:first-child::before { display: none }
  table.stack td.hide-sm, table.stack td:empty { display: none }
  table.stack td.empty { grid-column: 1 / -1; text-align: center }
}
.plays { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 520px), 1fr)); gap: 12px }
.pk { background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--brand); border-radius: 10px; box-shadow: 0 1px 2px rgba(10,44,73,.06); padding: 14px 16px; display: grid; gap: 12px; min-width: 0 }
.pk.small { border-left-color: var(--amber) } .pk.tz { border-left-color: var(--blue) }
.pk-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px }
.pk-pick { font-size: 18px; font-weight: 700; letter-spacing: -.01em } .pk-pick .price { color: var(--muted); font-weight: 500 }
.pk-game { color: var(--muted); font-size: 12px; margin-top: 2px }
.pk-stake { text-align: right; flex: none }
.pk-stake b { display: block; font-size: 18px; font-weight: 700 }
.pk-stake span { font-size: 11px; font-weight: 600; color: var(--muted) }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); border: 1px solid var(--line); border-radius: 6px; overflow: hidden }
.stat { padding: 8px 10px; min-width: 0; border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); margin: 0 -1px -1px 0 }
.stat span { display: block; font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--muted) }
.stat b { display: block; font-size: 14px; font-weight: 600; margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis }
.stat b.pos { color: var(--turf) }
.pk-why { color: var(--muted); font-size: 12px }
.cta { display: inline-block; justify-self: start; font-weight: 600; font-size: 13px; padding: 7px 14px; border-radius: 6px; text-decoration: none;
  background: var(--ink); color: var(--surface) }
.cta:hover { opacity: .9 }
span.cta { background: var(--chalk); color: var(--muted) }
.empty { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 24px 16px; color: var(--muted); text-align: center }

/* Full slate: one card per game, books side by side */
.slate { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 520px), 1fr)); gap: 12px }
.game { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 1px 2px rgba(10,44,73,.06); padding: 14px 16px; display: grid; gap: 10px; min-width: 0 }
.game.has-play { border-left: 3px solid var(--brand) }
.g-top { display: flex; justify-content: space-between; align-items: center; gap: 8px }
.g-title { font-size: 15px; font-weight: 600 }
.g-kick { color: var(--muted); font-size: 12px }
.oddswrap { overflow-x: auto; margin: 0 -4px; padding: 0 4px; scrollbar-width: thin }
table.odds { font-size: 13px; width: 100%; border-collapse: separate; border-spacing: 3px; table-layout: auto }
table.odds.few { width: auto } table.odds.few td { min-width: 90px }
table.odds th, table.odds td { padding: 0; border: 0; white-space: nowrap }
table.odds thead th { position: static; background: none; cursor: default; font-size: 11px; font-weight: 500; color: var(--muted); text-align: center; padding: 0 2px 2px; text-transform: none; letter-spacing: 0 }
table.odds .abbr { display: none }
table.odds tbody th { text-align: left; font-size: 12px; font-weight: 500; color: var(--muted); padding-right: 6px; position: sticky; left: 0; background: var(--surface); z-index: 1; min-width: 46px; text-transform: none; letter-spacing: 0 }
table.odds tr.cap th { font-size: 11px; font-weight: 600; color: var(--pass); padding-top: 6px }
table.odds td { background: var(--chalk); border-radius: 6px; text-align: center; font-weight: 600; min-width: 58px; border-bottom: 0 }
table.odds td .t { display: block; padding: 5px 4px; color: inherit; text-decoration: none; line-height: 1.2 }
table.odds td small { display: block; color: var(--muted); font-size: 11px; font-weight: 500 }
table.odds td a.t:hover { outline: 1px solid var(--line-strong); border-radius: 6px }
table.odds td.best { color: var(--turf); background: var(--turf-soft) }
table.odds td.best small { color: var(--turf) }
table.odds td.none { background: transparent; color: var(--pass); font-weight: 400 }
table.odds tr.mine th { color: var(--ink); font-weight: 600 }
table.odds tr.mine th::before { content: "\2713 "; color: var(--turf) }
@media (max-width: 700px) { table.odds .full { display: none } table.odds .abbr { display: inline } table.odds td { min-width: 52px } }
.swipe { display: none; font-size: 12px; color: var(--pass) }
@media (max-width: 700px) { .swipe.on { display: block } }
.best-key { color: var(--turf) }
.g-model { font-size: 13px; color: var(--muted) } .g-model b { color: var(--ink); font-weight: 600 }
.g-play { font-size: 13px; font-weight: 600; color: var(--turf) }
.g-play a, a.bk { color: inherit; text-decoration: none } .g-play a:hover { text-decoration: underline }
.g-play.none { color: var(--pass); font-weight: 500 }

/* Verdict tags */
.v { display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 6px; border-radius: 4px; border: 1px solid var(--line-strong); color: var(--muted); white-space: nowrap }
.v.BET { color: #fff; border-color: var(--brand); background: var(--brand) }
.v.SMALL { color: var(--amber); border-color: var(--amber); background: var(--amber-soft) }
.v.H2H { color: var(--blue); border-color: var(--blue) }
.v.WATCH { color: var(--amber); border-color: var(--amber) }
.v.PASS { color: var(--pass) }
.v.FINAL, .v.LIVE { color: var(--muted) }
.v.LIVE { color: var(--neg); border-color: var(--neg) }
.game.done { opacity: .7; align-self: start; align-content: start }
.g-score { font-size: 15px; font-weight: 600 }

/* Tables */
.filters { display: flex; flex-wrap: wrap; gap: 6px }
.filters button { font-weight: 500 }
.filters button[aria-pressed="true"] { background: var(--ink); color: var(--surface); border-color: var(--ink) }
.tbl { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 8px }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums }
th, td { padding: 9px 12px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--line) }
th:first-child, td:first-child, th.l, td.l { text-align: left }
tbody tr:last-child td { border-bottom: 0 }
thead th { position: sticky; top: 0; background: var(--chalk); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); cursor: pointer; user-select: none }
thead th:hover, thead th:focus-visible { color: var(--ink); outline: none }
thead th[aria-sort="ascending"]::after { content: " \25B2"; font-size: 9px }
thead th[aria-sort="descending"]::after { content: " \25BC"; font-size: 9px }
tbody tr:hover { background: var(--chalk) }
td.pos { color: var(--turf); font-weight: 600 } td.neg { color: var(--neg) }
td.strong { font-weight: 600 }
details { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow: hidden }
details > :not(summary) { margin: 12px 16px }
details > .tbl { border-radius: 6px }
summary { cursor: pointer; list-style: none; padding: 12px 16px; font-size: 15px; font-weight: 600; display: flex; justify-content: space-between }
summary::after { content: "Show"; color: var(--blue); font-size: 13px; font-weight: 500 }
details[open] summary::after { content: "Hide" }
details[open] summary { border-bottom: 1px solid var(--line) }
.glossary { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px }
.glossary div { border: 1px solid var(--line); border-radius: 6px; padding: 12px; font-size: 13px; color: var(--muted) }
.glossary b { font-size: 13px; font-weight: 600; display: block; margin-bottom: 2px; color: var(--ink) }
.foot { padding: 8px 0 0 }
.follow { display: inline-flex; align-items: center; gap: 8px; font-weight: 700; color: var(--ink); text-decoration: none }
.follow img { width: 22px; height: 22px; border-radius: 5px }
@media (max-width: 860px) { .topin { gap: 10px } .status #upd-status { display: none } .wordmark { display: none } .tabs a { padding: 0 8px; font-size: 12px }
  .hide-sm { display: none } .show-sm { display: inline } h1 { font-size: 21px } }
</style>

<header class="top"><div class="topin">
  __BRAND__
  <nav class="tabs" aria-label="Sport"><a href="./" aria-current="page">NFL</a><a href="https://barrca12-rgb.github.io/NBA/">NBA</a><a href="https://barrca12-rgb.github.io/CBB/"><span class="hide-sm">College Basketball</span><span class="show-sm">CBB</span></a></nav>
  <div class="status"><span id="upd-status"></span><button id="upd-btn" type="button">Update</button></div>
</div></header>
<div class="wrap">
  <header class="head">
    <div class="eyebrow" id="eyebrow"></div>
    <h1 id="h1"></h1>
    <p class="lede" id="lede"></p>
    <div class="navrow"><nav class="weeks" id="weeks" aria-label="Weeks"></nav><a class="pagebtn" href="record.html">Record</a></div>
    <div class="meta" id="meta"></div>
  </header>

  <section class="panel" aria-labelledby="plays-h">
    <div class="panel-h"><h2 id="plays-h">Plays</h2><span class="note">Bet each play only at the number shown or better ("Bet to"). Cover % is the realistic rate, already adjusted for how often the model has been right; EV above 0 means the price is worth it. <span id="flagtext"></span></span></div>
    <div class="scroll"><table class="ct" id="t-plays"></table></div>
  </section>

  <section aria-labelledby="slate-h">
    <div class="sec-head"><h2 id="slate-h">Full slate</h2>
      <span class="note">Every game at every book. <b class="best-key">Green</b> is the best number for that bet; &#10003; marks a model play. Tap any price to open that bet at the book.</span></div>
    <div class="slate" id="slate"></div>
  </section>

  <details id="watch-sec"><summary>Watch list</summary>
    <p class="note">Not bets at today's number. Bet them only if the line reaches the number shown.</p>
    <div class="tbl"><table id="t-watch"></table></div></details>

  <details><summary>All games: model numbers</summary>
    <div class="filters" role="group" aria-label="Filter by verdict" id="vfilter"></div>
    <div class="tbl"><table id="t-games"></table></div>
    <p class="note">Click any column to sort. Spreads use sportsbook convention: -3 means that team is favored by 3. Edge is the model's margin minus the line, from the home team's side.</p></details>

  <details><summary>Winner picks</summary>
    <div class="tbl"><table id="t-win"></table></div>
    <p class="note">Win % blends the model (35%) with the line (65%). In the 2016-2025 backtest this picked 66.8% of winners; the closing-line favorite won 66.4%.</p></details>

  <details><summary>Totals (leans only)</summary>
    <p class="note">The model showed no reliable edge on totals in the backtest. Not bets.</p>
    <div class="tbl"><table id="t-tot"></table></div></details>

  <details id="tz-sec"><summary>Teaser legs</summary>
    <p class="note" id="tz-note">Every leg that qualifies this week. Only the pairings listed under Plays are bets; this list is for building your own if a book has a better teaser price. 6-point teasers through 3 and 7: grade A legs have won about 78% since 2006; break-even is 73.9% per leg at -120.</p>
    <div class="tbl"><table id="t-tz"></table></div></details>

  <details><summary>How to read this</summary>
    <div class="glossary">
      <div><b>Model cover %</b>How often the pick covers in 20,000 simulations if the model's projection is exactly right. Used to pick and rank plays; it runs high (a 70% here has historically won about 56%).</div>
      <div><b>Realistic cover %</b>The same simulation centered between the market consensus and the model (45% toward the model in Weeks 1-4, 20% after), which matches the model's real track record, then priced at each book's own number. Use this for whether a bet is worth it.</div>
      <div><b>EV / unit</b>Expected profit per unit staked at the posted odds, from the realistic cover %. Above 0 means the price is worth taking.</div>
      <div><b>Good down to</b>The worst number you can still take the bet at. If the line moves past it, pass.</div>
      <div><b>H2H</b>Coach head-to-head angle: fade the head coach whose teams have covered more in 4+ past meetings with this coach. Won 57% since 1999.</div>
      <div><b>Teaser</b>Two legs moved 6 points each, both must win. Break-even at -120 is 54.5% for the pair (73.9% per leg).</div>
    </div></details>

  <details id="inj-sec"><summary>Injuries and quarterbacks</summary><p class="note" id="inj-note"></p>
    <div class="tbl"><table id="t-inj"></table></div></details>

  <details><summary>Power ratings</summary>
    <p class="note">Points better or worse than an average team on a neutral field. Market = what past closing lines imply.</p>
    <div class="tbl"><table id="t-rat"></table></div></details>
  __FOLLOW__
</div>

<script id="report-data" type="application/json">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById("report-data").textContent);
  const $ = (id) => document.getElementById(id);
  // Betslip links: point state-specific books at the bettor's state, and use FanDuel's indexed betslip format
  const ST = (D.bet_state || "pa").toLowerCase();
  const fixLink = (u) => !u ? u : u
    .replace(/^https:\/\/sports\.[a-z]{2}\.betmgm\.com/, "https://sports." + ST + ".betmgm.com")
    .replace(/^(https:\/\/sportsbook\.caesars\.com\/us\/)[a-z]{2}\//, "$1" + ST + "/");
  D.games.forEach((g) => (g.book_lines || []).forEach((b) => Object.keys(b).forEach((k) => { if (k.endsWith("_link")) b[k] = fixLink(b[k]); })));
  const pct = (x) => x == null ? "–" : (x * 100).toFixed(1) + "%";
  const pct0 = (x) => x == null ? "–" : Math.round(x * 100) + "%";
  const sp = (x) => x == null ? "–" : (Math.abs(x) < 0.05 ? "PK" : (x > 0 ? "+" : "") + x.toFixed(1));
  const num1 = (x) => x == null ? "–" : x.toFixed(1);
  const sgn1 = (x) => x == null ? "–" : (x > 0 ? "+" : "") + x.toFixed(1);
  const ev = (x) => x == null ? "–" : (x > 0 ? "+" : "") + x.toFixed(3);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const units = (u) => u + "u";
  const ml = (x) => x == null ? "" : (x > 0 ? "+" : "") + Math.round(x);
  // ESPN logos by team abbreviation (nflverse LA and WAS are lar and wsh at ESPN); a logo that fails to load is removed
  const LOGO = {LA: "lar", WAS: "wsh"};
  const logo = (t, px) => { if (!t) return ""; const a = LOGO[t] || String(t).toLowerCase();
    return "<picture class='tl'><source media='(prefers-color-scheme: dark)' srcset='https://a.espncdn.com/i/teamlogos/nfl/500-dark/" + esc(a) +
      ".png'><img src='https://a.espncdn.com/i/teamlogos/nfl/500/" + esc(a) + ".png' alt='' loading='lazy' width='" + px + "' height='" + px +
      "' onerror=\"this.closest('picture').remove()\"></picture>"; };
  const stack = (id) => { const t = $(id); if (!t) return; const hs = [...t.querySelectorAll("thead th")].map((h) => h.textContent.trim());
    t.classList.add("stack"); t.querySelectorAll("tbody tr").forEach((r) => [...r.children].forEach((c, k) => { if (hs[k]) c.dataset.l = hs[k]; })); };
  const booksCell = (r) => (r.offers || []).map((o) => (o.book === r.best_book ? "<b>" : "") + esc(o.book.replace(/\.ag$/, "")) + " " +
    sp(o.line) + " " + ml(o.odds) + (o.book === r.best_book ? "</b>" : "")).join("<br>");

  // Header
  const days = D.games.map((g) => g.gameday).filter(Boolean).sort();
  const md = (d) => new Date(d + "T12:00:00").toLocaleDateString([], {month: "short", day: "numeric"});
  $("eyebrow").textContent = D.season + " NFL season" + (days.length ? " · " + md(days[0]) + (days[0] !== days[days.length - 1] ? " – " + md(days[days.length - 1]) : "") : "");
  $("h1").textContent = "Week " + D.week + " Bet Card";
  const bets = D.games.filter((g) => g.verdict === "BET" || g.verdict === "SMALL" || g.verdict === "H2H");
  const nPlays = bets.length + D.teaser_pairs.length;
  const units_total = Math.round((bets.reduce((s, g) => s + (g.stake_units || 0), 0) +
    D.teaser_pairs.reduce((s, t) => s + (t.stake_units || 0), 0)) * 100) / 100;
  $("lede").textContent = nPlays
    ? nPlays + (nPlays === 1 ? " play" : " plays") + " this week, " + units_total + " units total. Everything else is a pass."
    : "No bets at the current numbers. The full slate below shows every line.";
  const injLabel = {thursday: "Thursday practice report applied", final: "Final injury report applied", monday: "Injury report not out yet (Monday run)"};
  [D.n_sims.toLocaleString() + " sims per game", "Lines: " + D.line_source, injLabel[D.inj_mode] || "No injury report"]
    .forEach((t) => { const c = document.createElement("span"); c.className = "chip"; c.textContent = t; $("meta").appendChild(c); });
  $("flagtext").textContent = D.flag_text + " Break-even at -110 is 52.4%.";

  // Week switcher (weeks.json is written next to the pages by the site build)
  const navWeeks = (list, cur) => {
    $("weeks").innerHTML = list.map((w) => '<a href="' + D.season + "_week" + String(w).padStart(2, "0") + '.html"' +
      (w === D.week ? ' aria-current="page"' : "") + ">Week " + w + (w === cur ? '<span class="tag">Now</span>' : "") + "</a>").join("");
  };
  navWeeks([D.week], null);
  fetch("weeks.json", {cache: "no-store"}).then((r) => r.json()).then((w) => {
    if (w.season === D.season && w.weeks.length) navWeeks(w.weeks, w.current);
  }).catch(() => {});

  // Update bar. The button starts the "Update bet card" GitHub Actions workflow, which pulls live odds, re-runs the
  // model and republishes this page (about 5-10 minutes). It needs a fine-grained GitHub token with Actions
  // read/write on this repo, kept only in this browser.
  const REPO = "barrca12-rgb/NFL", WF = "update.yml", API = "https://api.github.com/repos/" + REPO + "/actions";
  const RUNS_URL = "https://github.com/" + REPO + "/actions/workflows/" + WF;
  const store = { get: () => { try { return localStorage.getItem("gh_token") || ""; } catch (e) { return ""; } },
                  set: (v) => { try { v ? localStorage.setItem("gh_token", v) : localStorage.removeItem("gh_token"); } catch (e) {} } };
  const gen = D.generated_at ? new Date(D.generated_at) : null;
  const ago = (d) => { const m = Math.round((Date.now() - d) / 60000);
    return m < 1 ? "just now" : m < 60 ? m + " min ago" : m < 1440 ? Math.round(m / 60) + " h ago" : Math.round(m / 1440) + " days ago"; };
  const idle = () => { $("upd-status").textContent = gen ? "Odds and picks as of " + gen.toLocaleString([], {weekday: "short", hour: "numeric", minute: "2-digit"}) + " (" + ago(gen) + ")" : ""; };
  idle();
  const askKey = () => {
    const t = prompt("Paste a GitHub fine-grained token for " + REPO + " with Actions: Read and write.\nIt is saved only in this browser. Leave blank to remove it.", store.get());
    if (t !== null) store.set(t.trim());
    return store.get();
  };
  const gh = (path, opts) => fetch(API + path, Object.assign({ headers: { Accept: "application/vnd.github+json",
    Authorization: "Bearer " + store.get(), "X-GitHub-Api-Version": "2022-11-28" } }, opts || {}));
  const followRun = (since) => {
    const btn = $("upd-btn"); btn.disabled = true;
    const t0 = Date.now();
    const tick = async () => {
      try {
        const r = await gh("/workflows/" + WF + "/runs?per_page=1");
        const run = (await r.json()).workflow_runs?.[0];
        const mins = Math.floor((Date.now() - t0) / 60000);
        if (!run || new Date(run.created_at) < since) { $("upd-status").textContent = "Starting run…"; }
        else if (run.status !== "completed") { $("upd-status").innerHTML = "Pulling live odds and running the model… " + mins + " min (usually 5–10). <a href='" + run.html_url + "' target='_blank' rel='noopener'>Progress</a>"; }
        else if (run.conclusion === "success") { $("upd-status").textContent = "Done. Loading the new card…"; setTimeout(() => location.replace(location.pathname + "?t=" + Date.now()), 4000); return; }
        else { $("upd-status").innerHTML = "The run failed (" + esc(run.conclusion) + "). <a href='" + run.html_url + "' target='_blank' rel='noopener'>See why</a>"; btn.disabled = false; return; }
      } catch (e) { $("upd-status").textContent = "Lost contact with GitHub, retrying…"; }
      setTimeout(tick, 10000);
    };
    tick();
  };
  $("upd-btn").onclick = async () => {
    if (!store.get() && !askKey()) { window.open(RUNS_URL, "_blank", "noopener"); $("upd-status").innerHTML = "No token saved: on GitHub, click <b>Run workflow</b>, then reload this page when it finishes."; return; }
    const since = new Date(Date.now() - 5000);
    $("upd-status").textContent = "Starting run…";
    const r = await gh("/workflows/" + WF + "/dispatches", { method: "POST", body: JSON.stringify({ ref: "main" }) }).catch(() => null);
    if (r && r.status === 204) followRun(since);
    else if (r && (r.status === 401 || r.status === 403 || r.status === 404)) { store.set(""); $("upd-status").textContent = "GitHub refused the saved token (" + r.status + "). Click Update now to enter a new one."; }
    else { $("upd-status").textContent = "Couldn't reach GitHub. Try again in a minute."; }
  };
  // a run already going (started from another device or the schedule): follow it
  if (store.get()) gh("/workflows/" + WF + "/runs?per_page=1").then((r) => r.json()).then((j) => {
    const run = j.workflow_runs?.[0];
    if (run && run.status !== "completed") followRun(new Date(run.created_at));
  }).catch(() => {});

  // Kickoff label, e.g. "Sun Nov 1 · 1:00 PM ET"
  const kick = (g) => {
    const d = new Date(g.gameday + "T12:00:00");
    const t = String(g.kickoff || "").split(" ")[1] || "";
    let time = "";
    if (/^\d{1,2}:\d{2}$/.test(t)) { let [h, m] = t.split(":").map(Number); time = (h % 12 || 12) + ":" + String(m).padStart(2, "0") + (h < 12 ? " AM" : " PM") + " ET"; }
    return d.toLocaleDateString([], {weekday: "short", month: "short", day: "numeric"}) + (time ? " · " + time : "");
  };
  const byKick = (a, b) => String(a.gameday).localeCompare(String(b.gameday)) || String(a.kickoff).localeCompare(String(b.kickoff));

  // Model plays: one row per bet, same table as the CBB card
  const pay = (o) => o == null ? -1 : (o > 0 ? o / 100 : 100 / -o);
  // other books' numbers for this side, best first
  const also = (g) => {
    const home = g.bet_side === g.home_team;
    const L = (g.book_lines || []).filter((b) => b.home_spread != null && b.book !== g.bet_book).map((b) => ({book: b.book,
      n: home ? b.home_spread : -b.home_spread, p: home ? b.home_spread_odds : b.away_spread_odds})).sort((x, y) => y.n - x.n || pay(y.p) - pay(x.p));
    return L.length ? "Also: " + L.map((x) => x.book + " " + sp(x.n) + (x.p != null ? " (" + ml(x.p) + ")" : "")).join(" · ") : "";
  };
  const ROWS = [];
  bets.sort((a, b) => byKick(a, b) || b.stake_units - a.stake_units).forEach((g) => {
    const h2h = g.verdict === "H2H";
    const home = g.bet_side === g.home_team;
    const modelLine = g.model_home_spread == null ? null : (home ? g.model_home_spread : -g.model_home_spread);
    const edge = modelLine == null ? null : g.bet_line - modelLine;
    const opp = home ? g.away_team : g.home_team;
    const kind = h2h ? "Spread · H2H angle" : "Spread" + (g.verdict === "SMALL" ? " · small" : "");
    ROWS.push("<tr><td><div class='b pick'>" + logo(g.bet_side, 20) + esc(g.bet_side) + " " + sp(g.bet_line) + "</div>" +
      "<div class='f' style='font-size:12px'>" + logo(g.away_team, 14) + esc(g.away_team) + " @ " + logo(g.home_team, 14) + esc(g.home_team) + " · " + kick(g) + " · " + kind + "</div>" +
      (also(g) ? "<div class='f' style='font-size:12px'>" + esc(also(g)) + "</div>" : "") +
      (g.qb_note ? "<div class='inj'>&#9888; QB: " + esc(g.qb_note) + "</div>" : "") +
      (h2h && g.reason ? "<div class='f' style='font-size:12px;white-space:normal'>" + esc(g.reason) + "</div>" : "") + "</td>" +
      "<td>" + esc(g.bet_book || "–") + (g.bet_odds != null ? " <span class='m'>" + ml(g.bet_odds) + "</span>" : "") + "</td>" +
      "<td class='r m hide-sm hide-md'>" + sp(modelLine) + "</td>" +
      "<td class='r " + (edge > 0 ? "pos" : "m") + "'>" + sgn1(edge) + "</td>" +
      "<td class='r m hide-sm'>" + pct(h2h ? g.h2h_p_blend : g.p_cover_blend) + "</td>" +
      "<td class='r'>" + (h2h || g.spread_ev == null ? "–" : (g.spread_ev > 0 ? "+" : "") + (100 * g.spread_ev).toFixed(1) + "%") + "</td>" +
      "<td class='r'>" + (h2h ? "–" : sp(g.bet_down_to)) + "</td>" +
      "<td class='r m hide-sm hide-md'>" + num1(g.proj_away) + "–" + num1(g.proj_home) + "</td>" +
      "<td class='r'>" + units(g.stake_units) + "</td></tr>");
  });
  D.teaser_pairs.forEach((t) => {
    const legs = t.legs || [];
    const books = [...new Set(legs.map((l) => l.book).filter(Boolean))];
    ROWS.push("<tr><td><div class='b pick'>" + legs.map((l) => logo(l.team, 20)).join("") + esc(t.teaser) + "</div>" +
      "<div class='f' style='font-size:12px'>2-team, 6-point teaser · both legs must win · grades " + esc(t.grades) + "</div>" +
      "<div class='f' style='font-size:12px'>" + legs.map((l) => esc(l.team) + " " + sp(l.line) + " → " + sp(l.teased_line)).join(" · ") + "</div></td>" +
      "<td>" + esc(books.length === 1 ? books[0] : books.length ? books.join(", ") : "Best teaser price") + " <span class='m'>-120</span></td>" +
      "<td class='r m hide-sm hide-md'>" + pct0(t.p_both_model) + "</td><td class='r m'>–</td>" +
      "<td class='r m hide-sm'>" + pct0(t.p_both) + "</td>" +
      "<td class='r'>" + (t.ev == null ? "–" : (t.ev > 0 ? "+" : "") + (100 * t.ev).toFixed(1) + "%") + "</td>" +
      "<td class='r'>" + legs.map((l) => sp(l.teased_line)).join(", ") + "</td><td class='r m hide-sm hide-md'>–</td>" +
      "<td class='r'>" + units(t.stake_units) + "</td></tr>");
  });
  $("t-plays").innerHTML = ROWS.length ? "<thead><tr><th>Bet</th><th>Book</th><th class='r hide-sm hide-md'>Model</th><th class='r'>Edge</th><th class='r hide-sm'>Cover %</th>" +
    "<th class='r'>EV</th><th class='r'>Bet to</th><th class='r hide-sm hide-md'>Proj</th><th class='r'>Units</th></tr></thead><tbody>" + ROWS.join("") + "</tbody>"
    : "<tbody><tr><td class='empty'>No game meets the betting rule at today's lines. That is the right call most weeks after Week 4.</td></tr></tbody>";
  stack("t-plays");

  // Full slate: one ticket per game; books side by side so the best number is easy to spot
  const slate = $("slate");
  const stOrder = {upcoming: 0, live: 1, final: 2};
  D.games.slice().sort((a, b) => (stOrder[a.state] || 0) - (stOrder[b.state] || 0) || byKick(a, b)).forEach((g) => {
    if (g.state && g.state !== "upcoming") {
      const el = document.createElement("article");
      el.className = "game done";
      const score = g.state === "final" && g.away_score != null
        ? '<div class="g-score">' + esc(g.away_team) + " " + g.away_score + " – " + g.home_score + " " + esc(g.home_team) + "</div>" : "";
      el.innerHTML = '<div class="g-top"><div><div class="g-title">' + logo(g.away_team, 20) + esc(g.away_team) + " @ " + logo(g.home_team, 20) + esc(g.home_team) + '</div><div class="g-kick">' +
        kick(g) + '</div></div><span class="v ' + g.verdict + '">' + (g.state === "final" ? "Final" : "In progress") + "</span></div>" + score +
        '<div class="g-model">Model had ' + esc(g.away_team) + " " + num1(g.proj_away) + ", " + esc(g.home_team) + " " + num1(g.proj_home) + "</div>";
      slate.appendChild(el);
      return;
    }
    const bl = g.book_lines || [];
    const A = g.away_team, H = g.home_team;
    const isPlay = ["BET", "SMALL", "H2H"].includes(g.verdict);
    // rows: [label, value(book) -> {num, price}, better(a, b) -> true if a beats b, mine?]
    const rows = [
      {lab: A, link: "away_spread_link", get: (b) => b.home_spread == null ? null : {n: -b.home_spread, p: b.away_spread_odds, txt: sp(-b.home_spread)}, mine: isPlay && g.bet_side === A},
      {lab: H, link: "home_spread_link", get: (b) => b.home_spread == null ? null : {n: b.home_spread, p: b.home_spread_odds, txt: sp(b.home_spread)}, mine: isPlay && g.bet_side === H},
      {lab: "Over", link: "over_link", get: (b) => b.total == null ? null : {n: -b.total, p: b.over_odds, txt: "O " + b.total}},
      {lab: "Under", link: "under_link", get: (b) => b.total == null ? null : {n: b.total, p: b.under_odds, txt: "U " + b.total}},
      {lab: A, link: "away_ml_link", get: (b) => b.away_ml == null ? null : {n: 0, p: b.away_ml, txt: ml(b.away_ml), noPrice: 1}},
      {lab: H, link: "home_ml_link", get: (b) => b.home_ml == null ? null : {n: 0, p: b.home_ml, txt: ml(b.home_ml), noPrice: 1}},
    ];
    const SHORT = {FanDuel: "FD", DraftKings: "DK", BetMGM: "MGM", Caesars: "CZR", Fanatics: "FAN", bet365: "365"};
    const head = (b) => '<span class="full">' + esc(b.book) + '</span><span class="abbr">' + esc(SHORT[b.book] || b.book) + "</span>";
    let tbl = '<div class="oddswrap"><table class="odds' + (bl.length < 3 ? " few" : "") + '"><thead><tr><th></th>' + bl.map((b) => '<th title="' + esc(b.book) + '">' + head(b) + "</th>").join("") + "</tr></thead><tbody>";
    const caps = {0: "Spread", 2: "Total", 4: "Moneyline"};
    rows.forEach((r, ri) => {
      if (caps[ri]) tbl += '<tr class="cap"><th colspan="' + (bl.length + 1) + '">' + caps[ri] + "</th></tr>";
      const vals = bl.map(r.get);
      let best = -1;
      if (bl.length > 1) vals.forEach((v, i) => { if (v && (best < 0 || v.n > vals[best].n || (v.n === vals[best].n && pay(v.p) > pay(vals[best].p)))) best = i; });
      const tie = best >= 0 && vals.every((v) => v && v.n === vals[best].n && v.p === vals[best].p);
      tbl += "<tr" + (r.mine ? ' class="mine"' : "") + '><th scope="row">' + esc(r.lab) + "</th>" + vals.map((v, i) => {
        if (!v) return '<td class="none">–</td>';
        const hi = r.mine ? bl[i].book === g.bet_book : i === best && !tie;
        const inner = (v.noPrice ? v.txt : v.txt + "<small>" + ml(v.p) + "</small>");
        const link = bl[i][r.link];
        return '<td class="' + (hi ? "best" : "") + '">' + (link
          ? '<a class="t" href="' + esc(link) + '"' + (bl[i].book === "FanDuel" ? "" : ' target="_blank"') + ' rel="noopener" title="Open in ' + esc(bl[i].book) + '">' + inner + "</a>"
          : '<span class="t">' + inner + "</span>") + "</td>";
      }).join("") + "</tr>";
    });
    tbl += '</tbody></table></div><div class="swipe' + (bl.length > 4 ? " on" : "") + '">Swipe for more books &rarr;</div>';
    const fav = g.model_home_spread == null ? "" : (g.model_home_spread <= 0 ? H + " " + sp(g.model_home_spread) : A + " " + sp(-g.model_home_spread));
    const el = document.createElement("article");
    el.className = "game" + (isPlay ? " has-play" : "");
    el.innerHTML = '<div class="g-top"><div><div class="g-title">' + logo(A, 20) + esc(A) + " @ " + logo(H, 20) + esc(H) + '</div><div class="g-kick">' + kick(g) + "</div></div>" +
      '<span class="v ' + g.verdict + '">' + g.verdict + "</span></div>" +
      (bl.length ? tbl : '<div class="note">No lines posted yet.</div>') +
      '<div class="g-model">Model: <b>' + esc(A) + " " + num1(g.proj_away) + ", " + esc(H) + " " + num1(g.proj_home) + "</b> · line " + esc(fav) +
      " · total " + num1(g.model_total) + " · " + esc(g.winner) + " wins " + pct0(g.win_pct) + "</div>" +
      (isPlay
        ? '<div class="g-play">&#10003; Play: ' + esc(g.bet_side) + " " + sp(g.bet_line) + " " + ml(g.bet_odds) + " at " + esc(g.bet_book) + " · " + units(g.stake_units) + "</div>"
        : g.verdict === "WATCH"
          ? '<div class="g-play none">Watch: bet ' + esc(g.spread_pick) + " only at " + sp(g.bet_at) + " or better</div>"
          : '<div class="g-play none">No play</div>');
    slate.appendChild(el);
  });

  // Generic sortable table
  function table(id, cols, rows, initial) {
    const t = $(id);
    let key = initial ? initial[0] : null, dir = initial ? initial[1] : -1, data = rows.slice();
    function render() {
      if (key) {
        const c = cols.find((c) => c.k === key);
        const val = c.sort || ((r) => r[key]);
        data.sort((a, b) => {
          const x = val(a), y = val(b);
          if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
          return (typeof x === "string" ? x.localeCompare(y) : x - y) * dir;
        });
      }
      t.innerHTML = "<thead><tr>" + cols.map((c) =>
        '<th scope="col" tabindex="0" data-k="' + c.k + '"' + (c.l ? ' class="l"' : "") +
        (c.k === key ? ' aria-sort="' + (dir > 0 ? "ascending" : "descending") + '"' : "") + ">" + c.h + "</th>").join("") +
        "</tr></thead><tbody>" + data.map((r) => "<tr>" + cols.map((c) => {
          const cls = [c.l ? "l" : "", c.cls ? c.cls(r) : ""].join(" ").trim();
          return "<td" + (cls ? ' class="' + cls + '"' : "") + ">" + (c.f ? c.f(r) : esc(r[c.k])) + "</td>";
        }).join("") + "</tr>").join("") + "</tbody>";
      t.querySelectorAll("th").forEach((th) => {
        const go = () => { const k = th.dataset.k; dir = key === k ? -dir : (cols.find((c) => c.k === k).l ? 1 : -1); key = k; render(); };
        th.addEventListener("click", go);
        th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
      });
    }
    render();
    return { set(rowsNew) { data = rowsNew.slice(); render(); } };
  }
  const verdictOrder = {BET: 4, H2H: 3, SMALL: 2, WATCH: 1, PASS: 0};
  const vpill = (r) => '<span class="v ' + r.verdict + '">' + r.verdict + "</span>";

  // Teaser legs (the recommended pairings are listed with the model plays)
  if (!D.teasers.length) {
    $("tz-note").textContent = "No qualifying teaser legs this week (underdogs +1.5 to +2.5, or favorites -7.5 to -8.5 the model likes).";
  } else {
    table("t-tz", [
      {k: "team", h: "Leg", l: 1, f: (r) => esc(r.team) + " · " + esc(r.matchup)},
      {k: "grade", h: "Grade", l: 1},
      {k: "line", h: "Line", f: (r) => sp(r.line)},
      {k: "teased_line", h: "Teased to", f: (r) => sp(r.teased_line), cls: () => "strong"},
      {k: "book", h: "Best number", l: 1},
      {k: "total", h: "Total", f: (r) => r.total == null ? "–" : r.total},
      {k: "p_leg", h: "Leg win % (realistic)", f: (r) => pct(r.p_leg)},
    ], D.teasers, ["p_leg", -1]);
  }

  // Watch list
  const watch = D.games.filter((g) => g.verdict === "WATCH");
  if (!watch.length) $("watch-sec").hidden = true;
  else table("t-watch", [
    {k: "matchup", h: "Game", l: 1},
    {k: "spread_pick", h: "Side", l: 1},
    {k: "pick_line", h: "Line now", f: (r) => sp(r.pick_line)},
    {k: "bet_at", h: "Bet at", f: (r) => '<span class="v WATCH">' + esc(r.spread_pick) + " " + sp(r.bet_at) + "</span>"},
    {k: "p_cover_raw", h: "Model cover", f: (r) => pct(r.p_cover_raw)},
    {k: "p_cover_blend", h: "Realistic cover", f: (r) => pct(r.p_cover_blend)},
    {k: "spread_ev", h: "EV now", f: (r) => ev(r.spread_ev), cls: (r) => r.spread_ev > 0 ? "pos" : "neg"},
  ], watch, ["p_cover_blend", -1]);

  // All games
  const gameCols = [
    {k: "matchup", h: "Game", l: 1},
    {k: "gameday", h: "Date", l: 1},
    {k: "verdict", h: "Call", l: 1, f: vpill, sort: (r) => verdictOrder[r.verdict] * 10 + (r.stake_units || 0)},
    {k: "bet_side", h: "Side (best price)", l: 1, f: (r) => esc(r.bet_side) + " " + sp(r.bet_line) + " " + ml(r.bet_odds) + (r.bet_book ? " · " + esc(r.bet_book) : ""), cls: (r) => r.verdict === "BET" || r.verdict === "H2H" ? "strong" : ""},
    {k: "stake_units", h: "Stake", f: (r) => r.stake_units ? units(r.stake_units) : "–"},
    {k: "proj", h: "Projected score", l: 1, f: (r) => esc(r.away_team) + " " + num1(r.proj_away) + " – " + esc(r.home_team) + " " + num1(r.proj_home), sort: (r) => r.proj_home - r.proj_away},
    {k: "model_home_spread", h: "Model line", f: (r) => esc(r.home_team) + " " + sp(r.model_home_spread)},
    {k: "home_spread", h: "Consensus", f: (r) => esc(r.home_team) + " " + sp(r.home_spread)},
    {k: "spread_edge", h: "Edge", f: (r) => sgn1(r.spread_edge), sort: (r) => r.spread_edge == null ? null : Math.abs(r.spread_edge)},
    {k: "p_cover_raw", h: "Model cover", f: (r) => pct(r.p_cover_raw)},
    {k: "p_cover_blend", h: "Realistic cover", f: (r) => pct(r.p_cover_blend)},
    {k: "spread_ev", h: "EV / unit", f: (r) => ev(r.spread_ev), cls: (r) => r.spread_ev > 0 ? "pos" : "neg"},
    {k: "bet_down_to", h: "Good down to", f: (r) => r.bet_down_to == null ? (r.bet_at == null ? "–" : "bet at " + sp(r.bet_at)) : sp(r.bet_down_to)},
    {k: "n_books", h: "Every book", l: 1, f: booksCell},
    {k: "reason", h: "Why", l: 1},
  ];
  const gt = table("t-games", gameCols, D.games, ["verdict", -1]);
  const states = ["All", "BET", "H2H", "SMALL", "WATCH", "PASS"];
  states.forEach((s) => {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = s === "All" ? "All games" : s; b.setAttribute("aria-pressed", s === "All");
    b.addEventListener("click", () => {
      $("vfilter").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b));
      gt.set(s === "All" ? D.games : D.games.filter((g) => g.verdict === s));
    });
    $("vfilter").appendChild(b);
  });

  // Winners
  table("t-win", [
    {k: "matchup", h: "Game", l: 1},
    {k: "winner", h: "Winner", l: 1, cls: () => "strong"},
    {k: "win_pct", h: "Win %", f: (r) => pct0(r.win_pct)},
    {k: "proj", h: "Projected score", l: 1, f: (r) => esc(r.away_team) + " " + num1(r.proj_away) + " – " + esc(r.home_team) + " " + num1(r.proj_home), sort: (r) => r.proj_home - r.proj_away},
  ], D.games, ["win_pct", -1]);

  // Totals
  table("t-tot", [
    {k: "matchup", h: "Game", l: 1},
    {k: "total_pick", h: "Lean", l: 1},
    {k: "total", h: "Line"},
    {k: "model_total", h: "Model", f: (r) => num1(r.model_total)},
    {k: "tedge", h: "Edge", f: (r) => sgn1(r.model_total - r.total), sort: (r) => Math.abs(r.model_total - r.total)},
    {k: "p_total_raw", h: "Model %", f: (r) => pct(r.p_total_raw)},
  ], D.games, ["tedge", -1]);

  // Injuries + QBs
  const injText = {
    monday: "Not applied to projections on Monday: these long-term absences are already in the ratings. Re-run Thursday or Friday to apply this week's report.",
    thursday: "Applied from the Thursday practice report.", final: "Applied from the final report with game statuses."};
  const qbs = D.games.filter((g) => g.qb_note).map((g) => g.qb_note);
  $("inj-note").textContent = (qbs.length ? "QB changes: " + qbs.join("; ") + ". " : "") + (injText[D.inj_mode] || "No injury report available.");
  if (D.injuries.length) table("t-inj", [
    {k: "team", h: "Team", l: 1}, {k: "player", h: "Player", l: 1}, {k: "position", h: "Pos", l: 1},
    {k: "status", h: "Status", l: 1, f: (r) => esc(String(r.status).replace("none_missed", "missed last game (likely IR)").replace(/^lag_/, "last week: ").replace(/^prac_/, "practice: ").replace(/_/g, " "))},
    {k: "p_miss", h: "Miss %", f: (r) => pct0(r.p_miss)}, {k: "usual", h: "Usual snaps", f: (r) => pct0(r.usual)},
  ], D.injuries, ["usual", -1]);

  // Ratings
  table("t-rat", [
    {k: "rank", h: "#"}, {k: "team", h: "Team", l: 1}, {k: "net_pts", h: "Net", f: (r) => sgn1(r.net_pts)},
    {k: "off_pts", h: "Offense", f: (r) => sgn1(r.off_pts)}, {k: "def_pts", h: "Defense", f: (r) => sgn1(r.def_pts)},
    {k: "off_epa", h: "Off EPA/play", f: (r) => r.off_epa == null ? "–" : r.off_epa.toFixed(3)},
    {k: "def_epa", h: "Def EPA/play", f: (r) => r.def_epa == null ? "–" : r.def_epa.toFixed(3)},
    {k: "st", h: "Special teams", f: (r) => sgn1(r.st)}, {k: "elo", h: "Elo", f: (r) => r.elo == null ? "–" : Math.round(r.elo)},
    {k: "market", h: "Market", f: (r) => sgn1(r.market)},
  ], D.ratings, ["net_pts", -1]);
})();
</script>
"""


def build_record_html(summary: dict, season: int) -> str:
    """The Record page: the model's graded plays (spreads and teasers) in the same theme as the bet card."""
    style = TEMPLATE[TEMPLATE.index("<style>"):TEMPLATE.index("</style>") + len("</style>")]
    payload = json.dumps({"season": season, **summary}, default=str).replace("</", "<\\/")
    return (RECORD_TEMPLATE.replace("__STYLE__", style).replace("__HEAD__", HEAD).replace("__BRAND__", BRAND)
            .replace("__FOLLOW__", FOLLOW).replace("__DATA__", payload))


RECORD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Off The Glass · NFL Record</title>
<meta property="og:title" content="Off The Glass · NFL Record">
__HEAD__
__STYLE__
<style>
/* Same layout as the CBB record page (barrca12-rgb/CBB cbb/record_page.py) */
.rhead { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px }
.sub { color: var(--muted); margin-top: 2px }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); border: 1px solid var(--line); border-top: 3px solid var(--ink); border-radius: 10px;
  background: var(--surface); overflow: hidden; box-shadow: 0 1px 2px rgba(10,44,73,.06) }
.kpi { padding: 12px 16px; border-right: 1px solid var(--line) } .kpi:last-child { border-right: 0 }
.kpi .k { color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em }
.kpi .kv { font-size: 21px; font-weight: 800; margin-top: 3px; letter-spacing: -.01em }
@media (max-width: 860px) { .kpi { border-right: 0; border-bottom: 1px solid var(--line) } .kpi:last-child { border-bottom: 0 } }
</style>
<header class="top"><div class="topin">
  __BRAND__
  <nav class="tabs" aria-label="Sport"><a href="./" aria-current="page">NFL</a><a href="https://barrca12-rgb.github.io/NBA/">NBA</a><a href="https://barrca12-rgb.github.io/CBB/"><span class="hide-sm">College Basketball</span><span class="show-sm">CBB</span></a></nav>
</div></header>
<div class="wrap">
  <div class="rhead"><div><div class="eyebrow" id="eyebrow"></div><h1>NFL Record</h1>
    <div class="sub">Every play the model recommended, at the logged number and price, graded against the final score.</div></div>
    <a class="pagebtn" href="./">&larr; Card</a></div>
  <div class="kpis" id="kpis"></div>
  <section class="panel"><div class="panel-h"><h2>Summary</h2><span class="note">Settled plays. CLV = points the line moved our way from the logged number to the close. Break-even at -110 is 52.4%.</span></div>
    <div class="scroll"><table class="ct" id="t-type"></table></div></section>
  <section class="panel"><div class="panel-h"><h2>By week</h2></div><div class="scroll"><table class="ct" id="t-week"></table></div></section>
  <section class="panel"><div class="panel-h"><h2>Bets placed</h2><span class="note">Newest first.</span></div><div class="scroll"><table class="ct" id="t-plays"></table></div></section>
  __FOLLOW__
</div>
<script id="record-data" type="application/json">__DATA__</script>
<script>
(function () {
  const R = JSON.parse(document.getElementById("record-data").textContent);
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const u = (x) => x == null ? "–" : (x > 0 ? "+" : "") + x.toFixed(2);
  const pc = (x) => x == null ? "–" : (x > 0 ? "+" : "") + (x * 100).toFixed(1) + "%";
  const rec = (a) => a.w + "–" + a.l + (a.p ? "–" + a.p : "");
  const wp = (a) => a.w + a.l ? ((a.w / (a.w + a.l)) * 100).toFixed(1) + "%" : "–";
  const ml = (x) => x == null ? "" : (x > 0 ? "+" : "") + Math.round(x);
  const cls = (x) => x > 0 ? "pos" : x < 0 ? "neg" : "";
  const LOGO = {LA: "lar", WAS: "wsh"};
  const logo = (t, px) => { if (!t) return ""; const a = LOGO[t] || String(t).toLowerCase();
    return "<picture class='tl'><source media='(prefers-color-scheme: dark)' srcset='https://a.espncdn.com/i/teamlogos/nfl/500-dark/" + esc(a) +
      ".png'><img src='https://a.espncdn.com/i/teamlogos/nfl/500/" + esc(a) + ".png' alt='' loading='lazy' width='" + px + "' height='" + px +
      "' onerror=\"this.closest('picture').remove()\"></picture>"; };
  $("eyebrow").textContent = R.season + " NFL season";
  const o = R.overall;
  $("kpis").innerHTML = [["Record", rec(o)], ["Win %", wp(o)], ["Units", "<span class='" + cls(o.units) + "'>" + u(o.units) + "</span>"], ["ROI", pc(o.roi)],
    ["Avg CLV", o.clv == null ? "–" : (o.clv > 0 ? "+" : "") + o.clv.toFixed(2) + " pts"], ["Open plays", String(o.pending)]]
    .map(([k, v]) => "<div class='kpi'><div class='k'>" + k + "</div><div class='kv'>" + v + "</div></div>").join("");
  const head = (first) => "<thead><tr><th>" + first + "</th><th class='r'>Record</th><th class='r'>Win %</th><th class='r'>Units</th><th class='r'>ROI</th><th class='r'>Avg CLV</th><th class='r'>Open</th></tr></thead>";
  const row = (label, a) => "<tr><td class='b'>" + esc(label) + "</td><td class='r'>" + rec(a) + "</td><td class='r'>" + wp(a) + "</td><td class='r " + cls(a.units) + "'>" + u(a.units) +
    "</td><td class='r'>" + pc(a.roi) + "</td><td class='r'>" + (a.clv == null ? "–" : (a.clv > 0 ? "+" : "") + a.clv.toFixed(2)) + "</td><td class='r m'>" + a.pending + "</td></tr>";
  const TYPE = {VALUE: "Spreads", SMALL: "Spreads (small)", H2H: "H2H angle", TEASER: "Teasers"};
  $("t-type").innerHTML = head("Bets") + "<tbody>" + row("All bets", o) + Object.entries(R.by_type).map(([k, a]) => row(TYPE[k] || k, a)).join("") + "</tbody>";
  $("t-week").innerHTML = head("Week") + "<tbody>" + Object.entries(R.by_week).sort((a, b) => b[0].localeCompare(a[0])).map(([k, a]) => row("Week " + Number(k.split("-")[1]), a)).join("") + "</tbody>";
  const plays = R.plays.slice().sort((a, b) => String(b.kickoff).localeCompare(String(a.kickoff)));
  const now = new Date();
  const game = (p) => { const m = String(p.matchup || "").split(" @ "); return m.length === 2 ? logo(m[0], 16) + esc(m[0]) + " @ " + logo(m[1], 16) + esc(m[1]) : esc(p.matchup || "Teaser"); };
  $("t-plays").innerHTML = plays.length ? "<thead><tr><th>Date</th><th>Game</th><th>Play</th><th>Book</th><th class='r'>Close</th><th class='r'>CLV</th><th>Result</th><th class='r'>Units</th><th class='hide-sm'>Type</th></tr></thead><tbody>" +
    plays.map((p) => {
      const started = p.kickoff && new Date(p.kickoff) <= now;
      const res = p.result === "W" ? "<span class='pos b'>Win</span>" : p.result === "L" ? "<span class='neg b'>Loss</span>" : p.result === "P" ? "Push" : (started ? "Live" : "Open");
      const d = p.kickoff ? new Date(p.kickoff).toLocaleDateString("en-US", {month: "short", day: "numeric", timeZone: "America/New_York"}) : "";
      return "<tr><td class='m'>" + esc(d) + "</td><td>" + game(p) + "</td><td class='b'>" + (p.team ? logo(p.team, 16) : "") + esc(p.selection) + "</td><td>" + esc(p.book || "–") +
        " <span class='m'>" + ml(p.odds) + "</span></td><td class='r'>" + (p.close_line == null ? "–" : (p.close_line > 0 ? "+" : "") + p.close_line) + "</td><td class='r'>" +
        (p.clv_pts == null ? "–" : (p.clv_pts > 0 ? "+" : "") + p.clv_pts.toFixed(1)) + "</td><td>" + res + "</td><td class='r " + cls(p.profit) + "'>" +
        (p.profit == null ? "–" : u(p.profit)) + "</td><td class='m hide-sm'>" + esc(TYPE[p.angle] || (String(p.angle).startsWith("TEASER") ? "Teaser " + String(p.angle).slice(7) : p.angle || p.kind)) + " · " + p.stake + "u</td></tr>";
    }).join("") + "</tbody>" : "<tbody><tr><td class='empty'>No plays yet.</td></tr></tbody>";
  // phones: each row becomes a small card, each cell labeled with its column header
  document.querySelectorAll("table.ct").forEach((t) => { const hs = [...t.querySelectorAll("thead th")].map((h) => h.textContent.trim());
    t.classList.add("stack"); t.querySelectorAll("tbody tr").forEach((r) => [...r.children].forEach((c, k) => { if (hs[k]) c.dataset.l = hs[k]; })); });
})();
</script>
"""
