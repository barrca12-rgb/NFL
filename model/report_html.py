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
FOLLOW = ('<footer><div><a class="follow" href="https://x.com/' + X_HANDLE + '" target="_blank" rel="noopener">'
          '<img src="assets/otg-logo-512.png" alt="">Follow @' + X_HANDLE + ' on X for every play</a></div></footer>')


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
               tz: pd.DataFrame | None = None, tpairs: pd.DataFrame | None = None, record: dict | None = None) -> str:
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
        "record": json.loads(json.dumps(record, default=str)) if record else None,
    }
    payload = json.dumps(data).replace("</", "<\\/")
    title = f"Off The Glass · NFL · Week {week}"
    return (TEMPLATE.replace("__HEAD__", HEAD).replace("__BRAND__", BRAND).replace("__X__", X_HANDLE)
            .replace("__TITLE__", html.escape(title)).replace("__DATA__", payload))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<meta property="og:title" content="__TITLE__">
__HEAD__
<style>
/* Same stylesheet as the CBB card (barrca12-rgb/CBB cbb/report_html.py); keep them in step */
:root {
  --bg: #fbf6ea; --surface: #ffffff; --sunken: #f7f0e1; --line: #ebe1cd; --line-strong: #d9cbb0;
  --text: #0a2c49; --muted: #56627a; --faint: #8a93a3;
  --accent: #b84a08; --pos: #0f7b4b; --pos-bg: #e6f4ed; --neg: #c2352b; --warn: #8a5a00; --warn-bg: #fbf0d9;
  --brand: #fd6c0f; --bar: #0a2c49; --bar-text: #fbf6ea; --bar-muted: #b7c3d3;
  --font: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --display: "Graduate", "Rockwell", Georgia, serif;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #08090b; --surface: #12151a; --sunken: #181c22; --line: #242a33; --line-strong: #333b46;
    --text: #f6eedd; --muted: #a8b0bc; --faint: #6f7886;
    --accent: #ff8a3d; --pos: #3ccf8e; --pos-bg: #12301f; --neg: #ff6b5e; --warn: #f0b44a; --warn-bg: #33270f;
    --brand: #fd6c0f; --bar: #000000; --bar-text: #f6eedd; --bar-muted: #9aa2ae;
    color-scheme: dark;
  }
}
* { box-sizing: border-box }
.show-sm { display: none }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 var(--font); font-feature-settings: "tnum" 1, "cv11" 1; -webkit-font-smoothing: antialiased }
a { color: var(--accent); text-decoration: none } a:hover { text-decoration: underline }
.wrap { max-width: 1320px; margin: 0 auto; padding: 0 16px }

/* Top bar */
.top { background: var(--bar); border-bottom: 3px solid var(--brand); position: sticky; top: 0; z-index: 5 }
.top .wrap { display: flex; align-items: center; gap: 22px; height: 58px }
.brand { display: flex; align-items: center; gap: 10px; white-space: nowrap; color: var(--bar-text); text-decoration: none }
.brand:hover { text-decoration: none }
.brand img { width: 34px; height: 34px; border-radius: 7px; display: block }
.wordmark { font-family: var(--display); font-size: 19px; letter-spacing: .02em; text-transform: uppercase; color: var(--bar-text); line-height: 1 }
.tabs { display: flex; gap: 2px; height: 100% }
.tabs a { display: flex; align-items: center; padding: 0 12px; color: var(--bar-muted); font-weight: 600; font-size: 13px; letter-spacing: .04em;
  text-transform: uppercase; border-bottom: 3px solid transparent; margin-bottom: -3px; text-decoration: none }
.tabs a[hidden] { display: none }
.tabs a:hover { color: var(--bar-text); text-decoration: none }
.tabs a[aria-current="page"] { color: var(--bar-text); border-bottom-color: var(--brand) }
.status { margin-left: auto; display: flex; align-items: center; gap: 12px; color: var(--bar-muted); font-size: 13px; min-width: 0 }
.status a { color: var(--bar-text) }
#upd-btn { background: transparent; color: var(--bar-text); border-color: rgba(251,246,234,.35) } #upd-btn:hover { background: rgba(251,246,234,.1) }
.status span { white-space: nowrap; overflow: hidden; text-overflow: ellipsis }
button { font: inherit; font-weight: 600; font-size: 13px; color: var(--text); background: var(--surface); border: 1px solid var(--line-strong);
  border-radius: 6px; padding: 6px 12px; cursor: pointer } button:hover { background: var(--sunken) } button:disabled { opacity: .5; cursor: progress }

/* Page head */
.head { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 12px; padding: 24px 0 16px }
h1 { font-family: var(--display); font-weight: 400; font-size: 26px; letter-spacing: .01em; margin: 0; line-height: 1.15 }
.sub { color: var(--muted); margin-top: 2px }
.days { display: flex; gap: 4px; flex-wrap: wrap }
.days a, .pagebtn { padding: 5px 12px; border-radius: 6px; color: var(--text); font-weight: 600; font-size: 13px; text-decoration: none;
  border: 1px solid var(--line-strong); background: var(--surface); white-space: nowrap }
.days a:hover, .pagebtn:hover { background: var(--sunken); text-decoration: none }
.days a[aria-current="page"] { color: var(--surface); border-color: var(--text); background: var(--text) }
.notice { border: 1px solid var(--line); background: var(--warn-bg); color: var(--warn); border-radius: 8px; padding: 10px 12px; margin-bottom: 16px; font-size: 13px }

/* KPI strip */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); border: 1px solid var(--line); border-top: 3px solid var(--text); border-radius: 10px; background: var(--surface); margin-bottom: 24px; overflow: hidden; box-shadow: 0 1px 2px rgba(10,44,73,.06) }
.kpi { padding: 12px 16px; border-right: 1px solid var(--line) } .kpi:last-child { border-right: 0 }
.kpi .k { color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em } .kpi .v { font-size: 21px; font-weight: 800; margin-top: 3px; letter-spacing: -.01em }
.kpi .v small { font-size: 12px; color: var(--muted); font-weight: 500; margin-left: 4px }

/* Panels and tables */
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; margin-bottom: 24px; overflow: hidden; box-shadow: 0 1px 2px rgba(10,44,73,.06) }
.panel-h { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; padding: 12px 16px; border-bottom: 1px solid var(--line) }
.panel-h h2 { font-size: 13px; font-weight: 800; margin: 0; text-transform: uppercase; letter-spacing: .07em; display: flex; align-items: center; gap: 8px }
.panel-h h2::before { content: ""; width: 4px; height: 14px; background: var(--brand); border-radius: 2px } .panel-h .note { color: var(--muted); font-size: 12px }
.scroll { overflow-x: auto }
table { width: 100%; border-collapse: collapse; font-size: 13px }
th { text-align: left; font-weight: 700; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); padding: 8px 12px; border-bottom: 1px solid var(--line); background: var(--sunken); white-space: nowrap }
td { padding: 9px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; vertical-align: middle }
tbody tr:last-child td { border-bottom: 0 }
.r { text-align: right } .c { text-align: center } .m { color: var(--muted) } .f { color: var(--faint) }
.pos { color: var(--pos) } .neg { color: var(--neg) } .b { font-weight: 600 }
.empty { padding: 28px 16px; color: var(--muted); text-align: center }
.betlink { font-weight: 600; font-size: 12px; border: 1px solid var(--line-strong); border-radius: 6px; padding: 4px 10px; color: var(--text); display: inline-block }
.betlink:hover { background: var(--sunken); text-decoration: none }
.tag { display: inline-block; font-size: 11px; font-weight: 600; padding: 1px 6px; border-radius: 4px; border: 1px solid var(--line-strong); color: var(--muted) }
.tag.bet { color: #fff; border-color: var(--brand); background: var(--brand) }

/* Segmented control */
.seg { display: inline-flex; border: 1px solid var(--line-strong); border-radius: 6px; overflow: hidden }
.seg button { border: 0; border-radius: 0; padding: 5px 12px; color: var(--muted); font-weight: 500; background: transparent }
.seg button + button { border-left: 1px solid var(--line-strong) }
.seg button[aria-pressed="true"] { background: var(--text); color: var(--surface) }
.ctrls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center }
label.chk { display: inline-flex; gap: 6px; align-items: center; color: var(--muted); font-size: 13px; cursor: pointer }

/* Odds board: two rows per game (away, home) */
.board td { padding-top: 6px; padding-bottom: 6px }
.board tbody.g tr:first-child td { padding-top: 10px } .board tbody.g tr:last-child td { padding-bottom: 10px; border-bottom: 1px solid var(--line) }
.board tbody.g tr:first-child td { border-bottom: 0 }
.board tbody.g:last-child tr:last-child td { border-bottom: 0 }
.board .team { font-weight: 500 }
.inj { color: var(--warn); font-size: 11px; font-weight: 500; white-space: normal; max-width: 340px; margin-top: 1px } .board .tm { width: 72px; color: var(--muted); font-size: 12px }
.board .pts { color: var(--muted); width: 40px }
.odd { display: inline-flex; flex-direction: column; align-items: flex-end; min-width: 56px; padding: 3px 6px; border-radius: 6px; line-height: 1.25; color: var(--text); text-decoration: none }
.odd small { color: var(--faint); font-size: 11px }
a.odd:hover { background: var(--sunken); text-decoration: none }
.odd.best { background: var(--pos-bg); color: var(--pos) } .odd.best small { color: var(--pos) }
.na { color: var(--faint) }
.board th.bk, .board td.bk { text-align: right; width: 84px; padding-left: 6px; padding-right: 6px }
.board td.edge { white-space: normal; min-width: 110px; max-width: 170px } .tag { white-space: normal }
.bestc { display: none; text-align: right } .tm-sm { display: none; color: var(--faint); font-size: 11px }
@media (max-width: 1100px) { .bk-minor, .hide-md { display: none } .board .bestc { display: table-cell } }
.bestc small { display: block; color: var(--faint); font-size: 11px }
.board tbody.g.has-bet { background: linear-gradient(90deg, var(--brand) 0 3px, transparent 3px) }

details.panel > summary { list-style: none; cursor: pointer; padding: 12px 16px; font-weight: 600; font-size: 15px; display: flex; justify-content: space-between }
details.panel > summary::after { content: "Show"; color: var(--accent); font-size: 13px; font-weight: 500 }
details.panel[open] > summary::after { content: "Hide" }
details.panel[open] > summary { border-bottom: 1px solid var(--line) }
footer { color: var(--muted); font-size: 12px; padding: 8px 0 40px; max-width: 80ch }
.follow { display: inline-flex; align-items: center; gap: 8px; margin-top: 12px; font-weight: 700; color: var(--text) }
.follow img { width: 22px; height: 22px; border-radius: 5px }
.tl img { width: auto; vertical-align: -3px; margin-right: 5px; object-fit: contain }
.pick .tl img { vertical-align: -5px; margin-right: 6px }
.hero { display: block; height: 120px; width: auto; max-width: 100%; object-fit: contain; border-radius: 10px; margin-top: 18px; border: 1px solid var(--line) }
@media (max-width: 720px) { .hero { height: auto; width: 100% } }

@media (max-width: 860px) {
  .top .wrap { gap: 10px } .status span { display: none }
  .wordmark { display: none } .tabs a { padding: 0 8px; font-size: 12px }
  h1 { font-size: 21px }
  .kpi { border-right: 0; border-bottom: 1px solid var(--line) } .kpi:last-child { border-bottom: 0 }
  .hide-sm { display: none } .show-sm { display: inline }
  td, th { padding-left: 10px; padding-right: 10px }
  #t-plays td:first-child { white-space: normal; min-width: 150px }
  .board .bk, .board .pts { display: none } .bestc { display: table-cell }
  .board .tm { width: 52px; font-size: 11px } .board td.edge { min-width: 0; max-width: 104px }
  .board td, .board th { padding-left: 6px; padding-right: 6px } .board .team { white-space: normal; min-width: 0 }
  .board .tag { font-size: 10px; padding: 1px 4px } .board .tl img { width: 14px; height: 14px; margin-right: 4px }
  .board .tm { display: none } .tm-sm { display: block } .board td.edge { max-width: 104px }
  table.stack thead { display: none }
  table.stack tbody tr { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px 12px; padding: 12px; border-bottom: 1px solid var(--line) }
  table.stack tbody tr:last-child { border-bottom: 0 }
  table.stack td { display: block; border: 0; padding: 0; white-space: normal; text-align: left; min-width: 0 }
  table.stack td:first-child { grid-column: 1 / -1; min-width: 0 }
  table.stack td[data-l]::before { content: attr(data-l); display: block; font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .05em; color: var(--faint); margin-bottom: 1px }
  table.stack td:first-child::before { display: none }
  table.stack td.hide-sm, table.stack td:empty { display: none }
  table.stack td.empty { grid-column: 1 / -1; text-align: center }
}
</style>

<header class="top"><div class="wrap">
  __BRAND__
  <nav class="tabs" aria-label="Sport"><a href="./" aria-current="page">NFL</a><a href="https://barrca12-rgb.github.io/NBA/">NBA</a><a href="https://barrca12-rgb.github.io/CBB/"><span class="hide-sm">College Basketball</span><span class="show-sm">CBB</span></a></nav>
  <div class="status"><span id="upd-status"></span><button id="upd-btn" type="button">Update</button></div>
</div></header>

<main class="wrap">
  <div class="head">
    <div><h1 id="h1"></h1><div class="sub" id="sub"></div></div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><nav class="days" id="days" aria-label="Weeks"></nav><a class="pagebtn" href="record.html">Record</a></div>
  </div>

  <div class="kpis" id="kpis"></div>

  <section class="panel" aria-labelledby="plays-h">
    <div class="panel-h"><h2 id="plays-h">Plays</h2><span class="note" id="plays-note"></span></div>
    <div class="scroll"><table id="t-plays"></table></div>
  </section>

  <section class="panel" aria-labelledby="board-h">
    <div class="panel-h"><h2 id="board-h">Odds board</h2>
      <div class="ctrls">
        <div class="seg" role="group" aria-label="Market"><button data-m="spread" aria-pressed="true">Spread</button><button data-m="total">Total</button><button data-m="ml">Moneyline</button></div>
        <label class="chk"><input type="checkbox" id="only-plays"> Plays only</label>
      </div>
    </div>
    <div class="scroll"><table class="board" id="t-board"></table></div>
  </section>

  <section class="panel" aria-labelledby="watch-h" id="watch-sec">
    <div class="panel-h"><h2 id="watch-h">Watch list</h2><span class="note">Becomes a play if the line reaches the number shown.</span></div>
    <div class="scroll"><table id="t-watch"></table></div>
  </section>

  <section class="panel" aria-labelledby="rec-h" id="rec-sec" hidden>
    <div class="panel-h"><h2 id="rec-h">Record</h2><span class="note">Every play graded at the number it was logged. CLV = points better than the closing line.</span></div>
    <div class="scroll"><table id="t-rec"></table></div>
  </section>

  <details class="panel" id="recent-sec" hidden><summary>Recent results</summary><div class="scroll"><table id="t-recent"></table></div></details>
  <details class="panel" id="tz-sec"><summary>Teaser legs</summary><div class="scroll"><table id="t-tz"></table></div></details>
  <details class="panel"><summary>Injuries and quarterbacks</summary><div class="scroll"><table id="t-inj"></table></div></details>
  <details class="panel"><summary>Power ratings</summary><div class="scroll"><table id="t-rat"></table></div></details>

  <footer>
    <div><a class="follow" href="https://x.com/__X__" target="_blank" rel="noopener"><img src="assets/otg-logo-512.png" alt="">Follow @__X__ on X for every play</a></div>
  </footer>
</main>

<script id="report-data" type="application/json">__DATA__</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById("report-data").textContent);
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const sgn = (x, d = 1) => x == null ? "–" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d).replace(/\.0+$/, "");
  const ln = (x) => x == null ? "–" : Math.abs(x) < 0.05 ? "PK" : sgn(x);
  const num = (x, d = 1) => x == null ? "–" : Number(x).toFixed(d).replace(/\.0$/, "");
  const odds = (x) => x == null ? "" : (x > 0 ? "+" : "−") + Math.abs(Math.round(x));
  const pct = (x) => x == null ? "–" : (100 * x).toFixed(1) + "%";
  const pay = (o) => o == null ? -1 : (o > 0 ? o / 100 : 100 / -o);
  const units = (u) => u == null ? "–" : num(u, 2) + "u";
  const stack = (id) => { const t = $(id); if (!t) return; const hs = [...t.querySelectorAll("thead th")].map((h) => h.textContent.trim());
    t.classList.add("stack"); t.querySelectorAll("tbody tr").forEach((r) => [...r.children].forEach((c, k) => { if (hs[k]) c.dataset.l = hs[k]; })); };
  // ESPN logos by team abbreviation (nflverse LA and WAS are lar and wsh at ESPN); a logo that fails to load is removed
  const LOGO = {LA: "lar", WAS: "wsh"};
  const logo = (t, px) => { if (!t) return ""; const a = LOGO[t] || String(t).toLowerCase();
    return "<picture class='tl'><source media='(prefers-color-scheme: dark)' srcset='https://a.espncdn.com/i/teamlogos/nfl/500-dark/" + esc(a) +
      ".png'><img src='https://a.espncdn.com/i/teamlogos/nfl/500/" + esc(a) + ".png' alt='' loading='lazy' width='" + px + "' height='" + px +
      "' onerror=\"this.closest('picture').remove()\"></picture>"; };
  const matchupL = (g) => logo(g.away_team, 14) + esc(g.away_team) + " @ " + logo(g.home_team, 14) + esc(g.home_team);
  // Kickoff: "Sun Nov 1 · 1:00 PM ET" (kick) and "Sun 1:00 PM" (tip, for the board)
  const ktime = (g) => { const t = String(g.kickoff || "").split(" ")[1] || "";
    if (!/^\d{1,2}:\d{2}$/.test(t)) return ""; const [h, m] = t.split(":").map(Number); return (h % 12 || 12) + ":" + String(m).padStart(2, "0") + (h < 12 ? " AM" : " PM"); };
  const kday = (g, o) => new Date(g.gameday + "T12:00:00").toLocaleDateString("en-US", o);
  const kick = (g) => kday(g, {weekday: "short", month: "short", day: "numeric"}) + (ktime(g) ? " · " + ktime(g) + " ET" : "");
  const tip = (g) => kday(g, {weekday: "short"}) + " " + ktime(g);
  const byKick = (a, b) => String(a.gameday).localeCompare(String(b.gameday)) || String(a.kickoff).localeCompare(String(b.kickoff));
  const isPlay = (g) => ["BET", "SMALL", "H2H"].includes(g.verdict);
  const done = (g) => g.state && g.state !== "upcoming";

  // Head
  const days = D.games.map((g) => g.gameday).filter(Boolean).sort();
  const md = (d) => new Date(d + "T12:00:00").toLocaleDateString("en-US", {month: "short", day: "numeric"});
  $("h1").textContent = "Week " + D.week;
  const injLabel = {thursday: "Thursday practice report applied", final: "final injury report applied", monday: "injury report not out yet"};
  $("sub").textContent = (days.length ? md(days[0]) + (days[0] !== days[days.length - 1] ? " – " + md(days[days.length - 1]) : "") + " · " : "") +
    D.games.length + " games · " + D.season + " season · " + (injLabel[D.inj_mode] || "no injury report");
  const navWeeks = (list, cur) => {
    $("days").innerHTML = list.map((w) => '<a href="' + D.season + "_week" + String(w).padStart(2, "0") + '.html"' +
      (w === D.week ? ' aria-current="page"' : "") + ">Week " + w + "</a>").join("");
  };
  navWeeks([D.week]);
  fetch("weeks.json", {cache: "no-store"}).then((r) => r.json()).then((w) => { if (w.season === D.season && w.weeks.length) navWeeks(w.weeks.slice(-7), w.current); }).catch(() => {});

  // Plays: spreads (value, small, H2H) and teasers
  const bets = D.games.filter(isPlay);
  const also = (g) => {
    const home = g.bet_side === g.home_team;
    const L = (g.book_lines || []).filter((b) => b.home_spread != null && b.book !== g.bet_book).map((b) => ({book: b.book,
      n: home ? b.home_spread : -b.home_spread, p: home ? b.home_spread_odds : b.away_spread_odds})).sort((x, y) => y.n - x.n || pay(y.p) - pay(x.p));
    return L.length ? "Also: " + L.map((x) => x.book + " " + ln(x.n) + (x.p != null ? " (" + odds(x.p) + ")" : "")).join(" · ") : "";
  };
  const ROWS = [];
  bets.slice().sort(byKick).forEach((g) => {
    const h2h = g.verdict === "H2H", home = g.bet_side === g.home_team;
    const modelLine = g.model_home_spread == null ? null : (home ? g.model_home_spread : -g.model_home_spread);
    const edge = modelLine == null ? null : g.bet_line - modelLine;
    const kind = h2h ? "Spread · H2H angle" : "Spread" + (g.verdict === "SMALL" ? " · small" : "");
    ROWS.push("<tr><td><div class='b pick'>" + logo(g.bet_side, 20) + esc(g.bet_side + " " + ln(g.bet_line)) + "</div><div class='f' style='font-size:12px'>" + matchupL(g) + " · " + esc(kick(g)) + " · " + kind + "</div>" +
      (also(g) ? "<div class='f' style='font-size:12px'>" + esc(also(g)) + "</div>" : "") +
      (g.qb_note ? "<div class='inj'>&#9888; QB: " + esc(g.qb_note) + "</div>" : "") +
      (h2h && g.reason ? "<div class='f' style='font-size:12px;white-space:normal'>" + esc(g.reason) + "</div>" : "") + "</td><td>" +
      esc(g.bet_book || "–") + (g.bet_odds != null ? " <span class='m'>" + esc(odds(g.bet_odds)) + "</span>" : "") + "</td><td class='r m hide-sm hide-md'>" + ln(modelLine) +
      "</td><td class='r " + (edge > 0 ? "pos b" : "m") + "'>" + sgn(edge) + "</td><td class='r m hide-sm'>" + pct(h2h ? g.h2h_p_blend : g.p_cover_blend) +
      "</td><td class='r'>" + (h2h || g.spread_ev == null ? "–" : sgn(100 * g.spread_ev) + "%") + "</td><td class='r'>" + (h2h ? "–" : ln(g.bet_down_to)) +
      "</td><td class='r m hide-sm hide-md'>" + num(g.proj_away) + "–" + num(g.proj_home) + "</td><td class='r hide-sm hide-md'>" + units(g.stake_units) + "</td></tr>");
  });
  D.teaser_pairs.forEach((t) => {
    const legs = t.legs || [];
    const books = [...new Set(legs.map((l) => l.book).filter(Boolean))];
    ROWS.push("<tr><td><div class='b pick'>" + legs.map((l) => logo(l.team, 20)).join("") + esc(t.teaser) + "</div>" +
      "<div class='f' style='font-size:12px'>2-team, 6-point teaser · both legs must win · grades " + esc(t.grades) + "</div>" +
      "<div class='f' style='font-size:12px'>" + legs.map((l) => esc(l.team + " " + ln(l.line) + " → " + ln(l.teased_line))).join(" · ") + "</div></td><td>" +
      esc(books.length ? books.join(", ") : "Best teaser price") + " <span class='m'>−120</span></td><td class='r m hide-sm hide-md'>" + pct(t.p_both_model) +
      "</td><td class='r m'>–</td><td class='r m hide-sm'>" + pct(t.p_both) + "</td><td class='r'>" + (t.ev == null ? "–" : sgn(100 * t.ev) + "%") +
      "</td><td class='r'>" + legs.map((l) => ln(l.teased_line)).join(", ") + "</td><td class='r m hide-sm hide-md'>–</td><td class='r hide-sm hide-md'>" + units(t.stake_units) + "</td></tr>");
  });
  const nPlays = ROWS.length;
  const uTotal = Math.round((bets.reduce((s, g) => s + (g.stake_units || 0), 0) + D.teaser_pairs.reduce((s, t) => s + (t.stake_units || 0), 0)) * 100) / 100;
  $("plays-note").textContent = "Spreads where the realistic cover % beats the break-even of the best price, the coach H2H angle, and 6-point teasers through 3 and 7. Cover % is the realistic rate; bet only at the number shown or better (Bet to). " +
    (D.flag_text || "") + " Break-even at −110 is 52.4%.";
  $("t-plays").innerHTML = nPlays ? "<thead><tr><th>Bet</th><th>Book</th><th class='r hide-sm hide-md'>Model</th><th class='r'>Edge</th><th class='r hide-sm'>Cover %</th><th class='r'>EV</th><th class='r'>Bet to</th><th class='r hide-sm hide-md'>Proj</th><th class='r hide-sm hide-md'>Units</th></tr></thead><tbody>" + ROWS.join("") + "</tbody>"
    : "<tbody><tr><td class='empty'>No plays. No game meets the betting rule at today's lines; most weeks after Week 4 that is the right call.</td></tr></tbody>";

  // KPIs
  const R = D.record, o = R ? R.overall : null;
  const rec = (a) => a ? a.w + "–" + a.l + (a.p ? "–" + a.p : "") : "0–0";
  const wp = (a) => a && a.w + a.l ? pct(a.w / (a.w + a.l)) : "";
  $("kpis").innerHTML = [["Plays this week", nPlays + "<small>" + num(uTotal, 2) + "u</small>"], ["Record", rec(o) + "<small>" + wp(o) + "</small>"],
    ["Units", "<span class='" + (o && o.units > 0 ? "pos" : o && o.units < 0 ? "neg" : "") + "'>" + sgn(o ? o.units : 0, 2) + "</span>"],
    ["ROI", o && o.roi != null ? sgn(100 * o.roi) + "%" : "–"], ["Avg CLV", o && o.clv != null ? sgn(o.clv, 2) + "<small>pts</small>" : "–"]]
    .map(([k, v]) => "<div class='kpi'><div class='k'>" + k + "</div><div class='v'>" + v + "</div></div>").join("");

  // Odds board: two rows per game (away, home), one column per book
  const PREF = ["FanDuel", "DraftKings", "BetMGM", "Caesars", "Fanatics", "bet365", "theScore Bet"];
  const all = [...new Set(D.games.flatMap((g) => (g.book_lines || []).map((b) => b.book)))];
  const BOOKS = PREF.filter((b) => all.includes(b)).concat(all.filter((b) => !PREF.includes(b)).sort());
  const bl = (g, bk) => (g.book_lines || []).find((x) => x.book === bk);
  let market = "spread";
  const cell = (val, price, link, best) => val == null ? "<span class='na'>–</span>" :
    "<" + (link ? "a href='" + esc(link) + "' target='_blank' rel='noopener'" : "span") + " class='odd" + (best ? " best" : "") + "'>" + esc(val) +
    (price != null ? "<small>" + esc(odds(price)) + "</small>" : "") + "</" + (link ? "a" : "span") + ">";
  const best = (g, sd, i) => {
    let top = null;
    for (const x of (g.book_lines || [])) {
      let v, pr, txt;
      if (market === "spread") { if (x.home_spread == null) continue; v = sd === "home" ? x.home_spread : -x.home_spread; pr = x[sd + "_spread_odds"]; txt = ln(v); }
      else if (market === "total") { if (x.total == null) continue; const ou = i === 0 ? "over" : "under"; v = ou === "over" ? -x.total : x.total; pr = x[ou + "_odds"]; txt = ou[0] + num(x.total); }
      else { pr = x[sd + "_ml"]; if (pr == null) continue; v = pay(pr); txt = odds(pr); }
      const key = [v, pay(pr)];
      if (!top || key[0] > top.key[0] || (key[0] === top.key[0] && key[1] > top.key[1])) top = {key, txt, pr, book: x.book};
    }
    return top ? esc(top.txt) + (market !== "ml" && top.pr != null ? " <span class='m'>" + esc(odds(top.pr)) + "</span>" : "") + "<small>" + esc(top.book) + "</small>" : "<span class='na'>–</span>";
  };
  const board = () => {
    const only = $("only-plays").checked;
    const G = D.games.filter((g) => !only || isPlay(g)).sort((a, b) => (done(a) - done(b)) || byKick(a, b));
    const mh = {spread: "Model", total: "Model", ml: "Win %"}[market];
    let h = "<thead><tr><th class='tm'>Time</th><th>Team</th><th class='r pts'>Proj</th><th class='r'>" + mh + "</th>" +
      BOOKS.map((b, j) => "<th class='bk" + (j >= 4 ? " bk-minor" : "") + "'>" + esc(b) + "</th>").join("") + "<th class='bestc'>Best</th><th class='r edge'>Edge</th></tr></thead>";
    if (!G.length) { $("t-board").innerHTML = h + "<tbody><tr><td class='empty' colspan='" + (BOOKS.length + 6) + "'>No games.</td></tr></tbody>"; return; }
    h += G.map((g) => {
      const play = market === "spread" && isPlay(g);
      const rows = ["away", "home"].map((sd, i) => {
        const team = g[sd + "_team"];
        const pts = g["proj_" + sd] == null ? "–" : Math.round(g["proj_" + sd]);
        let model, bookCells, edge = "";
        if (market === "spread") {
          model = ln(sd === "home" ? g.model_home_spread : -g.model_home_spread);
          const pick = isPlay(g) && g.bet_side === team, watch = g.verdict === "WATCH" && g.spread_pick === team;
          bookCells = BOOKS.map((bk) => { const x = bl(g, bk); if (!x || x.home_spread == null) return cell(null);
            const v = sd === "home" ? x.home_spread : -x.home_spread;
            return cell(ln(v), x[sd + "_spread_odds"], x[sd + "_spread_link"], pick && g.bet_book === bk); });
          if (pick) edge = "<span class='tag bet'>" + (g.verdict === "H2H" ? "H2H" : "BET") + " " + units(g.stake_units) + "</span>";
          else if (watch) edge = "<span class='tag'>Watch · bet at " + ln(g.bet_at) + "</span>";
        } else if (market === "total") {
          const ou = i === 0 ? "over" : "under";
          model = i === 0 ? num(g.model_total) : "";
          bookCells = BOOKS.map((bk) => { const x = bl(g, bk); if (!x || x.total == null) return cell(null);
            return cell((i === 0 ? "o" : "u") + num(x.total), x[ou + "_odds"], x[ou + "_link"], false); });
          if (g.total_pick === ou && g.total_edge != null) edge = "<span class='f' style='font-size:12px'>Lean " + sgn(Math.abs(g.total_edge)) + " · not bet</span>";
        } else {
          const p = g.winner == null ? null : (g.winner === team ? g.win_pct : 1 - g.win_pct);
          model = p == null ? "–" : Math.round(100 * p) + "%";
          const prices = BOOKS.map((bk) => (bl(g, bk) || {})[sd + "_ml"]);
          const ok = prices.filter((v) => v != null), top = Math.max(...ok);
          bookCells = BOOKS.map((bk, j) => { const x = bl(g, bk); return x && x[sd + "_ml"] != null ? cell(odds(x[sd + "_ml"]), null, x[sd + "_ml_link"], prices[j] === top && ok.length > 1) : cell(null); });
        }
        const fin = g.state === "final";
        return "<tr>" + (i === 0 ? "<td class='tm' rowspan='2'>" + esc(tip(g)) + (fin ? "<div class='f'>Final</div>" : g.state === "live" ? "<div class='f'>Live</div>" : "") + "</td>" : "") +
          "<td class='team'>" + (i === 0 ? "<div class='tm-sm'>" + esc(tip(g)) + (fin ? " · Final" : "") + "</div>" : "") +
          (i === 1 ? "<span class='f'>@ </span>" : "") + logo(team, 18) + esc(team) + (fin && g[sd + "_score"] != null ? " <span class='m'>" + esc(g[sd + "_score"]) + "</span>" : "") +
          "</td><td class='r pts'>" + pts + "</td><td class='r b'>" + model + "</td>" +
          bookCells.map((c, j) => "<td class='bk" + (j >= 4 ? " bk-minor" : "") + "'>" + c + "</td>").join("") + "<td class='bestc'>" + best(g, sd, i) + "</td><td class='r edge'>" + edge + "</td></tr>";
      }).join("");
      return "<tbody class='g" + (play ? " has-bet" : "") + "'>" + rows + "</tbody>";
    }).join("");
    $("t-board").innerHTML = h;
  };
  document.querySelectorAll(".seg button").forEach((b) => b.onclick = () => {
    market = b.dataset.m; document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", x === b)); board(); });
  $("only-plays").onchange = board;
  board();

  // Watch list
  const W = D.games.filter((g) => g.verdict === "WATCH" && !done(g)).sort(byKick);
  if (!W.length) $("watch-sec").hidden = true;
  else $("t-watch").innerHTML = "<thead><tr><th>Game</th><th>Side</th><th class='r'>Now</th><th class='r'>Play at</th><th class='r'>Cover % now</th><th class='r'>EV now</th></tr></thead><tbody>" +
    W.map((g) => "<tr><td>" + matchupL(g) + "<div class='f' style='font-size:12px'>" + esc(kick(g)) + "</div></td><td class='b'>" + logo(g.spread_pick, 16) + esc(g.spread_pick) +
      "</td><td class='r'>" + ln(g.pick_line) + "</td><td class='r b'>" + ln(g.bet_at) + "</td><td class='r m'>" + pct(g.p_cover_blend) +
      "</td><td class='r m'>" + (g.spread_ev == null ? "–" : sgn(100 * g.spread_ev) + "%") + "</td></tr>").join("") + "</tbody>";

  // Record
  if (R) {
    const TYPE = {VALUE: "Spreads", SMALL: "Spreads (small)", H2H: "H2H angle", TEASER: "Teasers"};
    $("rec-sec").hidden = false;
    $("t-rec").innerHTML = "<thead><tr><th>Bets</th><th class='r'>Record</th><th class='r'>Win %</th><th class='r'>Units</th><th class='r'>ROI</th><th class='r'>Avg CLV</th><th class='r'>Open</th></tr></thead><tbody>" +
      [["All bets", o]].concat(Object.entries(R.by_type).map(([k, a]) => [TYPE[k] || k, a])).map(([k, s]) => "<tr><td>" + esc(k) + "</td><td class='r'>" + rec(s) + "</td><td class='r'>" + (wp(s) || "–") +
        "</td><td class='r " + (s.units > 0 ? "pos" : s.units < 0 ? "neg" : "") + "'>" + sgn(s.units, 2) + "</td><td class='r'>" + (s.roi == null ? "–" : sgn(100 * s.roi) + "%") +
        "</td><td class='r'>" + (s.clv == null ? "–" : sgn(s.clv, 2)) + "</td><td class='r m'>" + s.pending + "</td></tr>").join("") + "</tbody>";
    const P = (R.plays || []).filter((x) => x.result).sort((a, b) => String(b.kickoff).localeCompare(String(a.kickoff))).slice(0, 40);
    if (P.length) {
      $("recent-sec").hidden = false;
      $("t-recent").innerHTML = "<thead><tr><th>Date</th><th>Game</th><th>Bet</th><th>Book</th><th class='r'>Close</th><th class='r'>CLV</th><th class='r'>Result</th><th class='r'>Units</th></tr></thead><tbody>" +
        P.map((x) => { const m = String(x.matchup || "").split(" @ ");
          return "<tr><td class='m'>" + esc(x.kickoff ? new Date(x.kickoff).toLocaleDateString("en-US", {month: "short", day: "numeric", timeZone: "America/New_York"}) : "") + "</td><td>" +
          (m.length === 2 ? logo(m[0], 14) + esc(m[0]) + " @ " + logo(m[1], 14) + esc(m[1]) : "Teaser") + "</td><td class='b'>" + esc(x.selection) + "</td><td>" + esc(x.book || "–") +
          "</td><td class='r'>" + (x.close_line == null ? "–" : ln(x.close_line)) + "</td><td class='r'>" + (x.clv_pts == null ? "–" : sgn(x.clv_pts)) +
          "</td><td class='r b " + (x.result === "W" ? "pos" : x.result === "L" ? "neg" : "") + "'>" + esc(x.result) + "</td><td class='r'>" + sgn(x.profit, 2) + "</td></tr>"; }).join("") + "</tbody>";
    }
  }

  // Teaser legs, injuries, ratings
  $("t-tz").innerHTML = D.teasers.length ? "<thead><tr><th>Leg</th><th>Grade</th><th class='r'>Line</th><th class='r'>Teased to</th><th>Best number</th><th class='r'>Leg win %</th></tr></thead><tbody>" +
    D.teasers.slice().sort((a, b) => (b.p_leg || 0) - (a.p_leg || 0)).map((r) => "<tr><td class='b'>" + logo(r.team, 16) + esc(r.team) + " <span class='f'>" + esc(r.matchup) + "</span></td><td>" + esc(r.grade) +
      "</td><td class='r'>" + ln(r.line) + "</td><td class='r b'>" + ln(r.teased_line) + "</td><td>" + esc(r.book || "–") + "</td><td class='r'>" + pct(r.p_leg) + "</td></tr>").join("") + "</tbody>"
    : "<tbody><tr><td class='empty'>No qualifying teaser legs this week (underdogs +1.5 to +2.5, or favorites −7.5 to −8.5 the model likes).</td></tr></tbody>";
  const injText = {monday: "Not applied on Monday: long-term absences are already in the ratings.", thursday: "Applied from the Thursday practice report.", final: "Applied from the final report."};
  const qbs = D.games.filter((g) => g.qb_note).map((g) => g.qb_note);
  $("t-inj").innerHTML = "<thead><tr><th>Team</th><th>Player</th><th>Pos</th><th>Status</th><th class='r'>Miss %</th><th class='r'>Usual snaps</th></tr></thead><tbody>" +
    (D.injuries.length ? D.injuries.slice().sort((a, b) => (b.usual || 0) - (a.usual || 0)).map((r) => "<tr><td>" + logo(r.team, 16) + esc(r.team) + "</td><td class='b'>" + esc(r.player) + "</td><td>" + esc(r.position) +
      "</td><td>" + esc(String(r.status).replace("none_missed", "missed last game (likely IR)").replace(/^lag_/, "last week: ").replace(/^prac_/, "practice: ").replace(/_/g, " ")) +
      "</td><td class='r'>" + (r.p_miss == null ? "–" : Math.round(100 * r.p_miss) + "%") + "</td><td class='r'>" + (r.usual == null ? "–" : Math.round(100 * r.usual) + "%") + "</td></tr>").join("")
      : "<tr><td class='empty' colspan='6'>No injuries listed.</td></tr>") + "</tbody>";
  $("t-inj").insertAdjacentHTML("beforebegin", "<p class='f' style='font-size:12px;margin:10px 16px'>" + esc((qbs.length ? "QB changes: " + qbs.join("; ") + ". " : "") + (injText[D.inj_mode] || "No injury report available.")) + "</p>");
  $("t-rat").innerHTML = "<thead><tr><th class='r'>#</th><th>Team</th><th class='r'>Net</th><th class='r'>Offense</th><th class='r'>Defense</th><th class='r'>Special teams</th><th class='r'>Elo</th></tr></thead><tbody>" +
    D.ratings.map((r) => "<tr><td class='r m'>" + r.rank + "</td><td>" + logo(r.team, 16) + esc(r.team) + "</td><td class='r b'>" + sgn(r.net_pts) + "</td><td class='r'>" + sgn(r.off_pts) +
      "</td><td class='r'>" + sgn(r.def_pts) + "</td><td class='r'>" + sgn(r.st) + "</td><td class='r'>" + (r.elo == null ? "–" : Math.round(r.elo)) + "</td></tr>").join("") + "</tbody>";
  ["t-plays", "t-watch", "t-rec", "t-recent", "t-tz", "t-inj"].forEach(stack);

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
<header class="top"><div class="wrap">
  __BRAND__
  <nav class="tabs" aria-label="Sport"><a href="./" aria-current="page">NFL</a><a href="https://barrca12-rgb.github.io/NBA/">NBA</a><a href="https://barrca12-rgb.github.io/CBB/"><span class="hide-sm">College Basketball</span><span class="show-sm">CBB</span></a></nav>
</div></header>
<main class="wrap">
  <div class="head"><div><h1>NFL Record</h1>
    <div class="sub"><span id="eyebrow"></span> · Every play the model recommended, at the logged number and price, graded against the final score.</div></div>
    <a class="pagebtn" href="./">&larr; Card</a></div>
  <div class="kpis" id="kpis"></div>
  <section class="panel"><div class="panel-h"><h2>Summary</h2><span class="note">Settled plays. CLV = points the line moved our way from the logged number to the close. Break-even at -110 is 52.4%.</span></div>
    <div class="scroll"><table id="t-type"></table></div></section>
  <section class="panel"><div class="panel-h"><h2>By week</h2></div><div class="scroll"><table id="t-week"></table></div></section>
  <section class="panel"><div class="panel-h"><h2>Bets placed</h2><span class="note">Newest first.</span></div><div class="scroll"><table id="t-plays"></table></div></section>
  __FOLLOW__
</main>
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
    .map(([k, v]) => "<div class='kpi'><div class='k'>" + k + "</div><div class='v'>" + v + "</div></div>").join("");
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
  document.querySelectorAll("main table").forEach((t) => { const hs = [...t.querySelectorAll("thead th")].map((h) => h.textContent.trim());
    t.classList.add("stack"); t.querySelectorAll("tbody tr").forEach((r) => [...r.children].forEach((c, k) => { if (hs[k]) c.dataset.l = hs[k]; })); });
})();
</script>
"""
