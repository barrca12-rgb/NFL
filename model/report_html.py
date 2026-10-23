"""Weekly bet card as a self-contained HTML page: verdicts first, then sortable tables for everything else.

The page has no external dependencies except Google Fonts (with fallbacks), so it opens from a file or publishes
as-is.
"""
from __future__ import annotations

import html
import json
import math

import numpy as np
import pandas as pd


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
               flag_text: str, inj_mode: str | None, inj_players: pd.DataFrame | None, qb_change: dict) -> str:
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
        "best_book", "pick_odds", "offers", "n_books",
    ])
    p = pt.reset_index().rename(columns={"index": "team"})
    p["rank"] = range(1, len(p) + 1)
    ratings = _records(p, ["rank", "team", "net_pts", "off_pts", "def_pts", "off_epa", "def_epa", "st", "elo", "market"])
    players = _records(inj_players, ["team", "player", "position", "status", "p_miss", "usual"]) \
        if inj_players is not None and len(inj_players) else []

    data = {
        "season": season, "week": week, "n_sims": n_sims, "phase": phase, "flag_text": flag_text,
        "inj_mode": inj_mode,
        "line_source": ", ".join(sorted({b for x in out["line_source"].dropna() for b in str(x).split(", ")})),
        "games": games, "ratings": ratings, "injuries": players,
    }
    payload = json.dumps(data).replace("</", "<\\/")
    title = f"Week {week} Bet Card"
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)


TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Source+Sans+3:wght@400;600;700&display=swap">
<style>
/* Layout: a bet slip up top (the only thing to act on), then sortable reference tables below. */
:root {
  --bg: #f3f5f2; --surface: #ffffff; --ink: #18211b; --muted: #5d6a61; --line: #d9dfd9;
  --turf: #1f7a45; --turf-soft: #e3f1e8; --amber: #a86400; --amber-soft: #fbefdc;
  --chalk: #eef1ee; --pass: #8a948d; --neg: #b3402f;
  --display: "Barlow Condensed", "Arial Narrow", "Roboto Condensed", sans-serif;
  --body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111613; --surface: #18201b; --ink: #e6ece7; --muted: #9aa69e; --line: #2a352e;
  --turf: #4cc27c; --turf-soft: #183323; --amber: #f0a53c; --amber-soft: #33270f;
  --chalk: #1e2722; --pass: #76817a; --neg: #ef7b67; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111613; --surface: #18201b; --ink: #e6ece7; --muted: #9aa69e; --line: #2a352e;
  --turf: #4cc27c; --turf-soft: #183323; --amber: #f0a53c; --amber-soft: #33270f;
  --chalk: #1e2722; --pass: #76817a; --neg: #ef7b67; color-scheme: dark }
* { box-sizing: border-box }
body { background: var(--bg); color: var(--ink); font: 15px/1.5 var(--body); }
.wrap { max-width: 1180px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 36px; }
h1, h2, h3 { font-family: var(--display); font-weight: 700; letter-spacing: .01em; margin: 0; text-wrap: balance }
h1 { font-size: clamp(34px, 5vw, 52px); line-height: 1 }
h2 { font-size: 26px; line-height: 1.1 }
.eyebrow { font: 600 13px/1 var(--display); letter-spacing: .14em; text-transform: uppercase; color: var(--muted) }
.lede { color: var(--muted); max-width: 70ch; margin: 8px 0 0 }
header { display: grid; gap: 10px }
.meta { display: flex; flex-wrap: wrap; gap: 8px }
.chip { font: 600 13px/1 var(--body); padding: 6px 10px; border-radius: 999px; background: var(--chalk); color: var(--muted) }
section { display: grid; gap: 14px; min-width: 0 }
.sec-head { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px }
.note { color: var(--muted); font-size: 14px; max-width: 80ch; margin: 0 }

/* Bet slip */
.slip { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px }
.ticket { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px;
  display: grid; gap: 10px; border-top: 4px solid var(--turf) }
.ticket.small { border-top-color: var(--amber) }
.ticket .pick { font: 700 32px/1 var(--display) }
.ticket .game { color: var(--muted); font-size: 14px }
.ticket dl { display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; margin: 0; font-size: 14px }
.ticket dt { color: var(--muted) }
.ticket dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; font-weight: 600 }
.stake { font: 700 15px/1 var(--display); letter-spacing: .06em; text-transform: uppercase }
.ticket .price { font-size: 20px; color: var(--muted) }
.ticket .book { font-size: 14px }
table.books { font-size: 13px; width: 100% }
table.books td { padding: 3px 0; border: 0; text-align: left }
table.books td:nth-child(2), table.books td:nth-child(3) { text-align: right }
table.books tr.best td { font-weight: 700 }
.empty { background: var(--surface); border: 1px dashed var(--line); border-radius: 10px; padding: 18px; color: var(--muted) }

/* Verdict pills */
.v { display: inline-block; font: 700 12px/1 var(--display); letter-spacing: .08em; padding: 5px 8px; border-radius: 4px }
.v.BET { background: var(--turf); color: var(--surface) }
.v.SMALL { background: var(--amber-soft); color: var(--amber) }
.v.WATCH { background: var(--chalk); color: var(--amber); box-shadow: inset 0 0 0 1px var(--amber) }
.v.PASS { background: var(--chalk); color: var(--pass) }

/* Tables */
.filters { display: flex; flex-wrap: wrap; gap: 6px }
.filters button { font: 600 13px/1 var(--body); padding: 7px 11px; border-radius: 999px; border: 1px solid var(--line);
  background: var(--surface); color: var(--ink); cursor: pointer }
.filters button[aria-pressed="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink) }
.tbl { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 10px }
table { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums }
th, td { padding: 9px 10px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--line) }
th:first-child, td:first-child, th.l, td.l { text-align: left }
tbody tr:last-child td { border-bottom: 0 }
thead th { position: sticky; top: 0; background: var(--chalk); font: 600 12px/1.2 var(--body); letter-spacing: .04em;
  text-transform: uppercase; color: var(--muted); cursor: pointer; user-select: none }
thead th:hover, thead th:focus-visible { color: var(--ink); outline: none }
thead th[aria-sort="ascending"]::after { content: " \25B2"; font-size: 9px }
thead th[aria-sort="descending"]::after { content: " \25BC"; font-size: 9px }
tbody tr:hover { background: var(--chalk) }
td.pos { color: var(--turf); font-weight: 600 } td.neg { color: var(--neg) }
td.strong { font-weight: 700 }
details { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px }
summary { cursor: pointer; font: 700 18px/1.2 var(--display) }
details[open] summary { margin-bottom: 10px }
.glossary { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px }
.glossary div { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 14px; font-size: 14px }
.glossary b { font-family: var(--display); font-size: 17px; display: block; margin-bottom: 4px }
@media (prefers-reduced-motion: no-preference) { tbody tr { transition: background .12s } }
</style>

<div class="wrap">
  <header>
    <div class="eyebrow" id="eyebrow"></div>
    <h1 id="h1"></h1>
    <p class="lede" id="lede"></p>
    <div class="meta" id="meta"></div>
  </header>

  <section aria-labelledby="slip-h">
    <div class="sec-head"><h2 id="slip-h">Bets</h2><span class="note" id="slip-sum"></span></div>
    <div class="slip" id="slip"></div>
    <p class="note" id="flagtext"></p>
  </section>

  <section aria-labelledby="watch-h" id="watch-sec">
    <div class="sec-head"><h2 id="watch-h">Watch list</h2><span class="note">Not bets at today's number. Bet them only if the line reaches the number shown.</span></div>
    <div class="tbl"><table id="t-watch"></table></div>
  </section>

  <section aria-labelledby="all-h">
    <div class="sec-head"><h2 id="all-h">All games, against the spread</h2>
      <div class="filters" role="group" aria-label="Filter by verdict" id="vfilter"></div></div>
    <div class="tbl"><table id="t-games"></table></div>
    <p class="note">Click any column to sort. Spreads use sportsbook convention: -3 means that team is favored by 3. Edge is the model's margin minus the line, from the home team's side.</p>
  </section>

  <section aria-labelledby="win-h">
    <div class="sec-head"><h2 id="win-h">Winner picks</h2></div>
    <div class="tbl"><table id="t-win"></table></div>
    <p class="note">Win % blends the model (35%) with the line (65%). In the 2016-2025 backtest this picked 66.8% of winners; the closing-line favorite won 66.4%.</p>
  </section>

  <section aria-labelledby="tot-h">
    <div class="sec-head"><h2 id="tot-h">Totals</h2><span class="note">Leans only. The model showed no reliable edge on totals in the backtest.</span></div>
    <div class="tbl"><table id="t-tot"></table></div>
  </section>

  <section aria-labelledby="gl-h">
    <h2 id="gl-h">How to read this</h2>
    <div class="glossary">
      <div><b>Model cover %</b>How often the pick covers in 20,000 simulations if the model's projection is exactly right. Used to pick and rank plays; it runs high (a 70% here has historically won about 56%).</div>
      <div><b>Realistic cover %</b>The same simulation centered between the market consensus and the model (45% toward the model in Weeks 1-4, 20% after), which matches the model's real track record. It is then priced at each book's own number, so a better number at one book shows up as real value. Use this for whether a bet is worth it and how much.</div>
      <div><b>EV / unit</b>Expected profit per unit staked at the posted odds, from the realistic cover %. Above 0 means the price is worth taking.</div>
      <div><b>Good down to</b>The worst number you can still take the bet at, at the same price. If the line moves past it, pass.</div>
      <div><b>Every book</b>Each sportsbook's number and price for the pick. The bold one is the best; always bet there.</div>
    </div>
  </section>

  <details id="inj-sec"><summary>Injuries and quarterbacks</summary><p class="note" id="inj-note"></p>
    <div class="tbl"><table id="t-inj"></table></div></details>

  <details><summary>Power ratings</summary>
    <p class="note">Points better or worse than an average team on a neutral field. Market = what past closing lines imply.</p>
    <div class="tbl"><table id="t-rat"></table></div></details>
</div>

<script id="report-data" type="application/json">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById("report-data").textContent);
  const $ = (id) => document.getElementById(id);
  const pct = (x) => x == null ? "–" : (x * 100).toFixed(1) + "%";
  const pct0 = (x) => x == null ? "–" : Math.round(x * 100) + "%";
  const sp = (x) => x == null ? "–" : (Math.abs(x) < 0.05 ? "PK" : (x > 0 ? "+" : "") + x.toFixed(1));
  const num1 = (x) => x == null ? "–" : x.toFixed(1);
  const sgn1 = (x) => x == null ? "–" : (x > 0 ? "+" : "") + x.toFixed(1);
  const ev = (x) => x == null ? "–" : (x > 0 ? "+" : "") + x.toFixed(3);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
  const units = (u) => u + "u";
  const ml = (x) => x == null ? "" : (x > 0 ? "+" : "") + Math.round(x);
  const offersHtml = (g) => !g.offers || g.offers.length < 2 ? "" :
    '<table class="books"><tbody>' + g.offers.map((o) => "<tr" + (o.book === g.best_book ? ' class="best"' : "") + "><td>" +
      esc(o.book) + "</td><td>" + sp(o.line) + " " + ml(o.odds) + '</td><td class="' + (o.ev > 0 ? "pos" : "neg") + '">' + ev(o.ev) + "</td></tr>").join("") +
    "</tbody></table>";
  const booksCell = (r) => (r.offers || []).map((o) => (o.book === r.best_book ? "<b>" : "") + esc(o.book.replace(/\.ag$/, "")) + " " +
    sp(o.line) + " " + ml(o.odds) + (o.book === r.best_book ? "</b>" : "")).join("<br>");

  // Header
  $("eyebrow").textContent = D.season + " NFL season · " + (D.phase === "early" ? "Weeks 1-4: strongest part of the season" : "Week 5 on: thin edges, smaller stakes");
  $("h1").textContent = "Week " + D.week + " Bet Card";
  const bets = D.games.filter((g) => g.verdict === "BET" || g.verdict === "SMALL");
  const units_total = bets.reduce((s, g) => s + (g.stake_units || 0), 0);
  $("lede").textContent = bets.length
    ? bets.length + (bets.length === 1 ? " play" : " plays") + " this week, " + units_total + " units total. Everything else is a pass unless a watch-list line moves."
    : "No bets at the current numbers. Check the watch list for lines that would turn into bets.";
  const injLabel = {thursday: "Thursday practice report applied", final: "Final injury report applied", monday: "Injury report not out yet (Monday run)"};
  [D.n_sims.toLocaleString() + " sims per game", "Lines: " + D.line_source, injLabel[D.inj_mode] || "No injury report"]
    .forEach((t) => { const c = document.createElement("span"); c.className = "chip"; c.textContent = t; $("meta").appendChild(c); });
  $("flagtext").textContent = D.flag_text + " Breakeven at -110 is 52.4%.";

  // Bet slip
  if (!bets.length) {
    $("slip").innerHTML = '<div class="empty">No game meets the betting rule at today\'s lines. That is the right call most weeks after Week 4.</div>';
  } else {
    bets.sort((a, b) => b.stake_units - a.stake_units || b.p_cover_raw - a.p_cover_raw).forEach((g) => {
      const el = document.createElement("article");
      el.className = "ticket" + (g.verdict === "SMALL" ? " small" : "");
      el.innerHTML = '<div class="stake" style="color:var(' + (g.verdict === "SMALL" ? "--amber" : "--turf") + ')">' +
        esc(g.verdict) + " · " + units(g.stake_units) + (g.flag ? " · " + esc(g.flag) : "") + "</div>" +
        '<div class="pick">' + esc(g.spread_pick) + " " + sp(g.pick_line) + ' <span class="price">' + ml(g.pick_odds) + "</span></div>" +
        '<div class="book">Bet at <b>' + esc(g.best_book) + "</b></div>" +
        '<div class="game">' + esc(g.matchup) + " · " + esc(g.gameday) + "</div>" +
        "<dl><dt>Good down to</dt><dd>" + esc(g.spread_pick) + " " + sp(g.bet_down_to) + "</dd>" +
        "<dt>Realistic cover</dt><dd>" + pct(g.p_cover_blend) + "</dd>" +
        "<dt>Model cover</dt><dd>" + pct(g.p_cover_raw) + "</dd>" +
        "<dt>EV / unit</dt><dd>" + ev(g.spread_ev) + "</dd>" +
        "<dt>Projection</dt><dd>" + esc(g.away_team) + " " + num1(g.proj_away) + " – " + esc(g.home_team) + " " + num1(g.proj_home) + "</dd></dl>" +
        offersHtml(g) +
        (g.qb_note ? '<div class="game">QB: ' + esc(g.qb_note) + "</div>" : "");
      $("slip").appendChild(el);
    });
  }
  $("slip-sum").textContent = bets.length ? units_total + " units across " + bets.length + " plays" : "";

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
  const verdictOrder = {BET: 3, SMALL: 2, WATCH: 1, PASS: 0};
  const vpill = (r) => '<span class="v ' + r.verdict + '">' + r.verdict + "</span>";

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
    {k: "spread_pick", h: "Side (best price)", l: 1, f: (r) => esc(r.spread_pick) + " " + sp(r.pick_line) + " " + ml(r.pick_odds) + (r.best_book ? " · " + esc(r.best_book) : ""), cls: (r) => r.verdict === "BET" ? "strong" : ""},
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
  const states = ["All", "BET", "SMALL", "WATCH", "PASS"];
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
