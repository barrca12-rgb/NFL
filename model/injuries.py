"""Injury adjustment from nflverse injury reports + snap counts.

For every team-game:
  * usage: each non-QB player's usual offense/defense snap share over the team's recent games (games he played).
    Regulars = usual share >= REGULAR_PCT. (QBs are handled by the QB rating.)
  * actual_missing: snap share of regulars who did not play in that game (known after the game).
  * expected_missing: snap share of regulars expected to miss the coming game, from the injury report:
    P(miss | report status / practice status) x usual share. Regulars who missed the last game and are not on the
    report are usually on IR and are counted as still out.
  * baseline: average actual_missing over the team's last few games. The team's ratings were earned with those
    players already missing, so only the CHANGE (expected - baseline) is new information.

Three report stages, each with its own empirically measured miss probabilities and its own trained model:
  monday   - this week's report isn't out; last week's final report carries long-term injuries forward
  thursday - practice participation only (game statuses aren't published yet)
  final    - Friday game statuses (Out / Doubtful / Questionable)
Historical reports in nflverse are the final versions, so the Thursday stage is approximated from final-week
practice statuses.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import CACHE, TEAM_MAP, _download

INJ_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"
SNAP_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.parquet"
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.parquet"

USAGE_WINDOW = 6      # team games used to measure a player's usual snap share
BASELINE_WINDOW = 3   # team games averaged for the baseline of already-missing regulars
REGULAR_PCT = 0.40


def _load(kind: str, url: str, seasons, refresh_current: bool) -> pd.DataFrame:
    frames = []
    current = max(seasons)
    for s in seasons:
        path = os.path.join(CACHE, kind, f"{kind}_{s}.parquet")
        if not os.path.exists(path) or (refresh_current and s == current):
            try:
                _download(url.format(season=s), path)
            except Exception as e:
                if not os.path.exists(path):
                    print(f"warning: no {kind} for {s} ({e})")
                    continue
        d = pd.read_parquet(path)
        if len(d):
            frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["team"] = out["team"].replace(TEAM_MAP)
    return out


def load_inputs(seasons, refresh: bool = False):
    inj = _load("inj", INJ_URL, seasons, refresh)
    snaps = _load("snaps", SNAP_URL, seasons, refresh)
    ppath = os.path.join(CACHE, "players.parquet")
    if refresh or not os.path.exists(ppath):
        _download(PLAYERS_URL, ppath)
    players = pd.read_parquet(ppath, columns=["gsis_id", "pfr_id"]).dropna().drop_duplicates("gsis_id")
    inj = inj.merge(players, on="gsis_id", how="left")
    return inj, snaps


STAGES = ("monday", "thursday", "final")


def _status(inj: pd.DataFrame, stage: str = "final") -> pd.Series:
    """Collapse report + practice status into one category for a report stage.

    final: game status (Out/Doubtful/Questionable, published Friday), else practice status.
    thursday: practice status only (prac_*), since game statuses don't exist yet.
    monday: last week's final status (lag_*); combined with whether he played last week in _join."""
    use_report = stage in ("final", "monday")
    rep = inj["report_status"].fillna("") if use_report else pd.Series("", index=inj.index)
    prac = inj["practice_status"].fillna("")
    pre = {"final": "", "thursday": "prac_", "monday": "lag_"}[stage]
    cat = np.select(
        [rep.eq("Out"), rep.eq("Doubtful"), rep.eq("Questionable"), rep.eq("Probable"),
         prac.str.startswith("Did Not"), prac.str.startswith("Limited"), prac.str.startswith("Full")],
        [pre + "out", pre + "doubtful", pre + "questionable", pre + "probable",
         pre + "dnp", pre + "limited", pre + "full"], "")
    return pd.Series(cat, index=inj.index)


def team_game_usage(snaps: pd.DataFrame, upcoming: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long table: team-game x player with the player's usual share entering that game and whether he played.

    upcoming: optional (game_id, team, season, week) rows for games not played yet; they get usage rows too
    (with played=False, which is never used for them)."""
    s = snaps[snaps["position"] != "QB"]
    order = s[["season", "week", "game_id", "team"]].drop_duplicates()
    if upcoming is not None:
        order = pd.concat([order, upcoming[["season", "week", "game_id", "team"]]]).drop_duplicates(["game_id", "team"])
    order = order.sort_values(["team", "season", "week"]).reset_index(drop=True)
    rows = []
    for team, tg in order.groupby("team"):
        ts = s[s["team"] == team]
        ids = tg["game_id"].tolist()
        seasons = tg["season"].to_numpy()
        po = ts.pivot_table(index="game_id", columns="pfr_player_id", values="offense_pct", aggfunc="max").reindex(ids)
        pdf = ts.pivot_table(index="game_id", columns="pfr_player_id", values="defense_pct", aggfunc="max").reindex(ids)
        played = (po.fillna(0) + pdf.fillna(0)) > 0
        for i in range(1, len(ids)):
            lo = max(0, i - USAGE_WINDOW)
            lo = lo + int(np.argmax(seasons[lo:i] == seasons[i])) if (seasons[lo:i] == seasons[i]).any() else i
            if lo >= i:
                continue  # no same-season history yet
            win_o = po.iloc[lo:i]
            win_d = pdf.iloc[lo:i]
            usual_o = win_o.where(win_o > 0).mean()
            usual_d = win_d.where(win_d > 0).mean()
            usual = pd.DataFrame({"usual_off": usual_o, "usual_def": usual_d}).fillna(0)
            usual = usual[(usual["usual_off"] >= REGULAR_PCT) | (usual["usual_def"] >= REGULAR_PCT)]
            if usual.empty:
                continue
            u = usual.reset_index().rename(columns={"pfr_player_id": "pfr_id"})
            u["played"] = played.iloc[i].reindex(u["pfr_id"]).fillna(False).to_numpy()
            u["played_last"] = played.iloc[i - 1].reindex(u["pfr_id"]).fillna(False).to_numpy()
            u["game_id"], u["team"], u["season"], u["week"] = ids[i], team, seasons[i], tg["week"].iloc[i]
            u["prev_week"] = tg["week"].iloc[i - 1] if seasons[i - 1] == seasons[i] else -1
            rows.append(u)
    return pd.concat(rows, ignore_index=True)


def miss_probabilities(usage: pd.DataFrame, inj: pd.DataFrame) -> pd.Series:
    """Empirical P(regular does not play | status) for every report stage, from history."""
    parts = []
    for stage in STAGES:
        rates = 1 - _join(usage, inj, stage).groupby("status")["played"].mean()
        parts.append(rates if stage == "final" else rates[~rates.index.str.startswith("none_")])
    return pd.concat(parts).rename("p_miss")


def _join(usage: pd.DataFrame, inj: pd.DataFrame, stage: str = "final") -> pd.DataFrame:
    i = inj[inj["pfr_id"].notna()].copy()
    i["status"] = _status(i, stage)
    i = i[i["status"] != ""]
    i = i.drop_duplicates(["season", "week", "team", "pfr_id"])[["season", "week", "team", "pfr_id", "status"]]
    if stage == "monday":
        # Last week's report for the team's previous game; split by whether he actually played that game.
        i = i.rename(columns={"week": "prev_week"})
        j = usage.merge(i, on=["season", "prev_week", "team", "pfr_id"], how="left")
        lag = j["status"].notna()
        j.loc[lag, "status"] = j.loc[lag, "status"] + np.where(j.loc[lag, "played_last"], "_played", "_missed")
    else:
        j = usage.merge(i, on=["season", "week", "team", "pfr_id"], how="left")
    # Not on the report: split by whether he played last game (missed = usually IR / long-term absence).
    j["status"] = j["status"].fillna(pd.Series(np.where(j["played_last"], "none_played", "none_missed"), index=j.index))
    return j


def injury_features(usage: pd.DataFrame, inj: pd.DataFrame, p_miss: pd.Series, stage: str = "final") -> pd.DataFrame:
    """Per team-game: expected missing usage (from the report at that stage), baseline missing usage, the delta."""
    j = _join(usage, inj, stage)
    p = j["status"].map(p_miss).fillna(0.0)
    j["exp_off"] = p * j["usual_off"]
    j["exp_def"] = p * j["usual_def"]
    j["act_off"] = (~j["played"]) * j["usual_off"]
    j["act_def"] = (~j["played"]) * j["usual_def"]
    g = j.groupby(["game_id", "team", "season", "week"])[["exp_off", "exp_def", "act_off", "act_def"]].sum()
    g = g.reset_index().sort_values(["team", "season", "week"])
    for side in ("off", "def"):
        base = g.groupby(["team", "season"])[f"act_{side}"].transform(
            lambda x: x.shift(1).rolling(BASELINE_WINDOW, min_periods=1).mean()).fillna(0)
        g[f"inj_{side}"] = g[f"exp_{side}"] - base
    return g[["game_id", "team", "season", "week", "exp_off", "exp_def", "inj_off", "inj_def"]]


SUFFIX = {"final": "", "thursday": "_thu", "monday": "_mon"}
# Model features per report stage: own offense's injury delta, opponent defense's injury delta.
FEATURES = {stage: [f"inj_off{sfx}", f"inj_def{sfx}_opp"] for stage, sfx in SUFFIX.items()}
_COLS = [f"inj_{side}{sfx}" for sfx in SUFFIX.values() for side in ("off", "def")]


def add_injury_features(f: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Merge a team-game injury table (inj_off/inj_def per stage suffix) onto team-perspective features."""
    t = table[["game_id", "team"] + [c for c in _COLS if c in table]]
    f = f.drop(columns=[c for c in f.columns if c.startswith("inj_")])
    f = f.merge(t, on=["game_id", "team"], how="left")
    opp = t.rename(columns={"team": "opp", **{c: f"{c}_opp" for c in _COLS}})
    f = f.merge(opp, on=["game_id", "opp"], how="left")
    inj_cols = [c for c in f.columns if c.startswith("inj_")]
    f[inj_cols] = f[inj_cols].fillna(0.0)
    return f


def _stage_table(usage, inj, p_miss, stage):
    t = injury_features(usage, inj, p_miss, stage)
    sfx = SUFFIX[stage]
    return t.rename(columns={"inj_off": f"inj_off{sfx}", "inj_def": f"inj_def{sfx}"})


def historical_table(seasons, refresh: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Injury features for every past team-game at every report stage, and the miss-probability table."""
    inj, snaps = load_inputs(seasons, refresh)
    usage = team_game_usage(snaps)
    p_miss = miss_probabilities(usage, inj)
    t = None
    for stage in STAGES:
        st = _stage_table(usage, inj, p_miss, stage)
        st = st[["game_id", "team"] + [c for c in st.columns if c.startswith("inj_")]]
        t = st if t is None else t.merge(st, on=["game_id", "team"])
    return t, p_miss


def report_stage(inj: pd.DataFrame, season: int, week: int) -> str | None:
    """Which report stage is available for (season, week):
    'final' once game statuses are published (Fri), 'thursday' with practice reports only,
    'monday' when only last week's report exists, None if there is nothing to use."""
    wk = inj[(inj["season"] == season) & (inj["week"] == week)]
    if not wk.empty:
        return "final" if wk["report_status"].isin(["Out", "Doubtful", "Questionable"]).any() else "thursday"
    prev = inj[(inj["season"] == season) & (inj["week"] < week)]
    return "monday" if not prev.empty else None


def live_table(week_games: pd.DataFrame, season: int, week: int, p_miss: pd.Series, refresh: bool = False):
    """Injury features + key absences for the coming week. Returns (stage, table, players) or (None, None, None)."""
    inj, snaps = load_inputs(range(season - 1, season + 1), refresh)
    stage = report_stage(inj, season, week)
    if stage is None:
        return None, None, None
    up = pd.concat([
        pd.DataFrame({"game_id": week_games["game_id"], "team": week_games[side], "season": season, "week": week})
        for side in ("home_team", "away_team")])
    usage = team_game_usage(snaps[snaps["season"] == season], upcoming=up)
    t = _stage_table(usage, inj, p_miss, stage)
    t = t[t["game_id"].isin(up["game_id"])]

    # Key absences for the report: regulars at least 50% likely to miss.
    j = _join(usage[usage["game_id"].isin(up["game_id"])], inj, stage)
    j["p_miss"] = j["status"].map(p_miss).fillna(0.0)
    names = snaps.drop_duplicates("pfr_player_id", keep="last").set_index("pfr_player_id")[["player", "position"]]
    j = j.join(names, on="pfr_id")
    j["usual"] = j[["usual_off", "usual_def"]].max(axis=1)
    players = j[j["p_miss"] >= 0.5].sort_values(["team", "usual"], ascending=[True, False])
    return stage, t, players[["team", "player", "position", "status", "p_miss", "usual"]]
