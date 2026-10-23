"""Data loading: schedules/results/lines (nflverse games.csv) and play-by-play (nflverse-data releases).

Everything is cached under model/cache. `refresh=True` re-downloads the files that change during a season.
"""
from __future__ import annotations

import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CACHE = os.path.join(HERE, "cache")
PBP_DIR = os.path.join(CACHE, "pbp")

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

# Relocated franchises -> current abbreviation, so a franchise's history is one rating series.
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
    "play_type", "epa", "success", "wp", "pass", "rush", "qb_dropback", "qb_epa",
    "passer_player_id", "passer_player_name", "fixed_drive", "fixed_drive_result", "special_teams_play",
    "fumble_lost", "interception", "down", "pass_oe", "cpoe", "sack", "qb_hit",
]


def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def load_games(refresh: bool = False) -> pd.DataFrame:
    """nflverse schedule with results and closing lines. spread_line > 0 means the HOME team is favored."""
    path = os.path.join(CACHE, "games.csv")
    if refresh or not os.path.exists(path):
        try:
            _download(GAMES_URL, path)
        except Exception as e:  # offline: fall back to the copy in this repo
            print(f"warning: could not download games.csv ({e}); using repo copy")
            path = os.path.join(REPO, "data", "games.csv")
    g = pd.read_csv(path)
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(TEAM_MAP)
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g


def load_pbp(seasons, refresh_current: bool = False) -> pd.DataFrame:
    frames = []
    current = max(seasons)
    for s in seasons:
        path = os.path.join(PBP_DIR, f"pbp_{s}.parquet")
        if not os.path.exists(path) or (refresh_current and s == current):
            try:
                _download(PBP_URL.format(season=s), path)
            except Exception as e:
                if not os.path.exists(path):
                    print(f"warning: no play-by-play for {s} ({e})")
                    continue
        frames.append(pd.read_parquet(path, columns=PBP_COLS))
    pbp = pd.concat(frames, ignore_index=True)
    for c in ("posteam", "defteam", "home_team", "away_team"):
        pbp[c] = pbp[c].replace(TEAM_MAP)
    return pbp


def team_games(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offense team) with the per-game stats the ratings are built from."""
    plays = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna() & pbp["posteam"].notna()].copy()
    # Down-weight garbage time (win prob outside 10-90%) instead of dropping it.
    plays["w"] = np.where(plays["wp"].between(0.10, 0.90), 1.0, 0.35)
    plays["wepa"] = plays["w"] * plays["epa"]
    plays["wsucc"] = plays["w"] * plays["success"]
    plays["is_pass"] = (plays["pass"] == 1).astype(float)
    plays["wpass_epa"] = plays["wepa"] * plays["is_pass"]
    plays["wpass"] = plays["w"] * plays["is_pass"]
    # Turnovers: fumble recoveries are close to a coin flip, so a lost fumble only counts half against the offense
    # in the luck-adjusted EPA. Non-turnover EPA and turnover rate are also kept separately.
    fl = plays["fumble_lost"].fillna(0).eq(1)
    to = fl | plays["interception"].fillna(0).eq(1)
    plays["wepa_adj"] = plays["w"] * np.where(fl, 0.5 * plays["epa"], plays["epa"])
    plays["w_nto"] = plays["w"] * (~to)
    plays["wepa_nto"] = plays["wepa"] * (~to)
    plays["wto"] = plays["w"] * to

    # Early downs (1st/2nd) are more predictive than 3rd/4th, which swing more on chance.
    ed = plays["down"].isin([1, 2])
    plays["w_ed"] = plays["w"] * ed
    plays["wepa_ed"] = plays["wepa"] * ed
    plays["wsucc_ed"] = plays["wsucc"] * ed
    # Pass rate over expected on early downs in neutral game states: coaching aggressiveness.
    neutral = ed & plays["wp"].between(0.2, 0.8) & plays["pass_oe"].notna()
    plays["proe_n"] = neutral.astype(float)
    plays["proe_sum"] = np.where(neutral, plays["pass_oe"], 0.0)

    # Matchup inputs: pass rush vs pass protection (sacks + QB hits per dropback), run efficiency, play mix.
    db_ = plays["qb_dropback"].fillna(0).eq(1)
    plays["db"] = db_.astype(float)
    plays["press"] = (db_ & (plays["sack"].fillna(0).eq(1) | plays["qb_hit"].fillna(0).eq(1))).astype(float)
    plays["is_rush"] = (plays["rush"] == 1).astype(float)
    plays["wrush_epa"] = plays["wepa"] * plays["is_rush"]
    plays["wrush"] = plays["w"] * plays["is_rush"]
    neutral_all = ed & plays["wp"].between(0.2, 0.8)
    plays["nd"] = neutral_all.astype(float)
    plays["nd_pass"] = (neutral_all & (plays["pass"] == 1)).astype(float)

    key = ["game_id", "posteam"]
    agg = plays.groupby(key).agg(
        db=("db", "sum"), press=("press", "sum"), wrush_epa=("wrush_epa", "sum"), wrush=("wrush", "sum"),
        nd=("nd", "sum"), nd_pass=("nd_pass", "sum"),
        w_ed=("w_ed", "sum"), wepa_ed=("wepa_ed", "sum"), wsucc_ed=("wsucc_ed", "sum"),
        proe_n=("proe_n", "sum"), proe_sum=("proe_sum", "sum"),
        defteam=("defteam", "first"), w=("w", "sum"), wepa=("wepa", "sum"), wsucc=("wsucc", "sum"),
        wpass_epa=("wpass_epa", "sum"), wpass=("wpass", "sum"), plays=("epa", "size"),
        wepa_adj=("wepa_adj", "sum"), w_nto=("w_nto", "sum"), wepa_nto=("wepa_nto", "sum"), wto=("wto", "sum"),
    ).reset_index()
    agg["epa"] = agg["wepa"] / agg["w"]
    agg["sr"] = agg["wsucc"] / agg["w"]
    agg["pass_epa"] = agg["wpass_epa"] / agg["wpass"].clip(lower=1)
    agg["epa_adj"] = agg["wepa_adj"] / agg["w"]
    agg["epa_nto"] = agg["wepa_nto"] / agg["w_nto"].clip(lower=1)
    agg["to_rate"] = agg["wto"] / agg["w"]
    agg["press_rate"] = agg["press"] / agg["db"].clip(lower=1)
    agg["rush_epa"] = agg["wrush_epa"] / agg["wrush"].clip(lower=1)
    agg["pass_share"] = (agg["nd_pass"] / agg["nd"].clip(lower=1)).where(agg["nd"] >= 8)
    agg["ed_epa"] = agg["wepa_ed"] / agg["w_ed"].clip(lower=1)
    agg["ed_sr"] = agg["wsucc_ed"] / agg["w_ed"].clip(lower=1)
    agg["proe"] = (agg["proe_sum"] / agg["proe_n"].clip(lower=1)).where(agg["proe_n"] >= 5)

    drives = (pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna()]
              .drop_duplicates(["game_id", "posteam", "fixed_drive"]))
    dr = drives.groupby(key).agg(drives=("fixed_drive", "size")).reset_index()
    res = drives.assign(td=(drives["fixed_drive_result"] == "Touchdown"),
                        fg=(drives["fixed_drive_result"] == "Field goal"))
    dr = dr.merge(res.groupby(key).agg(off_td=("td", "sum"), off_fg=("fg", "sum")).reset_index(), on=key)

    # Special teams: net EPA of kicking/punting/return plays from this team's perspective.
    st = pbp[pbp["special_teams_play"].eq(1) & pbp["epa"].notna() & pbp["posteam"].notna()]
    st_for = st.groupby(key)["epa"].sum().rename("st_for").reset_index()
    st_against = st.groupby(["game_id", "defteam"])["epa"].sum().rename("st_against").reset_index() \
        .rename(columns={"defteam": "posteam"})

    # Starting QB = most dropbacks; his EPA per dropback this game.
    db = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_player_id"].notna() & pbp["qb_epa"].notna()]
    db = db.assign(cpoe_n=db["cpoe"].notna().astype(float), cpoe_v=db["cpoe"].fillna(0.0))
    qb = db.groupby(key + ["passer_player_id", "passer_player_name"]).agg(
        qb_db=("qb_epa", "size"), qb_epa_sum=("qb_epa", "sum"),
        qb_cpoe_n=("cpoe_n", "sum"), qb_cpoe_sum=("cpoe_v", "sum")).reset_index()
    qb = qb.sort_values("qb_db", ascending=False).drop_duplicates(key)
    qb = qb.rename(columns={"passer_player_id": "qb_id", "passer_player_name": "qb_name"})

    tg = (agg.merge(dr, on=key, how="left").merge(st_for, on=key, how="left")
          .merge(st_against, on=key, how="left").merge(qb, on=key, how="left"))
    tg["st_net"] = tg["st_for"].fillna(0) - tg["st_against"].fillna(0)
    tg = tg.rename(columns={"posteam": "team", "defteam": "opp"})

    g = games[["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
               "home_score", "away_score", "location"]]
    tg = tg.merge(g, on="game_id", how="inner")
    tg["is_home"] = np.where(tg["location"] == "Neutral", 0, np.where(tg["team"] == tg["home_team"], 1, -1))
    tg["points"] = np.where(tg["team"] == tg["home_team"], tg["home_score"], tg["away_score"])
    tg["ppd"] = tg["points"] / tg["drives"]
    keep = ["game_id", "season", "week", "game_type", "gameday", "team", "opp", "is_home", "points",
            "plays", "drives", "ppd", "off_td", "off_fg", "epa", "sr", "pass_epa", "st_net",
            "epa_adj", "epa_nto", "to_rate", "ed_epa", "ed_sr", "proe", "qb_cpoe_n", "qb_cpoe_sum",
            "press_rate", "rush_epa", "pass_share",
            "qb_id", "qb_name", "qb_db", "qb_epa_sum"]
    return tg[keep].dropna(subset=["points"]).sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)


def load_all(first_season: int, last_season: int, refresh: bool = False):
    games = load_games(refresh=refresh)
    tg_path = os.path.join(CACHE, f"team_games_{first_season}_{last_season}.parquet")
    if refresh or not os.path.exists(tg_path):
        pbp = load_pbp(range(first_season, last_season + 1), refresh_current=refresh)
        tg = team_games(pbp, games)
        tg.to_parquet(tg_path)
    tg = pd.read_parquet(tg_path)
    return games, tg


DEPTH_URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.parquet"
INJ_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"


def _cached_release(url: str, name: str, refresh: bool) -> pd.DataFrame | None:
    path = os.path.join(CACHE, name)
    if refresh or not os.path.exists(path):
        try:
            _download(url, path)
        except Exception as e:
            print(f"warning: could not download {name} ({e})")
            if not os.path.exists(path):
                return None
    return pd.read_parquet(path)


def depth_chart_qb1(season: int, refresh: bool = False) -> pd.DataFrame:
    """Each team's QB1 on the most recent nflverse (ESPN) depth chart snapshot."""
    d = _cached_release(DEPTH_URL.format(season=season), f"depth_charts_{season}.parquet", refresh)
    if d is None or d.empty or "pos_abb" not in d:
        return pd.DataFrame(columns=["team", "qb_id", "qb_name", "as_of"])
    q = d[(d["pos_abb"] == "QB") & d["gsis_id"].notna()].copy()
    q["team"] = q["team"].replace(TEAM_MAP)
    q = q[q["dt"] == q.groupby("team")["dt"].transform("max")]
    q = q.sort_values("pos_rank").drop_duplicates("team")
    return q.rename(columns={"gsis_id": "qb_id", "player_name": "qb_name", "dt": "as_of"})[
        ["team", "qb_id", "qb_name", "as_of"]]


def injury_report(season: int, refresh: bool = False) -> pd.DataFrame:
    d = _cached_release(INJ_URL.format(season=season), f"injuries_{season}.parquet", refresh)
    if d is None:
        return pd.DataFrame()
    d["team"] = d["team"].replace(TEAM_MAP)
    return d

