"""Monte Carlo game simulator: possession-by-possession with a game clock and game-state behavior.

Each of the n simulated games is played out in order, vectorized across simulations:
  1. Game-day strength: each team's expected points gets a shared (pace/weather/officiating) shock and an
     independent team shock. Sizes are calibrated in backtest so the sim's spread of outcomes matches the
     model's real out-of-sample error -> probabilities are honest about how wrong the model can be.
  2. Possessions alternate, each burning game clock (gamma-distributed, scaled to the matchup's pace). Halftime
     switches possession to the team that did not receive the opening kick.
  3. A possession ends in TD, FG, a defensive TD or safety for the other team, or nothing. The base rates are
     solved so the expected score equals the projection: FG rate is nearly flat across teams and TD rate carries
     team quality (fit on 2021-2025 nflverse drives). TDs get an XP or a 2-point try.
  4. Game script: a leading team's scoring rate eases and the trailer's rises (prevent defense, conservative play
     calling). Late in the 4th, a team needing a TD stops kicking FGs, a team trailing by a FG plays for one,
     and leaders milk the clock. Drives cut off by the end of a half rarely produce TDs.
  5. Regulation ties go to overtime (regular season can still end tied; playoffs cannot).

This structure is what produces the NFL's key numbers (3 and 7 above all), which drive spread pricing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

TD_PTS = 6.95             # average points of a TD including the try
P_OPP_TD = 0.020          # share of possessions ending in a defensive/return TD for the other team
P_SAFETY = 0.0022
FG_A, FG_B = 0.127, 0.013  # P(FG) per drive = FG_A + FG_B * offensive points per drive
P_TWO_TRY, P_TWO_MADE, P_XP = 0.08, 0.48, 0.945
P_OT_TIE = 0.07
LEN_SCORE, LEN_EMPTY = 1.3, 0.85   # drive length multipliers (TD/FG drives vs the rest)
GAME_MIN = 60.0


@dataclass
class SimConfig:
    n_sims: int = 20000
    team_sd: float = 3.0      # independent game-day shock to each team's expected points (calibrated)
    swing_sd: float = 2.0     # zero-sum shock (+ to one team, - to the other: turnovers, ST swings) (calibrated)
    shared_sd: float = 1.0    # common shock to both teams' expected points (calibrated)
    script: float = 0.10      # game-script strength: rate multiplier exp(-script * lead / 7)
    late_min: float = 5.0     # minutes left at which end-game logic switches on
    cut_min: float = 0.6      # minutes left in a half below which a drive is a hurry-up/end-of-half drive
    shape: float = 6.0        # gamma shape of drive length (higher = more regular pace)
    seed: int | None = 7
    # Importance weights by absolute final margin so key-number frequencies match real NFL games (calibrated).
    key_weights: dict = field(default_factory=dict)

    @classmethod
    def calibrated(cls, **overrides) -> "SimConfig":
        """Config with the values fit by model/calibrate.py, if that has been run."""
        kw = {}
        if os.path.exists(CALIBRATION_PATH):
            with open(CALIBRATION_PATH) as fh:
                cal = json.load(fh)
            kw = {k: v for k, v in cal.items() if k in cls.__dataclass_fields__}
            kw["key_weights"] = {int(k): v for k, v in kw.get("key_weights", {}).items()}
        kw.update(overrides)
        return cls(**kw)


def key_weight(margin: np.ndarray, key_weights: dict) -> np.ndarray:
    if not key_weights:
        return np.ones(len(margin))
    am = np.abs(margin)
    top = max(key_weights)
    lut = np.ones(top + 2)
    for k, v in key_weights.items():
        lut[k] = v
    return lut[np.minimum(am, top + 1)]


def _try_points(rng, tds):
    twos = rng.binomial(tds, P_TWO_TRY)
    return 6 * tds + 2 * rng.binomial(twos, P_TWO_MADE) + rng.binomial(tds - twos, P_XP)


def _play(rate, exp_drives, playoff, cfg, rng):
    """rate: (2, n) offensive points per possession for home/away. Returns integer scores (2, n)."""
    n = rate.shape[1]
    score = np.zeros((2, n), dtype=int)
    t = np.full(n, GAME_MIN)
    first = (rng.random(n) < 0.5).astype(int)   # who receives the opening kickoff
    pos = first.copy()
    in_first_half = np.ones(n, bool)
    mean_len = GAME_MIN / (2 * exp_drives)
    shape = cfg.shape
    ar = np.arange(n)

    for _ in range(60):
        live = t > 0
        if not live.any():
            break
        off, dfn = pos, 1 - pos
        lead = score[off, ar] - score[dfn, ar]
        r = rate[off, ar] * np.exp(-cfg.script * np.clip(lead, -21, 21) / 7.0)
        p_fg = np.clip(FG_A + FG_B * r, 0.04, 0.35)
        p_td = np.clip((r - 3 * p_fg) / TD_PTS, 0.005, 0.85)

        late = (~in_first_half) & (t <= cfg.late_min)
        need_td = late & (lead <= -4) & (lead >= -8)
        p_td = np.where(need_td, p_td + 0.45 * p_fg, p_td)
        p_fg = np.where(need_td, 0.0, p_fg)
        need_fg = late & (lead >= -3) & (lead <= 0)
        p_fg = np.where(need_fg, p_fg + 0.35 * p_td, p_fg)
        p_td = np.where(need_fg, 0.65 * p_td, p_td)

        half_end = np.where(in_first_half, GAME_MIN / 2, 0.0)
        # A drive that cannot finish before the half/game ends rarely produces a TD.
        time_left = t - half_end
        cut = time_left < cfg.cut_min
        p_td = np.where(cut, p_td * 0.35, p_td)
        p_fg = np.where(cut, p_fg * 0.8, p_fg)
        p_fg = np.minimum(p_fg, 1 - p_td - P_OPP_TD - P_SAFETY)

        u = rng.random(n)
        td = u < p_td
        fg = (u >= p_td) & (u < p_td + p_fg)
        otd = (u >= p_td + p_fg) & (u < p_td + p_fg + P_OPP_TD)
        saf = (u >= p_td + p_fg + P_OPP_TD) & (u < p_td + p_fg + P_OPP_TD + P_SAFETY)

        # Scoring drives are long, empty ones short: high-scoring games have fewer possessions.
        scale = np.where(td | fg, LEN_SCORE, LEN_EMPTY)
        length = rng.gamma(shape, mean_len / shape, n) * scale
        length = np.where(late & (lead > 0), length * 1.6, length)    # leader runs clock
        length = np.where(cut, time_left, length)

        pts_td = _try_points(rng, td.astype(int))
        pts_otd = _try_points(rng, otd.astype(int))
        gain_off = np.where(live, pts_td + 3 * fg, 0)
        gain_def = np.where(live, pts_otd + 2 * saf, 0)
        score[off, ar] += gain_off
        score[dfn, ar] += gain_def

        new_t = np.maximum(t - length, half_end)
        crossed = in_first_half & (new_t <= half_end) & live
        t = np.where(live, new_t, t)
        # Next possession: normally the other team; after a defensive TD the original offense receives.
        pos = np.where(live, np.where(otd, off, dfn), pos)
        pos = np.where(crossed, 1 - first, pos)
        in_first_half = in_first_half & ~crossed

    tied = score[0] == score[1]
    if tied.any():
        k = int(tied.sum())
        idx = np.flatnonzero(tied)
        edge = (rate[0, idx] - rate[1, idx]) * 1.2
        home_wins = rng.random(k) < 1 / (1 + np.exp(-edge))
        decided = np.ones(k, bool) if playoff else rng.random(k) > P_OT_TIE
        pts = np.where(rng.random(k) < 0.55, 3, 6)
        score[0, idx] += np.where(decided & home_wins, pts, 0)
        score[1, idx] += np.where(decided & ~home_wins, pts, 0)
    return score


def simulate_game(home_pts: float, away_pts: float, exp_drives: float = 11.0, playoff: bool = False,
                  cfg: SimConfig | None = None, rng=None) -> dict[str, np.ndarray]:
    """Simulate one matchup cfg.n_sims times. Returns arrays of home and away final scores."""
    cfg = cfg or SimConfig.calibrated()
    rng = rng or np.random.default_rng(cfg.seed)
    n = cfg.n_sims
    target = np.array([home_pts, away_pts], dtype=float)
    def_pts = exp_drives * (P_OPP_TD * TD_PTS + P_SAFETY * 2)

    def run(adj, nn):
        shared = rng.normal(0, cfg.shared_sd, nn)
        swing = rng.normal(0, cfg.swing_sd, nn)
        mu = np.vstack([adj[0] + shared + swing + rng.normal(0, cfg.team_sd, nn),
                        adj[1] + shared - swing + rng.normal(0, cfg.team_sd, nn)])
        rate = np.clip((np.clip(mu, 3.0, None) - def_pts) / exp_drives, 0.05, 6.0)
        return _play(rate, exp_drives, playoff, cfg, rng)

    # Game-state rules and clipping shift the mean slightly; one pilot run corrects the inputs so the
    # simulated average score equals the projection.
    adj = target.copy()
    for size in (5000, max(n, 5000)):
        pilot = run(adj, size)
        pw = key_weight(pilot[0] - pilot[1], cfg.key_weights)
        adj = adj + (target - np.average(pilot, axis=1, weights=pw))
    s = run(adj, n)
    return {"home": s[0], "away": s[1], "w": key_weight(s[0] - s[1], cfg.key_weights)}


def summarize(sim: dict[str, np.ndarray], spread_line: float | None = None, total_line: float | None = None) -> dict:
    """Spread convention matches nflverse: spread_line > 0 means home favored by that many points."""
    h, a = sim["home"], sim["away"]
    w = sim.get("w")
    w = np.ones(len(h)) if w is None else w
    w = w / w.sum()
    m, t = h - a, h + a

    def P(mask):
        return float(w[mask].sum())

    def sd(x):
        return float(np.sqrt(w @ (x - w @ x) ** 2))

    out = {
        "sim_home_pts": float(w @ h), "sim_away_pts": float(w @ a),
        "sim_margin": float(w @ m), "sim_total": float(w @ t), "sim_margin_sd": sd(m), "sim_total_sd": sd(t),
        "p_home_win": P(m > 0) + 0.5 * P(m == 0),
        "p_tie": P(m == 0),
    }
    out["fair_home_ml"] = american(out["p_home_win"])
    out["fair_away_ml"] = american(1 - out["p_home_win"])
    if spread_line is not None and not np.isnan(spread_line):
        out["p_home_cover"] = P(m > spread_line)
        out["p_away_cover"] = P(m < spread_line)
        out["p_spread_push"] = P(m == spread_line)
    if total_line is not None and not np.isnan(total_line):
        out["p_over"] = P(t > total_line)
        out["p_under"] = P(t < total_line)
        out["p_total_push"] = P(t == total_line)
    return out


def american(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p
