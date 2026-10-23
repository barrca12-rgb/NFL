"""6-point NFL teaser legs through the key numbers 3 and 7 (Wong teasers), graded by what held in backtests.

Closing lines 2006-2025, leg win rates (pushes excluded):
    underdog +1.5/+2.5 teased to +7.5/+8.5:  77.0% (2006-12), 75.6% (2013-18), 77.6% (2019-25)
        total <= 47.5: ~76% (2006-15), ~79% (2016-25);  total >= 48: 66-73%
        model likes the side 78.5%, model against it 76.1% (2016-25)
    favorite -7.5/-8.5 teased to -1.5/-2.5:  70.6% / 76.1% / 71.1%; 75.0% when the model likes the side, 69.2% when not
Break-even per leg: 2-team at -120 = 73.9%, at -130 = 75.2%; 3-team at +160 = 72.7%.

Grades:
    A - underdog +1.5..+2.5, total <= 47.5, model not against the side by 2+ points
    B - underdog +1.5..+2.5 with a higher total, or favorite -7.5..-8.5 that the model likes
"""
from __future__ import annotations

import pandas as pd

TEASE = 6.0
DOG_RANGE = (1.5, 2.5)
FAV_RANGE = (-8.5, -7.5)
MAX_TOTAL_A = 47.5
BREAKEVEN = {"2-team -120": 0.739, "2-team -130": 0.752, "3-team +160": 0.727}
GRADE_RECORD = {"A": "about 78% per leg since 2006", "B": "about 75% per leg"}


def grade_leg(line: float, total: float, model_edge: float) -> str | None:
    """line: sportsbook number for this side (+2 = getting 2). model_edge: model margin minus the line, from this
    side's view (+ = the model likes this side)."""
    if DOG_RANGE[0] <= line <= DOG_RANGE[1]:
        if (pd.isna(total) or total <= MAX_TOTAL_A) and not model_edge <= -2:
            return "A"
        return "B"
    if FAV_RANGE[0] <= line <= FAV_RANGE[1] and model_edge > 0:
        return "B"
    return None


HIST_RATE = {"A": 0.78, "B": 0.75}   # backtested leg win rates by grade (see module docstring)


def pair_legs(legs: pd.DataFrame) -> list[tuple]:
    """Suggested 2-team teasers: A legs paired best-first, never two legs from the same game (that would be a
    middle, not two independent legs); a leftover A leg is paired with the best B leg from another game."""
    if legs.empty:
        return []
    legs = legs.sort_values(["grade", "p_leg"], ascending=[True, False]).to_dict("records")
    a = [x for x in legs if x["grade"] == "A"]
    b = [x for x in legs if x["grade"] == "B"]
    pairs, used = [], set()
    for i, x in enumerate(a):
        if x["game_id"] in used:
            continue
        for y in a[i + 1:]:
            if y["game_id"] not in used and y["game_id"] != x["game_id"]:
                pairs.append((x, y))
                used |= {x["game_id"], y["game_id"]}
                break
    for x in a:
        if x["game_id"] not in used:
            y = next((y for y in b if y["game_id"] not in used and y["game_id"] != x["game_id"]), None)
            if y:
                pairs.append((x, y))
                used |= {x["game_id"], y["game_id"]}
    return pairs


def describe(leg: dict) -> str:
    sign = "+" if leg["teased_line"] > 0 else ""
    return f"{leg['team']} {sign}{leg['teased_line']:g}"


def pairs_table(pairs: list[tuple]) -> pd.DataFrame:
    rows = []
    for x, y in pairs:
        hist = HIST_RATE[x["grade"]] * HIST_RATE[y["grade"]]
        rows.append({"teaser": f"{describe(x)} / {describe(y)}", "grades": x["grade"] + y["grade"],
                     "legs": [{k: leg[k] for k in ("game_id", "team", "line", "teased_line", "book")} for leg in (x, y)],
                     "p_both": hist, "p_both_model": x["p_leg"] * y["p_leg"],
                     "ev": hist * (100 / 120) - (1 - hist),
                     "stake_units": 1.0 if x["grade"] + y["grade"] == "AA" else 0.5})
    return pd.DataFrame(rows)
