"""python -m unittest tests.test_bets: market resistance (run_week.resist_moves) and probability CLV (tracker)."""
import unittest

import pandas as pd

from model.run_week import resist_moves
from model.tracker import clv_prob


def _card(home_spread):
    return pd.DataFrame([{"game_id": "2026_08_TB_NO", "home_team": "NO", "away_team": "TB", "bet_side": "TB",
                          "verdict": "BET", "stake_units": 1.0, "reason": "VALUE at FanDuel",
                          "book_lines": [{"book": "FanDuel", "home_spread": home_spread},
                                         {"book": "DraftKings", "home_spread": home_spread + 0.5}]}])


LOGGED = pd.DataFrame([{"game_id": "2026_08_TB_NO", "team": "TB", "line": 3.5, "stake": 0.5}])  # TB +3.5, 0.5u


class MarketResistance(unittest.TestCase):
    def test_moved_against_caps_stake(self):
        out = resist_moves(_card(-3.0), LOGGED)      # TB now best +3 (DK) < logged +3.5
        self.assertEqual(out.loc[0, "stake_units"], 0.5)
        self.assertIn("line moved against: no add", out.loc[0, "reason"])

    def test_same_or_better_line_unchanged(self):
        out = resist_moves(_card(-3.5), LOGGED)      # TB +4 at DK: moved our way
        self.assertEqual(out.loc[0, "stake_units"], 1.0)
        self.assertNotIn("moved against", out.loc[0, "reason"])

    def test_no_logged_bets_unchanged(self):
        card = _card(-3.0)
        pd.testing.assert_frame_equal(resist_moves(card, LOGGED.iloc[0:0]), card)

    def test_other_side_not_capped(self):
        out = resist_moves(_card(-3.0).assign(bet_side="NO"), LOGGED)
        self.assertEqual(out.loc[0, "stake_units"], 1.0)


class ClvProb(unittest.TestCase):
    def test_key_numbers(self):
        through_3 = clv_prob(3.0, 2.5)               # +3 vs a +2.5 close: crosses the 3
        off_key = clv_prob(9.0, 8.5)
        self.assertGreater(through_3, 0.03)
        self.assertLess(off_key, through_3 / 2)
        self.assertAlmostEqual(clv_prob(-3.0, -3.0), 0.0)
        self.assertLess(clv_prob(-3.5, -3.0), 0)


if __name__ == "__main__":
    unittest.main()
