import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from monitor import trade_history


class HistoryNumerics(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "position": [0, 1, 1, 0, 0, 1, 1, 0, 0],
            "gold": [100, 150, 180, 162, 291.6, 554.04, 609.444, 487.5552, 780.08832],
            "score_final": [0.0] * 9,
        }, index=pd.date_range("2015-12-18", periods=9, freq="W-FRI"))
        self.equity = pd.Series([
            100, 99.9, 119.88, 107.77212, 107.77212,
            107.66434788, 118.430782668, 94.626195351732, 94.626195351732,
        ], index=self.frame.index)

    def test_existing_position_at_window_floor_is_not_a_trade(self):
        records = trade_history(self.frame, self.equity)
        self.assertEqual([r["start"].strftime("%Y-%m-%d") for r in records],
                         ["2016-02-05", "2016-01-22", "2016-01-08"])
        self.assertEqual(records[-1]["end"], pd.Timestamp("2016-01-22"))

    def test_cash_interval_includes_next_entry_cost(self):
        record = trade_history(self.frame, self.equity)[-1]
        self.assertEqual(record["position"], 0)
        self.assertAlmostEqual(record["return"], -0.001, places=13)

    def test_long_interval_has_delayed_exit_week_and_excludes_left_cost(self):
        record = trade_history(self.frame, self.equity)[1]
        self.assertEqual(record["position"], 1)
        self.assertAlmostEqual(record["return"], -0.1211, places=13)
        self.assertFalse(record["ongoing"])

    def test_current_cash_has_zero_gain_and_remains_ongoing(self):
        record = trade_history(self.frame, self.equity)[0]
        self.assertEqual(record["start"], pd.Timestamp("2016-02-05"))
        self.assertEqual(record["end"], pd.Timestamp("2016-02-12"))
        self.assertEqual(record["return"], 0)
        self.assertTrue(record["ongoing"])

    def test_change_on_latest_week_is_an_ongoing_zero_length_interval(self):
        record = trade_history(self.frame.iloc[:6], self.equity.iloc[:6])[0]
        self.assertEqual(record["start"], pd.Timestamp("2016-01-22"))
        self.assertEqual(record["end"], record["start"])
        self.assertEqual(record["return"], 0)
        self.assertTrue(record["ongoing"])

    def test_no_changes_in_window_does_not_invent_an_initial_trade(self):
        self.assertEqual(trade_history(self.frame.iloc[:3], self.equity.iloc[:3]), [])

    def test_invalid_equity_is_unavailable_instead_of_zero_or_infinite(self):
        equity = self.equity.copy()
        equity.loc["2016-01-22"] = np.nan
        self.assertIsNone(trade_history(self.frame, equity)[1]["return"])
        equity.loc["2016-01-22"] = 0
        self.assertIsNone(trade_history(self.frame, equity)[1]["return"])

    def test_empty_data_has_no_history(self):
        self.assertEqual(trade_history(self.frame.iloc[:0], self.equity.iloc[:0]), [])


if __name__ == "__main__":
    unittest.main()
