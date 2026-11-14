import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from validate_data import validate_series


class SourceValidation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "source.csv"

    def write(self, dates, values):
        pd.DataFrame({"observation_date": dates, "VALUE": values}).to_csv(self.path, index=False)

    def test_accepts_holiday_gap_and_trailing_missing_observation(self):
        self.write(["2026-11-12", "2026-11-13"], [10, None])
        self.assertEqual(validate_series(self.path, date(2026, 11, 15), 7, min_rows=1),
                         "2026-11-12")

    def test_staleness_uses_last_numeric_value(self):
        self.write(["2026-10-01", "2026-11-15"], [10, None])
        with self.assertRaisesRegex(ValueError, "stale"):
            validate_series(self.path, date(2026, 11, 15), 7, min_rows=1)

    def test_rejects_provider_error_page(self):
        self.path.write_text("<html>Too many requests</html>")
        with self.assertRaises(ValueError):
            validate_series(self.path, date(2026, 11, 15), 7, min_rows=1)

    def test_rejects_short_history(self):
        self.write(["2026-11-13"], [10])
        with self.assertRaisesRegex(ValueError, "history"):
            validate_series(self.path, date(2026, 11, 15), 7)

    def test_rejects_duplicate_or_unsorted_dates(self):
        for dates in [["2026-11-13", "2026-11-13"], ["2026-11-13", "2026-11-12"]]:
            with self.subTest(dates=dates):
                self.write(dates, [10, 11])
                with self.assertRaisesRegex(ValueError, "dates"):
                    validate_series(self.path, date(2026, 11, 15), 7, min_rows=1)

    def test_yahoo_falls_back_to_close_when_adjusted_close_is_missing(self):
        pd.DataFrame({"Date": ["2026-11-13"], "Close": [100], "AdjClose": [None]}).to_csv(self.path, index=False)
        self.assertEqual(validate_series(self.path, date(2026, 11, 15), 7, yahoo=True, min_rows=1),
                         "2026-11-13")

    def test_rejects_future_observation(self):
        self.write(["2026-11-20"], [10])
        with self.assertRaisesRegex(ValueError, "future"):
            validate_series(self.path, date(2026, 11, 15), 7, min_rows=1)


if __name__ == "__main__":
    unittest.main()
