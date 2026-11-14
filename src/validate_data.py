import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FRED = ["DGS10", "DGS2", "DFII10", "T10YIE", "BAA10Y", "BAMLH0A0HYM2",
        "BAMLC0A0CM", "NFCI", "NFCINONFINLEVERAGE", "VIXCLS"]
YAHOO = ["GLD", "DX-Y.NYB", "GSPC", "CL=F"]


def validate_series(path, today, max_age, yahoo=False, min_rows=260):
    frame = pd.read_csv(path)
    if yahoo:
        dates = pd.to_datetime(frame["Date"], errors="raise")
        values = pd.to_numeric(frame["AdjClose"], errors="coerce")
        values = values.fillna(pd.to_numeric(frame["Close"], errors="coerce"))
    else:
        if len(frame.columns) != 2 or frame.columns[0] != "observation_date":
            raise ValueError(f"{path.name}: invalid CSV schema")
        dates = pd.to_datetime(frame.iloc[:, 0], errors="raise")
        values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError(f"{path.name}: invalid dates")
    valid = dates[np.isfinite(values)]
    if len(valid) < min_rows:
        raise ValueError(f"{path.name}: insufficient history ({len(valid)} rows)")
    latest = valid.iloc[-1].date()
    age = (today - latest).days
    if age < 0:
        raise ValueError(f"{path.name}: future observation {latest}")
    if age > max_age:
        raise ValueError(f"{path.name}: stale; last numeric observation {latest}")
    return latest.isoformat()


def main():
    now = datetime.now(timezone.utc)
    raw = ROOT / "data" / "raw"
    dates = {}
    for sid in FRED:
        path = raw / f"fred_{sid}.csv"
        dates[path.name] = validate_series(path, now.date(), 21 if sid.startswith("NFCI") else 10)
    for symbol in YAHOO:
        path = raw / f"yahoo_{symbol}.csv"
        dates[path.name] = validate_series(path, now.date(), 7, yahoo=True, min_rows=1000)
        first = pd.read_csv(path, usecols=["Date"]).Date.iloc[0]
        if first > "2005-01-01":
            raise ValueError(f"{path.name}: truncated history starting {first}")
    status = {"checked_at": now.isoformat(), "latest_observations": dates}
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "refresh-status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
