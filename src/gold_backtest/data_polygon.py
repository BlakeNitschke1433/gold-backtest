# data_polygon.py
# Polygon FX data loader for XAUUSD (C:XAUUSD) into pandas.
# Requires: pip install polygon-api-client pandas

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import pandas as pd
from polygon import RESTClient


@dataclass
class PolygonConfig:
    api_key: str
    ticker: str = "C:XAUUSD"  # Polygon FX ticker format
    multiplier: int = 15
    timespan: str = "minute"
    max_per_call: int = 50000  # polygon max limit per call


def _bar_get(bar, name: str, fallback: str):
    # polygon Agg objects sometimes expose .open/.high etc OR .o/.h etc
    if hasattr(bar, name):
        return getattr(bar, name)
    if hasattr(bar, fallback):
        return getattr(bar, fallback)
    if isinstance(bar, dict):
        return bar.get(name, bar.get(fallback))
    return None


def _to_utc_dt(x) -> datetime:
    # Polygon aggregate timestamps are in ms since epoch (UTC)
    if isinstance(x, (int, float)):
        return datetime.fromtimestamp(x / 1000.0, tz=timezone.utc)
    if isinstance(x, str):
        return pd.to_datetime(x, utc=True).to_pydatetime()
    raise ValueError(f"Unknown timestamp type: {type(x)}")


class PolygonData:
    def __init__(self, cfg: PolygonConfig):
        if not cfg.api_key:
            raise ValueError("Polygon API key is required.")
        self.cfg = cfg
        self.client = RESTClient(cfg.api_key)

    @staticmethod
    def from_env() -> "PolygonData":
        key = os.getenv("POLYGON_API_KEY", "").strip()
        if not key:
            raise EnvironmentError(
                "Missing POLYGON_API_KEY environment variable.\n"
                "Set it (recommended):\n"
                "  setx POLYGON_API_KEY \"YOUR_KEY\"\n"
                "Then restart VSCode/terminal."
            )
        return PolygonData(PolygonConfig(api_key=key))

    def fetch_aggs_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch aggregates for [start_date, end_date] inclusive.
        Dates in 'YYYY-MM-DD'. Returns OHLCV DataFrame indexed UTC.
        """
        cfg = self.cfg

        tries = 0
        while True:
            try:
                aggs = self.client.get_aggs(
                    ticker=cfg.ticker,
                    multiplier=cfg.multiplier,
                    timespan=cfg.timespan,
                    from_=start_date,
                    to=end_date,
                    limit=cfg.max_per_call,
                )
                break
            except Exception:
                tries += 1
                if tries >= 6:
                    raise
                time.sleep(min(10 * tries, 60))

        rows: List[dict] = []
        for bar in aggs:
            ts_raw = _bar_get(bar, "timestamp", "t")
            o = _bar_get(bar, "open", "o")
            h = _bar_get(bar, "high", "h")
            l = _bar_get(bar, "low", "l")
            c = _bar_get(bar, "close", "c")
            v = _bar_get(bar, "volume", "v")

            if ts_raw is None or o is None or h is None or l is None or c is None:
                continue

            ts = _to_utc_dt(ts_raw)
            rows.append(
                {
                    "timestamp": ts,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v) if v is not None else 0.0,
                }
            )

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df

    def fetch_recent(self, days: int = 120) -> pd.DataFrame:
        """
        Pull last N days. Good for research and live loop.
        """
        end = datetime.now(timezone.utc).date()
        start = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        return self.fetch_aggs_range(start.isoformat(), end.isoformat())
