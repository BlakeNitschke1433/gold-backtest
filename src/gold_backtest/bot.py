# bot.py
# XAUUSD 15m: Polygon data -> indicators -> signals -> backtest -> plots
# Also exports raw candles to debug_market_data.csv so you can inspect them.

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_polygon import PolygonData


# =========================
# Indicators
# =========================
def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    mf = tp * df["volume"]

    pos = np.zeros(len(df), dtype=float)
    neg = np.zeros(len(df), dtype=float)

    tp_vals = tp.values
    mf_vals = mf.values

    for i in range(1, len(df)):
        if tp_vals[i] > tp_vals[i - 1]:
            pos[i] = mf_vals[i]
        else:
            neg[i] = mf_vals[i]

    pos_s = pd.Series(pos, index=df.index).rolling(period).sum()
    neg_s = pd.Series(neg, index=df.index).rolling(period).sum()

    mfr = pos_s / neg_s.replace(0, np.nan)
    out = 100 - (100 / (1 + mfr))
    return out


def add_true_range(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    df["TR"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return df


def detect_sweep(df: pd.DataFrame, lookback: int = 10) -> pd.DataFrame:
    df = df.copy()
    prev_low = df["low"].shift(1).rolling(lookback).min()
    prev_high = df["high"].shift(1).rolling(lookback).max()
    df["sweep_down"] = (df["low"] < prev_low) & (df["close"] > prev_low)
    df["sweep_up"] = (df["high"] > prev_high) & (df["close"] < prev_high)
    return df


def add_exhaustion_flags(
    df: pd.DataFrame,
    atr_k: float = 1.5,
    mfi_os: float = 20,
    mfi_ob: float = 80,
    sweep_lb: int = 10,
) -> pd.DataFrame:
    df = add_true_range(df)
    df = detect_sweep(df, lookback=sweep_lb)

    df["cond_mfi_long"] = df["MFI"] < mfi_os
    df["cond_mfi_short"] = df["MFI"] > mfi_ob
    df["cond_atr_spike"] = df["TR"] > (atr_k * df["ATR"])
    df["cond_sweep_long"] = df["sweep_down"]
    df["cond_sweep_short"] = df["sweep_up"]

    df["exh_score_long"] = (
        df["cond_mfi_long"].astype(int)
        + df["cond_atr_spike"].astype(int)
        + df["cond_sweep_long"].astype(int)
    )
    df["exh_score_short"] = (
        df["cond_mfi_short"].astype(int)
        + df["cond_atr_spike"].astype(int)
        + df["cond_sweep_short"].astype(int)
    )

    df["exhaust_long"] = df["exh_score_long"] >= 2
    df["exhaust_short"] = df["exh_score_short"] >= 2
    return df


def add_entry_signals_v2(df: pd.DataFrame, trigger_window: int = 3) -> pd.DataFrame:
    """
    Exhaustion -> quick trigger inside a short window:
      LONG trigger: close > prior high
      SHORT trigger: close < prior low
    """
    df = df.copy()
    df["trigger_long"] = df["close"] > df["high"].shift(1)
    df["trigger_short"] = df["close"] < df["low"].shift(1)

    exh_long_recent = df["exhaust_long"].rolling(trigger_window).max().fillna(0).astype(bool)
    exh_short_recent = df["exhaust_short"].rolling(trigger_window).max().fillna(0).astype(bool)

    df["entry_long"] = exh_long_recent & df["trigger_long"]
    df["entry_short"] = exh_short_recent & df["trigger_short"]
    return df


def add_session_filter(df: pd.DataFrame, start_hour_utc: int = 7, end_hour_utc: int = 17) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    hours = idx.hour
    df["session_ok"] = (hours >= start_hour_utc) & (hours <= end_hour_utc)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ATR"] = atr(df, 14)
    df["MFI"] = mfi(df, 14)

    df = add_exhaustion_flags(df, atr_k=1.5, mfi_os=20, mfi_ob=80, sweep_lb=10)
    df = add_entry_signals_v2(df, trigger_window=3)

    df = add_session_filter(df, 7, 17)
    df["entry_long"] = df["entry_long"] & df["session_ok"]
    df["entry_short"] = df["entry_short"] & df["session_ok"]
    return df


# =========================
# Data inspection helpers
# =========================
def export_and_validate_raw(df: pd.DataFrame, timeframe_minutes: int = 15, out_csv: str = "debug_market_data.csv"):
    df = df.copy()
    # Ensure UTC + sorted
    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Print quick sample
    print("\n===== RAW DATA SAMPLE (head) =====")
    print(df.head(5).to_string())
    print("\n===== RAW DATA SAMPLE (tail) =====")
    print(df.tail(5).to_string())

    # Basic stats
    print("\n===== RAW DATA STATS =====")
    print(f"Rows: {len(df)}")
    print(f"Start: {df.index.min()} | End: {df.index.max()}")
    print(f"Price range: low={df['low'].min():.2f} high={df['high'].max():.2f}")

    # Spacing / gaps
    expected = pd.Timedelta(minutes=timeframe_minutes)
    diffs = df.index.to_series().diff()
    gaps = diffs[diffs > expected * 1.5]  # gap threshold
    if len(gaps) == 0:
        print("Gaps: none detected (within threshold).")
    else:
        print(f"Gaps: detected {len(gaps)} gaps > {expected * 1.5}. Largest gap: {gaps.max()}")
        print("First few gaps:")
        print(gaps.head(10).to_string())

    # Export to CSV with timestamp column (Excel-friendly)
    export_df = df.reset_index().rename(columns={"timestamp": "timestamp"})
    export_df.rename(columns={export_df.columns[0]: "timestamp"}, inplace=True)  # safe if index name not set
    export_df.to_csv(out_csv, index=False)
    print(f"\nSaved raw candles → {out_csv}\n")


# =========================
# Backtest
# =========================
@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    risk_per_trade: float = 0.005
    stop_atr_mult: float = 1.5
    rr: float = 2.0
    fee_rate: float = 0.0006
    slippage_rate: float = 0.0002
    ambiguous_bar_rule: str = "stop"  # "stop" conservative or "tp"


def backtest(df: pd.DataFrame, cfg: BacktestConfig) -> Tuple[pd.DataFrame, pd.Series]:
    req = ["open", "high", "low", "close", "ATR", "entry_long", "entry_short"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    equity = cfg.initial_equity
    curve = []
    trades = []

    in_pos = False
    side = None
    entry_price = stop_price = tp_price = None
    entry_time = None
    size = None

    for i in range(1, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        curve.append((ts, equity))

        # Manage open trade
        if in_pos:
            high = row["high"]
            low = row["low"]

            if side == "LONG":
                hit_stop = low <= stop_price
                hit_tp = high >= tp_price
            else:
                hit_stop = high >= stop_price
                hit_tp = low <= tp_price

            if hit_stop and hit_tp:
                if cfg.ambiguous_bar_rule == "tp":
                    hit_stop = False
                else:
                    hit_tp = False

            exit_reason = None
            exit_level = None
            if hit_stop:
                exit_reason = "STOP"
                exit_level = stop_price
            elif hit_tp:
                exit_reason = "TP"
                exit_level = tp_price

            if exit_reason:
                if side == "LONG":
                    exit_fill = exit_level * (1 - cfg.slippage_rate)
                    gross = (exit_fill - entry_price) * size
                else:
                    exit_fill = exit_level * (1 + cfg.slippage_rate)
                    gross = (entry_price - exit_fill) * size

                entry_fee = entry_price * size * cfg.fee_rate
                exit_fee = exit_fill * size * cfg.fee_rate
                net = gross - entry_fee - exit_fee

                eq_before = equity
                equity = equity + net
                r_mult = net / (eq_before * cfg.risk_per_trade) if eq_before > 0 else 0.0

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "side": side,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "tp_price": tp_price,
                    "exit_price": exit_fill,
                    "exit_reason": exit_reason,
                    "size": size,
                    "gross_pnl": gross,
                    "net_pnl": net,
                    "r_multiple": r_mult,
                    "equity_after": equity,
                })

                in_pos = False
                side = None
                entry_price = stop_price = tp_price = None
                entry_time = None
                size = None

            continue

        # Enter on next candle open if signal on prev candle
        enter_long = bool(prev["entry_long"])
        enter_short = bool(prev["entry_short"])
        if enter_long and enter_short:
            continue
        if not (enter_long or enter_short):
            continue

        side = "LONG" if enter_long else "SHORT"

        open_price = row["open"]
        if side == "LONG":
            entry_fill = open_price * (1 + cfg.slippage_rate)
        else:
            entry_fill = open_price * (1 - cfg.slippage_rate)

        atr_val = row["ATR"]
        if pd.isna(atr_val) or atr_val <= 0:
            continue

        stop_dist = cfg.stop_atr_mult * atr_val
        if side == "LONG":
            stop = entry_fill - stop_dist
            tp = entry_fill + cfg.rr * stop_dist
        else:
            stop = entry_fill + stop_dist
            tp = entry_fill - cfg.rr * stop_dist

        risk_cash = equity * cfg.risk_per_trade
        risk_per_unit = abs(entry_fill - stop)
        if risk_per_unit <= 0:
            continue

        size = risk_cash / risk_per_unit

        # deduct entry fee immediately
        entry_fee = entry_fill * size * cfg.fee_rate
        equity = equity - entry_fee

        in_pos = True
        entry_time = ts
        entry_price = entry_fill
        stop_price = stop
        tp_price = tp

    trades_df = pd.DataFrame(trades)
    eq_curve = pd.Series([v for _, v in curve], index=[t for t, _ in curve], name="equity")
    return trades_df, eq_curve


def summarize(trades: pd.DataFrame, eq_curve: pd.Series, initial_equity: float):
    print("\n====================")
    print(" BACKTEST SUMMARY")
    print("====================")
    if trades.empty:
        print("No trades taken.")
        return

    n = len(trades)
    wins = int((trades["net_pnl"] > 0).sum())
    losses = n - wins
    winrate = wins / n * 100

    gp = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
    gl = -trades.loc[trades["net_pnl"] <= 0, "net_pnl"].sum()
    pf = gp / gl if gl > 0 else float("inf")

    final_eq = float(trades["equity_after"].iloc[-1])
    ret_pct = (final_eq / initial_equity - 1) * 100

    peak = eq_curve.cummax()
    dd = (eq_curve - peak) / peak
    max_dd = float(dd.min() * 100)

    print(f"Trades: {n} | Wins: {wins} | Losses: {losses} | Winrate: {winrate:.1f}%")
    print(f"Final equity: {final_eq:,.2f} ({ret_pct:.2f}%)")
    print(f"Profit factor: {pf:.2f}")
    print(f"Avg R: {trades['r_multiple'].mean():.2f} | Median R: {trades['r_multiple'].median():.2f}")
    print(f"Max drawdown (equity curve): {max_dd:.2f}%")

    show = ["entry_time","exit_time","side","exit_reason","entry_price","exit_price","net_pnl","r_multiple","equity_after"]
    print("\nLast 10 trades:")
    print(trades[show].tail(10).to_string(index=False))


# =========================
# Run
# =========================
def run_backtest():
    data = PolygonData.from_env()

    # Pull enough recent data for a proper look + indicators
    df = data.fetch_recent(days=365)  # ~1 year of M15
    if df.empty:
        raise RuntimeError("No data returned. Check Polygon plan access for C:XAUUSD.")

    # ✅ Inspect/export raw candles BEFORE features/backtest
    export_and_validate_raw(df, timeframe_minutes=15, out_csv="debug_market_data.csv")

    # Features + backtest
    df = build_features(df)

    cfg = BacktestConfig(
        initial_equity=10_000.0,
        risk_per_trade=0.005,
        stop_atr_mult=1.5,
        rr=2.0,
        fee_rate=0.0006,
        slippage_rate=0.0002,
        ambiguous_bar_rule="stop",
    )
    trades, eq = backtest(df, cfg)
    summarize(trades, eq, cfg.initial_equity)

    # Plot price + entries
    plt.figure(figsize=(16, 7))
    plt.plot(df.index, df["close"], linewidth=1)
    long_idx = df.index[df["entry_long"]]
    short_idx = df.index[df["entry_short"]]
    plt.scatter(long_idx, df.loc[long_idx, "close"], marker="^", s=90, label="Long entry")
    plt.scatter(short_idx, df.loc[short_idx, "close"], marker="v", s=90, label="Short entry")
    plt.title("XAUUSD (Polygon C:XAUUSD) M15 — Close with Entry Signals (Session filtered)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot equity
    plt.figure(figsize=(14, 5))
    plt.plot(eq.index, eq.values, linewidth=1)
    plt.title("Equity Curve")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_backtest()
