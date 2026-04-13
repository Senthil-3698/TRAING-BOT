from __future__ import annotations

import argparse
import pathlib

import pandas as pd

from position_sizing import calculate_xauusd_lot_size


def build_atr_14(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def run(csv_path: str, equity: float, risk_percent: float) -> None:
    path = pathlib.Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns {required}. Missing: {missing}")

    df = df.copy()
    df["atr14"] = build_atr_14(df)
    sample = df.dropna(subset=["atr14"]).tail(25)
    if sample.empty:
        raise ValueError("Not enough rows to compute ATR14")

    print("idx,close,atr14,lot,sl_distance,sl_price,allowed,reason")
    for idx, row in sample.iterrows():
        close = float(row["close"])
        atr = float(row["atr14"])

        result = calculate_xauusd_lot_size(
            equity=equity,
            risk_percent=risk_percent,
            atr_14=atr,
            entry_price=close,
            action="BUY",
            tick_value=1.0,
            tick_size=0.01,
            volume_step=0.01,
            volume_min=0.01,
            volume_max=100.0,
            point=0.01,
            sl_atr_multiplier=1.5,
        )

        print(
            f"{idx},{close:.2f},{atr:.4f},{result.lot_size:.2f},"
            f"{result.stop_loss_distance_price:.4f},{result.stop_loss_price:.2f},"
            f"{result.allowed},{result.reason}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate dynamic XAUUSD lot sizing with historical ATR data")
    parser.add_argument("--csv", required=True, help="Path to OHLC CSV (must contain open,high,low,close)")
    parser.add_argument("--equity", type=float, default=10000.0)
    parser.add_argument("--risk", type=float, default=1.0, help="Risk percent per trade")
    args = parser.parse_args()

    run(csv_path=args.csv, equity=args.equity, risk_percent=args.risk)
