"""
Sentinel AI — Backtesting Engine
=================================
Replays historical MT5 OHLCV data through the same signal-generation and
exit-management logic used in production, without placing real trades.

CLI usage:
    python backtester.py --from 2024-10-01 --to 2024-12-31 --initial-balance 10000
    python backtester.py --from 2024-10-01 --to 2024-12-31 --data-csv XAUUSD_M1_2024_clean.csv
"""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None


SYMBOL = "XAUUSD"
POINT = 0.01
CONTRACT_SIZE = 100
MAGIC = 123456

RISK_PER_TRADE = 0.02
ATR_PERIOD = 14
SL_MULTIPLIER = 2.5
MIN_LOT = 0.01
MAX_LOT = 100.0
MAX_POSITIONS = 5

DEFAULT_SPREAD_POINTS = 20
DEFAULT_SLIPPAGE_POINTS = 5
DEFAULT_COMMISSION_PER_LOT = 7.0

BE_BUFFER_POINTS = 10
R_BREAKEVEN = 0.5
R_PARTIAL = 1.5
R_TRAIL_START = 2.0
R_FULL_EXIT = 2.5

SESSION_OPEN_UTC = 8
SESSION_CLOSE_UTC = 21


@dataclass
class Signal:
    action: str
    timestamp: pd.Timestamp
    signal_price: float
    atr_m5: float
    h1_bias: str
    h4_bias: str
    integrated_bias: str
    session_label: str
    day_of_week: str


@dataclass
class Trade:
    trade_id: int
    action: str
    open_time: pd.Timestamp
    entry_price: float
    initial_sl: float
    current_sl: float
    tp: float
    lot_size: float
    remaining_lots: float
    initial_risk_points: float
    signal_time: pd.Timestamp
    h1_bias: str
    h4_bias: str
    integrated_bias: str
    bias_alignment: str
    session_label: str
    day_of_week: str

    stage: str = "ENTRY"
    pnl_partial: float = 0.0
    pnl_full: float = 0.0

    close_time: Optional[pd.Timestamp] = None
    close_price: Optional[float] = None
    close_reason: str = ""
    close_stage: str = ""
    r_multiple: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0


@dataclass
class BacktestState:
    open_trades: list = field(default_factory=list)
    _counter: int = 0

    @property
    def open_position_count(self) -> int:
        return len(self.open_trades)

    def next_id(self) -> int:
        self._counter += 1
        return self._counter


def generate_signal(
    df_m1: pd.DataFrame,
    df_m5: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame,
    state: BacktestState,
) -> Optional[Signal]:
    """
    Trend Continuation Pullback Strategy — mirrors autonomous_scanner.py exactly.

    Entry conditions (ALL must be true):
    1. H4 and H1 EMA50 both agree on direction (integrated_bias)
    2. M5 price pulled back within 0.5 ATR of M5 EMA20 in last 3 bars
    3. Current M1 bar shows rejection: close back through EMA with wick toward EMA
    4. M1 bar has body >= 30% of range (real bar, not doji)
    5. Session gate: 08:00-21:00 UTC
    6. Position cap not exceeded
    """
    if len(df_m5) < 25 or len(df_m1) < 5:
        return None

    # Session gate
    bar_ts = df_m1["time"].iloc[-1]
    hour_utc = pd.Timestamp(bar_ts).hour
    day_of_week = pd.Timestamp(bar_ts).day_name()
    in_session = SESSION_OPEN_UTC <= hour_utc < SESSION_CLOSE_UTC
    if not in_session:
        return None

    if 13 <= hour_utc < 17:
        session_label = "OVERLAP"
    elif 8 <= hour_utc < 13:
        session_label = "LONDON"
    elif 17 <= hour_utc < 21:
        session_label = "NEW_YORK"
    else:
        session_label = "OFF_SESSION"

    if state.open_position_count >= MAX_POSITIONS:
        return None

    # HTF bias
    h1_bias = _ema_bias(df_h1)
    h4_bias = _ema_bias(df_h4)
    integrated_bias = h1_bias if (h1_bias == h4_bias) else "NO_CONFLUENCE"
    if integrated_bias == "NO_CONFLUENCE":
        return None

    # M5 EMA20 pullback zone
    m5_close = df_m5["close"].astype(float)
    m5_ema20 = m5_close.ewm(span=20, adjust=False).mean()
    ema_now = float(m5_ema20.iloc[-1])
    atr = _atr(df_m5)
    pullback_zone = atr * 0.5

    recent_lows  = df_m5["low"].astype(float).iloc[-3:].values
    recent_highs = df_m5["high"].astype(float).iloc[-3:].values
    touched_ema_bullish = any(low  <= ema_now + pullback_zone for low  in recent_lows)
    touched_ema_bearish = any(high >= ema_now - pullback_zone for high in recent_highs)

    # M1 rejection bar
    m1_bar   = df_m1.iloc[-1]
    m1_open  = float(m1_bar["open"])
    m1_close = float(m1_bar["close"])
    m1_high  = float(m1_bar["high"])
    m1_low   = float(m1_bar["low"])
    m1_body  = abs(m1_close - m1_open)
    m1_range = m1_high - m1_low

    # Doji filter
    if m1_range > 0 and (m1_body / m1_range) < 0.3:
        return None

    current_price = m1_close
    atr_m5 = _atr(df_m5)
    timestamp = df_m1["time"].iloc[-1]

    if integrated_bias == "BULLISH" and touched_ema_bullish:
        if m1_close > ema_now and m1_close > m1_open:
            lower_wick = min(m1_open, m1_close) - m1_low
            if lower_wick > atr * 0.3:
                return Signal("BUY", timestamp, current_price, atr_m5, h1_bias, h4_bias,
                              integrated_bias, session_label, day_of_week)

    if integrated_bias == "BEARISH" and touched_ema_bearish:
        if m1_close < ema_now and m1_close < m1_open:
            upper_wick = m1_high - max(m1_open, m1_close)
            if upper_wick > atr * 0.3:
                return Signal("SELL", timestamp, current_price, atr_m5, h1_bias, h4_bias,
                              integrated_bias, session_label, day_of_week)

    return None


def _ema_bias(df: pd.DataFrame, span: int = 50) -> str:
    if df is None or len(df) < span:
        return "NO_CONFLUENCE"
    ema = df["close"].ewm(span=span, adjust=False).mean().iloc[-1]
    price = df["close"].iloc[-1]
    if price > ema * 1.001:
        return "BULLISH"
    if price < ema * 0.999:
        return "BEARISH"
    return "NO_CONFLUENCE"


def _atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    if df is None or len(df) < period + 1:
        return 1.0
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))


def calculate_lot_size(balance: float, sl_distance_points: float, risk_percent: float = RISK_PER_TRADE) -> float:
    risk_dollars = balance * risk_percent
    dollar_risk_per_lot = sl_distance_points * POINT * CONTRACT_SIZE
    if dollar_risk_per_lot <= 0:
        return MIN_LOT
    lot = risk_dollars / dollar_risk_per_lot
    return float(max(MIN_LOT, min(round(lot, 2), MAX_LOT)))


def simulate_fill(signal: Signal, next_bar: pd.Series, balance: float,
                  spread_points: float = DEFAULT_SPREAD_POINTS,
                  slippage_points: float = DEFAULT_SLIPPAGE_POINTS) -> Trade:
    atr_points = max((signal.atr_m5 / POINT) * SL_MULTIPLIER, 30.0)
    spread_price = spread_points * POINT
    slippage_price = slippage_points * POINT

    if signal.action == "BUY":
        entry = next_bar["open"] + spread_price + slippage_price
        sl = entry - atr_points * POINT
        tp = entry + atr_points * POINT * R_FULL_EXIT
    else:
        entry = next_bar["open"] - slippage_price
        sl = entry + atr_points * POINT
        tp = entry - atr_points * POINT * R_FULL_EXIT

    lot = calculate_lot_size(balance, atr_points)

    if signal.integrated_bias == "NO_CONFLUENCE":
        bias_alignment = "NO_CONFLUENCE"
    elif signal.action == "BUY" and signal.integrated_bias == "BULLISH":
        bias_alignment = "ALIGNED"
    elif signal.action == "SELL" and signal.integrated_bias == "BEARISH":
        bias_alignment = "ALIGNED"
    else:
        bias_alignment = "COUNTER_TREND"

    return Trade(
        trade_id=0,
        action=signal.action,
        open_time=next_bar["time"],
        entry_price=entry,
        initial_sl=sl,
        current_sl=sl,
        tp=tp,
        lot_size=lot,
        remaining_lots=lot,
        initial_risk_points=atr_points,
        signal_time=signal.timestamp,
        h1_bias=signal.h1_bias,
        h4_bias=signal.h4_bias,
        integrated_bias=signal.integrated_bias,
        bias_alignment=bias_alignment,
        session_label=signal.session_label,
        day_of_week=signal.day_of_week,
    )


def process_bar_for_trade(trade: Trade, bar: pd.Series,
                           commission_per_lot: float = DEFAULT_COMMISSION_PER_LOT,
                           spread_points: float = DEFAULT_SPREAD_POINTS) -> bool:
    is_buy = trade.action == "BUY"
    e = trade.entry_price
    r = trade.initial_risk_points * POINT
    hi = bar["high"]
    lo = bar["low"]
    spread_price = spread_points * POINT

    _update_excursion(trade, hi, lo)

    be_target = e + r if is_buy else e - r
    partial_target = e + r * R_PARTIAL if is_buy else e - r * R_PARTIAL
    trail_target = e + r * R_TRAIL_START if is_buy else e - r * R_TRAIL_START
    full_target = e + r * R_FULL_EXIT if is_buy else e - r * R_FULL_EXIT

    be_reached = (hi >= be_target) if is_buy else (lo <= be_target)
    partial_reached = (hi >= partial_target) if is_buy else (lo <= partial_target)
    trail_reached = (hi >= trail_target) if is_buy else (lo <= trail_target)
    full_reached = (hi >= full_target) if is_buy else (lo <= full_target)
    sl_hit = (lo <= trade.current_sl) if is_buy else (hi >= trade.current_sl)

    sl_fill = min(lo, trade.current_sl) if is_buy else max(hi, trade.current_sl)

    if trade.stage == "ENTRY" and be_reached:
        be_buffer = BE_BUFFER_POINTS * POINT
        trade.current_sl = (e + be_buffer) if is_buy else (e - be_buffer)
        trade.stage = "BREAKEVEN"
        sl_hit = (lo <= trade.current_sl) if is_buy else (hi >= trade.current_sl)
        sl_fill = min(lo, trade.current_sl) if is_buy else max(hi, trade.current_sl)

    if trade.stage == "BREAKEVEN" and partial_reached:
        partial_lots = round(trade.lot_size * 0.5, 2)
        if partial_lots >= MIN_LOT:
            close_price = (partial_target - spread_price) if is_buy else (partial_target + spread_price)
            commission = commission_per_lot * partial_lots
            trade.pnl_partial = _pnl(trade.action, e, close_price, partial_lots, commission)
            trade.remaining_lots = max(round(trade.lot_size - partial_lots, 2), MIN_LOT)
        trade.stage = "PARTIAL_CLOSED"

    if trade.stage == "PARTIAL_CLOSED" and trail_reached:
        trail_sl = (e + r * R_PARTIAL) if is_buy else (e - r * R_PARTIAL)
        trade.current_sl = max(trade.current_sl, trail_sl) if is_buy else min(trade.current_sl, trail_sl)
        trade.stage = "TRAILING"
        sl_hit = (lo <= trade.current_sl) if is_buy else (hi >= trade.current_sl)
        sl_fill = min(lo, trade.current_sl) if is_buy else max(hi, trade.current_sl)

    if trade.stage in {"PARTIAL_CLOSED", "TRAILING"} and full_reached:
        close_price = (full_target - spread_price) if is_buy else (full_target + spread_price)
        commission = commission_per_lot * trade.remaining_lots
        remainder_pnl = _pnl(trade.action, e, close_price, trade.remaining_lots, commission)
        _close_trade(trade, bar["time"], close_price, "FULL_EXIT", trade.pnl_partial + remainder_pnl)
        return True

    if sl_hit:
        commission = commission_per_lot * trade.remaining_lots
        sl_pnl = _pnl(trade.action, e, sl_fill, trade.remaining_lots, commission)
        reason = "SL" if trade.stage == "ENTRY" else "TRAIL_SL"
        _close_trade(trade, bar["time"], sl_fill, reason, trade.pnl_partial + sl_pnl)
        return True

    return False


def _pnl(action: str, entry: float, close: float, lots: float, commission: float) -> float:
    direction = 1.0 if action == "BUY" else -1.0
    gross = (close - entry) * direction * lots * CONTRACT_SIZE
    return gross - commission


def _update_excursion(trade: Trade, hi: float, lo: float):
    r = trade.initial_risk_points * POINT
    if r <= 0:
        return
    if trade.action == "BUY":
        favorable = max(0.0, hi - trade.entry_price)
        adverse = max(0.0, trade.entry_price - lo)
    else:
        favorable = max(0.0, trade.entry_price - lo)
        adverse = max(0.0, hi - trade.entry_price)
    trade.mfe_r = max(trade.mfe_r, favorable / r)
    trade.mae_r = max(trade.mae_r, adverse / r)


def _close_trade(trade: Trade, close_time, close_price: float, reason: str, total_pnl: float):
    trade.close_stage = trade.stage
    trade.close_time = close_time
    trade.close_price = close_price
    trade.close_reason = reason
    trade.pnl_full = total_pnl
    trade.stage = "CLOSED"
    risk_dollar = trade.initial_risk_points * POINT * trade.lot_size * CONTRACT_SIZE
    trade.r_multiple = (total_pnl / risk_dollar) if risk_dollar > 0 else 0.0


def fetch_historical_data(symbol: str, from_date: datetime, to_date: datetime) -> dict:
    if not _MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 package not installed.")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"MT5 symbol_select({symbol}) failed: {mt5.last_error()}")

    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}
    tf_minutes = {"M1": 1, "M5": 5, "H1": 60, "H4": 240}
    data: dict[str, pd.DataFrame] = {}
    try:
        for name, tf in tf_map.items():
            span_minutes = max(int((to_date - from_date).total_seconds() / 60), tf_minutes[name])
            bars_needed = int(span_minutes / tf_minutes[name]) + 500
            chunks = []
            start_pos = 0
            remaining = bars_needed
            while remaining > 0:
                take = min(50_000, remaining)
                rates = mt5.copy_rates_from_pos(symbol, tf, start_pos, take)
                if rates is None or len(rates) == 0:
                    break
                chunks.append(pd.DataFrame(rates))
                start_pos += take
                remaining -= take
            if not chunks:
                raise RuntimeError(f"No {name} data. MT5 error: {mt5.last_error()}")
            df = pd.concat(chunks, ignore_index=True)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.sort_values("time").reset_index(drop=True)
            df = df[(df["time"] >= pd.Timestamp(from_date)) & (df["time"] <= pd.Timestamp(to_date))]
            if df.empty:
                raise RuntimeError(f"No {name} data after clipping.")
            data[name] = df
            print(f"[DATA] {name}: {len(df):,} bars")
    finally:
        mt5.shutdown()
    return data


def load_csv_data(csv_path: str, from_date: datetime, to_date: datetime) -> dict:
    print(f"[DATA] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    df = df[(df["time"] >= pd.Timestamp(from_date)) & (df["time"] <= pd.Timestamp(to_date))]
    if df.empty:
        raise RuntimeError(f"No data in CSV for range {from_date.date()} to {to_date.date()}")

    df = df.set_index("time")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "tick_volume" in df.columns:
        agg["tick_volume"] = "sum"

    data = {}
    for name, rule in [("M1", "1min"), ("M5", "5min"), ("H1", "1h"), ("H4", "4h")]:
        resampled = df.resample(rule).agg(agg).dropna(subset=["close"]).reset_index()
        data[name] = resampled
        print(f"[DATA] {name}: {len(resampled):,} bars")
    return data


def run_backtest(
    from_date: datetime,
    to_date: datetime,
    initial_balance: float = 10_000.0,
    spread_points: float = DEFAULT_SPREAD_POINTS,
    slippage_points: float = DEFAULT_SLIPPAGE_POINTS,
    commission_per_lot: float = DEFAULT_COMMISSION_PER_LOT,
    data: Optional[dict] = None,
    csv_path: Optional[str] = None,
) -> dict:
    if data is None:
        if csv_path:
            data = load_csv_data(csv_path, from_date, to_date)
        else:
            data = fetch_historical_data(SYMBOL, from_date, to_date)

    df_m1 = data["M1"].reset_index(drop=True)
    df_m5 = data["M5"].reset_index(drop=True)
    df_h1 = data["H1"].reset_index(drop=True)
    df_h4 = data["H4"].reset_index(drop=True)

    m5_times = df_m5["time"].astype("int64").values
    h1_times = df_h1["time"].astype("int64").values
    h4_times = df_h4["time"].astype("int64").values

    balance = initial_balance
    equity_records: list[dict] = []
    all_trades: list[Trade] = []
    state = BacktestState()

    print(f"\n[BACKTEST] {from_date.date()} to {to_date.date()}")
    print(f"[CONFIG]   Spread={spread_points}pt  Slippage={slippage_points}pt  "
          f"Commission=${commission_per_lot}/lot  Balance=${initial_balance:,.2f}\n")

    WARMUP = 100

    for i in range(WARMUP, len(df_m1)):
        bar = df_m1.iloc[i]
        bar_time = bar["time"]

        still_open = []
        for trade in state.open_trades:
            closed = process_bar_for_trade(trade, bar, commission_per_lot, spread_points)
            if closed:
                balance += trade.pnl_full
                all_trades.append(trade)
            else:
                still_open.append(trade)
        state.open_trades = still_open

        m1_window = df_m1.iloc[max(0, i - 99): i + 1]
        bar_time_ns = int(pd.Timestamp(bar_time).value)
        m5_idx = int(np.searchsorted(m5_times, bar_time_ns, side="right"))
        h1_idx = int(np.searchsorted(h1_times, bar_time_ns, side="right"))
        h4_idx = int(np.searchsorted(h4_times, bar_time_ns, side="right"))
        m5_window = df_m5.iloc[max(0, m5_idx - 100): m5_idx]
        h1_window = df_h1.iloc[max(0, h1_idx - 100): h1_idx]
        h4_window = df_h4.iloc[max(0, h4_idx - 100): h4_idx]

        signal = generate_signal(m1_window, m5_window, h1_window, h4_window, state)

        if signal is not None and (i + 1) < len(df_m1):
            next_bar = df_m1.iloc[i + 1]
            trade = simulate_fill(signal, next_bar, balance, spread_points, slippage_points)
            trade.trade_id = state.next_id()
            state.open_trades.append(trade)

        open_pnl = sum(
            _pnl(t.action, t.entry_price, float(bar["close"]), t.remaining_lots, 0.0)
            for t in state.open_trades
        )
        equity_records.append({
            "time": bar_time,
            "balance": balance,
            "equity": balance + open_pnl,
            "open_trades": len(state.open_trades),
        })

    last_bar = df_m1.iloc[-1] if not df_m1.empty else None
    for trade in state.open_trades:
        close_px = float(last_bar["close"]) if last_bar is not None else trade.entry_price
        commission = commission_per_lot * trade.remaining_lots
        eod_pnl = _pnl(trade.action, trade.entry_price, close_px, trade.remaining_lots, commission)
        _close_trade(trade, last_bar["time"] if last_bar is not None else None,
                     close_px, "EOD", trade.pnl_partial + eod_pnl)
        balance += trade.pnl_full
        all_trades.append(trade)
    state.open_trades.clear()

    eq_df = pd.DataFrame(equity_records)
    metrics = _calculate_metrics(all_trades, eq_df, initial_balance)
    return {"metrics": metrics, "trades": all_trades, "equity_curve": eq_df}


def _calculate_metrics(trades: list[Trade], equity_df: pd.DataFrame, initial_balance: float) -> dict:
    if not trades:
        return {"error": "No trades executed during the test window."}

    pnls = [t.pnl_full for t in trades]
    r_vals = [t.r_multiple for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    win_rate = len(winners) / len(trades)
    gross_profit = sum(winners)
    gross_loss = abs(sum(losers)) if losers else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    final_balance = initial_balance + sum(pnls)
    sharpe, sortino = _risk_ratios(equity_df)

    equity = equity_df["equity"].values if not equity_df.empty else np.array([initial_balance])
    peak = np.maximum.accumulate(equity)
    dd_dollar = peak - equity
    dd_pct = np.where(peak > 0, dd_dollar / peak, 0.0)

    walk_forward = _walk_forward_split_metrics(trades)
    monte_carlo = _monte_carlo_sharpe_stability(trades)

    return {
        "total_trades": len(trades),
        "winners": len(winners),
        "losers": len(losers),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 3),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(float(np.max(dd_pct)) * 100, 2),
        "max_drawdown_dollar": round(float(np.max(dd_dollar)), 2),
        "average_r": round(float(np.mean(r_vals)), 3),
        "expectancy_usd": round(float(np.mean(pnls)), 2),
        "total_pnl": round(sum(pnls), 2),
        "initial_balance": round(initial_balance, 2),
        "final_balance": round(final_balance, 2),
        "total_return_pct": round((final_balance - initial_balance) / initial_balance * 100, 2),
        "avg_win_usd": round(float(np.mean(winners)), 2) if winners else 0.0,
        "avg_loss_usd": round(float(np.mean(losers)), 2) if losers else 0.0,
        "walk_forward": walk_forward,
        "monte_carlo": monte_carlo,
    }


def _trade_return_series(trades: list[Trade], initial_balance: float = 10_000.0) -> np.ndarray:
    ordered = sorted(trades, key=lambda t: t.close_time or t.open_time or pd.Timestamp.min)
    if not ordered:
        return np.array([], dtype=float)
    balance = float(initial_balance)
    returns = []
    for trade in ordered:
        denom = balance if abs(balance) > 1e-9 else 1.0
        returns.append(float(trade.pnl_full) / denom)
        balance += float(trade.pnl_full)
    return np.array(returns, dtype=float)


def _annualized_sharpe(returns: np.ndarray) -> float:
    if returns.size < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std <= 0:
        return 0.0
    return float((np.mean(returns) / std) * np.sqrt(252.0))


def _walk_forward_split_metrics(trades: list[Trade], split_ratio: float = 0.7) -> dict:
    returns = _trade_return_series(trades)
    n = int(returns.size)
    if n < 2:
        return {"split_ratio_in_sample": split_ratio, "in_sample_trades": n,
                "out_of_sample_trades": 0, "in_sample_sharpe": 0.0,
                "out_of_sample_sharpe": 0.0, "oos_drop_vs_insample_pct": 0.0,
                "flag_oos_drop_gt_30pct": False}
    split_idx = max(1, min(n - 1, int(round(n * split_ratio))))
    in_sample = returns[:split_idx]
    out_sample = returns[split_idx:]
    in_sharpe = _annualized_sharpe(in_sample)
    out_sharpe = _annualized_sharpe(out_sample)
    drop_pct = max(0.0, (in_sharpe - out_sharpe) / in_sharpe) if in_sharpe > 0 else 0.0
    return {
        "split_ratio_in_sample": split_ratio,
        "in_sample_trades": int(in_sample.size),
        "out_of_sample_trades": int(out_sample.size),
        "in_sample_sharpe": round(in_sharpe, 6),
        "out_of_sample_sharpe": round(out_sharpe, 6),
        "oos_drop_vs_insample_pct": round(drop_pct * 100.0, 4),
        "flag_oos_drop_gt_30pct": bool(drop_pct > 0.30),
    }


def _monte_carlo_sharpe_stability(trades: list[Trade], iterations: int = 1000, seed: int = 42) -> dict:
    returns = _trade_return_series(trades)
    n = int(returns.size)
    if n < 2:
        return {"iterations": iterations, "sample_size": n, "base_sharpe": 0.0,
                "mean_sharpe": 0.0, "std_sharpe": 0.0, "p05_sharpe": 0.0,
                "p50_sharpe": 0.0, "p95_sharpe": 0.0, "prob_sharpe_positive": 0.0}
    rng = np.random.default_rng(seed)
    base_sharpe = _annualized_sharpe(returns)
    sampled = np.array([_annualized_sharpe(rng.choice(returns, size=n, replace=True))
                        for _ in range(iterations)])
    return {
        "iterations": iterations, "sample_size": n,
        "base_sharpe": round(base_sharpe, 6),
        "mean_sharpe": round(float(np.mean(sampled)), 6),
        "std_sharpe": round(float(np.std(sampled, ddof=1)), 6),
        "p05_sharpe": round(float(np.percentile(sampled, 5)), 6),
        "p50_sharpe": round(float(np.percentile(sampled, 50)), 6),
        "p95_sharpe": round(float(np.percentile(sampled, 95)), 6),
        "prob_sharpe_positive": round(float(np.mean(sampled > 0.0)), 6),
    }


def _risk_ratios(equity_df: pd.DataFrame) -> tuple[float, float]:
    if equity_df.empty or "time" not in equity_df.columns:
        return 0.0, 0.0
    try:
        eq = equity_df.set_index("time")["equity"]
        daily = eq.resample("1D").last().ffill().dropna()
        rets = daily.pct_change().dropna()
        if len(rets) < 2:
            return 0.0, 0.0
        mean_r = rets.mean()
        std_r = rets.std()
        sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0
        downside = rets[rets < 0]
        std_d = downside.std()
        sortino = (mean_r / std_d) * np.sqrt(252) if len(downside) > 1 and std_d > 0 else 0.0
        return float(sharpe), float(sortino)
    except Exception:
        return 0.0, 0.0


def save_results(results: dict, run_id: str, output_dir: str = "./backtest_results") -> str:
    run_path = Path(output_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    trades = results["trades"]
    eq_df = results["equity_curve"]
    metrics = results["metrics"]

    if trades:
        rows = [{
            "trade_id": t.trade_id, "action": t.action,
            "open_time": t.open_time, "close_time": t.close_time,
            "entry_price": round(t.entry_price, 5), "initial_sl": round(t.initial_sl, 5),
            "tp": round(t.tp, 5), "lot_size": t.lot_size,
            "close_price": round(t.close_price, 5) if t.close_price else None,
            "close_reason": t.close_reason, "stage_at_close": t.close_stage,
            "pnl_usd": round(t.pnl_full, 2), "r_multiple": round(t.r_multiple, 3),
            "initial_risk_points": t.initial_risk_points, "signal_time": t.signal_time,
            "h1_bias": t.h1_bias, "h4_bias": t.h4_bias,
            "integrated_bias": t.integrated_bias, "bias_alignment": t.bias_alignment,
            "session_label": t.session_label, "day_of_week": t.day_of_week,
            "mfe_r": round(t.mfe_r, 3), "mae_r": round(t.mae_r, 3),
        } for t in trades]
        pd.DataFrame(rows).to_csv(run_path / "trades.csv", index=False)
        print(f"[OUTPUT] trades.csv ({len(rows)} rows)")

    if not eq_df.empty:
        eq_df.to_csv(run_path / "equity_curve.csv", index=False)

    with open(run_path / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    _plot_equity_curve(eq_df, metrics, run_path, run_id)
    return str(run_path)


def _plot_equity_curve(eq_df, metrics, run_path, run_id):
    if eq_df.empty:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#adb5bd")
        ax.spines[:].set_color("#30363d")

    times = eq_df["time"]
    equity = eq_df["equity"]
    balance = eq_df["balance"]
    ax1.plot(times, equity, color="#58a6ff", linewidth=1.0, label="Equity", zorder=3)
    ax1.plot(times, balance, color="#6e7681", linewidth=0.7, alpha=0.8, label="Balance", zorder=2)
    ax1.fill_between(times, balance, equity, alpha=0.12, color="#58a6ff")
    ax1.set_ylabel("USD", color="#adb5bd")
    ax1.legend(loc="upper left", fontsize=8, facecolor="#161b22", labelcolor="#adb5bd", framealpha=0.6)
    ax1.grid(True, alpha=0.15, color="#30363d")
    ax1.set_title(f"Sentinel AI — Backtest | {run_id}", color="#f0f6fc", fontsize=10, pad=8)

    peak = equity.cummax()
    dd_pct = ((equity - peak) / peak.replace(0, np.nan)) * 100
    ax2.fill_between(times, dd_pct, 0, color="#f85149", alpha=0.6, label="Drawdown %")
    ax2.set_ylabel("DD %", color="#adb5bd")
    ax2.set_xlabel("Date", color="#adb5bd")
    ax2.legend(loc="lower left", fontsize=8, facecolor="#161b22", labelcolor="#adb5bd", framealpha=0.6)
    ax2.grid(True, alpha=0.15, color="#30363d")

    m = metrics
    footer = (f"Trades: {m.get('total_trades',0)}  |  Win: {m.get('win_rate_pct',0):.1f}%  |  "
              f"PF: {m.get('profit_factor',0):.2f}  |  Sharpe: {m.get('sharpe_ratio',0):.2f}  |  "
              f"MaxDD: {m.get('max_drawdown_pct',0):.1f}%  |  Return: {m.get('total_return_pct',0):.1f}%")
    fig.text(0.5, 0.005, footer, ha="center", fontsize=8.5, color="#8b949e")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(run_path / "equity_curve.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("[OUTPUT] equity_curve.png")


def _print_summary(metrics: dict, run_path: str):
    w = 55
    print("\n" + "-" * w)
    print("  SENTINEL AI — BACKTEST RESULTS")
    print("-" * w)
    for k, v in metrics.items():
        if k in ("walk_forward", "monte_carlo"):
            continue
        print(f"  {k.replace('_',' ').title():<32} {v}")
    print("-" * w)
    print(f"  Saved to: {run_path}")
    print("-" * w + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sentinel AI Backtesting Engine")
    parser.add_argument("--from", dest="from_date", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--spread", type=float, default=DEFAULT_SPREAD_POINTS)
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE_POINTS)
    parser.add_argument("--commission", type=float, default=DEFAULT_COMMISSION_PER_LOT)
    parser.add_argument("--output-dir", default="./backtest_results")
    parser.add_argument("--data-csv", default=None, help="Path to OHLC CSV (offline mode)")
    args = parser.parse_args()

    from_dt = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    to_dt = datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    run_id = f"{args.from_date}__{args.to_date}__{uuid.uuid4().hex[:8]}"

    results = run_backtest(
        from_date=from_dt,
        to_date=to_dt,
        initial_balance=args.initial_balance,
        spread_points=args.spread,
        slippage_points=args.slippage,
        commission_per_lot=args.commission,
        csv_path=args.data_csv,
    )

    run_path = save_results(results, run_id, args.output_dir)
    _print_summary(results["metrics"], run_path)


if __name__ == "__main__":
    main()