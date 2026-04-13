"""
Sentinel AI — Backtesting Engine
=================================
Replays historical MT5 OHLCV data through the same signal-generation and
exit-management logic used in production, without placing real trades.

CLI usage:
    python backtester.py --from 2024-01-01 --to 2024-12-31 --initial-balance 10000

Architecture:
    generate_signal()        ← pure function, refactored from autonomous_scanner.analyze_and_trade()
    simulate_fill()          ← realistic spread/slippage/commission model
    process_bar_for_trade()  ← exact exit state machine from exit_manager.py
    run_backtest()           ← main loop; feeds bars to the above functions
    save_results()           ← writes trades.csv, metrics.json, equity_curve.png
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
matplotlib.use("Agg")  # Non-interactive backend — safe in headless envs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Guard MT5 import so the module can be imported in test environments
try:
    import MetaTrader5 as mt5  # type: ignore
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    mt5 = None  # type: ignore


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SYMBOL = "XAUUSD"
POINT = 0.01          # 1 point = $0.01 in XAUUSD price
CONTRACT_SIZE = 100   # 1 standard lot = 100 troy oz
MAGIC = 123456

# Risk / position-sizing (mirrors mt5_executor.py)
RISK_PER_TRADE = 0.02
ATR_PERIOD = 14
SL_MULTIPLIER = 1.5   # SL = 1.5 * ATR
MIN_LOT = 0.01
MAX_LOT = 100.0
MAX_POSITIONS = 5

# Fill friction
DEFAULT_SPREAD_POINTS = 20
DEFAULT_SLIPPAGE_POINTS = 5
DEFAULT_COMMISSION_PER_LOT = 7.0   # USD round-turn

# Exit state machine thresholds (in R units)
BE_BUFFER_POINTS = 10   # points above entry for breakeven SL
R_BREAKEVEN = 1.0
R_PARTIAL = 1.5         # close 50 % of position
R_TRAIL_START = 2.0     # begin trailing; lock SL at +1.5 R
R_FULL_EXIT = 2.5       # close remainder

# Session gate (UTC hours) — London + NY overlap for XAUUSD
SESSION_OPEN_UTC = 8
SESSION_CLOSE_UTC = 21


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class Signal:
    action: str            # "BUY" | "SELL"
    timestamp: pd.Timestamp
    signal_price: float    # close price at signal bar
    atr_m5: float          # ATR at signal time (used for lot sizing)


@dataclass
class Trade:
    trade_id: int
    action: str            # "BUY" | "SELL"
    open_time: pd.Timestamp
    entry_price: float     # actual fill price (after spread + slippage)
    initial_sl: float
    current_sl: float
    tp: float              # 2.5 R target
    lot_size: float
    remaining_lots: float
    initial_risk_points: float   # SL distance in points

    # Mutable state machine
    stage: str = "ENTRY"   # ENTRY → BREAKEVEN → PARTIAL_CLOSED → TRAILING → CLOSED

    # Accumulated P&L
    pnl_partial: float = 0.0   # P&L from 50 % partial close
    pnl_full: float = 0.0      # total closed P&L (partial + remainder)

    # Outcome (set on close)
    close_time: Optional[pd.Timestamp] = None
    close_price: Optional[float] = None
    close_reason: str = ""
    r_multiple: float = 0.0


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


# ─────────────────────────────────────────────────────────────
# Pure Signal Generation
# (refactored from autonomous_scanner.analyze_and_trade)
# ─────────────────────────────────────────────────────────────

def generate_signal(
    df_m1: pd.DataFrame,
    df_m5: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_h4: pd.DataFrame,
    state: BacktestState,
) -> Optional[Signal]:
    """
    Pure, side-effect-free signal generator.

    Replicates the triple-confluence logic from autonomous_scanner.analyze_and_trade():
      Confluence 1 — EMA 50 master trend (M1)
      Confluence 2 — SMA 5 / SMA 13 crossover (M1)
      Confluence 3 — Volume filter (M1)

    Additional filters present in the live orchestration layer:
      H1 / H4 EMA-50 integrated bias  (mirrors orchestrator get_integrated_bias)
      Session gate (London + NY hours, UTC)
      Max open-position gate
    """
    # ── Minimum data guard ──────────────────────────────────────
    if len(df_m1) < 50:
        return None

    # ── Confluence 1: Master Trend — EMA 50 (M1) ─────────────────
    ema_50 = df_m1["close"].ewm(span=50, adjust=False).mean().iloc[-1]
    current_price = df_m1["close"].iloc[-1]
    trend_bullish = current_price > ema_50
    trend_bearish = current_price < ema_50

    # ── Confluence 2: Entry Timing — SMA 5 / SMA 13 crossover ────
    sma_5 = df_m1["close"].rolling(window=5).mean()
    sma_13 = df_m1["close"].rolling(window=13).mean()

    sma_5_curr = sma_5.iloc[-1]
    sma_13_curr = sma_13.iloc[-1]
    sma_5_prev = sma_5.iloc[-2]
    sma_13_prev = sma_13.iloc[-2]

    bullish_cross = (sma_5_prev <= sma_13_prev) and (sma_5_curr > sma_13_curr)
    bearish_cross = (sma_5_prev >= sma_13_prev) and (sma_5_curr < sma_13_curr)

    # ── Confluence 3: Volume Filter ───────────────────────────────
    if "tick_volume" in df_m1.columns and df_m1["tick_volume"].iloc[-10:].sum() > 0:
        vol = df_m1["tick_volume"]
    else:
        vol = df_m1["high"] - df_m1["low"]   # spread-range proxy

    avg_vol_10 = vol.iloc[-10:].mean()
    volume_strong = vol.iloc[-1] > avg_vol_10

    # ── H1 / H4 Integrated Bias ───────────────────────────────────
    h1_bias = _ema_bias(df_h1)
    h4_bias = _ema_bias(df_h4)
    integrated_bias = h1_bias if (h1_bias == h4_bias) else "NO_CONFLUENCE"

    # ── Session Gate ──────────────────────────────────────────────
    bar_ts = df_m1["time"].iloc[-1]
    hour_utc = pd.Timestamp(bar_ts).hour
    in_session = SESSION_OPEN_UTC <= hour_utc < SESSION_CLOSE_UTC

    # ── Position Cap ──────────────────────────────────────────────
    if state.open_position_count >= MAX_POSITIONS:
        return None
    if not in_session:
        return None

    # ── Decision ──────────────────────────────────────────────────
    atr_m5 = _atr(df_m5)
    timestamp = df_m1["time"].iloc[-1]

    if (trend_bullish and bullish_cross and volume_strong
            and integrated_bias in ("BULLISH", "NO_CONFLUENCE")):
        return Signal("BUY", timestamp, current_price, atr_m5)

    if (trend_bearish and bearish_cross and volume_strong
            and integrated_bias in ("BEARISH", "NO_CONFLUENCE")):
        return Signal("SELL", timestamp, current_price, atr_m5)

    return None


def _ema_bias(df: pd.DataFrame, span: int = 50) -> str:
    """EMA-50 bias on any timeframe DataFrame. Returns BULLISH / BEARISH / NO_CONFLUENCE."""
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
    """Average True Range from a historical OHLCV DataFrame (no MT5 needed)."""
    if df is None or len(df) < period + 1:
        return 1.0   # safe fallback
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))


# ─────────────────────────────────────────────────────────────
# Fill Simulation
# ─────────────────────────────────────────────────────────────

def calculate_lot_size(
    balance: float,
    sl_distance_points: float,
    risk_percent: float = RISK_PER_TRADE,
) -> float:
    """
    Mirrors mt5_executor.calculate_dynamic_lot_size() without an MT5 connection.

    Formula:
        lot = (balance × risk%) / (sl_points × POINT × CONTRACT_SIZE)

    For XAUUSD: 1 lot = 100 oz; a 1-point ($0.01) move per lot = $1.00 P&L.
    """
    risk_dollars = balance * risk_percent
    dollar_risk_per_lot = sl_distance_points * POINT * CONTRACT_SIZE
    if dollar_risk_per_lot <= 0:
        return MIN_LOT
    lot = risk_dollars / dollar_risk_per_lot
    return float(max(MIN_LOT, min(round(lot, 2), MAX_LOT)))


def simulate_fill(
    signal: Signal,
    next_bar: pd.Series,
    balance: float,
    spread_points: float = DEFAULT_SPREAD_POINTS,
    slippage_points: float = DEFAULT_SLIPPAGE_POINTS,
) -> Trade:
    """
    Simulate a realistic market-order fill at the open of the bar AFTER the signal bar.

    Spread model (XAUUSD):
      BUY  → fills at ask = open + spread; additional slippage adds cost
      SELL → fills at bid = open;          slippage subtracts from fill

    SL  = entry ± (1.5 × ATR) in points   (same formula as mt5_executor)
    TP  = entry ± (2.5 R)                  (full exit target per exit machine)
    Minimum SL distance: 30 points (broker constraint proxy)
    """
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
    )


# ─────────────────────────────────────────────────────────────
# Exit State Machine
# (mirrors exit_manager.manage_exits — bar-level simulation)
# ─────────────────────────────────────────────────────────────

def process_bar_for_trade(
    trade: Trade,
    bar: pd.Series,
    commission_per_lot: float = DEFAULT_COMMISSION_PER_LOT,
    spread_points: float = DEFAULT_SPREAD_POINTS,
) -> bool:
    """
    Advance one M1 bar through the exit state machine for a single Trade.
    Mutates `trade` in-place.  Returns True when the trade is fully closed.

    Exit layers (identical to exit_manager.py):
      ENTRY       → BREAKEVEN    at +1 R  (SL moved to entry + 10 pts buffer)
      BREAKEVEN   → PARTIAL_CLOSED at +1.5 R  (close 50 %)
      PARTIAL_CLOSED → TRAILING  at +2 R  (trail SL to +1.5 R)
      TRAILING    → CLOSED       at +2.5 R (full exit)

    SL fills at the WORST price within the bar (conservative / realistic).
    TP / partial fills at the exact target price (mid of spread on exit).
    """
    is_buy = trade.action == "BUY"
    e = trade.entry_price
    r = trade.initial_risk_points * POINT   # 1 R in price units
    hi = bar["high"]
    lo = bar["low"]
    spread_price = spread_points * POINT

    # Pre-compute all trigger levels
    be_target = e + r if is_buy else e - r
    partial_target = e + r * R_PARTIAL if is_buy else e - r * R_PARTIAL
    trail_target = e + r * R_TRAIL_START if is_buy else e - r * R_TRAIL_START
    full_target = e + r * R_FULL_EXIT if is_buy else e - r * R_FULL_EXIT

    be_reached = (hi >= be_target) if is_buy else (lo <= be_target)
    partial_reached = (hi >= partial_target) if is_buy else (lo <= partial_target)
    trail_reached = (hi >= trail_target) if is_buy else (lo <= trail_target)
    full_reached = (hi >= full_target) if is_buy else (lo <= full_target)
    sl_hit = (lo <= trade.current_sl) if is_buy else (hi >= trade.current_sl)

    # SL worst-fill: clamp to bar extreme beyond SL
    if is_buy:
        sl_fill = min(lo, trade.current_sl)
    else:
        sl_fill = max(hi, trade.current_sl)

    # ── Layer 1: Breakeven ────────────────────────────────────────
    if trade.stage == "ENTRY" and be_reached:
        be_buffer = BE_BUFFER_POINTS * POINT
        trade.current_sl = (e + be_buffer) if is_buy else (e - be_buffer)
        trade.stage = "BREAKEVEN"
        # Re-evaluate SL hit against new SL on same bar
        sl_hit = (lo <= trade.current_sl) if is_buy else (hi >= trade.current_sl)
        if is_buy:
            sl_fill = min(lo, trade.current_sl)
        else:
            sl_fill = max(hi, trade.current_sl)

    # ── Layer 2: Partial Close at +1.5 R ─────────────────────────
    if trade.stage == "BREAKEVEN" and partial_reached:
        partial_lots = round(trade.lot_size * 0.5, 2)
        if partial_lots >= MIN_LOT:
            # Exit fill: at target price, adjusted for spread cost on exit
            close_price = (partial_target - spread_price) if is_buy else (partial_target + spread_price)
            commission = commission_per_lot * partial_lots
            trade.pnl_partial = _pnl(trade.action, e, close_price, partial_lots, commission)
            trade.remaining_lots = max(round(trade.lot_size - partial_lots, 2), MIN_LOT)
        trade.stage = "PARTIAL_CLOSED"

    # ── Layer 3: Trail at +2 R ────────────────────────────────────
    if trade.stage == "PARTIAL_CLOSED" and trail_reached:
        trail_sl = (e + r * R_PARTIAL) if is_buy else (e - r * R_PARTIAL)
        # Only tighten, never widen
        if is_buy:
            trade.current_sl = max(trade.current_sl, trail_sl)
        else:
            trade.current_sl = min(trade.current_sl, trail_sl)
        trade.stage = "TRAILING"
        sl_hit = (lo <= trade.current_sl) if is_buy else (hi >= trade.current_sl)
        if is_buy:
            sl_fill = min(lo, trade.current_sl)
        else:
            sl_fill = max(hi, trade.current_sl)

    # ── Layer 4: Full Exit at +2.5 R ─────────────────────────────
    if trade.stage in {"PARTIAL_CLOSED", "TRAILING"} and full_reached:
        close_price = (full_target - spread_price) if is_buy else (full_target + spread_price)
        commission = commission_per_lot * trade.remaining_lots
        remainder_pnl = _pnl(trade.action, e, close_price, trade.remaining_lots, commission)
        _close_trade(trade, bar["time"], close_price, "FULL_EXIT",
                     trade.pnl_partial + remainder_pnl)
        return True

    # ── SL Hit ────────────────────────────────────────────────────
    if sl_hit:
        commission = commission_per_lot * trade.remaining_lots
        sl_pnl = _pnl(trade.action, e, sl_fill, trade.remaining_lots, commission)
        reason = "SL" if trade.stage == "ENTRY" else "TRAIL_SL"
        _close_trade(trade, bar["time"], sl_fill, reason,
                     trade.pnl_partial + sl_pnl)
        return True

    return False


def _pnl(action: str, entry: float, close: float, lots: float, commission: float) -> float:
    """Net P&L for a lot-level closure on XAUUSD."""
    direction = 1.0 if action == "BUY" else -1.0
    gross = (close - entry) * direction * lots * CONTRACT_SIZE
    return gross - commission


def _close_trade(trade: Trade, close_time, close_price: float, reason: str, total_pnl: float):
    """Finalize a Trade object (mutates in-place)."""
    trade.close_time = close_time
    trade.close_price = close_price
    trade.close_reason = reason
    trade.pnl_full = total_pnl
    trade.stage = "CLOSED"
    risk_dollar = trade.initial_risk_points * POINT * trade.lot_size * CONTRACT_SIZE
    trade.r_multiple = (total_pnl / risk_dollar) if risk_dollar > 0 else 0.0


# ─────────────────────────────────────────────────────────────
# Data Fetching
# ─────────────────────────────────────────────────────────────

def fetch_historical_data(symbol: str, from_date: datetime, to_date: datetime) -> dict:
    """Pull M1, M5, H1, H4 OHLCV from MT5 for the requested date range."""
    if not _MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 package not installed.")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")

    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
    }
    data: dict[str, pd.DataFrame] = {}
    try:
        for name, tf in tf_map.items():
            rates = mt5.copy_rates_range(symbol, tf, from_date, to_date)
            if rates is None or len(rates) == 0:
                raise RuntimeError(
                    f"No {name} data for {symbol} ({from_date.date()} – {to_date.date()}). "
                    f"MT5 error: {mt5.last_error()}"
                )
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            data[name] = df
            print(f"[DATA] {name}: {len(df):,} bars")
    finally:
        mt5.shutdown()

    return data


# ─────────────────────────────────────────────────────────────
# Main Backtest Loop
# ─────────────────────────────────────────────────────────────

def run_backtest(
    from_date: datetime,
    to_date: datetime,
    initial_balance: float = 10_000.0,
    spread_points: float = DEFAULT_SPREAD_POINTS,
    slippage_points: float = DEFAULT_SLIPPAGE_POINTS,
    commission_per_lot: float = DEFAULT_COMMISSION_PER_LOT,
    data: Optional[dict] = None,
) -> dict:
    """
    Core backtest loop.

    For each completed M1 bar (chronological order):
      1. Advance all open trades through the exit state machine.
      2. Call generate_signal() on the indicator window ending at this bar.
      3. If a signal fires, simulate fill on the NEXT bar's open.

    `data` can be pre-loaded (dict of DataFrames keyed "M1","M5","H1","H4")
    — useful for unit tests that don't need an MT5 connection.
    """
    if data is None:
        data = fetch_historical_data(SYMBOL, from_date, to_date)

    df_m1 = data["M1"].reset_index(drop=True)
    df_m5 = data["M5"].reset_index(drop=True)
    df_h1 = data["H1"].reset_index(drop=True)
    df_h4 = data["H4"].reset_index(drop=True)

    # Pre-extract time arrays as int64 nanoseconds for TZ-safe O(log n) searchsorted.
    # pandas DatetimeTZDtype .values gives tz-naive datetime64; comparing directly to a
    # tz-aware Timestamp raises TypeError. Using .astype("int64") (ns since epoch) avoids this.
    m5_times = df_m5["time"].astype("int64").values
    h1_times = df_h1["time"].astype("int64").values
    h4_times = df_h4["time"].astype("int64").values

    balance = initial_balance
    equity_records: list[dict] = []
    all_trades: list[Trade] = []
    state = BacktestState()

    print(f"\n[BACKTEST] {from_date.date()} → {to_date.date()}")
    print(f"[CONFIG]   Spread={spread_points}pt  Slippage={slippage_points}pt  "
          f"Commission=${commission_per_lot}/lot  Balance=${initial_balance:,.2f}\n")

    WARMUP = 100  # bars needed before indicators are reliable

    for i in range(WARMUP, len(df_m1)):
        bar = df_m1.iloc[i]
        bar_time = bar["time"]

        # ── 1. Run exit machine for all open trades ───────────────
        still_open = []
        for trade in state.open_trades:
            closed = process_bar_for_trade(trade, bar, commission_per_lot, spread_points)
            if closed:
                balance += trade.pnl_full
                all_trades.append(trade)
            else:
                still_open.append(trade)
        state.open_trades = still_open

        # ── 2. Generate signal on bars [i-99 .. i] ───────────────
        m1_window = df_m1.iloc[max(0, i - 99): i + 1]

        # Align higher-timeframe windows to current bar time (no lookahead).
        # Convert bar_time to int64 ns to match the pre-extracted arrays.
        bar_time_ns = int(pd.Timestamp(bar_time).value)
        m5_idx = int(np.searchsorted(m5_times, bar_time_ns, side="right"))
        h1_idx = int(np.searchsorted(h1_times, bar_time_ns, side="right"))
        h4_idx = int(np.searchsorted(h4_times, bar_time_ns, side="right"))
        m5_window = df_m5.iloc[max(0, m5_idx - 100): m5_idx]
        h1_window = df_h1.iloc[max(0, h1_idx - 100): h1_idx]
        h4_window = df_h4.iloc[max(0, h4_idx - 100): h4_idx]

        signal = generate_signal(m1_window, m5_window, h1_window, h4_window, state)

        # ── 3. Fill on next bar open ──────────────────────────────
        if signal is not None and (i + 1) < len(df_m1):
            next_bar = df_m1.iloc[i + 1]
            trade = simulate_fill(signal, next_bar, balance, spread_points, slippage_points)
            trade.trade_id = state.next_id()
            state.open_trades.append(trade)

        # ── 4. Equity snapshot (balance + floating P&L) ───────────
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

    # ── Force-close remaining open trades at last bar close ───────
    if df_m1.empty:
        last_bar = None
    else:
        last_bar = df_m1.iloc[-1]

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

    return {
        "metrics": metrics,
        "trades": all_trades,
        "equity_curve": eq_df,
    }


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def _calculate_metrics(
    trades: list[Trade],
    equity_df: pd.DataFrame,
    initial_balance: float,
) -> dict:
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

    # Sharpe / Sortino — computed on daily equity returns
    sharpe, sortino = _risk_ratios(equity_df)

    # Maximum drawdown
    equity = equity_df["equity"].values if not equity_df.empty else np.array([initial_balance])
    peak = np.maximum.accumulate(equity)
    dd_dollar = peak - equity
    dd_pct = np.where(peak > 0, dd_dollar / peak, 0.0)

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
    }


def _risk_ratios(equity_df: pd.DataFrame) -> tuple[float, float]:
    """Compute annualised Sharpe and Sortino from daily equity snapshots."""
    if equity_df.empty or "time" not in equity_df.columns:
        return 0.0, 0.0
    try:
        eq = equity_df.set_index("time")["equity"]
        # Resample to daily; forward-fill weekend gaps
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


# ─────────────────────────────────────────────────────────────
# Output: CSV + PNG
# ─────────────────────────────────────────────────────────────

def save_results(
    results: dict,
    run_id: str,
    output_dir: str = "./backtest_results",
) -> str:
    run_path = Path(output_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=True)

    trades: list[Trade] = results["trades"]
    eq_df: pd.DataFrame = results["equity_curve"]
    metrics: dict = results["metrics"]

    # Trade log CSV
    if trades:
        rows = [
            {
                "trade_id": t.trade_id,
                "action": t.action,
                "open_time": t.open_time,
                "close_time": t.close_time,
                "entry_price": round(t.entry_price, 5),
                "initial_sl": round(t.initial_sl, 5),
                "tp": round(t.tp, 5),
                "lot_size": t.lot_size,
                "close_price": round(t.close_price, 5) if t.close_price else None,
                "close_reason": t.close_reason,
                "stage_at_close": t.stage,
                "pnl_usd": round(t.pnl_full, 2),
                "r_multiple": round(t.r_multiple, 3),
                "initial_risk_points": t.initial_risk_points,
            }
            for t in trades
        ]
        pd.DataFrame(rows).to_csv(run_path / "trades.csv", index=False)
        print(f"[OUTPUT] trades.csv  ({len(rows)} rows)")

    # Equity curve CSV
    if not eq_df.empty:
        eq_df.to_csv(run_path / "equity_curve.csv", index=False)

    # Metrics JSON
    with open(run_path / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    # Equity curve PNG
    _plot_equity_curve(eq_df, metrics, run_path, run_id)

    return str(run_path)


def _plot_equity_curve(
    eq_df: pd.DataFrame,
    metrics: dict,
    run_path: Path,
    run_id: str,
):
    if eq_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0d1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#adb5bd")
        ax.spines[:].set_color("#30363d")

    times = eq_df["time"]
    equity = eq_df["equity"]
    balance = eq_df["balance"]

    ax1.plot(times, equity, color="#58a6ff", linewidth=1.0, label="Equity", zorder=3)
    ax1.plot(times, balance, color="#6e7681", linewidth=0.7, alpha=0.8,
             label="Balance (realised)", zorder=2)
    ax1.fill_between(times, balance, equity, alpha=0.12, color="#58a6ff")
    ax1.set_ylabel("USD", color="#adb5bd")
    ax1.legend(loc="upper left", fontsize=8, facecolor="#161b22",
               labelcolor="#adb5bd", framealpha=0.6)
    ax1.grid(True, alpha=0.15, color="#30363d")
    ax1.set_title(
        f"Sentinel AI — Backtest  |  Run: {run_id}",
        color="#f0f6fc", fontsize=10, pad=8,
    )

    # Drawdown panel
    peak = equity.cummax()
    dd_pct = ((equity - peak) / peak.replace(0, np.nan)) * 100
    ax2.fill_between(times, dd_pct, 0, color="#f85149", alpha=0.6, label="Drawdown %")
    ax2.set_ylabel("DD %", color="#adb5bd")
    ax2.set_xlabel("Date", color="#adb5bd")
    ax2.legend(loc="lower left", fontsize=8, facecolor="#161b22",
               labelcolor="#adb5bd", framealpha=0.6)
    ax2.grid(True, alpha=0.15, color="#30363d")

    # Stats footer
    m = metrics
    footer = (
        f"Trades: {m.get('total_trades', 0)}  |  "
        f"Win: {m.get('win_rate_pct', 0):.1f}%  |  "
        f"PF: {m.get('profit_factor', 0):.2f}  |  "
        f"Sharpe: {m.get('sharpe_ratio', 0):.2f}  |  "
        f"Sortino: {m.get('sortino_ratio', 0):.2f}  |  "
        f"MaxDD: {m.get('max_drawdown_pct', 0):.1f}%  |  "
        f"Return: {m.get('total_return_pct', 0):.1f}%"
    )
    fig.text(0.5, 0.005, footer, ha="center", fontsize=8.5, color="#8b949e")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path = run_path / "equity_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OUTPUT] equity_curve.png")


def _print_summary(metrics: dict, run_path: str):
    w = 55
    print("\n" + "─" * w)
    print("  SENTINEL AI — BACKTEST RESULTS")
    print("─" * w)
    for k, v in metrics.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<32} {v}")
    print("─" * w)
    print(f"  Saved to: {run_path}")
    print("─" * w + "\n")


# ─────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sentinel AI Backtesting Engine — replay historical XAUUSD data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--from", dest="from_date", required=True,
                        help="Start date  YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", required=True,
                        help="End date    YYYY-MM-DD")
    parser.add_argument("--initial-balance", type=float, default=10_000.0,
                        help="Starting capital in USD")
    parser.add_argument("--spread", type=float, default=DEFAULT_SPREAD_POINTS,
                        help="Spread in points (1 pt = $0.01)")
    parser.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE_POINTS,
                        help="Slippage in points")
    parser.add_argument("--commission", type=float, default=DEFAULT_COMMISSION_PER_LOT,
                        help="Commission per lot round-turn (USD)")
    parser.add_argument("--output-dir", default="./backtest_results",
                        help="Root output directory")
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
    )

    run_path = save_results(results, run_id, args.output_dir)
    _print_summary(results["metrics"], run_path)


if __name__ == "__main__":
    main()
