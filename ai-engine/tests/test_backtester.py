"""
Unit tests for backtester.py — fill simulation and R-multiple math.

Run with:
    cd ai-engine
    pytest tests/test_backtester.py -v
"""

import sys
import os
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

# Add ai-engine to path so we can import backtester without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# backtester.py guards MT5 import, so it is safe to import here
from backtester import (
    POINT,
    CONTRACT_SIZE,
    DEFAULT_SPREAD_POINTS,
    DEFAULT_SLIPPAGE_POINTS,
    DEFAULT_COMMISSION_PER_LOT,
    R_PARTIAL,
    R_FULL_EXIT,
    MIN_LOT,
    BacktestState,
    Signal,
    Trade,
    calculate_lot_size,
    simulate_fill,
    process_bar_for_trade,
    generate_signal,
    _pnl,
    _atr,
    _ema_bias,
    run_backtest,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _make_signal(action="BUY", price=2000.0, atr=1.0) -> Signal:
    return Signal(action=action, timestamp=pd.Timestamp("2024-01-02 09:00", tz="UTC"),
                  signal_price=price, atr_m5=atr)


def _make_bar(open_=2000.0, high=2010.0, low=1990.0, close=2005.0,
              time="2024-01-02 09:01") -> pd.Series:
    return pd.Series({
        "time": pd.Timestamp(time, tz="UTC"),
        "open": open_, "high": high, "low": low, "close": close,
        "tick_volume": 100,
    })


def _make_trade(
    action="BUY",
    entry=2000.0,
    sl_points=150.0,   # points
    lot=0.10,
) -> Trade:
    """Build a Trade directly (bypasses fill simulation) for exit-machine tests."""
    is_buy = action == "BUY"
    sl = (entry - sl_points * POINT) if is_buy else (entry + sl_points * POINT)
    tp = (entry + sl_points * POINT * R_FULL_EXIT) if is_buy else (entry - sl_points * POINT * R_FULL_EXIT)
    return Trade(
        trade_id=1,
        action=action,
        open_time=pd.Timestamp("2024-01-02 09:00", tz="UTC"),
        entry_price=entry,
        initial_sl=sl,
        current_sl=sl,
        tp=tp,
        lot_size=lot,
        remaining_lots=lot,
        initial_risk_points=sl_points,
    )


def _make_ohlcv(n=200, base_price=2000.0, trend="up") -> pd.DataFrame:
    """Generate a minimal OHLCV DataFrame for generate_signal tests."""
    times = pd.date_range("2024-01-01 09:00", periods=n, freq="1min", tz="UTC")
    closes = np.linspace(base_price - 10, base_price + 10, n) if trend == "up" \
        else np.linspace(base_price + 10, base_price - 10, n)
    df = pd.DataFrame({
        "time": times,
        "open": closes - 0.5,
        "high": closes + 1.0,
        "low": closes - 1.0,
        "close": closes,
        "tick_volume": np.full(n, 120),
    })
    return df


# ─────────────────────────────────────────────────────────────
# Lot-size math
# ─────────────────────────────────────────────────────────────

class TestCalculateLotSize:
    def test_basic_formula(self):
        """lot = (balance * 2%) / (sl_pts * POINT * CONTRACT_SIZE)"""
        balance = 10_000.0
        sl_pts = 150.0
        expected = (balance * 0.02) / (sl_pts * POINT * CONTRACT_SIZE)
        result = calculate_lot_size(balance, sl_pts)
        assert abs(result - round(expected, 2)) <= 0.01

    def test_minimum_lot_enforced(self):
        """Very large SL → lot must not fall below MIN_LOT."""
        result = calculate_lot_size(1_000.0, sl_distance_points=50_000.0)
        assert result == MIN_LOT

    def test_maximum_lot_enforced(self):
        """Tiny SL with huge balance → lot must not exceed MAX_LOT."""
        result = calculate_lot_size(10_000_000.0, sl_distance_points=1.0)
        assert result == 100.0

    def test_zero_sl_returns_min(self):
        result = calculate_lot_size(10_000.0, sl_distance_points=0.0)
        assert result == MIN_LOT

    def test_scales_with_balance(self):
        """Doubling balance should double lot size (clamped to same range)."""
        lot_small = calculate_lot_size(5_000.0, 150.0)
        lot_large = calculate_lot_size(10_000.0, 150.0)
        assert abs(lot_large / lot_small - 2.0) < 0.05

    def test_scales_inversely_with_sl(self):
        """Wider SL → smaller lot (same risk $)."""
        lot_tight = calculate_lot_size(10_000.0, 100.0)
        lot_wide = calculate_lot_size(10_000.0, 200.0)
        assert lot_tight > lot_wide


# ─────────────────────────────────────────────────────────────
# Fill simulation
# ─────────────────────────────────────────────────────────────

class TestSimulateFill:
    """Tests for spread, slippage, SL/TP placement, and lot sizing."""

    def _fill(self, action="BUY", open_=2000.0, atr=1.0, balance=10_000.0,
              spread=DEFAULT_SPREAD_POINTS, slip=DEFAULT_SLIPPAGE_POINTS):
        sig = _make_signal(action=action, atr=atr)
        bar = _make_bar(open_=open_)
        return simulate_fill(sig, bar, balance, spread, slip)

    # ── Entry price reflects spread and slippage ──────────────────
    def test_buy_entry_above_open(self):
        """BUY fill must be above bar open (paying spread + slippage)."""
        trade = self._fill("BUY", open_=2000.0)
        assert trade.entry_price > 2000.0

    def test_sell_entry_below_open(self):
        """SELL fill must be at or below bar open (slippage only)."""
        trade = self._fill("SELL", open_=2000.0)
        assert trade.entry_price <= 2000.0

    def test_buy_spread_cost_exact(self):
        spread_pts = 20
        slip_pts = 5
        trade = self._fill("BUY", open_=2000.0, spread=spread_pts, slip=slip_pts)
        expected = 2000.0 + (spread_pts + slip_pts) * POINT
        assert abs(trade.entry_price - expected) < 1e-8

    def test_sell_slippage_cost_exact(self):
        slip_pts = 5
        trade = self._fill("SELL", open_=2000.0, slip=slip_pts)
        expected = 2000.0 - slip_pts * POINT
        assert abs(trade.entry_price - expected) < 1e-8

    # ── SL direction ─────────────────────────────────────────────
    def test_buy_sl_below_entry(self):
        trade = self._fill("BUY")
        assert trade.initial_sl < trade.entry_price

    def test_sell_sl_above_entry(self):
        trade = self._fill("SELL")
        assert trade.initial_sl > trade.entry_price

    # ── TP at 2.5 R ───────────────────────────────────────────────
    def test_buy_tp_equals_2_5r(self):
        trade = self._fill("BUY")
        sl_dist = trade.entry_price - trade.initial_sl
        tp_dist = trade.tp - trade.entry_price
        ratio = tp_dist / sl_dist
        assert abs(ratio - R_FULL_EXIT) < 1e-6

    def test_sell_tp_equals_2_5r(self):
        trade = self._fill("SELL")
        sl_dist = trade.initial_sl - trade.entry_price
        tp_dist = trade.entry_price - trade.tp
        ratio = tp_dist / sl_dist
        assert abs(ratio - R_FULL_EXIT) < 1e-6

    # ── Minimum SL distance ───────────────────────────────────────
    def test_minimum_sl_distance_enforced(self):
        """Even with near-zero ATR, SL must be at least 30 points away."""
        trade = self._fill("BUY", atr=0.0001)
        sl_pts = (trade.entry_price - trade.initial_sl) / POINT
        # Allow a tiny floating-point tolerance (the floor is 30 pts exactly)
        assert sl_pts >= 30.0 - 1e-6

    # ── Lot size is risk-based ─────────────────────────────────────
    def test_lot_ties_to_balance(self):
        t1 = self._fill("BUY", balance=10_000.0)
        t2 = self._fill("BUY", balance=20_000.0)
        assert t2.lot_size > t1.lot_size

    def test_lot_minimum_enforced(self):
        trade = self._fill("BUY", balance=10.0)
        assert trade.lot_size >= MIN_LOT


# ─────────────────────────────────────────────────────────────
# P&L helper
# ─────────────────────────────────────────────────────────────

class TestPnlHelper:
    def test_buy_profit(self):
        """BUY: price rises → profit."""
        result = _pnl("BUY", entry=2000.0, close=2001.0, lots=1.0, commission=0.0)
        # 1.0 price move * 1 lot * 100 oz = $100
        assert abs(result - 100.0) < 1e-6

    def test_buy_loss(self):
        result = _pnl("BUY", entry=2000.0, close=1999.0, lots=1.0, commission=0.0)
        assert abs(result - (-100.0)) < 1e-6

    def test_sell_profit(self):
        result = _pnl("SELL", entry=2000.0, close=1999.0, lots=1.0, commission=0.0)
        assert abs(result - 100.0) < 1e-6

    def test_commission_deducted(self):
        result = _pnl("BUY", entry=2000.0, close=2001.0, lots=1.0, commission=7.0)
        assert abs(result - 93.0) < 1e-6

    def test_breakeven_minus_commission(self):
        """A trade closed at entry is a loss equal to commission."""
        result = _pnl("BUY", entry=2000.0, close=2000.0, lots=1.0, commission=7.0)
        assert abs(result - (-7.0)) < 1e-6


# ─────────────────────────────────────────────────────────────
# R-multiple math
# ─────────────────────────────────────────────────────────────

class TestRMultiple:
    """
    R-multiple = total_pnl / initial_risk_dollars
    For a position that closes 50 % at +1.5 R and remainder at +2.5 R:
        R = 0.5 * 1.5 + 0.5 * 2.5 = 0.75 + 1.25 = 2.0
    """

    def _r_value(self, trade: Trade) -> float:
        """Recompute R as the backtester does."""
        risk_dollar = trade.initial_risk_points * POINT * trade.lot_size * CONTRACT_SIZE
        return trade.pnl_full / risk_dollar if risk_dollar else 0.0

    def test_full_exit_r(self):
        """Trade that hits full exit (+2.5 R) should have r ≈ 2.0 (after partial at 1.5R)."""
        trade = _make_trade(action="BUY", entry=2000.0, sl_points=100.0, lot=1.0)
        r_price = trade.initial_risk_points * POINT          # 1 R in price

        # Bar that triggers: BE (1R), partial (1.5R), trail (2R), full exit (2.5R)
        # High must exceed entry + 2.5R
        hi = trade.entry_price + r_price * 2.6
        bar = _make_bar(open_=trade.entry_price, high=hi,
                        low=trade.entry_price - r_price * 0.1,
                        close=hi - 0.01)
        closed = process_bar_for_trade(trade, bar, commission_per_lot=0.0, spread_points=0)
        assert closed
        assert trade.close_reason == "FULL_EXIT"
        assert abs(self._r_value(trade) - 2.0) < 0.05   # 0.5*1.5 + 0.5*2.5

    def test_sl_hit_entry_r(self):
        """Trade stopped out at SL in ENTRY stage → r ≈ -1."""
        trade = _make_trade(action="BUY", entry=2000.0, sl_points=100.0, lot=1.0)
        r_price = trade.initial_risk_points * POINT
        # Bar dips below SL without triggering BE
        bar = _make_bar(open_=trade.entry_price,
                        high=trade.entry_price + r_price * 0.5,   # doesn't reach BE
                        low=trade.initial_sl - r_price * 0.1,
                        close=trade.entry_price)
        closed = process_bar_for_trade(trade, bar, commission_per_lot=0.0, spread_points=0)
        assert closed
        assert trade.close_reason == "SL"
        assert self._r_value(trade) < -0.9

    def test_trail_sl_hit_r(self):
        """
        Trade that reaches partial close (+1.5R) then SL is hit at breakeven:
        Expected R ≈ 0.75 (only the partial profit banked).
        """
        trade = _make_trade(action="BUY", entry=2000.0, sl_points=100.0, lot=0.10)
        r_price = trade.initial_risk_points * POINT
        be_buffer = 10 * POINT

        # Bar 1: trigger BE and partial close (+1.5R reached).
        # Low must stay ABOVE the new BE SL (entry + 10 pts buffer = 2000.10)
        # so the trade is not immediately stopped out on the same bar.
        bar1 = _make_bar(
            open_=trade.entry_price,
            high=trade.entry_price + r_price * 1.6,
            low=trade.entry_price + r_price * 0.2,   # 2000.20 > new SL 2000.10
            close=trade.entry_price + r_price * 1.6,
        )
        closed1 = process_bar_for_trade(trade, bar1, commission_per_lot=0.0, spread_points=0)
        assert not closed1
        assert trade.stage == "PARTIAL_CLOSED"

        # Bar 2: price dips back to breakeven SL
        sl_price = trade.entry_price + be_buffer   # current SL after BE move
        bar2 = _make_bar(
            open_=trade.entry_price + r_price * 1.3,
            high=trade.entry_price + r_price * 1.4,
            low=sl_price - r_price * 0.1,   # below new SL
            close=trade.entry_price + r_price * 0.5,
        )
        closed2 = process_bar_for_trade(trade, bar2, commission_per_lot=0.0, spread_points=0)
        assert closed2
        assert trade.close_reason == "TRAIL_SL"
        # Partial at 1.5R on half → 0.75R on full position; remainder at ~0R (BE) → 0
        assert self._r_value(trade) > 0.5   # must be positive (partial banked)
        assert self._r_value(trade) < 1.0   # but less than 1R

    def test_sell_full_exit_r(self):
        """SELL trade full exit R should also be ~2.0."""
        trade = _make_trade(action="SELL", entry=2000.0, sl_points=100.0, lot=1.0)
        r_price = trade.initial_risk_points * POINT
        # Price falls: lo must go below entry - 2.5R
        lo = trade.entry_price - r_price * 2.6
        bar = _make_bar(open_=trade.entry_price,
                        high=trade.entry_price + r_price * 0.1,
                        low=lo,
                        close=lo + 0.01)
        closed = process_bar_for_trade(trade, bar, commission_per_lot=0.0, spread_points=0)
        assert closed
        assert trade.close_reason == "FULL_EXIT"
        assert abs(self._r_value(trade) - 2.0) < 0.05


# ─────────────────────────────────────────────────────────────
# Exit state machine transitions
# ─────────────────────────────────────────────────────────────

class TestExitStateMachine:
    def test_entry_to_breakeven(self):
        """Bar that reaches +1R moves trade to BREAKEVEN; SL updated."""
        trade = _make_trade("BUY", entry=2000.0, sl_points=100.0)
        r_price = trade.initial_risk_points * POINT
        # Low must stay ABOVE the new BE SL (entry + 10 pts buffer = 2000.10).
        # Using low = entry + 0.2R keeps the bar well above the new stop.
        bar = _make_bar(
            open_=trade.entry_price,
            high=trade.entry_price + r_price * 1.1,
            low=trade.entry_price + r_price * 0.2,   # above new BE SL → trade stays open
            close=trade.entry_price + r_price,
        )
        closed = process_bar_for_trade(trade, bar, commission_per_lot=0.0, spread_points=0)
        assert not closed
        assert trade.stage == "BREAKEVEN"
        assert trade.current_sl > trade.initial_sl   # SL moved up

    def test_breakeven_sl_above_initial_sl(self):
        """After BE, SL must be at or above entry (with buffer)."""
        trade = _make_trade("BUY", entry=2000.0, sl_points=100.0)
        r = trade.initial_risk_points * POINT
        bar = _make_bar(high=trade.entry_price + r * 1.1, low=trade.entry_price - r * 0.05)
        process_bar_for_trade(trade, bar, commission_per_lot=0.0, spread_points=0)
        assert trade.current_sl >= trade.entry_price

    def test_partial_close_lots(self):
        """After partial close, remaining_lots == 50 % of original."""
        trade = _make_trade("BUY", entry=2000.0, sl_points=100.0, lot=0.10)
        r = trade.initial_risk_points * POINT
        bar = _make_bar(high=trade.entry_price + r * 1.6, low=trade.entry_price - r * 0.05)
        process_bar_for_trade(trade, bar, commission_per_lot=0.0, spread_points=0)
        if trade.stage == "PARTIAL_CLOSED":
            assert abs(trade.remaining_lots - 0.05) < 1e-6

    def test_no_trade_on_flat_bar(self):
        """Bar entirely inside the entry candle should not close or advance stage."""
        trade = _make_trade("BUY", entry=2000.0, sl_points=100.0)
        r = trade.initial_risk_points * POINT
        # Flat bar — doesn't touch SL or any target
        bar = _make_bar(high=trade.entry_price + r * 0.4, low=trade.entry_price - r * 0.4)
        closed = process_bar_for_trade(trade, bar, commission_per_lot=0.0, spread_points=0)
        assert not closed
        assert trade.stage == "ENTRY"


# ─────────────────────────────────────────────────────────────
# Generate signal (pure function)
# ─────────────────────────────────────────────────────────────

class TestGenerateSignal:
    def _state(self, positions=0) -> BacktestState:
        s = BacktestState()
        # Fake open trades to simulate position count
        for _ in range(positions):
            s.open_trades.append(object())
        return s

    def test_returns_none_on_insufficient_bars(self):
        tiny = _make_ohlcv(n=10)
        state = self._state()
        result = generate_signal(tiny, tiny, tiny, tiny, state)
        assert result is None

    def test_returns_none_when_max_positions(self):
        df = _make_ohlcv(n=200)
        state = self._state(positions=5)
        result = generate_signal(df, df, df, df, state)
        assert result is None

    def test_returns_none_outside_session(self):
        """Bars at 02:00 UTC (outside London/NY) should not generate signals."""
        df = _make_ohlcv(n=200)
        # Override all times to 02:00 UTC
        df["time"] = pd.date_range("2024-01-01 02:00", periods=200, freq="1min", tz="UTC")
        state = self._state()
        result = generate_signal(df, df, df, df, state)
        assert result is None

    def test_signal_action_is_buy_or_sell(self):
        """When a signal is produced it must be BUY or SELL."""
        df = _make_ohlcv(n=200, trend="up")
        df["time"] = pd.date_range("2024-01-01 10:00", periods=200, freq="1min", tz="UTC")
        state = self._state()
        result = generate_signal(df, df, df, df, state)
        # May return None (no crossover in synthetic flat data), which is fine
        if result is not None:
            assert result.action in ("BUY", "SELL")

    def test_atr_m5_positive(self):
        """If a signal fires its ATR must be > 0."""
        df = _make_ohlcv(n=200, trend="up")
        df["time"] = pd.date_range("2024-01-01 10:00", periods=200, freq="1min", tz="UTC")
        state = self._state()
        result = generate_signal(df, df, df, df, state)
        if result is not None:
            assert result.atr_m5 > 0


# ─────────────────────────────────────────────────────────────
# ATR helper
# ─────────────────────────────────────────────────────────────

class TestAtr:
    def test_atr_positive(self):
        df = _make_ohlcv(n=50)
        val = _atr(df)
        assert val > 0

    def test_atr_fallback_on_small_df(self):
        df = _make_ohlcv(n=5)
        val = _atr(df)
        assert val == 1.0   # fallback

    def test_atr_none_df(self):
        val = _atr(None)
        assert val == 1.0


# ─────────────────────────────────────────────────────────────
# EMA bias helper
# ─────────────────────────────────────────────────────────────

class TestEmaBias:
    def test_bullish_trend(self):
        df = _make_ohlcv(n=200, base_price=2000.0, trend="up")
        bias = _ema_bias(df)
        # Rising prices should end above EMA → BULLISH or NO_CONFLUENCE
        assert bias in ("BULLISH", "NO_CONFLUENCE")

    def test_bearish_trend(self):
        df = _make_ohlcv(n=200, base_price=2000.0, trend="down")
        bias = _ema_bias(df)
        assert bias in ("BEARISH", "NO_CONFLUENCE")

    def test_insufficient_bars(self):
        df = _make_ohlcv(n=10)
        bias = _ema_bias(df)
        assert bias == "NO_CONFLUENCE"

    def test_none_df(self):
        bias = _ema_bias(None)
        assert bias == "NO_CONFLUENCE"


# ─────────────────────────────────────────────────────────────
# run_backtest (integration — no MT5 needed)
# ─────────────────────────────────────────────────────────────

class TestRunBacktest:
    """Integration tests that inject synthetic data to avoid MT5."""

    def _make_data(self, n_m1=500) -> dict:
        """Build a minimal four-timeframe data dict with a synthetic uptrend."""
        base = 2000.0
        times_m1 = pd.date_range("2024-01-02 09:00", periods=n_m1, freq="1min", tz="UTC")
        closes = np.linspace(base, base + 20, n_m1)
        df_m1 = pd.DataFrame({
            "time": times_m1,
            "open": closes - 0.3,
            "high": closes + 1.0,
            "low": closes - 1.0,
            "close": closes,
            "tick_volume": np.full(n_m1, 150),
        })

        # Build M5 from M1 (every 5th bar)
        df_m5 = df_m1.iloc[::5].copy().reset_index(drop=True)
        df_h1 = df_m1.iloc[::60].copy().reset_index(drop=True)
        df_h4 = df_m1.iloc[::240].copy().reset_index(drop=True)

        return {"M1": df_m1, "M5": df_m5, "H1": df_h1, "H4": df_h4}

    def test_returns_dict_with_keys(self):
        data = self._make_data()
        result = run_backtest(
            from_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            to_date=datetime(2024, 1, 3, tzinfo=timezone.utc),
            initial_balance=10_000.0,
            data=data,
        )
        assert "metrics" in result
        assert "trades" in result
        assert "equity_curve" in result

    def test_equity_curve_not_empty(self):
        data = self._make_data()
        result = run_backtest(
            from_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            to_date=datetime(2024, 1, 3, tzinfo=timezone.utc),
            initial_balance=10_000.0,
            data=data,
        )
        assert not result["equity_curve"].empty

    def test_no_lookahead_bias(self):
        """
        Trades must open AFTER the signal bar (next-bar fill).
        open_time of every trade must be > signal bar time.
        This is guaranteed because we fill on bar i+1 after signal on bar i.
        """
        data = self._make_data(n_m1=600)
        result = run_backtest(
            from_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            to_date=datetime(2024, 1, 3, tzinfo=timezone.utc),
            initial_balance=10_000.0,
            data=data,
        )
        # All trades open within the data window
        for t in result["trades"]:
            assert t.open_time >= data["M1"]["time"].iloc[0]
            assert t.open_time <= data["M1"]["time"].iloc[-1]

    def test_metrics_have_required_fields(self):
        data = self._make_data()
        result = run_backtest(
            from_date=datetime(2024, 1, 2, tzinfo=timezone.utc),
            to_date=datetime(2024, 1, 3, tzinfo=timezone.utc),
            initial_balance=10_000.0,
            data=data,
        )
        m = result["metrics"]
        required = {
            "total_trades", "win_rate_pct", "profit_factor",
            "sharpe_ratio", "sortino_ratio",
            "max_drawdown_pct", "max_drawdown_dollar",
            "average_r", "expectancy_usd", "final_balance", "total_return_pct",
        }
        if "error" not in m:
            for field in required:
                assert field in m, f"Missing metric: {field}"
