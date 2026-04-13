"""
Sentinel AI — Autonomous Scanner (Trend Continuation Pullback Strategy)
========================================================================
Strategy: Trend-following pullback on XAUUSD
- H4 + H1 must agree on direction (integrated bias)
- Wait for M5 price to pull back to EMA20
- Enter on first M1 bar showing rejection from EMA20
- Exit machine handles SL/TP/trail (unchanged)

Replaces: Bollinger Band mean reversion (proven unprofitable on trending Gold)
"""

import argparse
import asyncio
import os
import time
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from orchestrator import on_signal_received
from trade_journal import TradeJournal

journal = TradeJournal()
MAX_SCALP_SPREAD_POINTS = float(os.getenv("MAX_SCALP_SPREAD_POINTS", "30"))
ATR_PERIOD = 14
PULLBACK_EMA_SPAN = int(os.getenv("PULLBACK_EMA_SPAN", "20"))
REJECTION_ATR_THRESHOLD = float(os.getenv("REJECTION_ATR_THRESHOLD", "0.3"))
SESSION_OPEN_UTC = int(os.getenv("SESSION_OPEN_UTC", "8"))
SESSION_CLOSE_UTC = int(os.getenv("SESSION_CLOSE_UTC", "21"))
SCAN_INTERVAL_SECONDS = float(os.getenv("SCAN_INTERVAL_SECONDS", "10"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))


def count_open_positions(symbol: str = "XAUUSD") -> int:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    return len([p for p in positions if p.magic == 123456])


def filter_spread(symbol: str, max_spread_points: float = MAX_SCALP_SPREAD_POINTS) -> tuple[bool, str, float | None]:
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None or not info.point:
        return False, "spread_filter: missing tick/symbol info", None
    spread_points = float((tick.ask - tick.bid) / info.point)
    if spread_points > max_spread_points:
        return False, f"spread_filter: reject ({spread_points:.1f} > {max_spread_points:.1f})", spread_points
    return True, f"spread_filter: pass ({spread_points:.1f})", spread_points


def _ema_bias(rates_df: pd.DataFrame, span: int = 50) -> str:
    if rates_df is None or len(rates_df) < span:
        return "NO_CONFLUENCE"
    close = rates_df["close"].astype(float)
    ema = close.ewm(span=span, adjust=False).mean().iloc[-1]
    price = float(close.iloc[-1])
    if price > ema * 1.001:
        return "BULLISH"
    if price < ema * 0.999:
        return "BEARISH"
    return "NO_CONFLUENCE"


def _atr(rates_df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    if rates_df is None or len(rates_df) < period + 1:
        return 1.0
    h = rates_df["high"].values.astype(float)
    l = rates_df["low"].values.astype(float)
    c = rates_df["close"].values.astype(float)
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]),
                               np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-period:]))


def _get_integrated_bias(symbol: str) -> tuple[str, str, str]:
    h1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 60)
    h4_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 60)
    h1_df = pd.DataFrame(h1_rates) if h1_rates is not None else None
    h4_df = pd.DataFrame(h4_rates) if h4_rates is not None else None
    h1_bias = _ema_bias(h1_df)
    h4_bias = _ema_bias(h4_df)
    integrated = h1_bias if h1_bias == h4_bias else "NO_CONFLUENCE"
    return h1_bias, h4_bias, integrated


def _detect_pullback_rejection(symbol: str, bias: str) -> tuple[str | None, dict]:
    """
    Trend continuation pullback detection on M5/M1.

    Logic:
    - Compute M5 EMA20 — defines the pullback zone
    - Price must have pulled back within 0.5 ATR of EMA20 in last 3 M5 bars
    - Current M1 bar must show rejection: close back above EMA20 (for buys)
      or below EMA20 (for sells) with a wick toward the EMA
    - M1 momentum: last close stronger than open in signal direction
    """
    m5_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 40)
    m1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 20)

    if m5_rates is None or len(m5_rates) < 25:
        return None, {"reason": "insufficient_m5_data"}
    if m1_rates is None or len(m1_rates) < 5:
        return None, {"reason": "insufficient_m1_data"}

    m5_df = pd.DataFrame(m5_rates)
    m1_df = pd.DataFrame(m1_rates)

    m5_close = m5_df["close"].astype(float)
    m5_ema20 = m5_close.ewm(span=PULLBACK_EMA_SPAN, adjust=False).mean()
    ema_now = float(m5_ema20.iloc[-1])
    atr = _atr(m5_df)

    # Pullback zone: price was within 0.5 ATR of EMA20 in last 3 M5 bars
    recent_lows = m5_df["low"].astype(float).iloc[-3:].values
    recent_highs = m5_df["high"].astype(float).iloc[-3:].values
    pullback_zone = atr * 0.5

    touched_ema_bullish = any(low <= ema_now + pullback_zone for low in recent_lows)
    touched_ema_bearish = any(high >= ema_now - pullback_zone for high in recent_highs)

    # M1 rejection bar
    m1_bar = m1_df.iloc[-1]
    m1_open = float(m1_bar["open"])
    m1_close = float(m1_bar["close"])
    m1_high = float(m1_bar["high"])
    m1_low = float(m1_bar["low"])
    m1_body = abs(m1_close - m1_open)
    m1_range = m1_high - m1_low

    current_price = m1_close

    details = {
        "ema20_m5": round(ema_now, 3),
        "atr_m5": round(atr, 3),
        "current_price": round(current_price, 3),
        "pullback_zone": round(pullback_zone, 3),
        "touched_ema_bullish": touched_ema_bullish,
        "touched_ema_bearish": touched_ema_bearish,
        "m1_body": round(m1_body, 3),
        "m1_range": round(m1_range, 3),
    }

    # Body must be at least 30% of range (real bar, not doji)
    if m1_range > 0 and (m1_body / m1_range) < 0.3:
        return None, {**details, "reason": "doji_bar_rejected"}

    if bias == "BULLISH" and touched_ema_bullish:
        # Rejection: close above EMA, bullish bar
        if m1_close > ema_now and m1_close > m1_open:
            # Lower wick present (tested EMA and bounced)
            lower_wick = min(m1_open, m1_close) - m1_low
            if lower_wick > atr * REJECTION_ATR_THRESHOLD:
                return "BUY", {**details, "reason": "bullish_rejection_confirmed",
                                "lower_wick": round(lower_wick, 3)}

    if bias == "BEARISH" and touched_ema_bearish:
        # Rejection: close below EMA, bearish bar
        if m1_close < ema_now and m1_close < m1_open:
            upper_wick = m1_high - max(m1_open, m1_close)
            if upper_wick > atr * REJECTION_ATR_THRESHOLD:
                return "SELL", {**details, "reason": "bearish_rejection_confirmed",
                                 "upper_wick": round(upper_wick, 3)}

    return None, {**details, "reason": "no_rejection_pattern"}


def _session_label(hour_utc: int) -> str:
    if 13 <= hour_utc < 17:
        return "OVERLAP"
    if 8 <= hour_utc < 13:
        return "LONDON"
    if 17 <= hour_utc < 21:
        return "NEW_YORK"
    return "OFF_SESSION"


def _calculate_confidence(action: str, current_price: float, ema: float, atr: float) -> float:
    distance = abs(current_price - ema)
    confidence = min(100.0, (distance / max(atr * 0.5, 1e-6)) * 70.0)
    return round(max(0.0, confidence), 1)


def analyze_and_trade() -> None:
    symbol = "XAUUSD"

    print("[SCANNER] TREND CONTINUATION PULLBACK ENGINE ACTIVATED")
    print(f"[CONFIG] Strategy: M5 EMA{PULLBACK_EMA_SPAN} pullback + M1 rejection bar")
    print(f"[CONFIG] Bias: H4+H1 EMA50 integrated bias required")
    print(f"[CONFIG] Spread Shield: <={MAX_SCALP_SPREAD_POINTS:.1f} points | Max Positions: {MAX_POSITIONS}")
    print(f"[CONFIG] Sessions: {SESSION_OPEN_UTC}:00–{SESSION_CLOSE_UTC}:00 UTC")
    print()

    last_telemetry = 0.0

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            hour_utc = now_utc.hour

            # Session gate
            if not (SESSION_OPEN_UTC <= hour_utc < SESSION_CLOSE_UTC):
                time.sleep(30.0)
                continue

            session = _session_label(hour_utc)

            # Position cap
            current_positions = count_open_positions(symbol)
            if current_positions >= MAX_POSITIONS:
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            # Spread gate
            spread_pass, spread_reason, spread_points = filter_spread(symbol)
            if not spread_pass:
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            # HTF bias
            h1_bias, h4_bias, integrated_bias = _get_integrated_bias(symbol)
            if integrated_bias == "NO_CONFLUENCE":
                now = time.time()
                if (now - last_telemetry) >= 30.0:
                    print(f"[WAIT] No confluence — H1={h1_bias} H4={h4_bias} | Session={session}")
                    last_telemetry = now
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            # Pullback + rejection detection
            action, signal_details = _detect_pullback_rejection(symbol, integrated_bias)

            now = time.time()
            if (now - last_telemetry) >= 10.0:
                tick = mt5.symbol_info_tick(symbol)
                price = float((tick.bid + tick.ask) / 2.0) if tick else 0.0
                ema = signal_details.get("ema20_m5", 0.0)
                reason = signal_details.get("reason", "")
                print(
                    f"[TELEMETRY] Price={price:.2f} EMA20={ema:.2f} "
                    f"H1={h1_bias} H4={h4_bias} Bias={integrated_bias} "
                    f"Session={session} | {reason}"
                )
                last_telemetry = now

            if action is None:
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            # Confidence
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            current_price = float((tick.bid + tick.ask) / 2.0)
            ema_now = signal_details.get("ema20_m5", current_price)
            atr = signal_details.get("atr_m5", 1.0)
            confidence = _calculate_confidence(action, current_price, ema_now, atr)

            if confidence <= 70.0:
                print(f"[LOW CONF] {action} confidence={confidence:.1f} — skipped")
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue

            signal = {
                "source": "autonomous_scanner",
                "setup_type": "TREND_PULLBACK",
                "symbol": symbol,
                "timeframe": "M5",
                "action": action,
                "timestamp": now_utc.timestamp(),
                "signal_bar_time": now_utc.isoformat(),
                "signal_bar_relation": "m1_rejection",
                "intended_price": float(current_price),
                "confidence_score": confidence,
                "indicators": {
                    "ema20_m5": ema_now,
                    "atr_m5": atr,
                    "h1_bias": h1_bias,
                    "h4_bias": h4_bias,
                    "integrated_bias": integrated_bias,
                    "spread_points": float(spread_points) if spread_points is not None else None,
                    **signal_details,
                },
            }

            asyncio.run(on_signal_received(signal))
            print(
                f"[SIGNAL] {action} | Price={current_price:.2f} EMA20={ema_now:.2f} "
                f"Conf={confidence:.1f} | H1={h1_bias} H4={h4_bias} | {session}"
            )

        except Exception as error:
            print(f"[ERROR] {error}")

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentinel AI — Trend Continuation Pullback Scanner")
    args = parser.parse_args()

    if not mt5.initialize():
        print("[ERROR] Failed to initialize MT5")
    else:
        try:
            print("[SCANNER] Autonomous Agent Activated. Monitoring XAUUSD...")
            analyze_and_trade()
        finally:
            mt5.shutdown()