import argparse
import asyncio
import os
import time
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from orchestrator import on_signal_received
from trade_journal import TradeJournal

journal = TradeJournal()
MAX_SCALP_SPREAD_POINTS = float(os.getenv("MAX_SCALP_SPREAD_POINTS", "20"))
ATR_PERIOD = 14
MIN_VOLATILITY_THRESHOLD = float(os.getenv("MIN_VOLATILITY_THRESHOLD", "0.50"))


def count_open_positions(symbol: str = "XAUUSD") -> int:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    return len([p for p in positions if p.magic == 123456])


def filter_micro_spread(symbol: str = "XAUUSD", max_spread_points: float = MAX_SCALP_SPREAD_POINTS) -> tuple[bool, str, float | None]:
    """Allow scalps only when spread is tight enough for micro targets."""
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None or not info.point:
        return False, "spread_filter: missing tick/symbol info", None

    spread_points = float((tick.ask - tick.bid) / info.point)
    if spread_points > max_spread_points:
        return False, f"spread_filter: reject ({spread_points:.1f} > {max_spread_points:.1f} points)", spread_points
    return True, f"spread_filter: pass ({spread_points:.1f} points)", spread_points


def get_tick_data(symbol: str, count: int = 1000) -> pd.DataFrame | None:
    """Fetch recent ticks for high-frequency momentum decisions."""
    # Requested primary API path.
    ticks = mt5.copy_ticks_from(symbol, datetime.now(timezone.utc), count, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) == 0:
        # Fallback in case broker API doesn't return records at exact current timestamp.
        ticks = mt5.copy_ticks_from(symbol, datetime.now(timezone.utc) - timedelta(minutes=10), count, mt5.COPY_TICKS_ALL)
    if ticks is None or len(ticks) < 120:
        return None

    df = pd.DataFrame(ticks)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    if "last" in df.columns and (df["last"] > 0).any():
        df["tick_price"] = np.where(df["last"] > 0, df["last"], (df["bid"] + df["ask"]) / 2.0)
    else:
        df["tick_price"] = (df["bid"] + df["ask"]) / 2.0
    return df


def compute_tick_ema_crossover(ticks_df: pd.DataFrame) -> tuple[str | None, dict]:
    """Signal only when fast tick EMA crosses slow tick EMA."""
    prices = ticks_df["tick_price"]
    ema_fast = prices.ewm(span=50, adjust=False).mean()
    ema_slow = prices.ewm(span=100, adjust=False).mean()

    if len(ema_fast) < 101:
        return None, {"reason": "insufficient_ticks"}

    fast_prev = float(ema_fast.iloc[-2])
    slow_prev = float(ema_slow.iloc[-2])
    fast_now = float(ema_fast.iloc[-1])
    slow_now = float(ema_slow.iloc[-1])

    bullish_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
    bearish_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)

    action = "BUY" if bullish_cross else "SELL" if bearish_cross else None
    return action, {
        "ema_fast_50": fast_now,
        "ema_slow_100": slow_now,
        "ema_gap": fast_now - slow_now,
        "bullish_cross": bullish_cross,
        "bearish_cross": bearish_cross,
    }


def calculate_m1_atr(symbol: str, period: int = ATR_PERIOD) -> float | None:
    """Calculate 1-minute ATR using a simple rolling true-range mean."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, period + 2)
    if rates is None or len(rates) < period + 1:
        return None

    df = pd.DataFrame(rates)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean().iloc[-1]
    if pd.isna(atr):
        return None
    return float(atr)


def analyze_and_trade() -> None:
    symbol = "XAUUSD"
    max_positions = 5

    print("[SCANNER] RUBBER BAND ENGINE ACTIVATED")
    print("[CONFIG] Mean Reversion: Bollinger(20,1) + RSI(2)")
    print("[CONFIG] BUY: tick < LowerBand and RSI2 < 30 | SELL: tick > UpperBand and RSI2 > 70")
    print(f"[CONFIG] Spread Shield: <={MAX_SCALP_SPREAD_POINTS:.1f} points | Max Positions: 5 | Poll: 1.0s")
    print()
    last_telemetry = 0.0

    while True:
        try:
            # Build 1-minute indicator context (Bollinger 20,2 + RSI 2)
            m1_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 60)
            if m1_rates is None or len(m1_rates) < 25:
                time.sleep(1.0)
                continue

            m1_df = pd.DataFrame(m1_rates)
            close = m1_df["close"].astype(float)
            bb_mid = close.rolling(window=20).mean().iloc[-1]
            bb_std = close.rolling(window=20).std(ddof=0).iloc[-1]
            if pd.isna(bb_mid) or pd.isna(bb_std):
                time.sleep(1.0)
                continue

            bb_upper = float(bb_mid + (1.0 * bb_std))
            bb_lower = float(bb_mid - (1.0 * bb_std))

            delta = close.diff()
            gain = delta.clip(lower=0.0)
            loss = -delta.clip(upper=0.0)
            avg_gain = gain.rolling(window=2).mean().iloc[-1]
            avg_loss = loss.rolling(window=2).mean().iloc[-1]
            if pd.isna(avg_gain) or pd.isna(avg_loss):
                time.sleep(1.0)
                continue
            if avg_loss == 0:
                rsi2 = 100.0
            else:
                rs = float(avg_gain / avg_loss)
                rsi2 = float(100.0 - (100.0 / (1.0 + rs)))

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                time.sleep(1.0)
                continue
            current_price = float((tick.bid + tick.ask) / 2.0)

            buy_signal = current_price < bb_lower and rsi2 < 30.0
            sell_signal = current_price > bb_upper and rsi2 > 70.0
            signal_action = "BUY" if buy_signal else "SELL" if sell_signal else None

            now = time.time()
            if (now - last_telemetry) >= 5.0:
                print(
                    f"[TELEMETRY] Price={current_price:.2f} BB_L={bb_lower:.2f} BB_U={bb_upper:.2f} RSI2={rsi2:.1f}"
                )
                last_telemetry = now

            if signal_action is None:
                time.sleep(1.0)
                continue

            tick_time = datetime.now(timezone.utc)
            current_positions = count_open_positions(symbol)
            spread_pass, spread_reason, spread_points = filter_micro_spread(symbol)
            spread_txt = f"{spread_points:.1f}" if spread_points is not None else "NA"

            if current_positions >= max_positions or not spread_pass:
                rejection_reasons = []
                if current_positions >= max_positions:
                    rejection_reasons.append("Max positions reached")
                if not spread_pass:
                    rejection_reasons.append(spread_reason)

                signal_ts_value = tick_time.to_pydatetime() if hasattr(tick_time, "to_pydatetime") else tick_time

                journal.log_signal(
                    source="autonomous_scanner",
                    symbol=symbol,
                    action=signal_action,
                    timeframe="1m",
                    signal_ts=signal_ts_value,
                    rsi_value=float(rsi2),
                    ema_distance=None,
                    atr_value=None,
                    m5_trend=None,
                    h1_bias=None,
                    h4_bias=None,
                    integrated_bias=None,
                    news_context=None,
                    ai_decision="REJECTED",
                    ai_reasoning="Tick scanner safety veto",
                    ai_confidence=0.0,
                    decision_status="REJECTED",
                    rejection_reason="; ".join(rejection_reasons),
                    metadata={
                        "setup_type": "RUBBER_BAND_SCALP",
                        "bb_upper": bb_upper,
                        "bb_lower": bb_lower,
                        "rsi2": float(rsi2),
                        "spread_points": float(spread_points) if spread_points is not None else None,
                        "spread_reason": spread_reason,
                        "current_positions": current_positions,
                    },
                )
                print(f"[MONITOR] Snap-back={signal_action} vetoed | Spread={spread_txt} | Pos={current_positions}/5")
                time.sleep(1.0)
                continue

            # Confidence scales with band excursion distance.
            if signal_action == "BUY":
                excursion = max(0.0, bb_lower - current_price)
            else:
                excursion = max(0.0, current_price - bb_upper)
            confidence = max(0.0, min(100.0, (excursion / max(current_price * 0.0005, 1e-6)) * 100.0))

            if confidence <= 70.0:
                print(
                    f"[MONITOR] Snap-back={signal_action} low confidence={confidence:.1f} | "
                    f"Price={current_price:.2f} BB_L={bb_lower:.2f} BB_U={bb_upper:.2f} RSI2={rsi2:.1f}"
                )
                time.sleep(1.0)
                continue

            signal = {
                "source": "autonomous_scanner",
                "setup_type": "RUBBER_BAND_SCALP",
                "symbol": symbol,
                "timeframe": "1m",
                "action": signal_action,
                "timestamp": tick_time.timestamp(),
                "signal_bar_time": tick_time.isoformat(),
                "signal_bar_relation": "tick",
                "intended_price": float(current_price),
                "confidence_score": round(confidence, 1),
                "indicators": {
                    "bb_upper": bb_upper,
                    "bb_lower": bb_lower,
                    "rsi2": float(rsi2),
                    "filter_spread_reason": spread_reason,
                    "spread_points": float(spread_points) if spread_points is not None else None,
                },
            }

            asyncio.run(on_signal_received(signal))
            print(
                f"[RUBBER BAND] {signal_action} | Price={current_price:.2f} | "
                f"BB_L={bb_lower:.2f} BB_U={bb_upper:.2f} RSI2={rsi2:.1f} | Spread={spread_txt}"
            )

        except Exception as error:
            print(f"[ERROR] {error}")

        time.sleep(1.0)


def run_filter_ablation_backtest(symbol: str = "XAUUSD", days: int = 90, lookahead_bars: int = 15) -> None:
    print(
        "[ABLATION] This scanner is now tick-native for Phase 3. "
        "Use the previous candle-based branch for legacy filter ablation runs."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous scanner with tick-data EMA crossover")
    parser.add_argument("--ablation", action="store_true", help="Show legacy ablation notice")
    parser.add_argument("--days", type=int, default=90, help="Backtest window in days for ablation")
    args = parser.parse_args()

    if not mt5.initialize():
        print("[ERROR] Failed to initialize MT5")
    else:
        try:
            if args.ablation:
                run_filter_ablation_backtest(days=args.days)
            else:
                print("[SCANNER] Autonomous Agent Activated. Monitoring XAUUSD ticks...")
                analyze_and_trade()
        finally:
            mt5.shutdown()
