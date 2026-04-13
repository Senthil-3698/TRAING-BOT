import argparse
import asyncio
import time
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from orchestrator import on_signal_received
from trade_journal import TradeJournal

journal = TradeJournal()


def count_open_positions(symbol: str = "XAUUSD") -> int:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    return len([p for p in positions if p.magic == 123456])


def filter_micro_spread(symbol: str = "XAUUSD", max_spread_points: float = 15.0) -> tuple[bool, str, float | None]:
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


def analyze_and_trade() -> None:
    symbol = "XAUUSD"
    max_positions = 5

    print("[SCANNER] TICK-MOMENTUM SNIPER ACTIVATED")
    print("[CONFIG] Trigger: 50-tick EMA cross 100-tick EMA | Input: last 1000 ticks")
    print("[CONFIG] Spread Shield: <=15 points | Max Positions: 5 | Poll: 0.2s")
    print()

    while True:
        try:
            ticks_df = get_tick_data(symbol, count=1000)
            if ticks_df is None:
                time.sleep(0.2)
                continue

            signal_action, tick_details = compute_tick_ema_crossover(ticks_df)
            if signal_action is None:
                time.sleep(0.2)
                continue

            tick_time = ticks_df["time"].iloc[-1]
            current_price = float(ticks_df["tick_price"].iloc[-1])
            current_positions = count_open_positions(symbol)
            spread_pass, spread_reason, spread_points = filter_micro_spread(symbol)
            spread_txt = f"{spread_points:.1f}" if spread_points is not None else "NA"

            if current_positions >= max_positions or not spread_pass:
                rejection_reasons = []
                if current_positions >= max_positions:
                    rejection_reasons.append("Max positions reached")
                if not spread_pass:
                    rejection_reasons.append(spread_reason)

                journal.log_signal(
                    source="autonomous_scanner",
                    symbol=symbol,
                    action=signal_action,
                    timeframe="tick",
                    signal_ts=tick_time.to_pydatetime(),
                    rsi_value=None,
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
                        "setup_type": "TICK_EMA_CROSS_SCALP",
                        "tick_ema_fast_50": float(tick_details.get("ema_fast_50", 0.0)),
                        "tick_ema_slow_100": float(tick_details.get("ema_slow_100", 0.0)),
                        "tick_ema_gap": float(tick_details.get("ema_gap", 0.0)),
                        "spread_points": float(spread_points) if spread_points is not None else None,
                        "spread_reason": spread_reason,
                        "current_positions": current_positions,
                    },
                )
                print(f"[MONITOR] Tick cross={signal_action} vetoed | Spread={spread_txt} | Pos={current_positions}/5")
                time.sleep(0.2)
                continue

            ema_gap = abs(float(tick_details.get("ema_gap", 0.0)))
            confidence = max(0.0, min(1.0, ema_gap / max(current_price * 0.0001, 1e-6)))

            signal = {
                "source": "autonomous_scanner",
                "setup_type": "TICK_EMA_CROSS_SCALP",
                "symbol": symbol,
                "timeframe": "tick",
                "action": signal_action,
                "timestamp": tick_time.timestamp(),
                "signal_bar_time": tick_time.isoformat(),
                "signal_bar_relation": "tick",
                "intended_price": float(current_price),
                "confidence_score": round(confidence, 4),
                "indicators": {
                    "tick_ema_fast_50": float(tick_details.get("ema_fast_50", 0.0)),
                    "tick_ema_slow_100": float(tick_details.get("ema_slow_100", 0.0)),
                    "tick_ema_gap": float(tick_details.get("ema_gap", 0.0)),
                    "filter_spread_reason": spread_reason,
                    "spread_points": float(spread_points) if spread_points is not None else None,
                },
            }

            asyncio.run(on_signal_received(signal))
            print(
                f"[TICK SIGNAL] {signal_action} | Price={current_price:.2f} | "
                f"EMA50={tick_details['ema_fast_50']:.4f} EMA100={tick_details['ema_slow_100']:.4f} | "
                f"Spread={spread_txt}"
            )

        except Exception as error:
            print(f"[ERROR] {error}")

        time.sleep(0.2)


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
