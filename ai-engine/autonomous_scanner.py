import MetaTrader5 as mt5
import pandas as pd
import time
import numpy as np

from mt5_executor import execute_trade


def get_market_data(symbol, timeframe, count=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def count_open_positions(symbol="XAUUSD"):
    """Count open positions for the symbol to prevent overloading."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    # Only count positions with our magic number
    return len([p for p in positions if p.magic == 123456])


def analyze_and_trade():
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M1  # M1 for precision entry timing
    max_positions = 5

    print("[SCANNER] TRIPLE CONFLUENCE SNIPER ACTIVATED")
    print("[CONFIG] Baseline: 50-EMA (Master Trend) | Trigger: 5/13-SMA Cross | Filter: Volume")
    print("[CONFIG] Polling: Every 2 seconds | Max Positions: 5 | Timeframe: M1")
    print()

    while True:
        try:
            df = get_market_data(symbol, timeframe, count=100)
            if df is not None and len(df) >= 50:
                
                # ===== CONFLUENCE #1: MASTER TREND (50-Period EMA) =====
                df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
                current_price = df["close"].iloc[-1]
                ema_50 = df["ema_50"].iloc[-1]
                
                # Determine trend direction
                trend_bullish = current_price > ema_50
                trend_bearish = current_price < ema_50
                
                # ===== CONFLUENCE #2: ENTRY TIMING (5/13 SMA Crossover) =====
                df["sma_5"] = df["close"].rolling(window=5).mean()
                df["sma_13"] = df["close"].rolling(window=13).mean()
                
                sma_5_current = df["sma_5"].iloc[-1]
                sma_13_current = df["sma_13"].iloc[-1]
                sma_5_prev = df["sma_5"].iloc[-2]
                sma_13_prev = df["sma_13"].iloc[-2]
                
                # Detect crossovers
                bullish_cross = (sma_5_prev <= sma_13_prev) and (sma_5_current > sma_13_current)
                bearish_cross = (sma_5_prev >= sma_13_prev) and (sma_5_current < sma_13_current)
                
                # ===== CONFLUENCE #3: VOLUME FILTER =====
                if "tick_volume" in df.columns:
                    volume_data = df["tick_volume"]
                else:
                    # Fallback: use spread as pseudo-volume metric
                    volume_data = (df["high"] - df["low"])
                
                avg_volume_10 = volume_data.iloc[-10:].mean()
                current_volume = volume_data.iloc[-1]
                volume_strong = current_volume > avg_volume_10
                
                # ===== DECISION LOGIC: ALL THREE MUST ALIGN =====
                current_positions = count_open_positions(symbol)
                
                # BUY: Price above EMA + 5/13 bullish cross + Strong volume
                if trend_bullish and bullish_cross and volume_strong and current_positions < max_positions:
                    print(f"[HIGH PROBABILITY SIGNAL] BUY SETUP DETECTED!")
                    print(f"  [TREND] Price {current_price:.2f} > EMA50 {ema_50:.2f} (BULLISH)")
                    print(f"  [CROSS] SMA5 {sma_5_current:.2f} > SMA13 {sma_13_current:.2f} (BULLISH CROSS)")
                    print(f"  [VOLUME] Current {current_volume:.0f} > Avg10 {avg_volume_10:.0f} (STRONG)")
                    print(f"  >>> EXECUTING BUY ORDER <<<")
                    result = execute_trade("BUY", symbol, "1m")
                    print(f"  [RESULT] {result}\n")
                
                # SELL: Price below EMA + 5/13 bearish cross + Strong volume
                elif trend_bearish and bearish_cross and volume_strong and current_positions < max_positions:
                    print(f"[HIGH PROBABILITY SIGNAL] SELL SETUP DETECTED!")
                    print(f"  [TREND] Price {current_price:.2f} < EMA50 {ema_50:.2f} (BEARISH)")
                    print(f"  [CROSS] SMA5 {sma_5_current:.2f} < SMA13 {sma_13_current:.2f} (BEARISH CROSS)")
                    print(f"  [VOLUME] Current {current_volume:.0f} > Avg10 {avg_volume_10:.0f} (STRONG)")
                    print(f"  >>> EXECUTING SELL ORDER <<<")
                    result = execute_trade("SELL", symbol, "1m")
                    print(f"  [RESULT] {result}\n")
                
                else:
                    # Show why we're NOT trading (confluence status)
                    trend_ok = "✓" if (trend_bullish and bullish_cross) or (trend_bearish and bearish_cross) else "✗"
                    cross_ok = "✓" if bullish_cross or bearish_cross else "✗"
                    volume_ok = "✓" if volume_strong else "✗"
                    
                    print(f"[MONITOR] Trend={trend_ok} Cross={cross_ok} Volume={volume_ok} | Price={current_price:.2f} EMA={ema_50:.2f} | Pos={current_positions}/5")

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(1)  # Poll every 1 second for non-stop action on demo


if __name__ == "__main__":
    if mt5.initialize():
        print("[SCANNER] Autonomous Agent Activated. Monitoring XAUUSD trends...")
        try:
            analyze_and_trade()
        finally:
            # Clean shutdown at the very end
            mt5.shutdown()
    else:
        print("[ERROR] Failed to initialize MT5")