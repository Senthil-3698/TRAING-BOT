import MetaTrader5 as mt5
import pandas as pd
import time
import numpy as np
from datetime import datetime, timezone

from mt5_executor import execute_trade
from state_manager import auto_update_bias, get_integrated_bias

# ── Configuration ──────────────────────────────────────────────────────────────
SYMBOL = "XAUUSD"
MAX_POSITIONS = 5
MAGIC = 123456
POLL_INTERVAL = 2           # seconds between scans
BIAS_REFRESH_BARS = 30      # refresh higher-TF bias every N bars (~1 min at 2s)
COOLDOWN_SECONDS = 300      # minimum gap between trades on same symbol (5 min)
RSI_OVERBOUGHT = 65         # don't BUY above this RSI
RSI_OVERSOLD = 35           # don't SELL below this RSI
MIN_EMA_DISTANCE_PCT = 0.0002  # price must be ≥0.02% away from EMA (avoid ranging)

# ── Helpers ─────────────────────────────────────────────────────────────────────

def get_market_data(symbol, timeframe, count=120):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def count_open_positions(symbol=SYMBOL):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    return len([p for p in positions if p.magic == MAGIC])


def compute_rsi(closes, period=14):
    """Standard Wilder RSI."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def is_active_session():
    """
    Only trade during London (07:00–16:00 UTC) or New York (13:00–21:00 UTC).
    These sessions have the highest liquidity and tightest spreads for XAUUSD.
    """
    hour = datetime.now(timezone.utc).hour
    london = 7 <= hour < 16
    new_york = 13 <= hour < 21
    return london or new_york


def get_m5_trend(symbol):
    """
    M5 trend confirmation: returns 'BULLISH', 'BEARISH', or 'NEUTRAL'
    based on price vs EMA50 on the M5 chart.
    """
    df = get_market_data(symbol, mt5.TIMEFRAME_M5, count=60)
    if df is None or len(df) < 55:
        return "NEUTRAL"
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    price = df["close"].iloc[-1]
    ema = df["ema_50"].iloc[-1]
    if price > ema * (1 + MIN_EMA_DISTANCE_PCT):
        return "BULLISH"
    if price < ema * (1 - MIN_EMA_DISTANCE_PCT):
        return "BEARISH"
    return "NEUTRAL"


# ── Main Loop ───────────────────────────────────────────────────────────────────

def analyze_and_trade():
    print("[SCANNER] TRIPLE CONFLUENCE SNIPER v2 ACTIVATED")
    print("[CONFIG] Trend: EMA50 M1+M5 | Entry: SMA5/13 Cross | Filter: RSI14 + Volume")
    print(f"[CONFIG] Session: London/NY only | Cooldown: {COOLDOWN_SECONDS}s | Max Pos: {MAX_POSITIONS}")
    print()

    last_trade_time = 0.0   # epoch seconds of last executed trade
    bias_tick = 0           # counter for periodic bias refresh

    while True:
        try:
            # ── 0. Session gate ────────────────────────────────────────────
            if not is_active_session():
                current_hour = datetime.now(timezone.utc).hour
                print(f"[SESSION] Outside trading hours (UTC {current_hour:02d}:xx). Waiting...")
                time.sleep(60)
                continue

            # ── 1. Refresh higher-TF bias periodically ─────────────────────
            bias_tick += 1
            if bias_tick >= BIAS_REFRESH_BARS:
                auto_update_bias(SYMBOL)
                bias_tick = 0

            higher_tf_bias = get_integrated_bias(SYMBOL)

            # ── 2. Fetch M1 data ───────────────────────────────────────────
            df = get_market_data(SYMBOL, mt5.TIMEFRAME_M1, count=120)
            if df is None or len(df) < 55:
                time.sleep(POLL_INTERVAL)
                continue

            # ── 3. Indicators ──────────────────────────────────────────────
            df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
            df["sma_5"] = df["close"].rolling(window=5).mean()
            df["sma_13"] = df["close"].rolling(window=13).mean()

            current_price = df["close"].iloc[-1]
            ema_50 = df["ema_50"].iloc[-1]

            sma_5_cur = df["sma_5"].iloc[-1]
            sma_13_cur = df["sma_13"].iloc[-1]
            sma_5_prev = df["sma_5"].iloc[-2]
            sma_13_prev = df["sma_13"].iloc[-2]

            rsi = compute_rsi(df["close"].values)

            # ── 4. Volume filter ───────────────────────────────────────────
            vol_col = "tick_volume" if "tick_volume" in df.columns else None
            if vol_col:
                volume_data = df[vol_col]
            else:
                volume_data = df["high"] - df["low"]
            avg_vol = volume_data.iloc[-10:].mean()
            cur_vol = volume_data.iloc[-1]
            volume_strong = cur_vol > avg_vol

            # ── 5. Confluence checks ───────────────────────────────────────
            ema_distance_pct = abs(current_price - ema_50) / ema_50
            price_not_ranging = ema_distance_pct >= MIN_EMA_DISTANCE_PCT

            trend_bullish = current_price > ema_50 and price_not_ranging
            trend_bearish = current_price < ema_50 and price_not_ranging

            bullish_cross = (sma_5_prev <= sma_13_prev) and (sma_5_cur > sma_13_cur)
            bearish_cross = (sma_5_prev >= sma_13_prev) and (sma_5_cur < sma_13_cur)

            m5_trend = get_m5_trend(SYMBOL)

            # Higher-TF bias must align (or be absent)
            htf_allows_buy = higher_tf_bias in ("BULLISH", "NO_CONFLUENCE")
            htf_allows_sell = higher_tf_bias in ("BEARISH", "NO_CONFLUENCE")

            rsi_allows_buy = rsi < RSI_OVERBOUGHT
            rsi_allows_sell = rsi > RSI_OVERSOLD

            current_positions = count_open_positions(SYMBOL)
            now = time.time()
            cooldown_ok = (now - last_trade_time) >= COOLDOWN_SECONDS

            # ── 6. Decision ────────────────────────────────────────────────
            buy_signal = (
                trend_bullish
                and bullish_cross
                and volume_strong
                and rsi_allows_buy
                and m5_trend != "BEARISH"
                and htf_allows_buy
                and current_positions < MAX_POSITIONS
                and cooldown_ok
            )

            sell_signal = (
                trend_bearish
                and bearish_cross
                and volume_strong
                and rsi_allows_sell
                and m5_trend != "BULLISH"
                and htf_allows_sell
                and current_positions < MAX_POSITIONS
                and cooldown_ok
            )

            if buy_signal:
                print(f"[SIGNAL] BUY | Price {current_price:.2f} > EMA50 {ema_50:.2f} | "
                      f"SMA cross UP | RSI {rsi:.1f} | M5:{m5_trend} | HTF:{higher_tf_bias}")
                result = execute_trade("BUY", SYMBOL, "1m")
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    last_trade_time = now
                    print(f"  [OK] Ticket {result.order} executed\n")
                else:
                    retcode = result.retcode if result else "None"
                    print(f"  [FAIL] retcode={retcode}\n")

            elif sell_signal:
                print(f"[SIGNAL] SELL | Price {current_price:.2f} < EMA50 {ema_50:.2f} | "
                      f"SMA cross DOWN | RSI {rsi:.1f} | M5:{m5_trend} | HTF:{higher_tf_bias}")
                result = execute_trade("SELL", SYMBOL, "1m")
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    last_trade_time = now
                    print(f"  [OK] Ticket {result.order} executed\n")
                else:
                    retcode = result.retcode if result else "None"
                    print(f"  [FAIL] retcode={retcode}\n")

            else:
                t_ok = "✓" if (trend_bullish or trend_bearish) else "✗"
                c_ok = "✓" if (bullish_cross or bearish_cross) else "✗"
                v_ok = "✓" if volume_strong else "✗"
                r_ok = "✓" if (rsi_allows_buy or rsi_allows_sell) else "✗(RSI)"
                cd_ok = "✓" if cooldown_ok else f"✗(cd {int(COOLDOWN_SECONDS-(now-last_trade_time))}s)"
                print(
                    f"[WAIT] Trend={t_ok} Cross={c_ok} Vol={v_ok} RSI={r_ok} "
                    f"CD={cd_ok} | P={current_price:.2f} EMA={ema_50:.2f} "
                    f"RSI={rsi:.1f} M5={m5_trend} HTF={higher_tf_bias} Pos={current_positions}/{MAX_POSITIONS}"
                )

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    if mt5.initialize():
        print("[SCANNER] Autonomous Agent v2 Activated. Monitoring XAUUSD...")
        try:
            analyze_and_trade()
        finally:
            mt5.shutdown()
    else:
        print("[ERROR] Failed to initialize MT5")
