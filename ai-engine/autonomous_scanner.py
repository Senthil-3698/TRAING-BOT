import MetaTrader5 as mt5
import pandas as pd
import time
import numpy as np
import asyncio
import argparse
from datetime import datetime, timedelta, timezone

from orchestrator import on_signal_received
from trade_journal import TradeJournal

journal = TradeJournal()


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


def get_session_bounds(ts: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    ts = pd.Timestamp(ts)
    start = ts.normalize() + pd.Timedelta(hours=8)
    if ts < start:
        start = start - pd.Timedelta(days=1)
    end = start + pd.Timedelta(hours=13)
    return start, end


def compute_market_structure_state(m5_df: pd.DataFrame) -> str:
    if m5_df is None or len(m5_df) < 40:
        return "INSUFFICIENT"

    highs = m5_df["high"].values
    lows = m5_df["low"].values
    swing_highs = []
    swing_lows = []

    for i in range(2, len(m5_df) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            swing_lows.append(lows[i])

    if len(swing_highs) < 5 or len(swing_lows) < 5:
        return "INSUFFICIENT"

    last_highs = swing_highs[-5:]
    last_lows = swing_lows[-5:]

    highs_up = all(np.diff(last_highs) > 0)
    lows_up = all(np.diff(last_lows) > 0)
    highs_down = all(np.diff(last_highs) < 0)
    lows_down = all(np.diff(last_lows) < 0)

    if highs_up and lows_up:
        return "HH_HL"
    if highs_down and lows_down:
        return "LH_LL"
    return "CONSOLIDATING"


def filter_market_structure(m5_df: pd.DataFrame, action: str) -> tuple[bool, str]:
    state = compute_market_structure_state(m5_df)
    if state == "INSUFFICIENT":
        return False, "market_structure: insufficient swings"
    if action == "BUY" and state == "HH_HL":
        return True, f"market_structure: pass ({state})"
    if action == "SELL" and state == "LH_LL":
        return True, f"market_structure: pass ({state})"
    return False, f"market_structure: reject ({state})"


def compute_volume_profile_levels(m15_df: pd.DataFrame, bar_time: pd.Timestamp) -> tuple[float, float, float]:
    if m15_df is None or len(m15_df) < 20:
        return float("nan"), float("nan"), float("nan")

    session_start, session_end = get_session_bounds(bar_time)
    session_df = m15_df[(m15_df["time"] >= session_start) & (m15_df["time"] <= min(bar_time, session_end))]
    if len(session_df) < 10:
        return float("nan"), float("nan"), float("nan")

    typical_price = (session_df["high"] + session_df["low"] + session_df["close"]) / 3.0
    weights = session_df["tick_volume"] if "tick_volume" in session_df.columns else (session_df["high"] - session_df["low"]).clip(lower=0.0001)

    bins = 24
    hist, edges = np.histogram(typical_price.values, bins=bins, weights=weights.values)
    if hist.sum() <= 0:
        return float("nan"), float("nan"), float("nan")

    poc_idx = int(np.argmax(hist))
    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0)

    total = float(hist.sum())
    target = 0.70 * total
    selected = {poc_idx}
    acc = float(hist[poc_idx])
    left = poc_idx - 1
    right = poc_idx + 1

    while acc < target and (left >= 0 or right < len(hist)):
        left_val = hist[left] if left >= 0 else -1
        right_val = hist[right] if right < len(hist) else -1
        if right_val >= left_val:
            if right < len(hist):
                selected.add(right)
                acc += float(hist[right])
                right += 1
            elif left >= 0:
                selected.add(left)
                acc += float(hist[left])
                left -= 1
        else:
            if left >= 0:
                selected.add(left)
                acc += float(hist[left])
                left -= 1
            elif right < len(hist):
                selected.add(right)
                acc += float(hist[right])
                right += 1

    lower_edge = min(selected)
    upper_edge = max(selected) + 1
    val = float(edges[lower_edge])
    vah = float(edges[upper_edge])
    return poc, vah, val


def filter_volume_profile_m15(
    m15_df: pd.DataFrame,
    bar_time: pd.Timestamp,
    current_price: float,
    action: str,
    structure_state: str,
    atr_value: float | None,
) -> tuple[bool, str]:
    poc, vah, val = compute_volume_profile_levels(m15_df, bar_time)
    if np.isnan(poc) or np.isnan(vah) or np.isnan(val):
        return False, "volume_profile: insufficient session profile"

    if structure_state != "CONSOLIDATING":
        return True, f"volume_profile: pass (trend market, POC={poc:.2f})"

    proximity = max((atr_value or 0.0) * 0.25, 0.15)
    if action == "BUY" and current_price >= (vah - proximity):
        return False, f"volume_profile: reject BUY near VAH ({vah:.2f}) in range"
    if action == "SELL" and current_price <= (val + proximity):
        return False, f"volume_profile: reject SELL near VAL ({val:.2f}) in range"
    return True, f"volume_profile: pass (POC={poc:.2f}, VAH={vah:.2f}, VAL={val:.2f})"


def compute_orderflow_delta(m1_df: pd.DataFrame, bars: int = 20) -> float:
    window = m1_df.tail(bars)
    if len(window) < bars:
        return 0.0
    volume = window["tick_volume"] if "tick_volume" in window.columns else (window["high"] - window["low"]).clip(lower=0.0001)
    direction = np.sign(window["close"].diff().fillna(0.0))
    return float((direction * volume).sum())


def filter_orderflow_delta(m1_df: pd.DataFrame, action: str) -> tuple[bool, str]:
    delta_value = compute_orderflow_delta(m1_df, bars=20)
    if action == "BUY" and delta_value > 0:
        return True, f"orderflow_delta: pass ({delta_value:.1f})"
    if action == "SELL" and delta_value < 0:
        return True, f"orderflow_delta: pass ({delta_value:.1f})"
    return False, f"orderflow_delta: reject ({delta_value:.1f})"


def compute_session_atr_regime(m15_df: pd.DataFrame, bar_time: pd.Timestamp) -> tuple[float | None, float | None]:
    if m15_df is None or len(m15_df) < 1200:
        return None, None

    # 13 trading hours (08:00-21:00 UTC) = 52 bars on M15.
    session_bars = 52
    df = m15_df[m15_df["time"] <= bar_time].tail(session_bars * 25).copy()
    if len(df) < session_bars * 21:
        return None, None

    df["range"] = df["high"] - df["low"]
    current_session = df.tail(session_bars)
    if len(current_session) < session_bars:
        return None, None
    current_session_atr = float(current_session["range"].mean())

    previous = df.iloc[: -session_bars]
    day_atrs = []
    for k in range(20):
        seg = previous.iloc[-session_bars * (k + 1) : -session_bars * k if k > 0 else None]
        if len(seg) < session_bars:
            break
        day_atrs.append(float(seg["range"].mean()))

    if len(day_atrs) < 20:
        return current_session_atr, None

    baseline = float(np.mean(day_atrs))
    return current_session_atr, baseline


def filter_session_volatility_regime(
    m15_df: pd.DataFrame,
    bar_time: pd.Timestamp,
    ema_distance: float,
    volume_strong: bool,
) -> tuple[bool, str]:
    current_atr, baseline_atr = compute_session_atr_regime(m15_df, bar_time)
    if current_atr is None or baseline_atr is None or baseline_atr <= 0:
        return False, "session_volatility: insufficient 20-day baseline"

    low_vol = current_atr < baseline_atr
    stronger_confluence = abs(ema_distance) >= 0.12 and volume_strong

    if low_vol and not stronger_confluence:
        return False, f"session_volatility: reject low-vol ({current_atr:.4f} < {baseline_atr:.4f}) without stronger confluence"
    regime = "LOW_VOL" if low_vol else "NORMAL_OR_HIGH_VOL"
    return True, f"session_volatility: pass ({regime}, sessionATR={current_atr:.4f}, baseline={baseline_atr:.4f})"


def analyze_and_trade():
    symbol = "XAUUSD"
    timeframe = mt5.TIMEFRAME_M1  # M1 for precision entry timing
    max_positions = 5

    print("[SCANNER] TRIPLE CONFLUENCE SNIPER ACTIVATED")
    print("[CONFIG] Baseline: 50-EMA (Master Trend) | Trigger: 5/13-SMA Cross | Filter: Volume")
    print("[CONFIG] Polling: Every 2 seconds | Max Positions: 5 | Timeframe: M1")
    print()
    last_signal_bar_time = None

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

                # ===== EXTRA CONTEXT FOR JOURNAL =====
                delta = df["close"].diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                avg_gain = gain.rolling(window=14).mean().iloc[-1]
                avg_loss = loss.rolling(window=14).mean().iloc[-1]
                if avg_loss and avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi_value = 100 - (100 / (1 + rs))
                else:
                    rsi_value = 100.0

                ema_distance = ((current_price - ema_50) / ema_50) * 100 if ema_50 else 0.0

                m5_df = get_market_data(symbol, mt5.TIMEFRAME_M5, count=900)
                if m5_df is not None and len(m5_df) >= 50:
                    m5_ema = m5_df["close"].ewm(span=50, adjust=False).mean().iloc[-1]
                    m5_price = m5_df["close"].iloc[-1]
                    m5_trend = "BULLISH" if m5_price > m5_ema else "BEARISH"
                    atr_value = (
                        m5_df["high"] - m5_df["low"]
                    ).rolling(window=14).mean().iloc[-1]
                else:
                    m5_trend = "UNKNOWN"
                    atr_value = None

                m15_df = get_market_data(symbol, mt5.TIMEFRAME_M15, count=2600)

                h1_df = get_market_data(symbol, mt5.TIMEFRAME_H1, count=120)
                h4_df = get_market_data(symbol, mt5.TIMEFRAME_H4, count=120)

                def tf_bias(frame):
                    if frame is None or len(frame) < 50:
                        return "NO_CONFLUENCE"
                    ema = frame["close"].ewm(span=50, adjust=False).mean().iloc[-1]
                    price = frame["close"].iloc[-1]
                    if price > ema * 1.001:
                        return "BULLISH"
                    if price < ema * 0.999:
                        return "BEARISH"
                    return "NO_CONFLUENCE"

                h1_bias = tf_bias(h1_df)
                h4_bias = tf_bias(h4_df)
                integrated_bias = h1_bias if h1_bias == h4_bias else "NO_CONFLUENCE"

                bar_time = df["time"].iloc[-1]
                
                # ===== DECISION LOGIC: ALL THREE MUST ALIGN =====
                current_positions = count_open_positions(symbol)
                
                signal_action = "BUY" if bullish_cross else "SELL" if bearish_cross else None

                if signal_action and last_signal_bar_time == bar_time:
                    time.sleep(1)
                    continue

                structure_state = compute_market_structure_state(m5_df)
                structure_pass, structure_reason = filter_market_structure(m5_df, signal_action) if signal_action else (False, "market_structure: no action")
                vp_pass, vp_reason = filter_volume_profile_m15(
                    m15_df,
                    bar_time,
                    float(current_price),
                    signal_action,
                    structure_state,
                    float(atr_value) if atr_value is not None else None,
                ) if signal_action else (False, "volume_profile: no action")
                delta_pass, delta_reason = filter_orderflow_delta(df, signal_action) if signal_action else (False, "orderflow_delta: no action")
                regime_pass, regime_reason = filter_session_volatility_regime(
                    m15_df,
                    bar_time,
                    float(ema_distance),
                    bool(volume_strong),
                ) if signal_action else (False, "session_volatility: no action")

                all_filters_pass = all([structure_pass, vp_pass, delta_pass, regime_pass])

                # BUY: baseline + microstructure filters
                if signal_action == "BUY" and trend_bullish and volume_strong and current_positions < max_positions and all_filters_pass:
                    print(f"[HIGH PROBABILITY SIGNAL] BUY SETUP DETECTED!")
                    print(f"  [TREND] Price {current_price:.2f} > EMA50 {ema_50:.2f} (BULLISH)")
                    print(f"  [CROSS] SMA5 {sma_5_current:.2f} > SMA13 {sma_13_current:.2f} (BULLISH CROSS)")
                    print(f"  [VOLUME] Current {current_volume:.0f} > Avg10 {avg_volume_10:.0f} (STRONG)")
                    print(f"  >>> EXECUTING BUY ORDER <<<")
                    confidence = min(1.0, max(0.0, (abs(ema_distance) / 0.5) + (1.0 if volume_strong else 0.0)))
                    signal = {
                        "source": "autonomous_scanner",
                        "setup_type": "SMA_CROSS_SCALP",
                        "symbol": symbol,
                        "timeframe": "1m",
                        "action": "BUY",
                        "timestamp": bar_time.timestamp(),
                        "signal_bar_time": bar_time.isoformat(),
                        "signal_bar_relation": "current",
                        "intended_price": float(current_price),
                        "confidence_score": round(confidence, 4),
                        "indicators": {
                            "rsi": float(rsi_value),
                            "ema_distance": float(ema_distance),
                            "atr": float(atr_value) if atr_value is not None else None,
                            "m5_trend": m5_trend,
                            "h1_bias": h1_bias,
                            "h4_bias": h4_bias,
                            "integrated_bias": integrated_bias,
                            "market_structure": structure_state,
                            "orderflow_delta": float(compute_orderflow_delta(df, bars=20)),
                            "filter_structure_reason": structure_reason,
                            "filter_volume_profile_reason": vp_reason,
                            "filter_orderflow_reason": delta_reason,
                            "filter_vol_regime_reason": regime_reason,
                        },
                    }
                    asyncio.run(on_signal_received(signal))
                    last_signal_bar_time = bar_time
                    print("  [RESULT] SIGNAL SENT TO ORCHESTRATOR\n")
                
                # SELL: baseline + microstructure filters
                elif signal_action == "SELL" and trend_bearish and volume_strong and current_positions < max_positions and all_filters_pass:
                    print(f"[HIGH PROBABILITY SIGNAL] SELL SETUP DETECTED!")
                    print(f"  [TREND] Price {current_price:.2f} < EMA50 {ema_50:.2f} (BEARISH)")
                    print(f"  [CROSS] SMA5 {sma_5_current:.2f} < SMA13 {sma_13_current:.2f} (BEARISH CROSS)")
                    print(f"  [VOLUME] Current {current_volume:.0f} > Avg10 {avg_volume_10:.0f} (STRONG)")
                    print(f"  >>> EXECUTING SELL ORDER <<<")
                    confidence = min(1.0, max(0.0, (abs(ema_distance) / 0.5) + (1.0 if volume_strong else 0.0)))
                    signal = {
                        "source": "autonomous_scanner",
                        "setup_type": "SMA_CROSS_SCALP",
                        "symbol": symbol,
                        "timeframe": "1m",
                        "action": "SELL",
                        "timestamp": bar_time.timestamp(),
                        "signal_bar_time": bar_time.isoformat(),
                        "signal_bar_relation": "current",
                        "intended_price": float(current_price),
                        "confidence_score": round(confidence, 4),
                        "indicators": {
                            "rsi": float(rsi_value),
                            "ema_distance": float(ema_distance),
                            "atr": float(atr_value) if atr_value is not None else None,
                            "m5_trend": m5_trend,
                            "h1_bias": h1_bias,
                            "h4_bias": h4_bias,
                            "integrated_bias": integrated_bias,
                            "market_structure": structure_state,
                            "orderflow_delta": float(compute_orderflow_delta(df, bars=20)),
                            "filter_structure_reason": structure_reason,
                            "filter_volume_profile_reason": vp_reason,
                            "filter_orderflow_reason": delta_reason,
                            "filter_vol_regime_reason": regime_reason,
                        },
                    }
                    asyncio.run(on_signal_received(signal))
                    last_signal_bar_time = bar_time
                    print("  [RESULT] SIGNAL SENT TO ORCHESTRATOR\n")
                
                else:
                    if signal_action:
                        rejection_reasons = []
                        if signal_action == "BUY" and not trend_bullish:
                            rejection_reasons.append("Trend filter failed for BUY")
                        if signal_action == "SELL" and not trend_bearish:
                            rejection_reasons.append("Trend filter failed for SELL")
                        if not volume_strong:
                            rejection_reasons.append("Volume filter failed")
                        if current_positions >= max_positions:
                            rejection_reasons.append("Max positions reached")
                        if not structure_pass:
                            rejection_reasons.append(structure_reason)
                        if not vp_pass:
                            rejection_reasons.append(vp_reason)
                        if not delta_pass:
                            rejection_reasons.append(delta_reason)
                        if not regime_pass:
                            rejection_reasons.append(regime_reason)

                        journal.log_signal(
                            source="autonomous_scanner",
                            symbol=symbol,
                            action=signal_action,
                            timeframe="1m",
                            signal_ts=bar_time.to_pydatetime(),
                            rsi_value=float(rsi_value),
                            ema_distance=float(ema_distance),
                            atr_value=float(atr_value) if atr_value is not None else None,
                            m5_trend=m5_trend,
                            h1_bias=h1_bias,
                            h4_bias=h4_bias,
                            integrated_bias=integrated_bias,
                            news_context=None,
                            ai_decision="REJECTED",
                            ai_reasoning="Scanner-level filter veto",
                            ai_confidence=0.0,
                            decision_status="REJECTED",
                            rejection_reason="; ".join(rejection_reasons) if rejection_reasons else "No actionable confluence",
                            metadata={
                                "trend_bullish": trend_bullish,
                                "trend_bearish": trend_bearish,
                                "bullish_cross": bullish_cross,
                                "bearish_cross": bearish_cross,
                                "volume_strong": volume_strong,
                                "current_positions": current_positions,
                                "structure_state": structure_state,
                                "structure_reason": structure_reason,
                                "volume_profile_reason": vp_reason,
                                "orderflow_reason": delta_reason,
                                "volatility_regime_reason": regime_reason,
                            },
                        )
                        last_signal_bar_time = bar_time

                    # Show why we're NOT trading (confluence status)
                    trend_ok = "✓" if (trend_bullish and bullish_cross) or (trend_bearish and bearish_cross) else "✗"
                    cross_ok = "✓" if bullish_cross or bearish_cross else "✗"
                    volume_ok = "✓" if volume_strong else "✗"
                    
                    print(f"[MONITOR] Trend={trend_ok} Cross={cross_ok} Volume={volume_ok} Struct={'✓' if structure_pass else '✗'} VP={'✓' if vp_pass else '✗'} Delta={'✓' if delta_pass else '✗'} VolReg={'✓' if regime_pass else '✗'} | Price={current_price:.2f} EMA={ema_50:.2f} | Pos={current_positions}/5")

        except Exception as e:
            print(f"[ERROR] {e}")

        time.sleep(1)  # Poll every 1 second for non-stop action on demo


def run_filter_ablation_backtest(symbol="XAUUSD", days=90, lookahead_bars=15):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    if not mt5.symbol_select(symbol, True):
        print("[ABLATION] Failed to select symbol in MT5")
        return

    def fetch_chunked(timeframe, total_bars):
        frames = []
        start_pos = 0
        remaining = total_bars
        chunk_size = 50000
        while remaining > 0:
            take = min(chunk_size, remaining)
            rates = mt5.copy_rates_from_pos(symbol, timeframe, start_pos, take)
            if rates is None or len(rates) == 0:
                break
            frames.append(pd.DataFrame(rates))
            start_pos += take
            remaining -= take
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
        return df

    m1_bars = max(days * 24 * 60 + 3000, 10000)
    m5_bars = max(days * 24 * 12 + 2000, 5000)
    m15_bars = max(days * 24 * 4 + 3000, 5000)

    m1_rates = fetch_chunked(mt5.TIMEFRAME_M1, m1_bars)
    m5_rates = fetch_chunked(mt5.TIMEFRAME_M5, m5_bars)
    m15_rates = fetch_chunked(mt5.TIMEFRAME_M15, m15_bars)
    if m1_rates is None or m5_rates is None or m15_rates is None:
        print("[ABLATION] Failed to load data from MT5")
        return

    m1 = pd.DataFrame(m1_rates)
    m5 = pd.DataFrame(m5_rates)
    m15 = pd.DataFrame(m15_rates)
    m1["time"] = pd.to_datetime(m1["time"], unit="s", utc=True)
    m5["time"] = pd.to_datetime(m5["time"], unit="s", utc=True)
    m15["time"] = pd.to_datetime(m15["time"], unit="s", utc=True)
    m1 = m1[m1["time"] >= start].reset_index(drop=True)
    m5 = m5[m5["time"] >= (start - timedelta(days=5))].reset_index(drop=True)
    m15 = m15[m15["time"] >= (start - timedelta(days=25))].reset_index(drop=True)

    stats = {
        "baseline": {"trades": 0, "wins": 0, "ret_sum": 0.0},
        "market_structure": {"trades": 0, "wins": 0, "ret_sum": 0.0},
        "volume_profile": {"trades": 0, "wins": 0, "ret_sum": 0.0},
        "orderflow_delta": {"trades": 0, "wins": 0, "ret_sum": 0.0},
        "session_volatility": {"trades": 0, "wins": 0, "ret_sum": 0.0},
    }

    for i in range(120, len(m1) - lookahead_bars):
        m1_win = m1.iloc[i - 119 : i + 1]
        close = m1_win["close"]
        ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        trend_bullish = close.iloc[-1] > ema_50
        trend_bearish = close.iloc[-1] < ema_50

        sma_5 = close.rolling(5).mean()
        sma_13 = close.rolling(13).mean()
        bullish_cross = (sma_5.iloc[-2] <= sma_13.iloc[-2]) and (sma_5.iloc[-1] > sma_13.iloc[-1])
        bearish_cross = (sma_5.iloc[-2] >= sma_13.iloc[-2]) and (sma_5.iloc[-1] < sma_13.iloc[-1])
        action = "BUY" if bullish_cross else "SELL" if bearish_cross else None
        if action is None:
            continue

        volume = m1_win["tick_volume"] if "tick_volume" in m1_win.columns else (m1_win["high"] - m1_win["low"])
        volume_strong = volume.iloc[-1] > volume.iloc[-10:].mean()

        baseline_pass = (action == "BUY" and trend_bullish and volume_strong) or (action == "SELL" and trend_bearish and volume_strong)
        if not baseline_pass:
            continue

        bar_time = m1_win["time"].iloc[-1]
        m5_win = m5[m5["time"] <= bar_time].tail(900)
        m15_win = m15[m15["time"] <= bar_time].tail(2600)

        ema_distance = ((close.iloc[-1] - ema_50) / ema_50) * 100 if ema_50 else 0.0
        atr_value = (m5_win["high"] - m5_win["low"]).rolling(14).mean().iloc[-1] if len(m5_win) >= 20 else None
        structure_state = compute_market_structure_state(m5_win)

        struct_pass, _ = filter_market_structure(m5_win, action)
        vp_pass, _ = filter_volume_profile_m15(m15_win, bar_time, float(close.iloc[-1]), action, structure_state, float(atr_value) if atr_value is not None else None)
        delta_pass, _ = filter_orderflow_delta(m1_win, action)
        vol_pass, _ = filter_session_volatility_regime(m15_win, bar_time, float(ema_distance), bool(volume_strong))

        entry = float(close.iloc[-1])
        future = float(m1["close"].iloc[i + lookahead_bars])
        ret = (future - entry) if action == "BUY" else (entry - future)
        win = ret > 0

        stats["baseline"]["trades"] += 1
        stats["baseline"]["wins"] += int(win)
        stats["baseline"]["ret_sum"] += ret

        if struct_pass:
            stats["market_structure"]["trades"] += 1
            stats["market_structure"]["wins"] += int(win)
            stats["market_structure"]["ret_sum"] += ret
        if vp_pass:
            stats["volume_profile"]["trades"] += 1
            stats["volume_profile"]["wins"] += int(win)
            stats["volume_profile"]["ret_sum"] += ret
        if delta_pass:
            stats["orderflow_delta"]["trades"] += 1
            stats["orderflow_delta"]["wins"] += int(win)
            stats["orderflow_delta"]["ret_sum"] += ret
        if vol_pass:
            stats["session_volatility"]["trades"] += 1
            stats["session_volatility"]["wins"] += int(win)
            stats["session_volatility"]["ret_sum"] += ret

    rows = []
    baseline_wr = (stats["baseline"]["wins"] / stats["baseline"]["trades"] * 100.0) if stats["baseline"]["trades"] else 0.0
    baseline_ret = (stats["baseline"]["ret_sum"] / stats["baseline"]["trades"]) if stats["baseline"]["trades"] else 0.0

    for name, data in stats.items():
        trades = data["trades"]
        win_rate = (data["wins"] / trades * 100.0) if trades else 0.0
        avg_ret = (data["ret_sum"] / trades) if trades else 0.0
        rows.append(
            {
                "filter": name,
                "trades": trades,
                "win_rate_pct": round(win_rate, 2),
                "avg_return_points": round(avg_ret, 4),
                "marginal_win_rate_vs_baseline": round(win_rate - baseline_wr, 2),
                "marginal_avg_return_vs_baseline": round(avg_ret - baseline_ret, 4),
            }
        )

    result_df = pd.DataFrame(rows)
    out_dir = "../backtest_results"
    out_file = f"{out_dir}/FILTER_ABLATION_{symbol}_{days}d.csv"
    result_df.to_csv(out_file, index=False)
    print("[ABLATION] Marginal contribution summary")
    print(result_df.to_string(index=False))
    print(f"[ABLATION] Saved {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous scanner with microstructure filters")
    parser.add_argument("--ablation", action="store_true", help="Run filter ablation backtest instead of live scanner")
    parser.add_argument("--days", type=int, default=90, help="Backtest window in days for ablation")
    args = parser.parse_args()

    if not mt5.initialize():
        print("[ERROR] Failed to initialize MT5")
    else:
        try:
            if args.ablation:
                run_filter_ablation_backtest(days=args.days)
            else:
                print("[SCANNER] Autonomous Agent Activated. Monitoring XAUUSD trends...")
                analyze_and_trade()
        finally:
            mt5.shutdown()