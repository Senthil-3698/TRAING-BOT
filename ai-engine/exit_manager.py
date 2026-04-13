import MetaTrader5 as mt5
import time
import redis
import json
import os
import httpx
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from mfe_analyzer import auto_tune_partial_r_if_due
from mt5_executor import partial_close_position
from state_manager import update_trade_stage
from risk_engine import RiskEngine

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6380")),
    db=0,
)
risk_engine = RiskEngine()

TREND_REFRESH_SECONDS = 30
TIME_EXIT_MINUTES_M1 = 45
STRONG_TREND_THRESHOLD = 70.0
PARTIAL_R_KEY = os.getenv("EXIT_PARTIAL_R_KEY", "exit:partial_r_multiple")
PARTIAL_R_MIN = 1.2
PARTIAL_R_MAX = 2.0
PARTIAL_R_DEFAULT = 1.5
AUTO_TUNE_CHECK_SECONDS = 300
EXIT_SCAN_INTERVAL_SECONDS = 0.5
BE_TRIGGER_POINTS = 50  # 5 pips = 50 points
BE_LOCK_POINTS = 10     # 1 pip = 10 points


def _get_tracked_trade(ticket):
    trade_raw = r.get(f"trade:{ticket}")
    if not trade_raw:
        return None
    try:
        return json.loads(trade_raw.decode("utf-8"))
    except (ValueError, TypeError):
        return None


def _get_trade_stage(ticket, tracked_trade=None):
    stage_raw = r.get(f"trade_stage:{ticket}")
    if stage_raw:
        return stage_raw.decode("utf-8")

    if tracked_trade and tracked_trade.get("stage"):
        return tracked_trade.get("stage")

    return "ENTRY"


def _update_ema(key, current_value, alpha=0.2):
    previous_raw = r.get(key)
    if previous_raw:
        try:
            previous_value = float(previous_raw.decode("utf-8"))
        except ValueError:
            previous_value = current_value
    else:
        previous_value = current_value

    updated = (alpha * current_value) + ((1 - alpha) * previous_value)
    r.set(key, str(updated))
    return updated


def _safe_to_datetime(value):
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _position_direction(pos_type):
    return "BUY" if pos_type == mt5.ORDER_TYPE_BUY else "SELL"


def _to_dataframe(symbol, timeframe=mt5.TIMEFRAME_M1, bars=120):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) < 30:
        return None
    df = pd.DataFrame(rates)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def _compute_atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    if atr.dropna().empty:
        return 0.0
    return float(atr.iloc[-1])


def _wilder(values, period):
    out = np.zeros_like(values, dtype=float)
    if len(values) < period:
        return out
    out[period - 1] = np.sum(values[:period])
    for i in range(period, len(values)):
        out[i] = out[i - 1] - (out[i - 1] / period) + values[i]
    return out


def _compute_adx(df, period=14):
    if len(df) < period + 5:
        return 0.0

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)

    for i in range(1, len(high)):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0.0
        minus_dm[i] = down if down > up and down > 0 else 0.0

        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i - 1])
        tr3 = abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)

    atr_w = _wilder(tr, period)
    plus_w = _wilder(plus_dm, period)
    minus_w = _wilder(minus_dm, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * (plus_w / atr_w)
        minus_di = 100.0 * (minus_w / atr_w)
        dx = 100.0 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, np.nan, (plus_di + minus_di))

    dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
    if len(dx) < period * 2:
        return float(dx[-1]) if len(dx) else 0.0

    adx = np.zeros_like(dx)
    adx[period * 2 - 1] = np.mean(dx[period - 1 : period * 2 - 1])
    for i in range(period * 2, len(dx)):
        adx[i] = ((adx[i - 1] * (period - 1)) + dx[i]) / period
    return float(adx[-1])


def _trend_strength_score(symbol):
    df = _to_dataframe(symbol, timeframe=mt5.TIMEFRAME_M1, bars=140)
    if df is None or len(df) < 60:
        return 50.0, {"reason": "insufficient_data"}

    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    adx = _compute_adx(df, period=14)
    atr = _compute_atr(df, period=14)
    atr = atr if atr > 0 else max(1e-6, float(close.iloc[-1]) * 0.0001)

    distance_from_ema20 = abs(float(close.iloc[-1]) - float(ema20.iloc[-1]))
    distance_norm = distance_from_ema20 / atr

    slope = float(ema50.iloc[-1] - ema50.iloc[-6]) if len(ema50) > 6 else 0.0
    slope_norm = abs(slope) / atr

    adx_component = min(100.0, max(0.0, adx * 2.0))
    distance_component = min(100.0, distance_norm * 100.0)
    slope_component = min(100.0, slope_norm * 180.0)
    score = (0.5 * adx_component) + (0.3 * distance_component) + (0.2 * slope_component)

    details = {
        "adx": round(adx, 3),
        "distance_from_ema20": round(distance_from_ema20, 6),
        "ema50_slope": round(slope, 6),
        "atr": round(atr, 6),
        "adx_component": round(adx_component, 3),
        "distance_component": round(distance_component, 3),
        "slope_component": round(slope_component, 3),
    }
    return float(round(min(100.0, max(0.0, score)), 3)), details


def _trend_exit_profile(score):
    if score > STRONG_TREND_THRESHOLD:
        return 4.0, 1.5, "STRONG"
    return 2.0, 0.7, "WEAK"


def _r_progress(pos_type, entry, current_price, risk_points):
    if risk_points <= 0:
        return 0.0
    if pos_type == mt5.ORDER_TYPE_BUY:
        return (current_price - entry) / risk_points
    return (entry - current_price) / risk_points


def _swing_points(df, lookback=60, n=2):
    sample = df.tail(lookback).reset_index(drop=True)
    highs = sample["high"].to_numpy(dtype=float)
    lows = sample["low"].to_numpy(dtype=float)
    points = []
    for i in range(n, len(sample) - n):
        window_h = highs[i - n : i + n + 1]
        window_l = lows[i - n : i + n + 1]
        if highs[i] == np.max(window_h):
            points.append(("H", i, highs[i]))
        elif lows[i] == np.min(window_l):
            points.append(("L", i, lows[i]))
    return points


def _opposing_structure_shift(symbol, pos_type):
    df = _to_dataframe(symbol, timeframe=mt5.TIMEFRAME_M1, bars=120)
    if df is None:
        return False, "structure: no data"

    points = _swing_points(df, lookback=80, n=2)
    highs = [p for p in points if p[0] == "H"]
    lows = [p for p in points if p[0] == "L"]

    if len(highs) < 2 or len(lows) < 2:
        return False, "structure: insufficient swings"

    last_high, prev_high = highs[-1][2], highs[-2][2]
    last_low, prev_low = lows[-1][2], lows[-2][2]

    bearish_shift = (last_high < prev_high) and (last_low < prev_low)
    bullish_shift = (last_high > prev_high) and (last_low > prev_low)

    if pos_type == mt5.ORDER_TYPE_BUY and bearish_shift:
        return True, "Opposing bearish structure shift detected (LH+LL)."
    if pos_type == mt5.ORDER_TYPE_SELL and bullish_shift:
        return True, "Opposing bullish structure shift detected (HH+HL)."
    return False, "structure: aligned"


def _close_position_market(ticket):
    result = partial_close_position(ticket, 1.0)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return True
    return False


def _get_partial_r_multiple() -> float:
    raw = r.get(PARTIAL_R_KEY)
    if not raw:
        return PARTIAL_R_DEFAULT
    try:
        value = float(raw.decode("utf-8"))
    except ValueError:
        return PARTIAL_R_DEFAULT
    return max(PARTIAL_R_MIN, min(PARTIAL_R_MAX, value))


def get_market_volatility(symbol):
    """
    Classifies volatility into NORMAL / ELEVATED / EXTREME.
    Uses spread expansion and 5-minute price-range spikes.
    """
    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if symbol_info is None or tick is None or symbol_info.point == 0:
        return "NORMAL", 1.0

    point = symbol_info.point
    spread_points = (tick.ask - tick.bid) / point

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 5)
    if rates is None or len(rates) == 0:
        range_points = 0.0
    else:
        high_5m = max(rate[2] for rate in rates)
        low_5m = min(rate[3] for rate in rates)
        range_points = (high_5m - low_5m) / point

    avg_spread = _update_ema(f"vol:{symbol}:avg_spread", spread_points)
    avg_range = _update_ema(f"vol:{symbol}:avg_range", range_points)

    if avg_range > 0 and range_points > (3.0 * avg_range):
        return "EXTREME", 0.5
    if avg_spread > 0 and spread_points > (2.0 * avg_spread):
        return "ELEVATED", 0.7
    return "NORMAL", 1.0


def log_trade_event(ticket, event_type):
    payload = {
        "ticketId": str(ticket),
        "eventType": event_type,
    }

    try:
        httpx.post("http://localhost:8080/events", json=payload, timeout=5.0)
    except httpx.HTTPError as error:
        print(f"EVENT LOG FAILED for {ticket}: {error}")

def manage_exits():
    if not mt5.initialize(): return

    last_tune_check_at = 0.0

    while True:
        now_epoch = time.time()
        if (now_epoch - last_tune_check_at) >= AUTO_TUNE_CHECK_SECONDS:
            try:
                tune_result = auto_tune_partial_r_if_due(min_new_trades=100)
                status = tune_result.get("status") if isinstance(tune_result, dict) else "unknown"
                if status == "tuned":
                    print(f"[PARTIAL-TUNE] Updated partial target to {tune_result.get('best_partial_r')}R")
            except Exception as error:
                print(f"[PARTIAL-TUNE] check failed: {error}")
            last_tune_check_at = now_epoch

        # 1. Get all open positions managed by our bot (using Magic Number)
        positions = mt5.positions_get(magic=123456) or []
        
        for pos in positions:
            symbol = pos.symbol
            ticket = pos.ticket
            entry = pos.price_open
            current_price = pos.price_current
            current_sl = pos.sl
            point = mt5.symbol_info(symbol).point

            tracked_trade = _get_tracked_trade(ticket)
            trade_stage = _get_trade_stage(ticket, tracked_trade)
            if trade_stage in {"BREAKEVEN", "TRAILING"}:
                # Allow partial-close management to run after breakeven.
                pass

            initial_sl = None
            if tracked_trade and tracked_trade.get("sl") is not None:
                try:
                    initial_sl = float(tracked_trade.get("sl"))
                except (ValueError, TypeError):
                    initial_sl = None

            if initial_sl is None:
                initial_sl = current_sl

            now = datetime.now(timezone.utc)

            if tracked_trade is None:
                tracked_trade = {}

            opened_at = _safe_to_datetime(tracked_trade.get("opened_at"))
            if opened_at is None:
                opened_at = datetime.fromtimestamp(int(getattr(pos, "time", time.time())), tz=timezone.utc)
                tracked_trade["opened_at"] = opened_at.isoformat()
                r.set(f"trade:{ticket}", json.dumps(tracked_trade, default=str))

            trend_score = tracked_trade.get("entry_trend_score")
            if trend_score is None:
                entry_score, entry_details = _trend_strength_score(symbol)
                tracked_trade["entry_trend_score"] = float(entry_score)
                tracked_trade["trend_score"] = float(entry_score)
                tracked_trade["trend_score_details"] = entry_details
                tracked_trade["trend_score_updated_at"] = now.isoformat()
                trend_score = entry_score
                r.set(f"trade:{ticket}", json.dumps(tracked_trade, default=str))

            updated_at = _safe_to_datetime(tracked_trade.get("trend_score_updated_at"))
            if updated_at is None or (now - updated_at).total_seconds() >= TREND_REFRESH_SECONDS:
                live_score, live_details = _trend_strength_score(symbol)
                tracked_trade["trend_score"] = float(live_score)
                tracked_trade["trend_score_details"] = live_details
                tracked_trade["trend_score_updated_at"] = now.isoformat()
                trend_score = live_score
                r.set(f"trade:{ticket}", json.dumps(tracked_trade, default=str))
            else:
                trend_score = float(tracked_trade.get("trend_score", trend_score))
            
            # Calculate Risk (R) dynamically from the original SL stored in Redis.
            risk_points = abs(entry - initial_sl)
            if risk_points <= 0:
                continue

            progress_r = _r_progress(pos.type, entry, current_price, risk_points)
            max_r = float(tracked_trade.get("max_r", -999.0))
            if progress_r > max_r:
                tracked_trade["max_r"] = float(progress_r)
                r.set(f"trade:{ticket}", json.dumps(tracked_trade, default=str))

            full_exit_r, runner_trail_r, trend_profile = _trend_exit_profile(float(trend_score))

            shield_mode, _ = get_market_volatility(symbol)
            be_lock_distance = BE_LOCK_POINTS * point
            if pos.type == mt5.ORDER_TYPE_BUY:
                profit_points = (current_price - entry) / point
            else:
                profit_points = (entry - current_price) / point

            # Time-based hard exit for M1 scalps: no +1R within 45 min.
            timeframe = str(tracked_trade.get("timeframe", "1m")).lower()
            if timeframe in {"1m", "m1"} and opened_at is not None:
                elapsed_min = (now - opened_at).total_seconds() / 60.0
                if elapsed_min >= TIME_EXIT_MINUTES_M1 and float(tracked_trade.get("max_r", progress_r)) < 1.0:
                    if _close_position_market(ticket):
                        update_trade_stage(ticket, "TIME_EXIT")
                        log_trade_event(ticket, "TIME_EXIT_45M_NO_1R")
                        print(f"TIME EXIT: {ticket} closed after {elapsed_min:.1f}m without reaching +1R.")
                    continue

            # Structure-based hard exit on opposing shift.
            shift_detected, shift_reason = _opposing_structure_shift(symbol, pos.type)
            if shift_detected:
                if _close_position_market(ticket):
                    update_trade_stage(ticket, "STRUCTURE_EXIT")
                    log_trade_event(ticket, "STRUCTURE_SHIFT_EXIT")
                    print(f"STRUCTURE EXIT: {ticket} closed. {shift_reason}")
                continue
            
            # 2. Hyper-Aggressive Layer 1: move to break-even + 1 pip as soon as +5 pips are reached.
            if pos.type == mt5.ORDER_TYPE_BUY:
                target_sl = entry + be_lock_distance
                if profit_points > BE_TRIGGER_POINTS and current_sl < target_sl:
                    modify_sl(ticket, target_sl, pos.tp)
                    update_trade_stage(ticket, "BREAKEVEN")
                    log_trade_event(ticket, "BE")
                    print(f"Shield {shield_mode}: BUY {ticket} moved to BE+1pip at +{profit_points:.1f} points.")
            
            elif pos.type == mt5.ORDER_TYPE_SELL:
                target_sl = entry - be_lock_distance
                if profit_points > BE_TRIGGER_POINTS and ((current_sl == 0.0) or (current_sl > target_sl)):
                    modify_sl(ticket, target_sl, pos.tp)
                    update_trade_stage(ticket, "BREAKEVEN")
                    log_trade_event(ticket, "BE")
                    print(f"Shield {shield_mode}: SELL {ticket} moved to BE+1pip at +{profit_points:.1f} points.")

            # 3. Layer 2: Partial Close (auto-tuned R level)
            stage = _get_trade_stage(ticket, tracked_trade)

            if stage == "BREAKEVEN":
                partial_r = _get_partial_r_multiple()
                take_profit_target = entry + (risk_points * partial_r) if pos.type == mt5.ORDER_TYPE_BUY else entry - (risk_points * partial_r)

                reached_target = (current_price >= take_profit_target) if pos.type == mt5.ORDER_TYPE_BUY else (current_price <= take_profit_target)

                if reached_target:
                    print(f"Target {partial_r:.2f}R reached for {ticket}. Scaling out 50%.")
                    res = partial_close_position(ticket, 0.5)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        update_trade_stage(ticket, "PARTIAL_CLOSED")
                        log_trade_event(ticket, f"PARTIAL_CLOSE_{partial_r:.2f}R")

            stage = _get_trade_stage(ticket, tracked_trade)
            if stage == "PARTIAL_CLOSED":
                # Dynamic full target on runner.
                full_target_price = entry + (risk_points * full_exit_r) if pos.type == mt5.ORDER_TYPE_BUY else entry - (risk_points * full_exit_r)
                full_target_hit = (current_price >= full_target_price) if pos.type == mt5.ORDER_TYPE_BUY else (current_price <= full_target_price)
                if full_target_hit:
                    if _close_position_market(ticket):
                        update_trade_stage(ticket, "FULL_EXIT")
                        log_trade_event(ticket, f"FULL_EXIT_{full_exit_r}R_{trend_profile}")
                        print(f"FULL EXIT: {ticket} closed at dynamic target {full_exit_r}R ({trend_profile}).")
                    continue

                # Adaptive trailing on runner based on trend profile.
                trail_distance = risk_points * runner_trail_r
                if pos.type == mt5.ORDER_TYPE_BUY:
                    proposed_sl = current_price - trail_distance
                    if proposed_sl > current_sl and proposed_sl > entry:
                        modify_sl(ticket, proposed_sl, pos.tp)
                        update_trade_stage(ticket, "TRAILING")
                        log_trade_event(ticket, f"TRAIL_UPDATE_{runner_trail_r}R_{trend_profile}")
                else:
                    proposed_sl = current_price + trail_distance
                    if (current_sl == 0.0) or (proposed_sl < current_sl and proposed_sl < entry):
                        modify_sl(ticket, proposed_sl, pos.tp)
                        update_trade_stage(ticket, "TRAILING")
                        log_trade_event(ticket, f"TRAIL_UPDATE_{runner_trail_r}R_{trend_profile}")

        time.sleep(EXIT_SCAN_INTERVAL_SECONDS)  # Hyper-aggressive scan cadence

def modify_sl(ticket, new_sl, current_tp):
    position = mt5.positions_get(ticket=ticket)
    symbol = position[0].symbol if position else "UNKNOWN"
    decision = risk_engine.pre_trade_check(
        symbol=symbol,
        action="MODIFY",
        timeframe="1m",
        source="exit_manager",
        purpose="MODIFY",
    )
    if not decision.allowed:
        print(f"[RISK BLOCK] {decision.code}: {decision.message}")
        return

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl": new_sl,
        "tp": current_tp,
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"✅ Position {ticket}: SL moved to Breakeven.")

if __name__ == "__main__":
    print("🛡️ Exit Manager Active: Guarding active trades...")
    manage_exits()