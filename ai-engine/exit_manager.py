import MetaTrader5 as mt5
import time
import redis
import json
import os
import httpx
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from mfe_analyzer import auto_tune_partial_r_if_due
from mt5_executor import partial_close_position
from state_manager import update_trade_stage
from risk_engine import RiskEngine
from alerts import send_telegram_alert

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
BOT_MAGIC_ID = int(os.getenv("BOT_MAGIC_IDS", "123456").split(",")[0].strip())
BE_TRIGGER_POINTS = 50  # 5 pips = 50 points
BE_LOCK_POINTS = 10     # 1 pip = 10 points
BE_TRIGGER_ATR_MULTIPLE = 1.0
# XAUUSD shadow trail: 100-150 points trigger (10-15 pips), 80-100 points trail (8-10 pips)
SHADOW_TRIGGER_POINTS = 100  # ~10 pips for XAUUSD
SHADOW_TRAIL_POINTS = 80     # ~8 pips for XAUUSD
# TIME_BOMB: Only close if trade is at or below breakeven after this duration
TIME_BOMB_SECONDS = 180      # 3 minutes
TIME_BOMB_BE_TOLERANCE_POINTS = 20  # Allow up to +2 pips before force-close


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


def _send_event_async(ticket, event_type):
    """Fire-and-forget event logging to prevent blocking exit loop."""
    try:
        payload = {
            "ticketId": str(ticket),
            "eventType": event_type,
        }
        httpx.post("http://localhost:8080/events", json=payload, timeout=2.0)
        event_key = str(event_type).upper()
        if event_key == "ENTRY":
            send_telegram_alert(
                "TRADE_ENTRY",
                f"Trade entry event logged (ticket={ticket}).",
                level="INFO",
                extra={"ticket": ticket, "event_type": event_key},
            )
        elif event_key in {"EXIT", "EOD", "TIME_EXPIRED", "SL", "TP", "TRAIL_SL", "FULL_EXIT", "PARTIAL_CLOSE"}:
            send_telegram_alert(
                "TRADE_EXIT",
                f"Trade exit event logged: {event_key}.",
                level="INFO",
                extra={"ticket": ticket, "event_type": event_key},
            )
    except Exception as error:
        print(f"[EVENT LOG ASYNC FAILED] {ticket}: {error}")

def log_trade_event(ticket, event_type):
    """Non-blocking wrapper: dispatch event logging on background thread."""
    thread = threading.Thread(target=_send_event_async, args=(ticket, event_type), daemon=True)
    thread.start()

def manage_exits():
    if not mt5.initialize():
        return

    last_auto_tune = 0.0

    while True:
        now = time.time()
        if (now - last_auto_tune) >= AUTO_TUNE_CHECK_SECONDS:
            try:
                auto_tune_partial_r_if_due()
            except Exception as error:
                print(f"[EXIT AUTO TUNE] skipped: {error}")
            last_auto_tune = now

        positions = mt5.positions_get(magic=BOT_MAGIC_ID) or []

        for pos in positions:
            symbol = pos.symbol
            ticket = pos.ticket
            entry = float(pos.price_open)
            current_price = float(pos.price_current)
            current_sl = float(pos.sl or 0.0)
            tp = float(pos.tp or 0.0)
            tracked_trade = _get_tracked_trade(ticket)
            trade_stage = _get_trade_stage(ticket, tracked_trade)

            # Get symbol info early for use in all checks
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None or symbol_info.point <= 0:
                continue

            point = symbol_info.point
            volatility_mode, volatility_multiplier = get_market_volatility(symbol)
            trend_score, _trend_details = _trend_strength_score(symbol)
            _trend_full_exit_r, trend_trail_multiplier, trend_label = _trend_exit_profile(trend_score)

            opened_at_epoch = float(getattr(pos, "time", 0) or 0)
            open_seconds = max(0.0, time.time() - opened_at_epoch) if opened_at_epoch > 0 else 0.0
            
            # TIME_BOMB: Only close if trade is at or below breakeven after TIME_BOMB_SECONDS
            if open_seconds >= TIME_BOMB_SECONDS:
                if pos.type == mt5.ORDER_TYPE_BUY:
                    pnl_points = (current_price - entry) / point
                else:
                    pnl_points = (entry - current_price) / point
                
                # Close only if breakeven or losing (pnl <= tolerance threshold)
                if pnl_points <= TIME_BOMB_BE_TOLERANCE_POINTS:
                    if _close_position_market(ticket):
                        update_trade_stage(ticket, "TIME_EXPIRED")
                        log_trade_event(ticket, "TIME_EXPIRED")
                        send_telegram_alert(
                            "TRADE_EXIT",
                            f"Position closed: {open_seconds:.0f}s stagnant, {pnl_points:.1f} points pnl (<=breakeven).",
                            level="INFO",
                            extra={"ticket": ticket, "symbol": symbol, "reason": "TIME_EXPIRED", "seconds": open_seconds, "pnl_points": pnl_points},
                        )
                        print(f"[TIME EXPIRED] {ticket} closed after {open_seconds:.0f}s, pnl={pnl_points:.1f} points")
                    else:
                        print(f"[TIME EXPIRED] Failed to close stagnant position {ticket}.")
                        send_telegram_alert(
                            "TRADE_ERROR",
                            f"Failed to close stagnant position after {open_seconds:.0f}s.",
                            level="ERROR",
                            extra={"ticket": ticket, "symbol": symbol},
                        )
                    continue
                else:
                    # Trade is profitable, don't force-close even though it's been open > 3 min
                    print(f"[TIME BOMB SKIP] {ticket} profitable at {pnl_points:.1f} points, allowing to run")

            be_lock_distance = BE_LOCK_POINTS * point
            trigger_distance = SHADOW_TRIGGER_POINTS * point * volatility_multiplier
            trail_distance = SHADOW_TRAIL_POINTS * point * trend_trail_multiplier

            # Structural failure exit: if market structure shifts against an already-protected trade, flatten.
            if trade_stage in {"BREAKEVEN", "TRAILING", "PARTIAL"}:
                shift, reason = _opposing_structure_shift(symbol, pos.type)
                if shift:
                    if _close_position_market(ticket):
                        update_trade_stage(ticket, "STRUCTURE_EXIT")
                        log_trade_event(ticket, "FULL_EXIT")
                        print(f"[STRUCTURE EXIT] {ticket}: {reason}")
                    continue

            if pos.type == mt5.ORDER_TYPE_BUY:
                profit_distance = current_price - entry
                be_sl = entry + be_lock_distance

                if profit_distance >= trigger_distance:
                    if current_sl < be_sl:
                        modify_sl(ticket, be_sl, tp)
                        update_trade_stage(ticket, "BREAKEVEN")
                        log_trade_event(ticket, "BE")
                        print(f"[SHADOW] BUY {ticket} locked BE+1pip at +{profit_distance/point:.1f} points")
                        current_sl = be_sl

                    proposed_sl = max(be_sl, current_price - trail_distance)
                    if proposed_sl > current_sl:
                        modify_sl(ticket, proposed_sl, tp)
                        update_trade_stage(ticket, "TRAILING")
                        log_trade_event(ticket, "SHADOW_TRAIL")
                        print(f"[SHADOW] BUY {ticket} trailed to {proposed_sl:.2f} ({volatility_mode}/{trend_label})")

            elif pos.type == mt5.ORDER_TYPE_SELL:
                profit_distance = entry - current_price
                be_sl = entry - be_lock_distance

                if profit_distance >= trigger_distance:
                    if (current_sl == 0.0) or (current_sl > be_sl):
                        modify_sl(ticket, be_sl, tp)
                        update_trade_stage(ticket, "BREAKEVEN")
                        log_trade_event(ticket, "BE")
                        print(f"[SHADOW] SELL {ticket} locked BE+1pip at +{profit_distance/point:.1f} points")
                        current_sl = be_sl

                    proposed_sl = min(be_sl, current_price + trail_distance)
                    if (current_sl == 0.0) or (proposed_sl < current_sl):
                        modify_sl(ticket, proposed_sl, tp)
                        update_trade_stage(ticket, "TRAILING")
                        log_trade_event(ticket, "SHADOW_TRAIL")
                        print(f"[SHADOW] SELL {ticket} trailed to {proposed_sl:.2f} ({volatility_mode}/{trend_label})")

            # PARTIAL CLOSE: Execute if position has reached target R-multiple
            try:
                partial_r_multiple = _get_partial_r_multiple()
                if trade_stage not in {"PARTIAL", "TIME_EXPIRED", "CLOSED"}:
                    risk_points = 0.0
                    if tracked_trade and tracked_trade.get("stop_loss_pips"):
                        risk_points = float(tracked_trade["stop_loss_pips"]) * point
                    elif current_sl > 0:
                        risk_points = abs(entry - current_sl)

                    if risk_points <= 0:
                        continue

                    r_progress = _r_progress(pos.type, entry, current_price, risk_points)
                    
                    # If position has reached partial close R-multiple threshold
                    if r_progress >= partial_r_multiple:
                        result = partial_close_position(ticket, percentage=0.5)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            update_trade_stage(ticket, "PARTIAL")
                            log_trade_event(ticket, "PARTIAL_CLOSE")
                            send_telegram_alert(
                                "TRADE_EXIT",
                                f"Partial close: {ticket} @ {r_progress:.2f}R (target={partial_r_multiple:.2f}R)",
                                level="INFO",
                                extra={"ticket": ticket, "symbol": symbol, "r_progress": r_progress},
                            )
                            print(f"[PARTIAL] {ticket} @ {r_progress:.2f}R, 50% closed")
            except Exception as e:
                print(f"[PARTIAL CLOSE ERROR] {ticket}: {e}")

        time.sleep(EXIT_SCAN_INTERVAL_SECONDS)

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
        send_telegram_alert(
            "TRADE_ERROR",
            "Risk engine blocked SL/TP modification.",
            level="ERROR",
            extra={"ticket": ticket, "risk_code": decision.code},
        )
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
    else:
        send_telegram_alert(
            "TRADE_ERROR",
            "Failed to modify SL/TP for position.",
            level="ERROR",
            extra={"ticket": ticket, "retcode": getattr(result, "retcode", None)},
        )

if __name__ == "__main__":
    print("🛡️ Exit Manager Active: Guarding active trades...")
    manage_exits()