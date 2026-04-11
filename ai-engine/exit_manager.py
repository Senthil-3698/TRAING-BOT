import MetaTrader5 as mt5
import time
import redis
import json
import os
import httpx
from mt5_executor import partial_close_position
from state_manager import update_trade_stage

r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6380")),
    db=0,
)


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

    while True:
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
            
            # Calculate Risk (R) dynamically from the original SL stored in Redis.
            risk_points = abs(entry - initial_sl)
            if risk_points <= 0:
                continue

            shield_mode, risk_trigger_multiplier = get_market_volatility(symbol)
            trigger_distance = risk_points * risk_trigger_multiplier
            be_buffer = 10 * point
            
            # 2. Layer 1: Move to Breakeven (+1R)
            if pos.type == mt5.ORDER_TYPE_BUY:
                if current_price >= (entry + trigger_distance) and current_sl < entry:
                    modify_sl(ticket, entry + be_buffer, pos.tp)
                    update_trade_stage(ticket, "BREAKEVEN")
                    log_trade_event(ticket, "BE")
                    print(f"Shield {shield_mode}: BUY {ticket} moved to breakeven.")
            
            elif pos.type == mt5.ORDER_TYPE_SELL:
                if current_price <= (entry - trigger_distance) and current_sl > entry:
                    modify_sl(ticket, entry - be_buffer, pos.tp)
                    update_trade_stage(ticket, "BREAKEVEN")
                    log_trade_event(ticket, "BE")
                    print(f"Shield {shield_mode}: SELL {ticket} moved to breakeven.")

            # 3. Layer 2: Partial Close (+1.5R)
            stage = _get_trade_stage(ticket, tracked_trade)

            if stage == "BREAKEVEN":
                take_profit_target = entry + (risk_points * 1.5) if pos.type == mt5.ORDER_TYPE_BUY else entry - (risk_points * 1.5)

                reached_target = (current_price >= take_profit_target) if pos.type == mt5.ORDER_TYPE_BUY else (current_price <= take_profit_target)

                if reached_target:
                    print(f"💰 Target 1.5R reached for {ticket}. Scaling out 50%...")
                    res = partial_close_position(ticket, 0.5)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        update_trade_stage(ticket, "PARTIAL_CLOSED")
                        log_trade_event(ticket, "PARTIAL_CLOSE")

        time.sleep(1) # Check every second

def modify_sl(ticket, new_sl, current_tp):
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