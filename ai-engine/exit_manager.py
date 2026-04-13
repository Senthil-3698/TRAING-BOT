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

# Java execution-engine is mapped to host port 8081 (docker-compose 8081:8080)
EXECUTION_ENGINE_URL = os.getenv("EXECUTION_ENGINE_URL", "http://localhost:8081")

# Trailing stop: after partial close, trail SL by this many R multiples behind price
TRAIL_R_MULTIPLE = 1.0
# Full exit target from entry (R multiple)
FULL_EXIT_R = 2.5


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
    payload = {"ticketId": str(ticket), "eventType": event_type}
    try:
        httpx.post(f"{EXECUTION_ENGINE_URL}/events", json=payload, timeout=5.0)
    except httpx.HTTPError as error:
        print(f"[EVENT LOG FAILED] {ticket}: {error}")


def modify_sl(ticket, new_sl, current_tp):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl": new_sl,
        "tp": current_tp,
    }
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return True
    return False


def _trail_stop(pos, risk_points, point):
    """
    After partial close, trail the SL so it stays TRAIL_R_MULTIPLE*R behind price.
    Only moves SL in the profitable direction (never widens it).
    """
    ticket = pos.ticket
    current_price = pos.price_current
    current_sl = pos.sl
    entry = pos.price_open

    if pos.type == mt5.ORDER_TYPE_BUY:
        desired_sl = current_price - (risk_points * TRAIL_R_MULTIPLE)
        # Only move SL up, never down; must be above entry (we're already at breakeven)
        desired_sl = max(desired_sl, entry)
        if desired_sl > current_sl + point:
            if modify_sl(ticket, desired_sl, pos.tp):
                print(f"[TRAIL] BUY {ticket}: SL trailed to {desired_sl:.5f}")
    else:
        desired_sl = current_price + (risk_points * TRAIL_R_MULTIPLE)
        desired_sl = min(desired_sl, entry)
        if desired_sl < current_sl - point:
            if modify_sl(ticket, desired_sl, pos.tp):
                print(f"[TRAIL] SELL {ticket}: SL trailed to {desired_sl:.5f}")


def manage_exits():
    if not mt5.initialize():
        return

    while True:
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

            initial_sl = None
            if tracked_trade and tracked_trade.get("sl") is not None:
                try:
                    initial_sl = float(tracked_trade["sl"])
                except (ValueError, TypeError):
                    initial_sl = None

            if initial_sl is None:
                initial_sl = current_sl

            risk_points = abs(entry - initial_sl)
            if risk_points <= 0:
                continue

            shield_mode, risk_trigger_multiplier = get_market_volatility(symbol)
            trigger_distance = risk_points * risk_trigger_multiplier
            be_buffer = 10 * point

            # ── Layer 1: Move to Breakeven (+1R) ──────────────────────────
            if trade_stage == "ENTRY":
                if pos.type == mt5.ORDER_TYPE_BUY:
                    if current_price >= (entry + trigger_distance) and current_sl < entry:
                        if modify_sl(ticket, entry + be_buffer, pos.tp):
                            update_trade_stage(ticket, "BREAKEVEN")
                            log_trade_event(ticket, "BE")
                            print(f"[BE] {shield_mode}: BUY {ticket} moved to breakeven.")
                elif pos.type == mt5.ORDER_TYPE_SELL:
                    if current_price <= (entry - trigger_distance) and current_sl > entry:
                        if modify_sl(ticket, entry - be_buffer, pos.tp):
                            update_trade_stage(ticket, "BREAKEVEN")
                            log_trade_event(ticket, "BE")
                            print(f"[BE] {shield_mode}: SELL {ticket} moved to breakeven.")

            # ── Layer 2: Partial Close at +1.5R ───────────────────────────
            elif trade_stage == "BREAKEVEN":
                # Refresh stage after potential BE update above
                trade_stage = _get_trade_stage(ticket, tracked_trade)
                if trade_stage == "BREAKEVEN":
                    if pos.type == mt5.ORDER_TYPE_BUY:
                        target = entry + (risk_points * 1.5)
                        reached = current_price >= target
                    else:
                        target = entry - (risk_points * 1.5)
                        reached = current_price <= target

                    if reached:
                        print(f"[SCALE] +1.5R hit for {ticket}. Closing 50%...")
                        res = partial_close_position(ticket, 0.5)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            update_trade_stage(ticket, "PARTIAL_CLOSED")
                            log_trade_event(ticket, "PARTIAL_CLOSE")

            # ── Layer 3: Trail remaining 50% after partial close ──────────
            elif trade_stage == "PARTIAL_CLOSED":
                # Full exit at FULL_EXIT_R
                if pos.type == mt5.ORDER_TYPE_BUY:
                    full_exit_target = entry + (risk_points * FULL_EXIT_R)
                    if current_price >= full_exit_target:
                        print(f"[EXIT] +{FULL_EXIT_R}R hit for {ticket}. Full close.")
                        res = partial_close_position(ticket, 1.0)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            update_trade_stage(ticket, "CLOSED")
                            log_trade_event(ticket, "FULL_CLOSE")
                            continue
                else:
                    full_exit_target = entry - (risk_points * FULL_EXIT_R)
                    if current_price <= full_exit_target:
                        print(f"[EXIT] +{FULL_EXIT_R}R hit for {ticket}. Full close.")
                        res = partial_close_position(ticket, 1.0)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            update_trade_stage(ticket, "CLOSED")
                            log_trade_event(ticket, "FULL_CLOSE")
                            continue

                # Trail stop on remaining position
                _trail_stop(pos, risk_points, point)

        time.sleep(1)


if __name__ == "__main__":
    print("[EXIT MANAGER] Active: Guarding active trades...")
    manage_exits()
