import redis
import json
import os

# Connection to the Docker Redis instance we started
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6380")),
    db=0,
)

def set_market_bias(symbol, timeframe, bias):
    """
    Saves the trend for a specific timeframe.
    Example: set_market_bias('XAUUSD', '4h', 'BULLISH')
    """
    key = f"{symbol}:{timeframe}:bias"
    r.set(key, bias)
    print(f"STATE UPDATED: {symbol} {timeframe} is now {bias}")

def get_integrated_bias(symbol):
    """
    The Scalper's Filter: Checks if 1h and 4h are aligned.
    """
    h1 = r.get(f"{symbol}:1h:bias")
    h4 = r.get(f"{symbol}:4h:bias")
    
    # Decodes bytes to string if they exist
    h1 = h1.decode('utf-8') if h1 else "NEUTRAL"
    h4 = h4.decode('utf-8') if h4 else "NEUTRAL"
    
    if h1 == h4 and h1 != "NEUTRAL":
        return h1 # Strong Trend
    return "NO_CONFLUENCE"


def track_active_trade(
    ticket,
    entry_price,
    sl,
    tp,
    *,
    symbol=None,
    action=None,
    timeframe=None,
    setup_type=None,
    opened_at=None,
):
    """
    Stores a newly opened trade in Redis so the Exit Manager can manage stages.
    """
    key = f"trade:{ticket}"
    trade_state = {
        "ticket": str(ticket),
        "entry_price": entry_price,
        "sl": sl,
        "tp": tp,
        "stage": "ENTRY",
        "symbol": symbol,
        "action": action,
        "timeframe": timeframe,
        "setup_type": setup_type,
        "opened_at": opened_at,
    }
    r.set(key, json.dumps(trade_state))
    r.set(f"trade_stage:{ticket}", "ENTRY")
    print(f"TRADE TRACKED: {ticket} set to ENTRY")


def update_trade_stage(ticket, new_stage):
    """
    Updates trade lifecycle stage, e.g. ENTRY -> BREAKEVEN -> TRAILING.
    """
    key = f"trade:{ticket}"
    trade_raw = r.get(key)

    if not trade_raw:
        print(f"TRADE NOT FOUND: {ticket}")
        return False

    trade_state = json.loads(trade_raw.decode("utf-8"))
    trade_state["stage"] = new_stage
    r.set(key, json.dumps(trade_state))
    r.set(f"trade_stage:{ticket}", new_stage)
    print(f"TRADE STAGE UPDATED: {ticket} -> {new_stage}")
    return True