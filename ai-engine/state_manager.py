import redis
import json
import os
import numpy as np

# Connection to the Docker Redis instance we started
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6380")),
    db=0,
)

def auto_update_bias(symbol):
    """
    Reads live H1 and H4 candles from MT5, computes EMA50 on each,
    and writes the resulting bias to Redis. Call this before checking
    get_integrated_bias() so the filter is always current.
    Returns the computed (h1_bias, h4_bias) tuple.
    """
    try:
        import MetaTrader5 as mt5

        _TF_MAP = {
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
        }

        results = {}
        for label, tf in _TF_MAP.items():
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, 60)
            if rates is None or len(rates) < 55:
                results[label] = "NEUTRAL"
                continue

            closes = np.array([r["close"] for r in rates], dtype=float)
            # EMA-50 using pandas-free Wilder smoothing
            k = 2.0 / (50 + 1)
            ema = closes[0]
            for c in closes[1:]:
                ema = c * k + ema * (1 - k)

            current_price = closes[-1]
            if current_price > ema * 1.0003:        # price at least 0.03% above EMA
                results[label] = "BULLISH"
            elif current_price < ema * 0.9997:      # price at least 0.03% below EMA
                results[label] = "BEARISH"
            else:
                results[label] = "NEUTRAL"

            set_market_bias(symbol, label, results[label])

        return results.get("1h", "NEUTRAL"), results.get("4h", "NEUTRAL")

    except Exception as e:
        print(f"[BIAS] auto_update_bias failed: {e}")
        return "NEUTRAL", "NEUTRAL"


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