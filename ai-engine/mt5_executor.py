import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()


def _initialize_mt5():
    return mt5.initialize(
        login=int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASS"),
        server=os.getenv("MT5_SERVER"),
    )

def execute_market_order(symbol, action, lot_size, sl_pips, tp_pips):
    # Initialize connection to the MT5 terminal on your laptop
    if not _initialize_mt5():
        print("MT5 initialize() failed")
        return None

    # Determine order type
    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    price = mt5.symbol_info_tick(symbol).ask if action == "BUY" else mt5.symbol_info_tick(symbol).bid
    
    # Calculate SL/TP levels based on your 7-year trading edge logic
    point = mt5.symbol_info(symbol).point
    sl = price - (sl_pips * point) if action == "BUY" else price + (sl_pips * point)
    tp = price + (tp_pips * point) if action == "BUY" else price - (tp_pips * point)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "magic": 123456,  # Your bot's unique ID
        "comment": "Sentinel AI Execution",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    mt5.shutdown()
    return result


def partial_close_position(ticket, percentage=0.5):
    """
    Institutional Exit: Closes a portion of the position to bank profits.
    """
    if not _initialize_mt5():
        print("MT5 initialize() failed")
        return None

    position = mt5.positions_get(ticket=ticket)
    if not position:
        mt5.shutdown()
        return None

    pos = position[0]
    symbol = pos.symbol
    lot_to_close = round(pos.volume * percentage, 2)

    # Ensure we don't try to close less than the minimum lot (0.01)
    if lot_to_close < 0.01:
        mt5.shutdown()
        return None

    # To close a BUY, we send a SELL for the partial lot
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_to_close,
        "type": order_type,
        "position": ticket, # CRITICAL: This links the order to the existing trade
        "price": price,
        "magic": 123456,
        "comment": f"Partial Close {int(percentage*100)}%",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    mt5.shutdown()
    return result