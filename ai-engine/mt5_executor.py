import MetaTrader5 as mt5
import os
import numpy as np
from datetime import datetime, timezone
from dotenv import load_dotenv
from risk_engine import RiskEngine
from execution_quality import ExecutionQualityMonitor

load_dotenv()

# ===== RISK MANAGEMENT PARAMETERS =====
RISK_PER_TRADE = 0.02  # Risk 2% of account per trade
MIN_LOT_SIZE = 0.01
MAX_LOT_SIZE = 100.0
ATR_PERIOD = 14
SL_MULTIPLIER = 1.5  # Stop Loss = 1.5 * ATR

# Global MT5 connection state
_mt5_initialized = False
risk_engine = RiskEngine()
quality_monitor = ExecutionQualityMonitor()

# Get bot magic number from environment, fallback to 123456
BOT_MAGIC_ID = int(os.getenv("BOT_MAGIC_IDS", "123456").split(",")[0].strip())

MAX_SLIPPAGE_POINTS = 5  # 0.5 pips on XAUUSD point convention
DEFAULT_FALLBACK_SL_PIPS = 50.0
DEFAULT_TP_RR = 2.0


def _initialize_mt5():
    """Initialize MT5 if not already initialized. Keeps persistent connection."""
    global _mt5_initialized
    if _mt5_initialized:
        return True
    
    if mt5.initialize(
        login=int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASS"),
        server=os.getenv("MT5_SERVER"),
    ):
        _mt5_initialized = True
        return True
    return False


def calculate_atr(symbol, timeframe, period=ATR_PERIOD):
    """
    Calculate Average True Range (ATR) to determine dynamic Stop Loss distance.
    ATR measures market volatility - higher ATR = wider SL needed = smaller lots.
    """
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 1)
    if rates is None or len(rates) < period + 1:
        print(f"[ATR] Failed to fetch {period + 1} candles for {symbol}")
        return None
    
    # Calculate True Range for each candle (skip index 0, no previous close for it)
    high = rates['high'][1:]
    low = rates['low'][1:]
    prev_close = rates['close'][:-1]

    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)

    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.mean(tr[-period:])  # Average of last `period` TR values
    
    return atr


def _atr_stop_loss_pips(symbol, period=ATR_PERIOD, sl_multiplier=SL_MULTIPLIER):
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info or not symbol_info.point:
        return DEFAULT_FALLBACK_SL_PIPS

    atr = calculate_atr(symbol, mt5.TIMEFRAME_M5, period=period)
    if atr is None:
        return DEFAULT_FALLBACK_SL_PIPS

    point_value = 1 if symbol == "XAUUSD" else 10
    atr_pips = (atr / symbol_info.point) / point_value
    sl_pips = atr_pips * sl_multiplier
    return max(1.0, sl_pips)


def calculate_dynamic_lot_size(symbol, risk_percent=RISK_PER_TRADE, sl_multiplier=SL_MULTIPLIER):
    """
    Institutional Risk Engine: Calculate exact lot size based on:
    1. Account equity (2% per trade)
    2. ATR-based Stop Loss distance (1.5 * ATR)
    3. Broker constraints (min 0.01, max 100.0)
    
    Formula: Lot Size = (Account Equity * Risk%) / (SL Distance in Currency)
    """
    if not _initialize_mt5():
        print("[RISK] MT5 connection failed")
        return MIN_LOT_SIZE
    
    try:
        # Get account info
        account = mt5.account_info()
        if not account:
            print("[RISK] Failed to get account info")
            return MIN_LOT_SIZE
        
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            print(f"[RISK] Failed to get {symbol} info, using fixed lot 0.01")
            return MIN_LOT_SIZE
        
        point = symbol_info.point
        
        # Calculate ATR-based SL distance (in pips)
        atr = calculate_atr(symbol, mt5.TIMEFRAME_M5, period=ATR_PERIOD)
        if atr is None:
            print("[RISK] ATR calculation failed, using fixed lot 0.01")
            return MIN_LOT_SIZE
        
        # Convert ATR (in price units) to pips
        point_value = 1 if symbol == "XAUUSD" else 10  # Gold scales differently
        atr_pips = (atr / point) / point_value
        
        # Stop Loss = 1.5 * ATR (in pips)
        sl_pips = atr_pips * sl_multiplier
        
        lot_size = risk_engine.calculate_position_size(
            symbol=symbol,
            stop_loss_pips=sl_pips,
        )

        print(f"[RISK ENGINE] {symbol} | Equity: ${account.equity:.2f} | ATR: {atr:.5f} | SL: {sl_pips:.1f} pips | Lot: {lot_size}")
        
        return lot_size
    
    except Exception as e:
        print(f"[RISK ERROR] {e}")
        return MIN_LOT_SIZE


def execute_market_order(symbol, action, lot_size=None, sl_pips=None, tp_pips=None):
    # Initialize connection (or reuse persistent one)
    if not _initialize_mt5():
        print("[ERROR] MT5 initialize() failed")
        return None

    # Use dynamic lot sizing if not specified
    if lot_size is None:
        lot_size = calculate_dynamic_lot_size(symbol)
    
    # Use volatility-aware SL by default so risk breathes with the market.
    if sl_pips is None:
        sl_pips = _atr_stop_loss_pips(symbol)
    
    if tp_pips is None:
        tp_pips = sl_pips * DEFAULT_TP_RR

    decision = risk_engine.pre_trade_check(
        symbol=symbol,
        action=action,
        timeframe="1m",
        source="mt5_executor",
        purpose="OPEN",
    )
    if not decision.allowed:
        print(f"[RISK BLOCK] {decision.code}: {decision.message}")
        return None

    # Determine order type
    order_type = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if action == "BUY" else tick.bid

    symbol_info = mt5.symbol_info(symbol)
    filling_mode = mt5.ORDER_FILLING_IOC
    if symbol_info is not None:
        allowed_mode = getattr(symbol_info, "filling_mode", None)
        if allowed_mode in (mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK):
            filling_mode = allowed_mode
        elif allowed_mode == mt5.ORDER_FILLING_RETURN:
            # RETURN is less strict for scalping; prefer IOC if available in environment.
            filling_mode = mt5.ORDER_FILLING_IOC
    
    # Calculate SL/TP levels
    point = symbol_info.point if symbol_info else mt5.symbol_info(symbol).point
    sl = price - (sl_pips * point) if action == "BUY" else price + (sl_pips * point)
    tp = price + (tp_pips * point) if action == "BUY" else price - (tp_pips * point)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": MAX_SLIPPAGE_POINTS,
        "sl": sl,
        "tp": tp,
        "magic": BOT_MAGIC_ID,
        "comment": "Sentinel AI Execution",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }

    spread_points = ((tick.ask - tick.bid) / symbol_info.point) if symbol_info and symbol_info.point else 0.0
    order_send_ts = datetime.now(timezone.utc)
    result = mt5.order_send(request)
    order_send_done_ts = datetime.now(timezone.utc)

    if result is not None and result.retcode in {
        mt5.TRADE_RETCODE_REQUOTE,
        mt5.TRADE_RETCODE_PRICE_CHANGED,
        mt5.TRADE_RETCODE_PRICE_OFF,
    }:
        print("[BROKER REJECT] Slippage exceeded limit. Order Cancelled.")
        return result

    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        quality_monitor.record_fill(
            source="mt5_executor",
            symbol=symbol,
            action=action,
            timeframe="1m",
            order_ticket=int(result.order) if getattr(result, "order", 0) else None,
            deal_ticket=int(result.deal) if getattr(result, "deal", 0) else None,
            signal_ts=None,
            signal_bar_time=None,
            signal_bar_relation=None,
            order_send_ts=order_send_ts,
            order_send_done_ts=order_send_done_ts,
            intended_price=float(price),
            spread_points=float(spread_points),
        )
    # NOTE: Do NOT shutdown() here - keeps persistent connection for scanner
    return result


def execute_trade(action, symbol, timeframe, lot_size=None, sl_pips=None, tp_pips=None):
    """
    Compatibility wrapper for autonomous scanners that expect execute_trade().
    Uses dynamic positioning by default (calculates lot based on ATR and account equity).
    """
    if sl_pips is None:
        sl_pips = _atr_stop_loss_pips(symbol)
    if tp_pips is None:
        tp_pips = sl_pips * DEFAULT_TP_RR

    return execute_market_order(symbol, action, lot_size, sl_pips, tp_pips)


def partial_close_position(ticket, percentage=0.5):
    """
    Institutional Exit: Closes a portion of the position to bank profits.
    """
    global _mt5_initialized
    
    if not _initialize_mt5():
        print("MT5 initialize() failed")
        return None

    position = mt5.positions_get(ticket=ticket)
    if not position:
        mt5.shutdown()
        _mt5_initialized = False
        return None

    pos = position[0]
    symbol = pos.symbol
    lot_to_close = round(pos.volume * percentage, 2)

    # Ensure we don't try to close less than the minimum lot (0.01)
    if lot_to_close < 0.01:
        mt5.shutdown()
        _mt5_initialized = False
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
        "magic": BOT_MAGIC_ID,
        "comment": f"Partial Close {int(percentage*100)}%",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    decision = risk_engine.pre_trade_check(
        symbol=symbol,
        action="CLOSE",
        timeframe="1m",
        source="mt5_executor",
        purpose="CLOSE",
    )
    if not decision.allowed:
        print(f"[RISK BLOCK] {decision.code}: {decision.message}")
        return None

    result = mt5.order_send(request)
    mt5.shutdown()
    _mt5_initialized = False
    return result