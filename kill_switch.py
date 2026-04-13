import MetaTrader5 as mt5
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent / "ai-engine"))
from risk_engine import RiskEngine

load_dotenv()
risk_engine = RiskEngine()

def emergency_shutdown():
    print("🚨 EMERGENCY SHUTDOWN INITIATED...")
    
    if not mt5.initialize():
        print("MT5 not running.")
        return

    # 1. Close all open positions managed by this bot
    positions = mt5.positions_get(magic=123456)
    if positions:
        for pos in positions:
            tick = mt5.symbol_info_tick(pos.symbol)
            type_dict = {
                mt5.ORDER_TYPE_BUY: mt5.ORDER_TYPE_SELL,
                mt5.ORDER_TYPE_SELL: mt5.ORDER_TYPE_BUY
            }
            price_dict = {
                mt5.ORDER_TYPE_BUY: tick.bid,
                mt5.ORDER_TYPE_SELL: tick.ask
            }
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": type_dict[pos.type],
                "price": price_dict[pos.type],
                "magic": 123456,
                "comment": "EMERGENCY CLOSE",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            decision = risk_engine.pre_trade_check(
                symbol=pos.symbol,
                action="CLOSE",
                timeframe="1m",
                source="kill_switch",
                purpose="CLOSE",
            )
            if not decision.allowed:
                print(f"Risk blocked close for {pos.ticket}: {decision.code} {decision.message}")
                continue
            result = mt5.order_send(request)
            print(f"Closed Ticket {pos.ticket}: {result.comment}")
    else:
        print("No active trades found.")

    # 2. Shut down connection
    mt5.shutdown()
    print("🛑 All connections severed. Bot Safely Disarmed.")

if __name__ == "__main__":
    emergency_shutdown()