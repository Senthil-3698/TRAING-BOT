import MetaTrader5 as mt5
import os
from pathlib import Path
import sys
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent / "ai-engine"))
from risk_engine import RiskEngine

load_dotenv(Path(__file__).resolve().parent / "ai-engine" / ".env")
risk_engine = RiskEngine()

def test_connection():
    # 1. Initialize MT5 with your .env credentials
    if not mt5.initialize(
        login=int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASS"),
        server=os.getenv("MT5_SERVER")
    ):
        print(f"❌ MT5 Initialize Failed: {mt5.last_error()}")
        return

    print("✅ MT5 Connected Successfully!")

    # 2. Try to place a tiny 0.01 lot Buy on XAUUSD
    candidate_symbols = ["XAUUSD", "XAUUSDm", "XAUUSD.pro", "GOLD"]
    symbol = None

    for candidate in candidate_symbols:
        mt5.symbol_select(candidate, True)
        tick = mt5.symbol_info_tick(candidate)
        if tick is not None:
            symbol = candidate
            break

    if symbol is None:
        print("❌ No tradable XAUUSD symbol found. Check Market Watch -> Show All and broker symbol suffix.")
        mt5.shutdown()
        return

    lot = 0.01
    price = mt5.symbol_info_tick(symbol).ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "magic": 999999,
        "comment": "PIPE_TEST",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    decision = risk_engine.pre_trade_check(
        symbol=symbol,
        action="BUY",
        timeframe="1m",
        source="debug_mt5_pipe",
        purpose="OPEN",
    )
    if not decision.allowed:
        print(f"❌ Risk blocked test order: {decision.code} {decision.message}")
        mt5.shutdown()
        return

    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Trade Failed! Error Code: {result.retcode}")
    else:
        print(f"🚀 SUCCESS! Trade placed. Ticket: {result.order}")

    mt5.shutdown()

if __name__ == "__main__":
    test_connection()