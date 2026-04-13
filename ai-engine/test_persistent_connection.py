"""
Test script to verify MT5 stays connected and trades execute properly.
This keeps a persistent connection instead of init/shutdown each time.
"""
import MetaTrader5 as mt5
from dotenv import load_dotenv
import os
from pathlib import Path
import time

load_dotenv(Path('.env'))

print("[TEST] Starting persistent MT5 connection test...")

if not mt5.initialize(
    login=int(os.getenv("MT5_LOGIN")),
    password=os.getenv("MT5_PASS"),
    server=os.getenv("MT5_SERVER"),
):
    print("[ERROR] MT5 initialization failed")
    exit(1)

print("[SUCCESS] MT5 connected")

# Check account
account = mt5.account_info()
print(f"[ACCOUNT] Login: {account.login}, Balance: {account.balance}, Equity: {account.equity}")

# Send test order
print("\n[TRADE] Sending BUY order...")
request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": "XAUUSD",
    "volume": 0.01,
    "type": mt5.ORDER_TYPE_BUY,
    "price": mt5.symbol_info_tick("XAUUSD").ask,
    "sl": mt5.symbol_info_tick("XAUUSD").ask - (50 * mt5.symbol_info("XAUUSD").point),
    "tp": mt5.symbol_info_tick("XAUUSD").ask + (100 * mt5.symbol_info("XAUUSD").point),
    "magic": 123456,
    "comment": "Persistent Connection Test",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_IOC,
}

result = mt5.order_send(request)
print(f"[RESULT] retcode={result.retcode}, comment={result.comment}")
if result.retcode == 10009:
    print(f"[SUCCESS] Order deal={result.deal}")
else:
    print(f"[FAILED] Order rejected")

# Wait a moment
time.sleep(2)

# Check positions WITHOUT disconnecting
print("\n[CHECK] Checking active positions...")
positions = mt5.positions_get(symbol="XAUUSD")
print(f"Active XAUUSD positions: {len(positions) if positions else 0}")
if positions:
    for pos in positions:
        print(f"  Ticket: {pos.ticket}, Type: {'BUY' if pos.type == 0 else 'SELL'}, Volume: {pos.volume}")

# Check history
print("\n[HISTORY] Checking trade history...")
from datetime import datetime, timedelta
deals = mt5.history_deals_get(datetime.now() - timedelta(hours=1), datetime.now())
print(f"Deals in last 1 hour: {len(deals) if deals else 0}")

# Shutdown
mt5.shutdown()
print("\n[DONE] Test complete")
