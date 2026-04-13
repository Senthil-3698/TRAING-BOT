#!/usr/bin/env python3
"""Test execution bridge to diagnose broker rejection reasons."""

from mt5_executor import execute_trade
import MetaTrader5 as mt5

print("[TEST] Sending forced BUY order to diagnose broker response...\n")

# Make a direct test trade
result = execute_trade("BUY", "XAUUSD", "1m")

if result:
    print(f"[RESULT] Order Response:")
    print(f"  Retcode: {result.retcode}")
    print(f"  Deal: {result.deal}")
    print(f"  Order: {result.order}")
    print(f"  Volume: {result.volume}")
    print(f"  Price: {result.price}")
    print(f"  Bid: {result.bid}")
    print(f"  Ask: {result.ask}")
    
    # Translate error codes
    error_map = {
        10009: "SUCCESS - Order executed",
        10016: "FAIL - Stop loss too close to market price",
        10004: "FAIL - Request expired (re-quote)",
        10015: "FAIL - Unable to process request",
        10017: "FAIL - Pending request",
    }
    
    error_msg = error_map.get(result.retcode, f"Unknown error code {result.retcode}")
    print(f"\n[DIAGNOSIS] {error_msg}")
    
    if result.retcode == 10009:
        print("✓ EXECUTION BRIDGE WORKING! Trade was accepted by broker.")
    elif result.retcode == 10016:
        print("✗ KILLER #1 HIT: Stop Loss distance too tight for broker limits.")
    elif result.retcode == 10004:
        print("✗ KILLER #2 HIT: Re-quote block - price changed during transmission.")
    else:
        print(f"✗ Unknown rejection. Check MT5 Journal for details.")
else:
    print("[ERROR] execute_trade() returned None")
