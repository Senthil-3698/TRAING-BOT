import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "ai-engine"))
from orchestrator import on_signal_received

async def run_test():
    print("🚀 Initializing Live Pipe Test...")
    
    # We are simulating a high-probability XAUUSD Buy signal
    test_signal = {
        "symbol": "XAUUSD",
        "timeframe": "5m",
        "action": "BUY",
        "price": 2350.50 # Placeholder price
    }
    
    # This will trigger: 
    # 1. AI Analysis (Strategist)
    # 2. Risk Check (Sentinel)
    # 3. MT5 Execution (Executor)
    await on_signal_received(test_signal)

if __name__ == "__main__":
    asyncio.run(run_test())