import asyncio
import time

import orjson  # noqa: F401

from zmq_bridge import get_bridge, publish_signal_async


async def run_latency_test() -> None:
    bridge = get_bridge()
    await bridge.start()

    # Give the ZMQ sockets a moment to bind.
    await asyncio.sleep(1.0)

    print("[TEST] Firing 1,000 high-speed signals to MT5...")

    for i in range(1, 1001):
        current_time_ms = time.time() * 1000.0
        signal = {
            "symbol": "XAUUSD",
            "action": "BUY",
            "timeframe": "M1",
            "test_id": i,
            "py_timestamp_ms": current_time_ms,
        }
        await publish_signal_async(signal)

        # 10ms delay between signals to simulate high-frequency bursts.
        await asyncio.sleep(0.01)

    print("[TEST] 1,000 signals dispatched. Check MT5 Experts Journal for latency metrics.")
    await bridge.stop()


if __name__ == "__main__":
    asyncio.run(run_latency_test())
