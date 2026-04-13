from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

# INSTITUTIONAL PATCH: Use orjson for asynchronous, zero-jitter serialization
try:
    import orjson  # type: ignore[import-not-found]
except ImportError:
    raise RuntimeError("orjson is not installed. Run: pip install orjson")

try:
    import zmq
    import zmq.asyncio
except Exception:  # pragma: no cover
    zmq = None  # type: ignore

load_dotenv()


@dataclass
class ZmqBridgeConfig:
    signal_pub_endpoint: str = os.getenv("ZMQ_SIGNAL_PUB_ENDPOINT", "tcp://127.0.0.1:5556")
    heartbeat_rep_endpoint: str = os.getenv("ZMQ_HEARTBEAT_REP_ENDPOINT", "tcp://127.0.0.1:5557")
    heartbeat_timeout_seconds: float = float(os.getenv("ZMQ_HEARTBEAT_TIMEOUT_SECONDS", "5"))
    heartbeat_sweep_seconds: float = float(os.getenv("ZMQ_HEARTBEAT_SWEEP_SECONDS", "1"))
    send_hwm: int = int(os.getenv("ZMQ_SEND_HWM", "20000"))


class ZmqSignalBridge:
    """Asynchronous signal PUB + heartbeat REP server for MT5 client EAs."""

    def __init__(self, config: ZmqBridgeConfig | None = None) -> None:
        self.config = config or ZmqBridgeConfig()
        self._clients_last_seen: dict[str, float] = {}
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._sweep_task: asyncio.Task | None = None

        if zmq is None:
            self._ctx = None
            self._pub = None
            self._rep = None
            return

        self._ctx = zmq.asyncio.Context.instance()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.SNDHWM, self.config.send_hwm)
        self._pub.setsockopt(zmq.LINGER, 0)

        self._rep = self._ctx.socket(zmq.REP)
        self._rep.setsockopt(zmq.LINGER, 0)
        # INSTITUTIONAL PATCH: Prevent REQ/REP deadlocks.
        self._rep.setsockopt(zmq.RCVTIMEO, 2000)
        self._rep.setsockopt(zmq.SNDTIMEO, 2000)

    @property
    def available(self) -> bool:
        return zmq is not None and self._ctx is not None

    async def start(self) -> None:
        if not self.available:
            print("[ZMQ] pyzmq unavailable. Bridge disabled.")
            return
        if self._running:
            return

        try:
            self._pub.bind(self.config.signal_pub_endpoint)
            self._rep.bind(self.config.heartbeat_rep_endpoint)
        except Exception as error:
            print(f"[ZMQ] bind failed: {error}")
            raise

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_server_loop())
        self._sweep_task = asyncio.create_task(self._heartbeat_sweep_loop())
        print(
            f"[ZMQ] bridge started PUB={self.config.signal_pub_endpoint} "
            f"REP={self.config.heartbeat_rep_endpoint}"
        )

    async def stop(self) -> None:
        self._running = False
        for task in (self._heartbeat_task, self._sweep_task):
            if task and not task.done():
                task.cancel()
        self._heartbeat_task = None
        self._sweep_task = None

        if self._pub is not None:
            self._pub.close(0)
        if self._rep is not None:
            self._rep.close(0)
        print("[ZMQ] bridge stopped")

    async def publish_signal(self, signal: dict[str, Any]) -> str:
        if not self.available:
            return "DISABLED"
        if not self._running:
            await self.start()

        required = ("symbol", "action", "timeframe")
        missing = [k for k in required if not signal.get(k)]
        if missing:
            raise ValueError(f"ZMQ signal missing required keys: {missing}")

        message_id = str(uuid.uuid4())
        envelope = {
            "message_id": message_id,
            "published_at": datetime.now(timezone.utc).isoformat(),
            "source": "ai-engine",
            "type": "TRADE_SIGNAL",
            "payload": signal,
        }

        try:
            # INSTITUTIONAL PATCH: orjson dumps straight to bytes natively.
            payload_bytes = orjson.dumps(envelope)
            await self._pub.send_multipart(
                [b"trade.signal", payload_bytes]
            )
            return message_id
        except Exception as error:
            print(f"[ZMQ] publish failed: {error}")
            raise

    async def _heartbeat_server_loop(self) -> None:
        while self._running:
            try:
                # INSTITUTIONAL PATCH: Non-blocking receives prevent complete loop freezes.
                raw = await self._rep.recv(flags=zmq.NOBLOCK)
                now_ts = asyncio.get_running_loop().time()
                request = orjson.loads(raw) if raw else {}
                client_id = str(request.get("client_id") or "unknown")
                self._clients_last_seen[client_id] = now_ts

                response = {
                    "ok": True,
                    "server_ts": datetime.now(timezone.utc).isoformat(),
                    "heartbeat_timeout_seconds": self.config.heartbeat_timeout_seconds,
                }
                # INSTITUTIONAL PATCH: Non-blocking sends.
                await self._rep.send(orjson.dumps(response), flags=zmq.NOBLOCK)
            except zmq.Again:
                # Normal behavior: no heartbeat received this tick.
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as error:
                print(f"[ZMQ] Heartbeat loop error: {error}")
                await asyncio.sleep(0.1)

    async def _heartbeat_sweep_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_sweep_seconds)
                now_ts = asyncio.get_running_loop().time()
                stale = [
                    client_id
                    for client_id, seen_ts in self._clients_last_seen.items()
                    if (now_ts - seen_ts) > self.config.heartbeat_timeout_seconds
                ]
                for client_id in stale:
                    print(
                        f"[ZMQ][HEARTBEAT_ALERT] client={client_id} "
                        f"silent>{self.config.heartbeat_timeout_seconds:.1f}s"
                    )
                    # Keep entry so repeated alerts are possible until client recovers.
                    self._clients_last_seen[client_id] = now_ts - self.config.heartbeat_timeout_seconds
            except asyncio.CancelledError:
                break


_bridge_singleton: ZmqSignalBridge | None = None


def get_bridge() -> ZmqSignalBridge:
    global _bridge_singleton
    if _bridge_singleton is None:
        _bridge_singleton = ZmqSignalBridge()
    return _bridge_singleton


async def publish_signal_async(signal: dict[str, Any]) -> str:
    bridge = get_bridge()
    return await bridge.publish_signal(signal)


async def _run_forever() -> None:
    bridge = get_bridge()
    await bridge.start()
    try:
        while True:
            await asyncio.sleep(1.0)
    except asyncio.CancelledError:
        pass
    finally:
        await bridge.stop()


def main() -> None:
    if zmq is None:
        raise RuntimeError("pyzmq is not installed. Run: pip install pyzmq")
    asyncio.run(_run_forever())


if __name__ == "__main__":
    main()
