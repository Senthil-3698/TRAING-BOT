from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid

import zmq


def run_benchmark(
    *,
    count: int,
    topic: str,
    pub_endpoint: str,
    echo_rep_endpoint: str,
    timeout_ms: int,
    inter_signal_delay_ms: int,
) -> None:
    context = zmq.Context.instance()

    pub = context.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(pub_endpoint)

    rep = context.socket(zmq.REP)
    rep.setsockopt(zmq.LINGER, 0)
    rep.bind(echo_rep_endpoint)

    # Give the MT5 SUB socket a moment to establish subscription.
    time.sleep(0.35)

    poller = zmq.Poller()
    poller.register(rep, zmq.POLLIN)

    send_times: dict[str, int] = {}
    rtt_ms: list[float] = []
    timeouts = 0

    print("[LATENCY] Starting ZeroMQ RTT benchmark")
    print(f"[LATENCY] count={count} topic={topic} pub={pub_endpoint} echo_rep={echo_rep_endpoint}")

    try:
        for seq in range(1, count + 1):
            message_id = str(uuid.uuid4())
            t0_ns = time.perf_counter_ns()
            send_times[message_id] = t0_ns

            envelope = {
                "type": "LATENCY_PING",
                "sequence": seq,
                "message_id": message_id,
                "python_send_perf_ns": t0_ns,
            }

            pub.send_multipart([topic.encode("utf-8"), json.dumps(envelope).encode("utf-8")])

            ready = dict(poller.poll(timeout_ms))
            if rep not in ready:
                timeouts += 1
                continue

            raw = rep.recv()
            recv_ns = time.perf_counter_ns()

            try:
                echo = json.loads(raw.decode("utf-8"))
                echoed_id = str(echo.get("message_id", ""))
                if not echoed_id or echoed_id not in send_times:
                    rep.send_json({"ok": False, "error": "unknown_message_id"})
                    continue

                t_send_ns = send_times.pop(echoed_id)
                rtt = (recv_ns - t_send_ns) / 1_000_000.0
                rtt_ms.append(rtt)
                rep.send_json({"ok": True, "message_id": echoed_id})
            except Exception as error:
                rep.send_json({"ok": False, "error": str(error)})

            if inter_signal_delay_ms > 0:
                time.sleep(inter_signal_delay_ms / 1000.0)

    finally:
        rep.close(0)
        pub.close(0)

    if not rtt_ms:
        print("[LATENCY] No successful RTT samples. Is MT5 echo EA running and connected?")
        print(f"[LATENCY] timeouts={timeouts}")
        return

    avg = statistics.fmean(rtt_ms)
    min_v = min(rtt_ms)
    max_v = max(rtt_ms)
    p50 = statistics.median(rtt_ms)
    p95_idx = max(0, min(len(rtt_ms) - 1, int(round((len(rtt_ms) - 1) * 0.95))))
    p95 = sorted(rtt_ms)[p95_idx]

    print("\n--- ZMQ RTT RESULTS (ms) ---")
    print(f"Samples: {len(rtt_ms)}/{count}")
    print(f"Timeouts: {timeouts}")
    print(f"Average: {avg:.6f}")
    print(f"Minimum: {min_v:.6f}")
    print(f"Maximum: {max_v:.6f}")
    print(f"P50:     {p50:.6f}")
    print(f"P95:     {p95:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ZeroMQ RTT benchmark between Python and MT5 echo EA")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--topic", default="trade.signal")
    parser.add_argument("--pub-endpoint", default="tcp://127.0.0.1:5556")
    parser.add_argument("--echo-rep-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--timeout-ms", type=int, default=2500)
    parser.add_argument("--inter-signal-delay-ms", type=int, default=0)
    args = parser.parse_args()

    run_benchmark(
        count=args.count,
        topic=args.topic,
        pub_endpoint=args.pub_endpoint,
        echo_rep_endpoint=args.echo_rep_endpoint,
        timeout_ms=args.timeout_ms,
        inter_signal_delay_ms=args.inter_signal_delay_ms,
    )


if __name__ == "__main__":
    main()
