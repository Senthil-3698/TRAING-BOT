from __future__ import annotations

import argparse
import statistics
import time

import zmq


def _make_socket(context: zmq.Context, endpoint: str, timeout_ms: int) -> zmq.Socket:
    req = context.socket(zmq.REQ)
    req.setsockopt(zmq.LINGER, 0)
    req.setsockopt(zmq.RCVTIMEO, timeout_ms)
    req.setsockopt(zmq.SNDTIMEO, timeout_ms)
    req.connect(endpoint)
    return req


def run_benchmark(*, count: int, echo_endpoint: str, timeout_ms: int, inter_signal_delay_ms: int) -> None:
    context = zmq.Context.instance()
    req = _make_socket(context, echo_endpoint, timeout_ms)

    time.sleep(0.35)

    rtt_ms: list[float] = []
    timeouts = 0

    print("[LATENCY] Starting ZeroMQ RTT benchmark")
    print(f"[LATENCY] count={count} echo_endpoint={echo_endpoint}")

    for seq in range(1, count + 1):
        t0_ns = time.perf_counter_ns()

        payload = (
            f'{{"type":"LATENCY_PING","sequence":{seq},"python_send_perf_ns":{t0_ns}}}'
            .encode("utf-8")
        )

        try:
            req.send(payload)
        except zmq.Again:
            timeouts += 1
            # Send itself timed out — recreate socket
            req.close(0)
            req = _make_socket(context, echo_endpoint, timeout_ms)
            continue

        try:
            _echo = req.recv()
            recv_ns = time.perf_counter_ns()
        except zmq.Again:
            timeouts += 1
            # Recv timed out — REQ socket is now stuck, must recreate
            req.close(0)
            req = _make_socket(context, echo_endpoint, timeout_ms)
            continue

        rtt = (recv_ns - t0_ns) / 1_000_000.0
        rtt_ms.append(rtt)

        if inter_signal_delay_ms > 0:
            time.sleep(inter_signal_delay_ms / 1000.0)

    req.close(0)

    if not rtt_ms:
        print("[LATENCY] No successful RTT samples. Is MT5 echo EA running and connected?")
        print(f"[LATENCY] timeouts={timeouts}")
        return

    avg = statistics.fmean(rtt_ms)
    min_v = min(rtt_ms)
    max_v = max(rtt_ms)
    p50 = statistics.median(rtt_ms)
    p95 = sorted(rtt_ms)[max(0, min(len(rtt_ms) - 1, int(round((len(rtt_ms) - 1) * 0.95))))]

    print("\n--- ZMQ RTT RESULTS (ms) ---")
    print(f"Samples:  {len(rtt_ms)}/{count}")
    print(f"Timeouts: {timeouts}")
    print(f"Average:  {avg:.6f}")
    print(f"Minimum:  {min_v:.6f}")
    print(f"Maximum:  {max_v:.6f}")
    print(f"P50:      {p50:.6f}")
    print(f"P95:      {p95:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ZeroMQ RTT benchmark between Python and MT5 echo EA")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--echo-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--timeout-ms", type=int, default=2500)
    parser.add_argument("--inter-signal-delay-ms", type=int, default=0)
    args = parser.parse_args()

    run_benchmark(
        count=args.count,
        echo_endpoint=args.echo_endpoint,
        timeout_ms=args.timeout_ms,
        inter_signal_delay_ms=args.inter_signal_delay_ms,
    )


if __name__ == "__main__":
    main()