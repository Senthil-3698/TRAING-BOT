from __future__ import annotations
import argparse, statistics, time
import zmq

def _make_socket(context, endpoint, timeout_ms):
    req = context.socket(zmq.REQ)
    req.setsockopt(zmq.LINGER, 0)
    req.setsockopt(zmq.RCVTIMEO, timeout_ms)
    req.setsockopt(zmq.SNDTIMEO, timeout_ms)
    req.connect(endpoint)
    return req

def run_benchmark(count, timeout_ms, echo_endpoint="tcp://127.0.0.1:5555", inter_signal_delay_ms=0):
    context = zmq.Context.instance()
    req = _make_socket(context, echo_endpoint, timeout_ms)
    time.sleep(0.35)
    rtt_ms = []
    timeouts = 0
    print("[LATENCY] Starting ZeroMQ RTT benchmark")
    for seq in range(1, count + 1):
        t0_ns = time.perf_counter_ns()
        payload = f'{{"type":"LATENCY_PING","seq":{seq}}}'.encode()
        try:
            req.send(payload)
        except zmq.Again:
            timeouts += 1
            req.close(0)
            req = _make_socket(context, echo_endpoint, timeout_ms)
            continue
        try:
            req.recv()
            rtt_ms.append((time.perf_counter_ns() - t0_ns) / 1e6)
        except zmq.Again:
            timeouts += 1
            req.close(0)
            req = _make_socket(context, echo_endpoint, timeout_ms)
    req.close(0)
    if not rtt_ms:
        print(f"[LATENCY] No samples. Timeouts={timeouts}")
        return
    s = sorted(rtt_ms)
    print(f"\n--- ZMQ RTT RESULTS (ms) ---")
    print(f"Samples:  {len(rtt_ms)}/{count}")
    print(f"Timeouts: {timeouts}")
    print(f"Average:  {statistics.fmean(rtt_ms):.3f}")
    print(f"Min:      {min(rtt_ms):.3f}")
    print(f"Max:      {max(rtt_ms):.3f}")
    print(f"P50:      {statistics.median(rtt_ms):.3f}")
    print(f"P95:      {s[int(len(s)*0.95)]:.3f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--timeout-ms", type=int, default=2500)
    p.add_argument("--echo-endpoint", default="tcp://127.0.0.1:5555")
    a = p.parse_args()
    run_benchmark(a.count, a.timeout_ms, a.echo_endpoint)
