import argparse
import time
import statistics
import json
import sys

try:
    import zmq
except ImportError:
    print("pyzmq is not installed. Run: pip install pyzmq")
    sys.exit(1)


def run_benchmark(count: int, timeout_ms: int):
    context = zmq.Context()

    # Setup PUB socket (Sending to MT5)
    pub = context.socket(zmq.PUB)
    pub.bind("tcp://127.0.0.1:5556")

    # Setup REP socket (Receiving Echo from MT5)
    rep = context.socket(zmq.REP)
    rep.RCVTIMEO = timeout_ms
    rep.bind("tcp://127.0.0.1:5557")

    print("[ZMQ] Benchmark Engine Online.")
    print("[ZMQ] PUB bound to 5556 | REP bound to 5557")
    print("Waiting 2 seconds for MT5 EA to establish connection...")
    time.sleep(2.0)

    latencies = []
    timeouts = 0

    print(f"Firing {count} signals...")

    for _ in range(count):
        # We use time.perf_counter() for extreme microsecond precision
        start_time = time.perf_counter()

        # Send multipart message exactly as MT5 expects
        pub.send_multipart([b"trade.signal", b'{"test": "ping"}'])

        try:
            # Wait for the MT5 REQ socket to bounce the echo back
            _echo = rep.recv()
            end_time = time.perf_counter()

            # Send an acknowledgment back to MT5 to complete the REQ/REP cycle
            rep.send(b'{"ok": true}')

            # Calculate RTT in milliseconds
            rtt_ms = (end_time - start_time) * 1000
            latencies.append(rtt_ms)

        except zmq.Again:
            timeouts += 1

        # 1ms delay between ticks to simulate high-frequency bursts without overflowing buffers
        time.sleep(0.001)

    # Output Institutional Metrics
    print("\n" + "=" * 30)
    print(" ZERO-MQ LATENCY BENCHMARK ")
    print("=" * 30)
    print(f"Samples Received : {len(latencies)} / {count}")
    print(f"Timeouts         : {timeouts}")

    if latencies:
        print(f"Average Latency  : {statistics.mean(latencies):.3f} ms")
        print(f"Minimum Latency  : {min(latencies):.3f} ms")
        print(f"Maximum Latency  : {max(latencies):.3f} ms")
        print("=" * 30)

        if statistics.mean(latencies) < 5.0:
            print("[STATUS] INSTITUTIONAL GRADE. Cleared for MT5 deployment.")
        else:
            print("[STATUS] WARNING: Latency exceeds 5ms. Optimization required.")
    else:
        print("\n[!] CRITICAL FAILURE: 0 samples received.")
        print("Checklist:")
        print("1. Is ZeroMqLatencyEchoEA.mq5 running on an MT5 chart?")
        print("2. Did you check 'Allow DLL imports' in MT5 settings?")

    rep.close(0)
    pub.close(0)
    context.term()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZMQ Latency RTT Benchmark")
    parser.add_argument("--count", type=int, default=1000, help="Number of signals to send")
    parser.add_argument("--timeout-ms", type=int, default=2500, help="Receive timeout in milliseconds")
    args = parser.parse_args()

    run_benchmark(args.count, args.timeout_ms)
