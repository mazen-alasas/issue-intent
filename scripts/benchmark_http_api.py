"""Measure end-to-end JSON HTTP prediction latency without third-party dependencies.

Examples:
    python scripts/benchmark_http_api.py --url http://127.0.0.1:8002 --output docs/results/http_benchmark_c1.json
    python scripts/benchmark_http_api.py --url http://127.0.0.1:8002 --concurrency 4 --output docs/results/http_benchmark_c4.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

LABELS = {"bug", "documentation", "feature", "question"}
PAYLOAD = {
    "title": "Application crashes on startup",
    "body": "RuntimeError after updating.",
}


def predict(base_url: str, timeout: float) -> float:
    request = Request(
        f"{base_url}/predict",
        data=json.dumps(PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urlopen(request, timeout=timeout) as response:
        data = response.read()
        status = response.status
    latency_ms = (time.perf_counter() - start) * 1000
    if status != 200:
        raise RuntimeError(f"POST /predict returned HTTP {status}")
    parsed = json.loads(data)
    if parsed.get("label") not in LABELS:
        raise RuntimeError(f"Unexpected label: {parsed.get('label')!r}")
    probs = parsed.get("probabilities", {})
    if set(probs) != LABELS or not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-4):
        raise RuntimeError("Invalid probability distribution")
    return latency_ms


def percentile(samples: list[float], fraction: float) -> float:
    """Linear interpolation between sorted observations (0 <= fraction <= 1)."""
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.requests < 1
        or args.warmup < 0
        or args.concurrency < 1
        or args.timeout <= 0
    ):
        parser.error(
            "requests/concurrency/timeout must be positive; warmup must be >= 0"
        )
    base_url = args.url.rstrip("/")
    try:
        health_request = Request(f"{base_url}/health")
        with urlopen(health_request, timeout=args.timeout) as response:
            health = json.loads(response.read())
            if response.status != 200 or not health.get("model_loaded"):
                raise RuntimeError(f"Service is not ready: {health}")
        for _ in range(args.warmup):
            predict(base_url, args.timeout)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(predict, base_url, args.timeout)
                for _ in range(args.requests)
            ]
            latencies = [future.result() for future in as_completed(futures)]
        elapsed = time.perf_counter() - start
    except (HTTPError, URLError, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")

    result = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "endpoint": "/predict",
        "payload": PAYLOAD,
        "requests": args.requests,
        "warmup": args.warmup,
        "concurrency": args.concurrency,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_requests_per_second": round(args.requests / elapsed, 3),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 3),
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "p99": round(percentile(latencies, 0.99), 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "success_count": len(latencies),
        "failure_count": 0,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
