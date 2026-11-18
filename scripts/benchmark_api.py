import argparse
import asyncio
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

PAYLOADS = {
    "short": {
        "title": "Application crashes on startup",
        "body": "The application crashes with RuntimeError after updating.",
    },
    "medium": {
        "title": "Application crashes on startup",
        "body": (
            "After updating the application, startup fails with RuntimeError. "
            "The issue occurs consistently on Ubuntu when launching the app. "
        )
        * 20,
    },
    "long": {
        "title": "Application crashes on startup",
        "body": (
            "After updating the application, startup fails with RuntimeError. "
            "The issue occurs consistently on Ubuntu when launching the app. "
        )
        * 100,
    },
}


def percentile(values: list[float], percent: float) -> float:
    """Calculate an interpolated percentile."""
    if not values:
        raise ValueError("Cannot calculate percentile of empty values.")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(latencies: list[float]) -> dict:
    """Summarize HTTP request latency in milliseconds."""
    if not latencies:
        raise ValueError("At least one latency measurement is required.")

    return {
        "count": len(latencies),
        "mean_ms": statistics.mean(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "p99_ms": percentile(latencies, 99),
    }


async def timed_request(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
) -> float:
    """Send one request and return elapsed time in milliseconds."""
    start = time.perf_counter()
    response = await client.post(url, json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    result = response.json()

    if not isinstance(result, dict) or "label" not in result:
        raise ValueError("Unexpected prediction response.")

    return elapsed_ms


async def benchmark_sequential(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    warm_requests: int,
) -> dict:
    """Measure one initial request followed by sequential requests."""
    first_ms = await timed_request(client, url, payload)

    latencies = [
        await timed_request(client, url, payload) for _ in range(warm_requests)
    ]

    return {
        "first_request_ms": first_ms,
        "warm": summarize(latencies),
    }


async def benchmark_concurrent(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    concurrency: int,
    total_requests: int,
) -> dict:
    """Measure bounded concurrent HTTP requests."""
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one() -> float:
        async with semaphore:
            return await timed_request(client, url, payload)

    start = time.perf_counter()

    latencies = await asyncio.gather(*(run_one() for _ in range(total_requests)))

    duration_seconds = time.perf_counter() - start

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "duration_seconds": duration_seconds,
        "throughput_rps": total_requests / duration_seconds,
        "latency": summarize(latencies),
    }


async def run_benchmark(
    base_url: str,
    warm_requests: int,
    concurrent_requests: int,
    concurrency_levels: list[int],
) -> dict:
    """Run sequential and concurrent benchmark scenarios."""
    url = f"{base_url.rstrip('/')}/predict"

    results = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "warm_requests": warm_requests,
        "payloads": {},
        "concurrency": [],
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        for name, payload in PAYLOADS.items():
            results["payloads"][name] = {
                "body_characters": len(payload["body"]),
                **await benchmark_sequential(client, url, payload, warm_requests),
            }

        for level in concurrency_levels:
            results["concurrency"].append(
                await benchmark_concurrent(
                    client,
                    url,
                    PAYLOADS["short"],
                    concurrency=level,
                    total_requests=concurrent_requests,
                )
            )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the issue-intent FastAPI service."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--warm-requests",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/benchmarks/api_baseline.json"),
    )

    args = parser.parse_args()

    if args.warm_requests < 1:
        parser.error("--warm-requests must be >= 1")

    if args.concurrent_requests < 1:
        parser.error("--concurrent-requests must be >= 1")

    if not args.concurrency or any(level < 1 for level in args.concurrency):
        parser.error("--concurrency values must be >= 1")

    return args


def main() -> None:
    args = parse_args()

    results = asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            warm_requests=args.warm_requests,
            concurrent_requests=args.concurrent_requests,
            concurrency_levels=args.concurrency,
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
