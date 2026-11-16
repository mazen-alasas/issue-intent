"""Tests for the API benchmark script."""

import asyncio
import json

import httpx
import pytest

from scripts.benchmark_api import (
    PAYLOADS,
    benchmark_concurrent,
    benchmark_sequential,
    percentile,
    summarize,
    timed_request,
)


def test_percentile_interpolation():
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 0) == 10.0
    assert percentile(values, 50) == 25.0
    assert percentile(values, 100) == 40.0


def test_percentile_rejects_empty_values():
    with pytest.raises(ValueError):
        percentile([], 95)


def test_summarize_latencies():
    result = summarize([10.0, 20.0, 30.0])

    assert result["count"] == 3
    assert result["mean_ms"] == 20.0
    assert result["p50_ms"] == 20.0
    assert result["min_ms"] == 10.0
    assert result["max_ms"] == 30.0


def test_summarize_rejects_empty_values():
    with pytest.raises(ValueError):
        summarize([])


def make_client(status_code=200, response_data=None):
    if response_data is None:
        response_data = {
            "label": "bug",
            "confidence": 0.9,
            "probabilities": {
                "bug": 0.9,
                "documentation": 0.02,
                "feature": 0.03,
                "question": 0.05,
            },
        }

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == "/predict"

        payload = json.loads(request.content)
        assert "title" in payload
        assert "body" in payload

        return httpx.Response(
            status_code,
            json=response_data,
        )

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://testserver",
    )


def test_timed_request_success():
    async def scenario():
        async with make_client() as client:
            elapsed = await timed_request(
                client,
                "http://testserver/predict",
                PAYLOADS["short"],
            )

        assert elapsed >= 0

    asyncio.run(scenario())


def test_timed_request_http_error():
    async def scenario():
        async with make_client(status_code=500) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await timed_request(
                    client,
                    "http://testserver/predict",
                    PAYLOADS["short"],
                )

    asyncio.run(scenario())


def test_timed_request_invalid_response():
    async def scenario():
        async with make_client(response_data={"unexpected": True}) as client:
            with pytest.raises(ValueError, match="Unexpected"):
                await timed_request(
                    client,
                    "http://testserver/predict",
                    PAYLOADS["short"],
                )

    asyncio.run(scenario())


def test_benchmark_sequential():
    async def scenario():
        async with make_client() as client:
            result = await benchmark_sequential(
                client,
                "http://testserver/predict",
                PAYLOADS["short"],
                warm_requests=5,
            )

        assert result["first_request_ms"] >= 0
        assert result["warm"]["count"] == 5

    asyncio.run(scenario())


def test_benchmark_concurrent():
    async def scenario():
        async with make_client() as client:
            result = await benchmark_concurrent(
                client,
                "http://testserver/predict",
                PAYLOADS["short"],
                concurrency=2,
                total_requests=6,
            )

        assert result["concurrency"] == 2
        assert result["total_requests"] == 6
        assert result["latency"]["count"] == 6
        assert result["throughput_rps"] > 0

    asyncio.run(scenario())


def test_payload_sizes():
    assert len(PAYLOADS["short"]["body"]) < len(PAYLOADS["medium"]["body"])
    assert len(PAYLOADS["medium"]["body"]) < len(PAYLOADS["long"]["body"])
    assert len(PAYLOADS["long"]["body"]) <= 20_000
