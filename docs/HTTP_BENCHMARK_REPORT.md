# ONNX-only Docker — HTTP Benchmark Report

## Summary

The ONNX INT8 FastAPI deployment was tested end-to-end over HTTP inside the ONNX-only CPU Docker image. Both measured runs completed without failed requests. Increasing client concurrency from 1 to 4 raised measured throughput from 56.986 to 136.316 requests/second (approximately 2.39x), while increasing response latency.

These results are from a short local microbenchmark, not production capacity measurements.

## Environment and scope

- Deployment: `issue-intent-api:onnx-only` Docker image, CPU ONNX Runtime, single Uvicorn worker.
- Host endpoint: `http://127.0.0.1:8002/predict`.
- Request: `{"title":"Application crashes on startup","body":"RuntimeError after updating."}`.
- Warmup: 10 requests per run.
- Measured requests: 100 per run.
- Concurrency settings: 1 and 4.
- Image size as reported by `docker image ls`: 556 MB, compared with 1.72 GB for the earlier transitional ONNX INT8 image retaining CPU PyTorch (approximately 67.7% smaller).
- Recorded at: 2026-11-18 20:47 UTC.

## Results

| Metric | Concurrency 1 | Concurrency 4 |
|---|---:|---:|
| Requests | 100 | 100 |
| Warmup requests | 10 | 10 |
| Elapsed (s) | 1.7548 | 0.7336 |
| Throughput (requests/s) | 56.986 | 136.316 |
| Mean latency (ms) | 17.314 | 28.735 |
| P50 latency (ms) | 16.952 | 28.109 |
| P95 latency (ms) | 21.388 | 35.309 |
| P99 latency (ms) | 24.560 | 42.953 |
| Min latency (ms) | 12.662 | 20.724 |
| Max latency (ms) | 34.295 | 50.792 |
| Successful requests | 100 | 100 |
| Failed requests | 0 | 0 |

Raw result files: `docs/results/http_onnx_only_c1.json` and `docs/results/http_onnx_only_c4.json`.

## Interpretation

- Four concurrent HTTP clients achieved approximately **2.39x** the observed throughput of one client; this is not a 4x scaling result.
- At concurrency 4, P95 latency was **35.309 ms**, versus **21.388 ms** at concurrency 1; throughput improved at the expense of per-request latency.
- Both runs reported zero failures, but each included only 100 measured requests.
- HTTP timings cover the request/response path, including API handling, preprocessing, tokenization, ONNX inference, and local networking. They are not directly comparable with model-only ONNX benchmark figures.

## Validations completed before this benchmark

- The ONNX-only container started and served `/health` and `/predict` with HTTP 200.
- The reference input returned `bug`, confidence `0.9955799012318574`, matching the previously tested container's response.
- PyTorch was verified absent from the ONNX-only Docker environment.
- After adding the HTTP benchmark script and guide, local checks passed: **372 pytest tests**, plus Ruff lint and format checks.

## Limitations and reproducibility

This is a single local run per concurrency setting, using one short repeated issue, one machine, and a single Uvicorn worker. Latency percentiles may vary with scheduler load, CPU throttling, background processes, Docker and WSL configuration, request lengths, and repetition. No confidence intervals, cold-start measurements, sustained-load tests, or multi-worker comparisons were performed. Do not infer production service-level objectives from this report.

For reproduction, use `scripts/benchmark_http_api.py` with `--requests 100 --warmup 10`, setting `--concurrency` to `1` or `4`, and save JSON to `docs/results/`.
