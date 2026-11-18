# ONNX-only Docker — HTTP inference benchmark

## Scope

Run a short, local end-to-end benchmark against the already built ONNX-only CPU Docker API. The HTTP request includes client JSON serialization, loopback networking, FastAPI request handling, preprocessing, tokenization, ONNX execution, and response transfer/validation. These numbers **cannot be directly compared** with model-only ONNX throughput.

## Prerequisite

Start the existing `issue-intent-api:onnx-only` container on `127.0.0.1:8002` with the same read-only model mounts used during the successful Docker smoke test. Do **not** rebuild or re-evaluate the entire test set.

## Commands

```bash
python scripts/benchmark_http_api.py --url http://127.0.0.1:8002 --requests 100 --warmup 10 --concurrency 1 --output docs/results/http_onnx_only_c1.json
python scripts/benchmark_http_api.py --url http://127.0.0.1:8002 --requests 100 --warmup 10 --concurrency 4 --output docs/results/http_onnx_only_c4.json
```

Results are automatically printed and saved to JSON. Do not conclude that concurrency improves performance unless the measurements support it. The repeated payload makes this a **fixed-input serving microbenchmark**, not a representative real-workload benchmark. Host hardware load, CPU thread settings and container setup affect results.

## Interpretations

- `latency_ms`: client-observed wall-clock times for one HTTP prediction (P50/P95/P99 are interpolated empirical percentiles).
- `throughput_requests_per_second`: number of completed requests divided by the total measured wall time; includes ThreadPoolExecutor scheduling, especially in small tests.
- `concurrency=1`: sequential requests; `concurrency=4`: at most four simultaneous requests.
- All measured requests must return HTTP 200, one of the four expected labels, and normalized probabilities. Any failed request terminates the benchmark, rather than silently excluding failures.
- This benchmark measures only a single repeated short issue and does **not** measure cold startup, model loading, different sequence lengths, memory consumption, or model accuracy.

Do not commit benchmark claims until you have captured actual numbers on the target machine. Consider retaining only the JSON results and guide in documentation; keep transient test artifacts out of version control.
