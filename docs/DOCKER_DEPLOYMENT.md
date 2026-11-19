# Docker Deployment — ONNX INT8 CPU

**Verified locally by November 19, 2026.** This is an **independent server deployment**; the static browser demo does not use Docker or FastAPI.

## Architecture

The ONNX-only image runs FastAPI + ONNX Runtime CPU with a directly loaded `tokenizer.json` (the `tokenizers` library) and backend-neutral shared prediction contracts. The PyTorch backend remains available in the development environment but **PyTorch is absent from the ONNX-only image**. No GPU/CUDA runtime is needed. Model weights are mounted read-only rather than embedded in the image.

The image uses `docker/requirements.txt` rather than the development `uv.lock`; its transitive dependency graph is not fully locked. The service runs as non-root UID 10001, one Uvicorn worker, on container port 8000.

## Prerequisites

- Docker Engine or Docker Desktop with WSL integration.
- Local tokenizer/checkpoint directory: `~/issue-intent-artifacts/models/distilbert/best`.
- ONNX graph: `~/issue-intent-artifacts/models/distilbert/onnx/model_int8.onnx`.
- Read permissions for the non-root container user on mounted paths.

## Build and run

From the repository root:

```bash
docker build -t issue-intent-api:onnx-only .
docker run --rm \
  --name issue-intent-onnx-only \
  -p 127.0.0.1:8002:8000 \
  -v "$HOME/issue-intent-artifacts/models/distilbert/best:/models/distilbert:ro" \
  -v "$HOME/issue-intent-artifacts/models/distilbert/onnx:/models/onnx:ro" \
  issue-intent-api:onnx-only
```

The image configures `ISSUE_INTENT_BACKEND=onnx-int8`, `ISSUE_INTENT_CHECKPOINT=/models/distilbert`, and `ISSUE_INTENT_ONNX_MODEL=/models/onnx/model_int8.onnx`. The published host binding `127.0.0.1:8002` is local-only.

## Smoke checks

In another terminal:

```bash
curl -i http://127.0.0.1:8002/health
curl -sS -X POST http://127.0.0.1:8002/predict \
  -H 'Content-Type: application/json' \
  -d '{"title":"Application crashes on startup","body":"RuntimeError after updating."}'
docker exec issue-intent-onnx-only python -c \
  'import importlib.util; assert importlib.util.find_spec("torch") is None; print("PASS: torch absent")'
docker exec issue-intent-onnx-only python -c \
  'import onnxruntime as ort; print(ort.get_available_providers())'
```

**Observed:** container startup succeeded, `/health` and `/predict` returned HTTP 200, and the reference input returned `bug` with confidence `0.9955799012318574`. `torch` was absent and `CPUExecutionProvider` was available. The full Python test suite passed **372 tests**.

## Measured image size and HTTP performance

| Measurement | Observed value |
| --- | ---: |
| Original PyTorch CPU image (earlier) | 1.64 GB |
| Transitional ONNX INT8 image retaining PyTorch | 1.72 GB |
| **Optimized ONNX-only image** | **556 MB** |
| HTTP throughput, concurrency 1 | 56.986 requests/s |
| HTTP throughput, concurrency 4 | 136.316 requests/s |
| HTTP p95 latency, concurrency 1 / 4 | 21.388 ms / 35.309 ms |

HTTP results used **100 measured requests and 10 warmups per run**, with one short repeated issue and one Uvicorn worker. They are not production capacity claims. See [HTTP Benchmark Report](HTTP_BENCHMARK_REPORT.md).

## Quality gate

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
git diff --check
```

## Limits and troubleshooting

- No weights are baked into the image; mount both directories and verify read permissions.
- If startup fails, inspect `docker logs issue-intent-onnx-only` while it is running. `--rm` removes the container on exit.
- If port 8002 is occupied, change only the host side of `-p`.
- The measured size is the local `docker image ls` figure, not an OCI transfer-size estimate.
- No multi-worker, sustained-load, cold-start or public-service SLA measurements were performed.
- For the browser-only portfolio path, use [Static Deployment](STATIC_DEPLOYMENT.md), not this container.
