# Docker Deployment — ONNX INT8 CPU

## Scope

This image defaults to ONNX Runtime **CPU** with the selected INT8 DistilBERT graph. The ONNX code path uses `tokenizers` to load the saved `tokenizer.json` directly, plus backend-neutral types in `src.api.shared`. The PyTorch backend is imported lazily only when selected. This image intentionally does not install PyTorch or CUDA.

The ONNX graph and tokenizer checkpoint are mounted read-only, not copied into the image or committed to Git. Local `pyproject.toml` and `uv.lock` describe the development environment and are **not** used by this Docker build.

## Prerequisites

- Docker Engine or Docker Desktop with WSL integration
- `~/issue-intent-artifacts/models/distilbert/best` containing `tokenizer.json` and the model configuration
- `~/issue-intent-artifacts/models/distilbert/onnx/model_int8.onnx` containing the quantized graph

## Build

From the repository root:

```bash
docker build -t issue-intent-api:onnx-only .
```

## Run

```bash
docker run --rm \
  --name issue-intent-api \
  -p 127.0.0.1:8001:8000 \
  -v "$HOME/issue-intent-artifacts/models/distilbert/best:/models/distilbert:ro" \
  -v "$HOME/issue-intent-artifacts/models/distilbert/onnx:/models/onnx:ro" \
  issue-intent-api:onnx-only
```

The image sets `ISSUE_INTENT_BACKEND=onnx-int8`, `ISSUE_INTENT_CHECKPOINT=/models/distilbert`, and `ISSUE_INTENT_ONNX_MODEL=/models/onnx/model_int8.onnx`. ONNX Runtime uses `CPUExecutionProvider`. The service runs as UID 10001 and exposes container port 8000. Port 8001 is bound to the host loopback interface only.

If the host uses restrictive file permissions, ensure the non-root container user can read both mounted directories and files. Avoid making private checkpoints world-writable.

## Verify

In another terminal:

```bash
curl -i http://127.0.0.1:8001/health

curl -sS -X POST http://127.0.0.1:8001/predict \
  -H 'Content-Type: application/json' \
  -d '{"title":"Application crashes on startup","body":"RuntimeError after updating."}'

uv run python scripts/smoke_test_api.py
```

Expected: `/health` returns HTTP 200 with `status=healthy`, and `/predict` returns a label, confidence, and a normalized probability distribution across bug, documentation, feature, and question. The local example previously returned `bug`; the Docker result must be checked rather than assumed.

Check the configured backend and provider:

```bash
docker exec issue-intent-api sh -c 'printf "Backend=%s\nGraph=%s\n" "$ISSUE_INTENT_BACKEND" "$ISSUE_INTENT_ONNX_MODEL"'
docker exec issue-intent-api python -c 'import onnxruntime as ort; print(ort.get_available_providers())'
docker logs --tail 40 issue-intent-api
```

`CPUExecutionProvider` must be available; the Docker environment is not configured for CUDA. The backend is selected at application startup. The `/health` endpoint does not currently expose the backend name.

## Quality checks

Run from the repository root before committing:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
git diff --check
```

The previous transitional Docker image passed its HTTP smoke tests and measured 1.72 GB. The optimized ONNX-only image must be rebuilt, measured, and smoke-tested separately; no size or parity claim is made before that verification.

## Constraints and follow-up

- The ONNX-only image does not include PyTorch or Transformers. The local development environment still supports the PyTorch backend.
- Inference is single-request oriented and uses a single Uvicorn worker by default.
- No model weights are embedded in the image.
- `docker/requirements.txt` has direct pins for serving packages, but the complete dependency graph is not locked.
- Container throughput and end-to-end HTTP latency have not yet been measured.
- Shared inference contracts are defined in `src/api/shared.py`. Confirm the image does not install or import PyTorch and verify prediction parity against the transitional image.

## Troubleshooting

If startup fails, inspect `docker logs issue-intent-api` and verify both volume paths exist and are readable. A missing graph or tokenizer causes startup failure. If port 8001 is occupied, change the host side of the `-p` mapping. With `--rm`, the container disappears after it stops.
