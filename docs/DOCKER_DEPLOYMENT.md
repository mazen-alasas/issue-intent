# Docker Deployment — ONNX INT8 CPU

## Scope

This image defaults to ONNX Runtime **CPU** with the selected INT8 DistilBERT graph. It is a transitional image: CPU-only PyTorch remains installed because `src.api.onnx_inference` currently imports shared objects from `src.api.inference`, which imports PyTorch. Removing PyTorch requires a separate shared-code refactor and regression testing. No CUDA runtime is required by this image.

The ONNX graph and tokenizer checkpoint are mounted read-only, not copied into the image or committed to Git. Local `pyproject.toml` and `uv.lock` describe the development environment and are **not** used by this Docker build.

## Prerequisites

- Docker Engine or Docker Desktop with WSL integration
- `~/issue-intent-artifacts/models/distilbert/best` containing the tokenizer and model configuration
- `~/issue-intent-artifacts/models/distilbert/onnx/model_int8.onnx` containing the quantized graph

## Build

From the repository root:

```bash
docker build -t issue-intent-api:onnx-int8 .
```

## Run

```bash
docker run --rm \
  --name issue-intent-api \
  -p 127.0.0.1:8001:8000 \
  -v "$HOME/issue-intent-artifacts/models/distilbert/best:/models/distilbert:ro" \
  -v "$HOME/issue-intent-artifacts/models/distilbert/onnx:/models/onnx:ro" \
  issue-intent-api:onnx-int8
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

The user reported 367 passing tests before this Docker-only change. Docker build and container smoke tests remain to be verified on the target machine.

## Constraints and follow-up

- The image retains CPU PyTorch for shared import compatibility; it is not yet an optimized ONNX-only image.
- Inference is single-request oriented and uses a single Uvicorn worker by default.
- No model weights are embedded in the image.
- `docker/requirements.txt` has direct pins for serving packages, but the complete dependency graph is not locked.
- Container throughput and end-to-end HTTP latency have not yet been measured.
- Future cleanup: extract `Prediction`, `LABEL_NAMES`, and `resolve_checkpoint_path` to a lightweight module, avoid importing PyTorch on the ONNX code path, verify tokenizer compatibility without torch, and then remove CPU PyTorch from the ONNX-only image.

## Troubleshooting

If startup fails, inspect `docker logs issue-intent-api` and verify both volume paths exist and are readable. A missing graph or tokenizer causes startup failure. If port 8001 is occupied, change the host side of the `-p` mapping. With `--rm`, the container disappears after it stops.
