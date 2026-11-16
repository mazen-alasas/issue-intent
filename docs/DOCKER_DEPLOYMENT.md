# Docker Deployment — Issue Intent API

This guide describes how to build, run, and verify the CPU-only Docker deployment of the Issue Intent FastAPI service.

## Prerequisites

- Docker Engine or Docker Desktop with WSL integration.
- A locally available, evaluated DistilBERT checkpoint.
- The checkpoint directory must contain the model configuration, tokenizer files, and model weights needed by Hugging Face Transformers.

The checkpoint is **not** bundled with the Docker image or committed to Git.

## Build the image

From the repository root:

```bash
docker build -t issue-intent-api:local .
```

The image uses Python 3.11 and CPU-only PyTorch, with serving dependencies defined in `docker/requirements.txt`.

To inspect its size:

```bash
docker image ls issue-intent-api
```

The locally tested image was **1.64 GB** (down from approximately **1.91 GB** before dependency cleanup). Image sizes may differ across builds and platforms.

## Run the container

From the repository root:

```bash
docker run --rm \
  --name issue-intent-api \
  -p 127.0.0.1:8001:8000 \
  -v "$HOME/issue-intent-artifacts/models/distilbert/best:/models/distilbert:ro" \
  issue-intent-api:local
```

The evaluated checkpoint is mounted **read-only** at `/models/distilbert`. The Dockerfile configures `ISSUE_INTENT_CHECKPOINT=/models/distilbert` and runs the service as the non-root user `appuser`.

The host binding `127.0.0.1:8001` restricts access to the local machine. The container listens on port `8000` internally.

Keep this terminal open while testing. Press `Ctrl+C` to stop the service; `--rm` removes the stopped container.

## Health check

From a second terminal:

```bash
curl -i http://127.0.0.1:8001/health
```

Expected HTTP status: `200 OK`.

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Prediction request

```bash
curl -sS -X POST http://127.0.0.1:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Application crashes on startup",
    "body": "The application crashes with RuntimeError after updating."
  }'
```

The response includes a predicted label, confidence, and probabilities for `bug`, `documentation`, `feature`, and `question`.

For the evaluated local checkpoint and reference request, the observed prediction was `bug` with confidence approximately `0.9957`. This is a smoke-test example, not a general accuracy guarantee.

## Automated smoke test

With the container running:

```bash
uv run python scripts/smoke_test_api.py
```

The script checks the health endpoint, a reference prediction, the expected label set, and probability normalization. Its default base URL is `http://127.0.0.1:8001`.

## Quality checks

Run these from the repository root:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The latest reported local run passed **340 tests**.

## Inspect the running container

```bash
docker ps --filter name=issue-intent-api
docker logs --tail 30 issue-intent-api
docker inspect issue-intent-api --format 'User={{.Config.User}} Image={{.Config.Image}}'
```

To verify that training and evaluation packages are absent:

```bash
docker exec -i issue-intent-api python - <<'PY'
import importlib.util

for package in ("pandas", "pyarrow", "sklearn"):
    installed = importlib.util.find_spec(package) is not None
    print(f"{package}: {'installed' if installed else 'not installed'}")
PY
```

The tested container reported all three as **not installed**.

## Architecture and dependency boundaries

The inference service uses lightweight shared modules for:

- Label definitions (`src/data/labels.py`).
- Device resolution and checkpoint label validation (`src/models/transformer_utils.py`).
- Structured preprocessing configurations (`src/preprocessing/experiment_configs.py`).
- Tokenization and input budgeting (`src/data/transformer_dataset.py`).

These shared modules prevent API imports from unnecessarily loading training and evaluation packages such as pandas, PyArrow, and scikit-learn.

The serving configuration preserves the trained checkpoint's label order and E4 body preprocessing behavior.

## Scope and limitations

- CPU-only deployment; GPU serving is not configured.
- One Uvicorn worker by default.
- Checkpoint files must be supplied externally.
- Intended for local deployment and reproducible inference experiments, not a production-hardened public service.
- Dependencies are version-constrained in the Docker build, but this setup does not use a fully locked deployment dependency graph.

## Troubleshooting

**Docker socket permission denied:** Verify that the current WSL shell has `docker` group membership (`id -nG`). Reopen the session after adding the user to the group; Docker group membership grants powerful host privileges.

**Container exits at startup:** Inspect the startup traceback and verify that the checkpoint mount contains the expected files, including `config.json`.

**Port already in use:** Keep the host port `8001`, or change only the host side of `-p 127.0.0.1:8001:8000` and pass the matching URL to the smoke test.

**No such container:** The `--rm` flag removes the container after it stops. Start a new one to inspect its live logs.
