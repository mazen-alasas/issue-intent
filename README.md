# issue-intent

**NLP-based GitHub Issue Type Classification**

`issue-intent` is an NLP/ML engineering project that classifies real GitHub issues as **bug**, **feature**, **question**, or **documentation**. It compares classical NLP, a PyTorch TextCNN baseline, and a fine-tuned DistilBERT, emphasizing data quality, reproducible evaluation, resource-aware model development, and practical model serving.

The repository also contains a **GitHub-inspired React web application** with a Markdown issue editor and a prediction panel connected to the FastAPI inference service. The app currently runs locally; a public demo is **not yet deployed**.

## Results

The selected model is a fine-tuned **DistilBERT** checkpoint from epoch 3, selected using validation **Macro F1**. Official NLBSE 2023 test inference was performed on Kaggle, and the saved outputs were audited locally.

| Model | Evaluation split | Micro F1 | Macro F1 |
|---|---|---:|---:|
| TF-IDF + Logistic Regression | Validation | 0.842650 | 0.706409 |
| TextCNN | Validation | 0.852650 | 0.724730 |
| DistilBERT | Validation | 0.883665 | 0.788362 |
| **DistilBERT** | **Official test** | **0.886558** | **0.791175** |

The official test has **142,320 issues**, including **126,175 correct** and **16,145 incorrect** predictions. DistilBERT achieved **88.66% accuracy** and **0.884702 weighted F1**. `question` and `documentation` remain the weaker classes. The models above were not all evaluated on the official test split, so this table is **not** a same-test-set architecture comparison.

See the [Final Evaluation Report](docs/FINAL_EVALUATION_REPORT.md) and [Model Card](docs/MODEL_CARD.md) for per-class metrics, the evaluation protocol, provenance caveats, and limitations.

## Dataset

The [NLBSE 2023 Issue Report Classification dataset](https://github.com/nlbse2023/issue-report-classification) contains **1,418,201** labeled issues:

| Official split | Issues |
|---|---:|
| Training | 1,275,881 |
| Test | 142,320 |

After exact title–body deduplication and exclusion of conflicting-label groups, **1,221,982** issues remained from the official training source. A stratified 85%/15% split (`random_state=42`) produced **1,038,684 training** and **183,298 validation** examples. The official test set was held out from model development and selection.

Selected preprocessing **E4** transforms only the body; the original title is preserved. The transformer uses a maximum sequence length of **256 tokens**, a **48-token title budget**, and head/tail body allocation.

## Approach

1. **Data feasibility and quality:** audit real issues, remove exact-text duplicates and conflicting-label groups, and verify train/validation integrity.
2. **Classical NLP:** compare title-only and title+body TF-IDF features, truncation ablations, structured preprocessing, Logistic Regression, and LinearSVC.
3. **Neural baseline:** train and validate a PyTorch TextCNN.
4. **Transformer:** fine-tune DistilBERT with gradient accumulation and validation-based checkpoint selection.
5. **Final evaluation:** run selected-checkpoint inference on the held-out official test and audit saved predictions against the source data.
6. **Serving:** load the selected checkpoint through FastAPI, benchmark local CPU HTTP requests, and validate a CPU-only Docker image.
7. **Web application:** implement a GitHub-inspired React/TypeScript/Vite editor with live API classification and automated frontend tests.

Python scripts implement reproducible pipelines, notebooks support exploration, and automated tests cover core behavior. Raw datasets, model weights, and generated evaluation artifacts are excluded from Git.

## Application Architecture

```text
Browser (React + TypeScript + Vite)
  ├── GitHub-inspired issue editor
  ├── Write / Preview, GitHub-flavored Markdown
  └── Prediction panel
          │ POST /predict (JSON)
          ▼
FastAPI (Python, Uvicorn)
  ├── Input validation and CORS
  ├── E4 structured body preprocessing
  ├── DistilBERT tokenization (256 tokens)
  └── PyTorch DistilBERT classifier
          │
          ▼
Predicted label + confidence + class probabilities
```

The current frontend communicates with FastAPI; **inference does not yet run in the browser**. There is no GitHub authentication or repository-write integration: entered issues are classified, not posted to GitHub.

## Web Application Features

- Responsive GitHub-inspired dark interface with issue title and description fields.
- Markdown authoring toolbar, Write/Preview views, and GitHub-flavored Markdown rendering.
- Client-side validation matching the API limits: title up to **500 characters**, body up to **20,000 characters**, and at least one nonblank field.
- A prediction panel displaying the predicted class, confidence, and all four class probabilities.
- Loading and error feedback, including unavailable or unreachable API states.
- A typed API client configured through `VITE_API_BASE_URL`, with request cancellation for stale predictions.

The returned confidence is a model softmax score and **has not been calibrated** as a probability of correctness.

## Technology Stack

**ML and data:** Python 3.11, `uv`, pandas, NumPy, PyArrow, scikit-learn, PyTorch, Hugging Face Transformers, and Kaggle GPU training.

**API and packaging:** FastAPI, Uvicorn, Docker (CPU-only inference image), pytest, and Ruff.

**Frontend:** React, TypeScript, Vite, GitHub-flavored Markdown rendering, Vitest, React Testing Library, ESLint, and Prettier.

**Automation:** GitHub Actions CI for the established Python quality gate. Frontend CI coverage should be verified against the workflow before claiming it.

## Development Setup

```bash
git clone git@github.com:mazen-alasas/issue-intent.git
cd issue-intent
uv sync
```

Run backend formatting, linting, and tests:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Some evaluation scripts require the source dataset or trained model, which are intentionally not distributed in Git. Audit preserved official-test outputs without rerunning inference:

```bash
uv run python -m src.models.audit_final_evaluation \
  --artifacts-dir "$HOME/issue-intent-artifacts/kaggle-backup/extracted/distilbert_final_test" \
  --source "$HOME/issue-intent-artifacts/kaggle-backup/extracted/official_test_structured_v1.parquet"
```

These are example local paths, not files automatically present after cloning. The audit validates saved artifacts, but cannot independently establish the exact weights used in the original Kaggle inference run.

## Running the Local Web Application

A trained checkpoint is required. The example path below refers to the development machine; set `ISSUE_INTENT_CHECKPOINT` to your own checkpoint directory.

**Terminal 1 — FastAPI backend:**

```bash
cd ~/issue-intent
ISSUE_INTENT_CHECKPOINT="$HOME/issue-intent-artifacts/models/distilbert/best" \
  uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — React frontend:**

```bash
cd ~/issue-intent/frontend
npm ci
cp -n .env.example .env.local
npm run dev
```

Open the development URL shown by Vite (typically `http://localhost:5173`). The example environment configures `VITE_API_BASE_URL=http://localhost:8000` for local requests. Both services must be running for the browser to classify issues.

For frontend setup, scripts, tests, and troubleshooting, see the [Frontend README](frontend/README.md).

**Frontend quality commands:**

```bash
cd frontend
npm run format
npm run format:check
npm run lint
npm run test
npm run build
```

The last reported frontend checks passed **19 tests across three files**, linting, formatting, and production build. These are historical results and should be rerun before merge.

## API and Docker

The API provides `GET /health` and `POST /predict`. `POST /predict` accepts JSON with `title` and `body`, then returns `label`, `confidence`, and `probabilities` for all four classes. A model checkpoint is loaded during FastAPI startup.

Example request:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"title":"Application crashes on startup","body":"The application raises a RuntimeError during initialization."}'
```

The CPU-only Docker image uses a non-root user and expects a **read-only external checkpoint mount**:

```bash
docker build -t issue-intent-api:local .
docker run --rm \
  --name issue-intent-api \
  -p 127.0.0.1:8001:8000 \
  -v "$HOME/issue-intent-artifacts/models/distilbert/best:/models/distilbert:ro" \
  issue-intent-api:local
```

In another terminal:

```bash
curl http://127.0.0.1:8001/health
uv run python scripts/smoke_test_api.py
```

The last reported Docker image measured **1.64 GB**, and an earlier reported backend quality gate passed **340 tests**. The local HTTP benchmark is documented in [Project Progress](docs/PROGRESS.md); it is not a Docker load test. For complete configuration, see the [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md).

## Project Status

| Stage | Status |
|---|---|
| Repository Foundation | Completed |
| Dataset Feasibility and Quality | Completed |
| Evaluation Protocol | Completed |
| Classical NLP Baselines | Completed |
| TextCNN Neural Baseline | Completed |
| DistilBERT Fine-Tuning | Completed |
| Official Test Evaluation and Artifact Audit | Completed |
| FastAPI, Benchmark, Docker, and Python CI | Implemented and tested |
| React Web Application | Implemented and locally tested; documentation and merge pending |
| ONNX Export, Quantization, Browser Inference | Planned |
| Public Static Demo and Portfolio Release | Planned |

## Documentation

- [Project Plan](docs/PROJECT_PLAN.md)
- [Progress and Experiment History](docs/PROGRESS.md)
- [Frontend Guide](frontend/README.md)
- [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md)
- [TextCNN Baseline Report](docs/TEXTCNN_BASELINE_REPORT.md)
- [Transformer Experiment Report](docs/TRANSFORMER_EXPERIMENT.md)
- [Final Evaluation Report](docs/FINAL_EVALUATION_REPORT.md)
- [DistilBERT Model Card](docs/MODEL_CARD.md)
- [Dataset Feasibility Notebook](notebooks/01_dataset_feasibility.ipynb)

## Next Planned Work

After documenting, verifying, and merging `web-application`, assess **ONNX export**, PyTorch/ONNX prediction parity, INT8 quantization, and **browser-based inference**. If validated, a static-hosted React demo could remove the need for an always-running Python server. This is a proposed path, **not** an implemented or publicly hosted service.

## Evaluation Principles and Limitations

**Micro F1** is the primary overall reporting metric; **Macro F1**, weighted F1, per-class precision/recall/F1, and confusion matrices are also reported. DistilBERT checkpoint selection used validation **Macro F1** to prioritize minority-class performance.

Labels can be ambiguous, the dataset is imbalanced, and token truncation can remove useful context. Generalization to private repositories, other issue trackers, and non-English issues is not established. Softmax confidence has not been calibrated. Public uptime, browser inference latency, and production traffic performance have **not** been measured.

## License

A code license has not yet been selected. Dataset usage and redistribution conditions require review before redistributing dataset-derived artifacts.
