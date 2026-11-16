# issue-intent

**NLP-based GitHub Issue Type Classification**

`issue-intent` is a machine learning engineering project for classifying real GitHub issues into four intent categories: **bug**, **feature**, **question**, and **documentation**. It compares classical NLP, a PyTorch neural baseline, and a fine-tuned transformer, with an emphasis on data quality, reproducible evaluation, and practical serving.

## Results

The selected model is a fine-tuned **DistilBERT**, chosen at epoch 3 using validation **Macro F1**. The official NLBSE 2023 test split was evaluated once on Kaggle; the saved prediction artifacts were subsequently audited locally against the processed official test dataset.

| Model | Evaluation split | Micro F1 | Macro F1 |
|---|---|---:|---:|
| TF-IDF + Logistic Regression | Validation | 0.842650 | 0.706409 |
| TextCNN | Validation | 0.852650 | 0.724730 |
| DistilBERT | Validation | 0.883665 | 0.788362 |
| **DistilBERT** | **Official test** | **0.886558** | **0.791175** |

**Official test:** 142,320 issues; 126,175 correct predictions; 16,145 incorrect predictions. The final model's test accuracy is **88.66%**, and its weighted F1 is **0.884702**. Question and documentation classification remain the weakest areas.

These are different evaluation splits: the official-test scores must not be interpreted as a direct test-set comparison of all three architectures. See the [Final Evaluation Report](docs/FINAL_EVALUATION_REPORT.md) and [Model Card](docs/MODEL_CARD.md) for the full protocol, class-level results, limitations, and audit scope.

## Dataset

The project uses the [NLBSE 2023 Issue Report Classification dataset](https://github.com/nlbse2023/issue-report-classification), containing **1,418,201** labeled GitHub issues:

| Official split | Issues |
|---|---:|
| Training | 1,275,881 |
| Test | 142,320 |

The official training set was audited for missing text, exact duplicates, conflicting labels, and class imbalance. After exact title–body deduplication and exclusion of conflicting-label groups, **1,221,982** training-source issues remained. These were split into **1,038,684 training** and **183,298 validation** examples using a stratified 85%/15% split (`random_state=42`). The official test set was held out from model development and selection.

The selected structured preprocessing configuration, **E4**, processes the **body only** and preserves the original title. The final transformer uses a maximum sequence length of 256 tokens, a title budget of 48 tokens, and head/tail body allocation.

## Approach

1. **Data feasibility and quality:** inspect real-world issues, remove exact-text duplicates and conflicting-label groups, and check train/validation integrity.
2. **Classical NLP:** evaluate title-only versus title-and-body TF-IDF representations, truncation ablations, structured preprocessing, Logistic Regression, and LinearSVC.
3. **Neural baseline:** train and evaluate a PyTorch TextCNN.
4. **Transformer:** fine-tune DistilBERT with gradient accumulation and validation-based checkpoint selection.
5. **Final evaluation:** evaluate the selected checkpoint on the official held-out test set, preserve predictions and reports, and audit saved outputs for consistency and source alignment.
6. **Serving:** expose the selected checkpoint through a tested FastAPI service, measure local CPU HTTP performance, and package inference in a CPU-only Docker image.

The repository uses Python scripts for reproducible pipelines, notebooks for exploration, and automated tests for core logic. Large datasets, model weights, and generated evaluation artifacts are kept outside Git.

## Technology Stack

Python 3.11, `uv`, pandas, NumPy, PyArrow, scikit-learn, PyTorch, Hugging Face Transformers, FastAPI, Uvicorn, Docker, pytest, and Ruff. Kaggle GPU compute was used for transformer training and official-test inference.

**Implemented:** FastAPI inference, CPU-only Docker deployment, API benchmarking, and automated smoke tests. **Not yet implemented:** GitHub Actions CI and portfolio-release improvements. MLflow remains optional rather than a completed integration.

## Development Setup

```bash
git clone git@github.com:mazen-alasas/issue-intent.git
cd issue-intent
uv sync
```

Run formatting, linting, and tests:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The official-test preparation and evaluation scripts require the source dataset and the saved trained checkpoint, which are not included in this repository. The existing final artifacts can be audited without rerunning inference:

```bash
uv run python -m src.models.audit_final_evaluation \
  --artifacts-dir "$HOME/issue-intent-artifacts/kaggle-backup/extracted/distilbert_final_test" \
  --source "$HOME/issue-intent-artifacts/kaggle-backup/extracted/official_test_structured_v1.parquet"
```

The example paths above refer to the project's local artifact layout and must be adjusted on another machine. The audit checks saved outputs; it does not independently establish the exact weights used for the original Kaggle inference run.

## Local API and Docker Deployment

The tested FastAPI service provides `GET /health` and `POST /predict`. The Docker image uses CPU-only PyTorch, runs as a non-root user, and loads the selected DistilBERT checkpoint from a read-only host mount rather than embedding model weights.

```bash
docker build -t issue-intent-api:local .
docker run --rm \
  --name issue-intent-api \
  -p 127.0.0.1:8001:8000 \
  -v "$HOME/issue-intent-artifacts/models/distilbert/best:/models/distilbert:ro" \
  issue-intent-api:local
```

With the container running, in another terminal:

```bash
curl http://127.0.0.1:8001/health
uv run python scripts/smoke_test_api.py
```

The example checkpoint path is specific to the local machine. See the [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md) for full instructions, requests, and limitations. The latest reported local checks passed **340 tests**, and the rebuilt image was **1.64 GB** (previously approximately **1.91 GB**). The local CPU HTTP benchmark is documented in [Project Progress](docs/PROGRESS.md); it is not a Docker-specific load benchmark.

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
| Final Evaluation Documentation | Completed and merged |
| FastAPI Inference and Local HTTP Benchmark | Implemented and tested; benchmark committed |
| Docker Containerization and Smoke Tests | Implemented and validated; commit pending |
| CI (GitHub Actions) | Planned |
| Portfolio Release | Planned |

## Documentation

- [Project Plan](docs/PROJECT_PLAN.md)
- [Progress and Experiment History](docs/PROGRESS.md)
- [Docker Deployment Guide](docs/DOCKER_DEPLOYMENT.md)
- [TextCNN Baseline Report](docs/TEXTCNN_BASELINE_REPORT.md)
- [Transformer Experiment Report](docs/TRANSFORMER_EXPERIMENT.md)
- [Final Evaluation Report](docs/FINAL_EVALUATION_REPORT.md)
- [DistilBERT Model Card](docs/MODEL_CARD.md)
- [Dataset Feasibility Notebook](notebooks/01_dataset_feasibility.ipynb)

## Evaluation Principles and Limitations

**Micro F1** is the project's primary overall metric; **Macro F1**, weighted F1, per-class precision/recall/F1, and confusion matrices are also reported. DistilBERT checkpoint selection used validation **Macro F1** to account for minority-class performance.

The dataset is imbalanced, labels can be semantically ambiguous, and truncation may discard useful context. Generalization to private repositories, other issue trackers, and non-English data has not been established. Confidence calibration and production inference latency remain to be evaluated.

## License

A code license has not yet been selected. Dataset usage and redistribution conditions require review before distributing dataset-derived artifacts.
