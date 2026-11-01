# issue-intent — Project Plan

## Project Overview

issue-intent is an NLP and ML engineering project focused on classifying real GitHub issues into four categories:

- Bug
- Feature
- Question
- Documentation

The project uses the NLBSE 2023 Issue Report Classification Competition as its intended benchmark.

The goal is to build a reproducible classification pipeline, compare different NLP approaches, evaluate their trade-offs, and deliver a tested inference API.

## Project Roadmap

| Stage | Branch | Main Deliverables | Status |
|---|---|---|---|
| Repository Foundation | main | Environment, configuration, documentation | In progress |
| Dataset Feasibility | dataset-feasibility | Dataset verification, schema, integrity, feasibility report | Planned |
| Data Quality | data-quality | Missing values, duplicates, leakage, label analysis | Planned |
| Evaluation Protocol | evaluation-protocol | Validation split, metrics, majority baseline | Planned |
| Classical NLP | classical-baselines | TF-IDF, Logistic Regression, LinearSVC | Planned |
| Neural NLP | neural-baseline | PyTorch TextCNN or BiLSTM | Planned |
| Transformers | transformer-finetuning | DistilBERT and optional RoBERTa | Planned |
| Final Evaluation | final-evaluation | Official test evaluation, error analysis, model card | Planned |
| API & Docker | api-serving | FastAPI, inference tests, Docker | Planned |
| Portfolio Release | portfolio-release | CI, documentation, reproducibility, release | Planned |

Optional stages: Similar Issue Retrieval and Cloud Deployment, only if justified.

## Engineering Principles

- Use Python 3.11 and uv for environment and dependency management.
- Maintain a flat `src/` layout with explicit import conventions.
- Keep implementations simple, modular, and testable.
- Use notebooks for exploration and scripts for stable workflows.
- Introduce dependencies only when needed.
- Preserve raw data and exclude large datasets and model artifacts from Git.
- Track reproducible experiments and document actual results.
- Keep the official test set separate from model development.
- Avoid unnecessary infrastructure and overengineering.

## Evaluation Strategy

The official NLBSE 2023 test set will be reserved for final evaluation.

Model development and hyperparameter selection will use a validation split derived exclusively from training data.

Primary metric: Micro F1.

Additional metrics:

- Macro F1 and Weighted F1
- Per-class Precision, Recall, and F1
- Confusion Matrix
- Inference Latency and Model Size

## Current Stage — Repository Foundation

### Completed

- [x] Initialize local Git repository.
- [x] Configure GitHub SSH remote.
- [x] Initialize Python 3.11 environment using uv.
- [x] Install pytest and Ruff.
- [x] Configure uv non-package mode.
- [x] Establish the flat `src/` structure.
- [x] Verify source imports.
- [x] Verify initial Ruff checks.

### Remaining

- [ ] Finalize initial documentation.
- [ ] Review staged changes.
- [ ] Create initial commit.
- [ ] Push main to GitHub.

### Exit Criteria

- Environment is reproducible with `uv sync`.
- Source imports work from the repository root.
- Ruff lint and formatting checks pass.
- Documentation reflects actual project status.
- Initial commit is pushed successfully to main.

Functional tests will be added when testable application behavior exists.

## Next Stage — Dataset Feasibility

1. Verify official dataset sources and download links.
2. Download the official training and test archives.
3. Record file sizes and SHA-256 hashes.
4. Inspect schema, labels, missing values, and class distributions.
5. Evaluate memory and processing requirements.
6. Review dataset provenance and redistribution conditions.
7. Produce a feasibility report and Go/No-Go decision.

No model training will begin before dataset feasibility is established.
