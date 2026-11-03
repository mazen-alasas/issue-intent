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
| Repository Foundation | main | Environment, configuration, documentation | Completed |
| Dataset Feasibility | dataset-feasibility | Dataset verification, schema, integrity, feasibility report | Completed |
| Data Quality | data-quality | Missing values, duplicates, leakage risk analysis, label analysis | Completed |
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
- Document functions with English Google-style docstrings, including parameters and return values where applicable.
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

## Completed Stages

### Repository Foundation — Completed

- Initialized the Git repository and configured the GitHub SSH remote.
- Set up Python 3.11 using uv.
- Configured the development environment, Ruff, and pytest.
- Established the flat `src/` layout and verified source imports.
- Created and pushed the initial project documentation and configuration.

### Dataset Feasibility — Completed

- Downloaded the official NLBSE'23 training and test archives.
- Audited the official training dataset using `notebooks/01_dataset_feasibility.ipynb`.
- Examined class imbalance, missing values, text lengths, structured text, duplicates, conflicting labels, and author metadata.
- Identified 1,222,422 unique title–body combinations and 440 conflicting text groups involving 4,146 rows.
- Confirmed dataset feasibility for the planned NLP experiments.

### Data Quality — Completed

**Objectives and implemented policy**

1. Implement reproducible data preparation using Python scripts.
2. Preserve the raw dataset without modification.
3. Normalize missing text values to empty strings without discarding otherwise usable issues.
4. Keep one representative from exact-text groups with consistent labels.
5. Exclude all rows from exact-text groups containing conflicting labels.
6. Produce a preparation summary with row counts and class distributions.
7. Add automated checks for the prepared dataset.

**Implementation**

- Added `src/data/prepare.py`, a two-pass chunked preparation pipeline.
- Used SHA-256 title–body hashes and a temporary disk-backed SQLite index to detect duplicates and label conflicts across chunks.
- Wrote `data/processed/issues_clean.csv` and `data/processed/preparation_report.json`.
- Wrote the output CSV header exactly once, including when the first processed chunk retains no rows.
- Added `tests/test_prepare.py` with three tests covering deduplication, missing titles, and cross-chunk label conflicts.

**Verified results**

| Metric | Value |
|---|---:|
| Original training rows | 1,275,881 |
| Prepared training rows | 1,221,982 |
| Duplicate rows removed | 49,753 |
| Conflicting rows removed | 4,146 |
| Conflicting text groups | 440 |
| Raw missing titles | 4 |
| Raw missing bodies | 4,681 |
| Prepared missing titles | 4 |
| Prepared missing bodies | 4,639 |
| Remaining exact title–body duplicates | 0 |

**Prepared class distribution**

| Label | Rows |
|---|---:|
| bug | 648,270 |
| feature | 444,745 |
| question | 74,474 |
| documentation | 54,493 |

**Validation and exit criteria**

- Reproduced preparation from the raw training dataset and obtained the expected row counts.
- Verified that no exact title–body duplicates remain in the prepared CSV.
- Verified that the row counts reconcile: 1,221,982 + 49,753 + 4,146 = 1,275,881.
- Verified the prepared CSV class distribution and allowed labels.
- Ruff lint checks passed, formatting applied, and all three unit tests passed.
- The raw dataset remains unchanged; the official test set has not been used for model development.

**Important scope distinction:** Data Quality removed exact-text label conflicts. It did not establish that all remaining labels are semantically correct, nor did it complete the future train/validation or official-test overlap audits.

## Current Stage — Evaluation Protocol (Next)

**Branch:** `evaluation-protocol`
**Status:** Planned

1. Create a reproducible, stratified training/validation split from the prepared training dataset.
2. Verify that exact title–body combinations do not overlap between splits.
3. Establish evaluation metrics and reporting conventions: Micro F1 as primary, with Macro/Weighted F1, per-class metrics, and confusion matrix.
4. Implement and report a majority-class baseline.
5. Record split seeds, counts, class proportions, and reproducibility details.
6. Keep the official test set reserved for final evaluation; plan an overlap audit without silently changing the benchmark.

## Planned Modeling Experiments

Experiments will progress from simpler to more complex approaches:

1. **Title Only:** TF-IDF with Logistic Regression and LinearSVC.
2. **Title + Body:** Measure the added value of issue descriptions.
3. **Input Truncation:** Compare shorter and longer input representations under controlled conditions.
4. **Preprocessing Ablations:** Compare minimal preprocessing against structured transformations, including code-block replacement with a `CODE` placeholder.
5. **Neural Models:** Evaluate an appropriate PyTorch neural text classifier (TextCNN or BiLSTM).
6. **Transformer Models:** Fine-tune DistilBERT and optionally RoBERTa, comparing predictive performance against computational cost.

All experiments will use a consistent validation protocol. The official test set will remain reserved for final evaluation.

## Data Quality Policy and Remaining Risks

- Keep raw data immutable.
- Deduplicate exact title–body matches with consistent labels, retaining one representative.
- Exclude entire exact-text groups containing conflicting labels from the primary modeling dataset.
- Do not use `id` as a standalone deduplication key until its semantics are established; duplicate IDs and ID-based disagreements require investigation.
- Exclude `author_association` from initial text-only model inputs; consider a separate ablation if justified.
- Preserve Markdown, code, stack traces, and other structured text initially; test transformations rather than assuming they help.
- Evaluate class imbalance with stratified splitting and per-class metrics.
- Study long-body truncation, near duplicates, label ambiguity, and possible repository-specific shortcuts during modeling and error analysis.
- Check exact train/validation overlap and audit official-test overlap without changing the benchmark silently.
- Fit learned preprocessing and vectorizers on the training split only.
