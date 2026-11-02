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
| Data Quality | data-quality | Missing values, duplicates, leakage, label analysis | Next |
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

## Current Stage — Data Quality

The Repository Foundation and Dataset Feasibility stages are complete.

### Completed Stages

**Repository Foundation**
- Initialized the Git repository and configured the GitHub SSH remote.
- Set up Python 3.11 using uv.
- Configured the development environment, Ruff, and pytest.
- Established the flat `src/` layout and verified source imports.
- Created and pushed the initial project documentation and configuration.

**Dataset Feasibility**
- Downloaded the official NLBSE'23 training and test archives.
- Audited the official training dataset using `notebooks/01_dataset_feasibility.ipynb`.
- Examined class imbalance, missing values, text lengths, structured text, duplicates, conflicting labels, and author metadata.
- Identified 1,222,422 unique title–body combinations and 440 conflicting text groups involving 4,146 rows.
- Confirmed dataset feasibility for the planned NLP experiments.

### Data Quality Objectives

1. Implement reproducible data preparation using Python scripts.
2. Preserve the raw dataset without modification.
3. Handle missing text values without unnecessarily discarding usable issues.
4. Keep one representative from exact-text groups with consistent labels.
5. Exclude all rows from exact-text groups containing conflicting labels.
6. Produce a preparation summary with row counts and class distributions.
7. Add automated checks for the prepared dataset.

### Exit Criteria

- Data preparation can be reproduced from the official raw training dataset.
- No exact title–body duplicates remain in the prepared dataset.
- No conflicting-label text groups remain in the prepared dataset.
- All row removals are accounted for in the preparation report.
- The original raw data remains unchanged.
- Tests and documentation reflect the implemented behavior.

### Next Stage — Evaluation Protocol

After completing Data Quality:

1. Create a reproducible, stratified training/validation split.
2. Verify that exact title–body combinations do not overlap between splits.
3. Establish evaluation metrics and reporting conventions.
4. Implement a majority-class baseline.
5. Reserve the official test set for final evaluation.


### Planned Modeling Experiments

Experiments will progress from simpler to more complex approaches:

1. **Title Only:** TF-IDF with Logistic Regression and LinearSVC.
2. **Title + Body:** Measure the added value of issue descriptions.
3. **Input Truncation:** Compare shorter and longer input representations under controlled conditions.
4. **Preprocessing Ablations:** Compare minimal preprocessing against structured transformations, including code-block replacement with a `CODE` placeholder.
5. **Neural Models:** Evaluate an appropriate neural text classifier.
6. **Transformer Models:** Fine-tune a pretrained transformer and compare predictive performance against computational cost.

All experiments will use a consistent validation protocol. The official test set will remain reserved for final evaluation.

### Data Quality Policy

- Keep raw data immutable.
- Deduplicate exact title–body matches with consistent labels, retaining one representative.
- Exclude entire exact-text groups containing conflicting labels from the primary modeling dataset.
- Do not use `id` as a standalone deduplication key until its semantics are established.
- Exclude `author_association` from initial text-only model inputs.
- Record preparation counts and verify that training and validation have no exact-text overlap.