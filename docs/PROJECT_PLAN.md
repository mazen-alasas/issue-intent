# issue-intent — Project Plan

## Project Overview

issue-intent is an NLP and ML engineering project for classifying real GitHub issues into four categories: `bug`, `feature`, `question`, and `documentation`.

The project uses the NLBSE 2023 Issue Report Classification Competition as its intended benchmark. The goal is to build a reproducible classification pipeline, compare NLP approaches and computational trade-offs, and deliver a tested inference API suitable for a technical portfolio.

## Project Roadmap

| Stage | Branch | Main Deliverables | Status |
|---|---|---|---|
| Repository Foundation | `main` | Environment, configuration, documentation | Completed |
| Dataset Feasibility | `dataset-feasibility` | Dataset audit, schema, integrity, feasibility | Completed |
| Data Quality | `data-quality` | Deduplication, conflicting labels, preparation pipeline | Completed and merged |
| Evaluation Protocol | `evaluation-protocol` | Stratified split, integrity checks, metrics, majority baseline | Implemented; documentation and merge pending |
| Classical NLP | `classical-baselines` | TF-IDF, Logistic Regression, LinearSVC | Planned — next |
| Neural NLP | `neural-baseline` | PyTorch TextCNN or BiLSTM | Planned |
| Transformers | `transformer-finetuning` | DistilBERT, optional RoBERTa | Planned |
| Final Evaluation | `final-evaluation` | Official test evaluation, error analysis, model card | Planned |
| API & Docker | `api-serving` | FastAPI, inference tests, Docker | Planned |
| Portfolio Release | `portfolio-release` | CI, documentation, reproducibility, release | Planned |

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
- Run Ruff auto-fix and formatting before lint/format verification and pytest.

## Evaluation Strategy

The official NLBSE 2023 test set is reserved for final evaluation. Development and model selection use the validation split derived exclusively from the official training data.

- **Primary metric:** Micro F1.
- **Additional metrics:** Macro F1, Weighted F1, Accuracy, per-class Precision/Recall/F1/Support, and confusion matrix.
- **Operational comparisons:** Training time, inference latency, peak RAM/VRAM where practical, and model size.
- **Reproducibility:** Stratified 85/15 train/validation split, `random_state=42`, recorded row counts, integrity checks, and file hashes.
- **Leakage prevention:** Fit learned preprocessing and vectorizers only on training data; do not tune using the official test set.

## Completed Stages

### Repository Foundation — Completed

- Initialized the Git repository and configured the GitHub SSH remote.
- Set up Python 3.11 using uv and configured development tooling.
- Established the flat `src/` layout and verified source imports.
- Created and pushed the initial project documentation and configuration.

### Dataset Feasibility — Completed

- Downloaded official NLBSE'23 training and test archives.
- Audited the training dataset in `notebooks/01_dataset_feasibility.ipynb`.
- Examined class imbalance, missing values, text lengths, structured text, duplicates, conflicting labels, and author metadata.
- Identified 1,222,422 unique title–body combinations, 440 conflicting text groups, and 4,146 rows in those groups.
- Confirmed suitability for planned NLP experiments, subject to data preparation and leakage-aware evaluation.

### Data Quality — Completed and Merged

**Policy:** Preserve raw data, normalize missing text to empty strings, retain one representative per exact-text group with consistent labels, and exclude all rows in conflicting-label groups. Do not deduplicate by `id` alone.

**Implementation:**

- Added `src/data/prepare.py`, a two-pass chunked pipeline with SHA-256 title–body hashes and a temporary SQLite index.
- Generated `data/processed/issues_clean.csv` and `data/processed/preparation_report.json`.
- Fixed a cross-chunk CSV header issue.
- Added `tests/test_prepare.py` with three focused tests at the stage's completion.

**Verified results:**

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

**Prepared class distribution:**

| Label | Rows |
|---|---:|
| bug | 648,270 |
| feature | 444,745 |
| question | 74,474 |
| documentation | 54,493 |

Row accounting reconciles: `1,221,982 + 49,753 + 4,146 = 1,275,881`. This process resolves exact-text conflicts, not all possible semantic label errors or near duplicates.

### Evaluation Protocol — Implemented; Documentation and Merge Pending

**Branch:** `evaluation-protocol`

**Implementation:**

- `src/data/split.py`: reproducible stratified 85/15 train/validation split, chunked CSV output, SHA-256 hashes, exact cross-split text overlap verification, and full-row source preservation checks.
- `src/models/evaluation.py`: reusable accuracy, Micro/Macro/Weighted F1, per-class metrics, and confusion matrix in JSON-compatible form.
- `src/models/dummy_baseline.py`: majority-class `DummyClassifier`, fitted on training labels and evaluated on validation labels.
- Tests: `tests/test_split.py`, `tests/test_evaluation.py`, and `tests/test_dummy_baseline.py`.

**Split configuration and results:**

| Metric | Value |
|---|---:|
| Source rows | 1,221,982 |
| Train rows (85%) | 1,038,684 |
| Validation rows (15%) | 183,298 |
| Random state | 42 |
| Exact train/validation title–body overlap | 0 |
| Source rows preserved | Yes |
| Row counts match | Yes |

**Per-class counts:**

| Label | Train | Validation |
|---|---:|---:|
| bug | 551,029 | 97,241 |
| feature | 378,033 | 66,712 |
| question | 63,303 | 11,171 |
| documentation | 46,319 | 8,174 |

**Majority-class baseline on validation:**

| Metric | Result |
|---|---:|
| Accuracy | 0.530508 |
| Micro F1 | 0.530508 |
| Macro F1 | 0.173311 |
| Weighted F1 | 0.367771 |

The model predicts `bug` for all validation rows, yielding zero F1 for the other three classes. Results are recorded in `data/processed/dummy_baseline_metrics.json`; split configuration and hashes are recorded in `data/processed/split_report.json`.

The actual split and baseline ran successfully. The latest explicitly reported full test run passed 34 tests. The evaluation implementation has been committed; documentation and branch merge are the remaining closure tasks.

**Exit criteria:** Final code-quality checks, documentation commit, and merge `evaluation-protocol` into `main`.

## Planned Modeling Experiments

Experiments progress from simpler to more complex approaches:

1. **Title Only:** TF-IDF with Logistic Regression and LinearSVC. Begin with a stratified training sample and assess runtime and RAM before scaling.
2. **Title + Body:** Measure the added value of issue descriptions.
3. **Input Truncation:** Compare shorter and longer input representations under controlled conditions.
4. **Preprocessing Ablations:** Compare minimal preprocessing against structured transformations, including code-block replacement with a `CODE` placeholder and selected Markdown/HTML handling.
5. **Neural Models:** Evaluate an appropriate PyTorch neural classifier (TextCNN or BiLSTM).
6. **Transformer Models:** Fine-tune DistilBERT and optionally RoBERTa, comparing predictive quality and computational cost.

All experiments will use the established validation protocol. Train-fitted transformations must not learn from validation or official test data. The official test set will remain reserved for final evaluation.

## Data Quality Policy and Remaining Risks

- Keep raw data immutable.
- Retain one representative for consistent-label exact title–body groups; exclude all conflicting-label exact-text groups.
- Do not use `id` as a standalone deduplication key until its semantics are established; repeated IDs and ID-based conflicts remain open research questions.
- Exclude `author_association` from initial text-only models; consider a later ablation only if justified.
- Preserve Markdown, code, stack traces, and other structured text initially; measure the effect of preprocessing rather than assuming it helps.
- Use stratification and per-class metrics to study class imbalance.
- Investigate long-body truncation, near duplicates, label ambiguity, and repository-specific shortcuts during modeling and error analysis.
- Audit official-test overlap without silently changing the benchmark.
- Fit learned preprocessing and vectorizers on training data only.
- Keep large data and model outputs out of Git; commit code, tests, and concise documentation of results.
