# Project Progress

## Current Status

- **Project:** issue-intent
- **Current stage:** Evaluation Protocol — implementation complete; documentation and merge pending
- **Current working branch:** `evaluation-protocol`
- **Next branch:** `classical-baselines` (after Evaluation Protocol merge)
- **Repository:** `git@github.com:mazen-alasas/issue-intent.git`
- **Local path:** `/home/mazen/issue-intent`
- **Last updated:** 2026-11-05

## Verified Environment

- WSL2 Ubuntu
- Python 3.11.16 (uv environment)
- System Python 3.12.3
- uv 0.12.19
- Git 2.43.0
- pytest 9.2.0
- Ruff 0.17.3
- NVIDIA GTX 1650 Ti (4GB VRAM)
- GitHub SSH authentication successful

## Completed Stages

### Repository Foundation — Completed

- Initialized the Git repository and configured the SSH remote.
- Created the Python environment using uv and configured non-package mode.
- Installed development dependencies and established the flat `src/` structure.
- Verified source imports, Ruff linting, formatting, and initial tests.
- Created and pushed the repository foundation and documentation.

### Dataset Feasibility — Completed

- Downloaded the official NLBSE'23 training and test archives.
- Audited the official training dataset only in `notebooks/01_dataset_feasibility.ipynb`.
- Examined schema, class imbalance, missing text, lengths, structured text, duplicates, conflicting labels, and author metadata.

**Key findings:**

| Metric | Value |
|---|---:|
| Original training rows | 1,275,881 |
| Bug | 670,951 (52.59%) |
| Feature | 472,216 (37.01%) |
| Question | 76,048 (5.96%) |
| Documentation | 56,666 (4.44%) |
| Missing titles | 4 |
| Missing bodies | 4,681 |
| Unique title–body combinations | 1,222,422 |
| Duplicate text occurrences | 53,459 |
| Conflicting text groups | 440 |
| Rows in conflicting groups | 4,146 |
| Median title length | 44 characters |
| Median body length | 490 characters |
| Duplicate ID occurrences | 33,353 |
| ID label conflicts | 2,059 |
| ID text conflicts | 1,714 |

**Decisions:** Preserve raw data; deduplicate by exact title–body text, retaining one representative for consistent-label groups; exclude all rows in conflicting-label text groups; do not deduplicate by `id` alone; retain structured text for later preprocessing ablations; exclude `author_association` from initial text-only models; reserve the official test set.

### Data Quality — Completed and Merged

**Objective:** Create a reproducible, leakage-aware modeling dataset without modifying the raw source.

**Implementation:**

- Added `src/data/prepare.py`, a two-pass chunked preparation pipeline.
- Used SHA-256 title–body hashes and a temporary SQLite index to detect duplicate and conflicting groups across chunks.
- Normalized missing titles and bodies to empty strings.
- Removed entire conflicting-label text groups and kept the first row of each consistent-label group.
- Preserved selected original columns.
- Generated `data/processed/issues_clean.csv` and `data/processed/preparation_report.json`.
- Fixed a cross-chunk CSV header bug so the header is written once, including when the first chunk has no retained rows.
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

**Validation:** Re-ran preparation successfully; verified class counts, zero exact duplicates, allowed labels, and row accounting (`1,221,982 + 49,753 + 4,146 = 1,275,881`). Ruff and the stage's unit tests passed.

**Scope:** Exact-text conflict removal does not guarantee semantic label correctness. Near duplicates, label noise, ID semantics, and repository shortcuts remain topics for later analysis.

### Evaluation Protocol — Implemented; Documentation and Merge Pending

**Branch:** `evaluation-protocol`

**Objectives:** Reproducible train/validation split, split integrity verification, consistent metrics, and a majority-class reference baseline.

**Implementation:**

- Added `src/data/split.py` with stratified 85% training / 15% validation splitting (`random_state=42`).
- Generated `data/processed/train.csv`, `data/processed/validation.csv`, and `data/processed/split_report.json`.
- Used chunked CSV writing, preserved source rows, verified class distributions and row counts, and checked exact title–body overlap across splits.
- Recorded SHA-256 hashes of the source, training, and validation CSV files.
- Added `src/models/evaluation.py` for accuracy, Micro/Macro/Weighted F1, per-class Precision/Recall/F1/Support, and an ordered confusion matrix.
- Added `src/models/dummy_baseline.py` using `DummyClassifier(strategy="most_frequent")`, fit on training labels only and evaluated on validation labels.
- Saved the baseline output to `data/processed/dummy_baseline_metrics.json`.
- Added unit tests in `tests/test_split.py`, `tests/test_evaluation.py`, and `tests/test_dummy_baseline.py`.

**Actual split results:**

| Metric | Value |
|---|---:|
| Prepared source rows | 1,221,982 |
| Training rows | 1,038,684 |
| Validation rows | 183,298 |
| Validation fraction | 0.15 |
| Random state | 42 |
| Exact cross-split title–body overlap | 0 |
| Source row multiset preserved | Yes |
| Row counts reconciled | Yes |

**Class distribution:**

| Label | Source | Train | Validation |
|---|---:|---:|---:|
| bug | 648,270 | 551,029 | 97,241 |
| feature | 444,745 | 378,033 | 66,712 |
| question | 74,474 | 63,303 | 11,171 |
| documentation | 54,493 | 46,319 | 8,174 |

**SHA-256 fingerprints (from `split_report.json`):**

- Source: `804e63362d1a8096864066191b42e41d3809ca5ff09050622dc5fa7f98d00ef3`
- Train: `41d703815435c2018a93bb9021e69083ce49ebe9b699ecb1aff03f0f7500055d`
- Validation: `7946749613bda611b96cb4871aaa38045195fa980744b1112fa2f9f9bd2f9b23`

**Majority-class baseline (validation set):**

| Metric | Result |
|---|---:|
| Accuracy | 0.530508 |
| Micro F1 (primary) | 0.530508 |
| Macro F1 | 0.173311 |
| Weighted F1 | 0.367771 |

The dummy model predicted `bug` for every validation sample. Its `bug` recall was 1.0, but F1 for `feature`, `question`, and `documentation` was 0.0. This establishes a reference, not evidence of meaningful issue understanding.

**Validation status:** The actual split and dummy baseline executed successfully. The last explicitly reported full suite passed **34 tests** after the confusion-matrix serialization fix. A later commit of the metrics and dummy-baseline implementation was confirmed; the exact final test count after that commit has not been separately reported.

**Stage exit criteria remaining:** Update documentation, run final quality checks, commit documentation, and merge `evaluation-protocol` into `main`.

## Next Stage — Classical NLP Baselines

**Branch:** `classical-baselines` (create after Evaluation Protocol merge)

1. Start with Title-only TF-IDF + Logistic Regression and LinearSVC.
2. Run a resource-conscious stratified training sample first; measure fit time, RAM, and validation metrics before scaling.
3. Compare Title-only against Title + Body.
4. Evaluate truncation and structured-text preprocessing ablations, including code-block replacement with `CODE`.
5. Keep vectorizer fitting strictly on the training split and use the same validation protocol for comparisons.
6. Do not use the official test set for model selection.

## Upcoming Stages

- Classical NLP Baselines
- Neural NLP
- Transformer Fine-Tuning
- Final Evaluation
- API & Docker
- Portfolio Release

## Notes and Open Risks

- Published benchmark results are not project results.
- The official test set remains reserved for final evaluation; audit possible exact overlap without silently modifying the benchmark.
- Near duplicates, semantic label noise, long bodies, Markdown/code/HTML, and repository-specific shortcuts remain to be investigated.
- Learned preprocessing and vectorizers must be fit on training data only.
- Large datasets, split files, and generated experiment artifacts should remain outside Git; record reproducible scripts and key results in documentation.
- For code quality, run `uv run ruff check --fix .` and `uv run ruff format .` before `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest -q`.
