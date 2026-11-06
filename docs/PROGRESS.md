# Project Progress

## Current Status

- **Project:** `issue-intent`
- **Current stage:** Classical NLP Baselines — in progress
- **Current working branch:** `classical-baselines` (expected; verify with `git branch --show-current`)
- **Repository:** `git@github.com:mazen-alasas/issue-intent.git`
- **Local path:** `/home/mazen/issue-intent`
- **Last updated:** 2026-11-06
- **Next experiment:** Title + Body truncation ablation, using the established raw full-body baseline for comparison

## Verified Environment

- WSL2 Ubuntu
- Python 3.11.16 (uv environment)
- System Python 3.12.3
- uv 0.12.19
- Git 2.43.0
- pytest 9.2.0
- Ruff 0.17.3
- NVIDIA GTX 1650 Ti (4 GB VRAM)
- GitHub SSH authentication successful

Version information reflects earlier verified environment reports; versions have not been rechecked for this update.

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

### Evaluation Protocol — Implementation Completed

**Branch used:** `evaluation-protocol`

**Objectives:** Reproducible train/validation split, split integrity verification, consistent metrics, and a majority-class reference baseline.

**Implementation:**

- Added `src/data/split.py` with stratified 85% training / 15% validation splitting (`random_state=42`).
- Generated `data/processed/train.csv`, `data/processed/validation.csv`, and `data/processed/split_report.json`.
- Used chunked CSV writing, preserved source rows, verified class distributions and row counts, and checked exact title–body overlap across splits.
- Recorded SHA-256 hashes of the source, training, and validation CSV files.
- Added `src/models/evaluation.py` for accuracy, Micro/Macro/Weighted F1, per-class Precision/Recall/F1/Support, and an ordered confusion matrix.
- Added `src/models/dummy_baseline.py` using `DummyClassifier(strategy="most_frequent")`, fit on training labels only and evaluated on validation labels.
- Saved baseline output to `data/processed/dummy_baseline_metrics.json`.
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

**Validation status:** The actual split and dummy baseline executed successfully. An earlier full suite passed 34 tests after the confusion-matrix serialization fix. Subsequent development reached 48 passing tests (see current-stage validation). The merge status of `evaluation-protocol` into `main` was not independently confirmed in the latest conversation; verify Git history before marking the merge complete.

## Current Stage — Classical NLP Baselines (In Progress)

**Working branch:** `classical-baselines` (verify locally)

**Experiment protocol:**

- Use the prepared training split for model fitting; keep the full validation split fixed at 183,298 issues.
- Reserve the official NLBSE'23 test set for final evaluation.
- Use `random_state=42`, stratified sampling, and nested training subsets for learning-curve comparisons.
- Primary model-selection metric: Micro F1; also track Macro/Weighted F1, per-class metrics, confusion matrix, and runtime.
- Fit TF-IDF on training text only. Do not use validation or test text for fitting.
- Begin with TF-IDF (`ngram_range=(1, 1)`, `max_features=50_000`, `min_df=2`, `dtype=float32`) and Logistic Regression (`max_iter=1000`, `random_state=42`); compare LinearSVC separately.

### 1. Title-Only Classical Baselines — Completed

**Implementation:**

- Added `src/data/sampling.py` for reproducible stratified sampling, including nested samples and optional text columns.
- Added `src/models/train_classical.py` to evaluate title-only TF-IDF with Logistic Regression or LinearSVC.
- Added `src/models/learning_curve.py` for a nested Logistic Regression learning curve.
- Added sampling and model-training unit tests.

**Initial independent-sample experiments:**

| Model | Training rows | Micro F1 | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 50,000 | 0.777864 | 0.595009 | 0.762348 |
| LinearSVC | 50,000 | 0.766850 | 0.598181 | 0.754801 |
| Logistic Regression | 100,000 | 0.786015 | 0.607988 | 0.771413 |
| LinearSVC | 100,000 | 0.779010 | 0.610171 | 0.766616 |

These initial 50K/100K runs were sampled independently and should not be conflated with the nested learning-curve samples.

**Nested Logistic Regression learning curve:**

| Training rows | Micro F1 | Macro F1 | Weighted F1 | Fit time |
|---|---:|---:|---:|---:|
| 50,000 | 0.776817 | 0.592338 | 0.760995 | 3.155 s |
| 100,000 | 0.785830 | 0.609182 | 0.771420 | 8.634 s |
| 200,000 | 0.792704 | 0.618379 | 0.778851 | 17.749 s |

**200K title-only per-class F1 and recall:**

| Label | F1 | Recall |
|---|---:|---:|
| bug | 0.849904 | 0.879310 |
| documentation | 0.543075 | 0.431490 |
| feature | 0.785123 | 0.811308 |
| question | 0.295414 | 0.192015 |

**Finding:** More training data improved validation performance, but the `question` class remained difficult. The 200K nested sample is the reference for the subsequent title-and-body experiment.

### 2. Body Structure EDA — Completed

**Notebook:** `notebooks/02_body_structure_eda.ipynb`

**Scope:** A reproducible stratified sample of 50,000 training issues; no official test data and no modification of processed CSVs.

**Sample class distribution:**

| Label | Rows |
|---|---:|
| bug | 26,525 |
| feature | 18,198 |
| question | 3,047 |
| documentation | 2,230 |

**Body lengths (characters):**

| Label | Median | P95 | P99 |
|---|---:|---:|---:|
| bug | 713 | 5,539.6 | 16,825.48 |
| documentation | 304 | 1,843.6 | 4,213.74 |
| feature | 312 | 1,797.3 | 4,246.03 |
| question | 590 | 4,794.8 | 15,259.16 |

Body lengths are strongly right-skewed, with substantial long tails, especially for bugs and questions.

**Detected text structures (heuristic regex):**

| Label | Fenced code blocks | Stack traces |
|---|---:|---:|
| bug | 32.67% | 5.60% |
| documentation | 11.17% | 0.40% |
| feature | 11.85% | 0.27% |
| question | 26.88% | 3.45% |

**Fenced-code character ratios:**

| Label | Mean across all issues | Issues with detected fenced code | Mean among code-containing issues | Median among code-containing issues |
|---|---:|---:|---:|---:|
| bug | 18.0% | 8,665 | 55.0% | 57.1% |
| documentation | 4.8% | 249 | 43.3% | 40.3% |
| feature | 5.0% | 2,156 | 42.0% | 37.1% |
| question | 14.4% | 819 | 53.5% | 53.1% |

**Hypothetical 1,000-character body truncation:**

| Label | Issues exceeding limit | Mean fraction of body removed (all issues) |
|---|---:|---:|
| bug | 37.55% | 17.94% |
| documentation | 13.14% | 5.22% |
| feature | 13.68% | 5.02% |
| question | 31.24% | 14.80% |

**Manual inspection:** Deliberately selected long, code-heavy bug issues contained logs, stack traces, exception types, and useful error messages within fenced blocks, sometimes beyond the first 1,000 characters. These examples are not representative random samples, but they demonstrate why blanket code-block removal or arbitrary truncation may discard useful information.

**Decision:** Do not assume all fenced blocks are irrelevant source code. Establish a raw full-body reference before testing selective preprocessing or truncation. Character loss alone does not measure semantic information loss; regex detection is incomplete.

### 3. Title + Raw Full Body Baseline — Completed

**Implementation:**

- Added `src/models/train_title_body.py` to combine title and full, untruncated body and evaluate TF-IDF + Logistic Regression.
- Added `tests/test_train_title_body.py` for text combination and missing-value handling.
- Used the same 200,000-row stratified training sample as the largest nested title-only learning-curve sample, with the same random seed and validation split.
- No custom cleaning, code-block replacement, or body truncation was applied. The TF-IDF vectorizer still performs its configured/default text tokenization and normalization.
- Saved `data/processed/title_body_logreg_200k_metrics.json`.

**Verified run:** `uv run python -m src.models.train_title_body`

| Metric | Title only, 200K | Title + raw body, 200K | Change |
|---|---:|---:|---:|
| Micro F1 | 0.792704 | 0.832186 | +3.95 percentage points |
| Macro F1 | 0.618379 | 0.676879 | +5.85 percentage points |
| Weighted F1 | 0.778851 | 0.822626 | +4.38 percentage points |
| Training time | 17.749 s | 85.633 s | ~4.82× |

**Raw-body per-class results:**

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| bug | 0.854399 | 0.915992 | 0.884124 | 97,241 |
| documentation | 0.738144 | 0.441767 | 0.552732 | 8,174 |
| feature | 0.818167 | 0.842217 | 0.830018 | 66,712 |
| question | 0.669281 | 0.328440 | 0.440641 | 11,171 |

**Key comparisons:**

- Question F1 improved from 0.295414 to 0.440641 (+14.52 percentage points).
- Question recall improved from 19.20% to 32.84% (+13.64 percentage points), but remains low.
- Documentation recall improved only modestly, from 43.15% to 44.18%.
- The raw-body model predicted 3,669 of 11,171 validation questions correctly; 4,628 were predicted as `bug` and 2,757 as `feature`.
- Prediction time for the full 183,298-row validation set was 26.965 seconds, including vectorization and classification.

**Finding:** Adding raw body text substantially improved both overall and minority-class F1, at a measurable computational cost. This is the untruncated reference against which subsequent truncation and cleaning experiments will be compared. Results are validation scores, not official-test benchmark results.

### Validation and Working-Tree Status

- Most recently reported test run: **48 passed in 3.53 seconds**.
- That same run initially reported Ruff `E402` in the body-EDA notebook because a `src` import followed a `sys.path` modification in one cell; the recommended fix was to split path setup and imports into separate cells.
- The notebook lint fix was not followed by a separately reported clean full-suite run in the supplied conversation. Re-run all checks before committing.
- The title + raw-body experiment ran successfully; no subsequent Git commit or merge was confirmed here.

## Next Steps — Controlled Truncation Ablation

1. Confirm the working branch and run quality checks.
2. Preserve the completed title-only and raw full-body experiment scripts/results.
3. Compare fixed body-length limits (initially 1,000 and 2,000 characters) against the raw full-body reference, keeping the same 200K sampled rows, validation set, TF-IDF configuration, and classifier.
4. Record Micro/Macro/Weighted F1, per-class recall/F1, training and prediction time, and input-length settings.
5. If useful, extend to 500 and 4,000 characters; do not assume that the longest input is necessarily best.
6. Evaluate structured preprocessing in a separate ablation, distinguishing source code from logs/error messages where practical. Do not silently mix cleaning and truncation changes.
7. Document outcomes, run tests, and commit the stage work when verified.

## Upcoming Stages

- Complete Classical NLP Baselines
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
- Keep the original title-only and raw-body baselines reproducible when introducing new experiments.
- Large datasets, split files, and generated experiment artifacts should remain outside Git; record reproducible scripts and key results in documentation.
- For code quality, run auto-fix/format before checks and tests:

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```
