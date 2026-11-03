# Project Progress

## Current Status

- **Project:** issue-intent
- **Current stage:** Data Quality completed; Evaluation Protocol next
- **Current working branch:** `data-quality` (pending review, commit, and pull request)
- **Next branch:** `evaluation-protocol` (after Data Quality merge)
- **Repository:** `git@github.com:mazen-alasas/issue-intent.git`
- **Local path:** `/home/mazen/issue-intent`
- **Last updated:** 2026-11-03

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
- Created the Python environment using uv.
- Installed development dependencies.
- Configured uv non-package mode.
- Established the flat `src/` directory structure.
- Verified source imports, Ruff linting, and formatting.
- Created and pushed the initial repository foundation and documentation.

### Dataset Feasibility — Completed

- Downloaded the official NLBSE'23 training and test archives.
- Extracted and audited the training dataset only.
- Created `notebooks/01_dataset_feasibility.ipynb`.
- Inspected schema, class distribution, missing values, text lengths, structured text, duplicate records, conflicting labels, and author metadata.

**Key findings:**

- Training rows: **1,275,881**
- Classes: `bug`, `feature`, `question`, `documentation`
- Original class distribution:
  - Bug: 670,951 (52.59%)
  - Feature: 472,216 (37.01%)
  - Question: 76,048 (5.96%)
  - Documentation: 56,666 (4.44%)
- Missing bodies: **4,681**
- Missing titles: **4**
- Unique title–body combinations: **1,222,422**
- Duplicate text occurrences: **53,459** (raw rows minus unique text groups)
- Conflicting text groups: **440**
- Rows belonging to conflicting groups: **4,146**
- Median title length: **44 characters**
- Median body length: **490 characters**
- Duplicate IDs: **33,353 occurrences**
- ID label conflicts: **2,059 occurrences**
- ID text conflicts: **1,714 occurrences**

**Decisions:**

- Preserve raw datasets unchanged.
- Keep one representative from each exact-text group with consistent labels.
- Exclude entire exact-text groups with conflicting labels from the primary modeling dataset.
- Do not deduplicate based solely on `id` until its semantics are established.
- Retain structured text initially and evaluate preprocessing variants experimentally.
- Exclude `author_association` from initial text-only models.
- Keep the official test set untouched until final evaluation.

**Outcome:** The dataset is suitable for the planned NLP classification experiments, subject to reproducible data preparation and a leakage-aware evaluation protocol.

### Data Quality — Completed

**Objective:** Prepare a reproducible training dataset by resolving exact-text duplicates and conflicting labels while preserving the raw data.

**Implementation:**

- Added `src/data/prepare.py` with a two-pass, chunk-based pipeline.
- Used a temporary SQLite index and SHA-256 title–body hashes.
- Normalized missing titles and bodies to empty strings.
- Removed all rows belonging to conflicting-label text groups.
- Retained the first occurrence of each consistent-label text group.
- Preserved the original selected columns.
- Generated `data/processed/issues_clean.csv` and `data/processed/preparation_report.json`.
- Added English Google-style function docstrings and three unit tests in `tests/test_prepare.py`.
- Fixed a CSV header bug discovered by the cross-chunk conflict test; the output header is now written exactly once, including when the first chunk has no retained rows.

**Results:**

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
| Remaining exact-text duplicates | 0 |

**Prepared class distribution:**

| Label | Rows |
|---|---:|
| bug | 648,270 |
| feature | 444,745 |
| question | 74,474 |
| documentation | 54,493 |

**Validation:**

- Ruff lint checks passed.
- Ruff formatting applied.
- All **3 unit tests passed**.
- Full preparation pipeline rerun successfully after the header fix, with unchanged counts.
- Prepared CSV verified for row count, exact duplicate removal, valid labels, and class distribution.
- Row accounting verified: **1,221,982 + 49,753 + 4,146 = 1,275,881**.

**Decisions:**

- Keep raw datasets immutable.
- Do not deduplicate using issue IDs alone.
- Do not apply aggressive text preprocessing at this stage.
- Exclude `author_association` from initial text-only models.
- Reserve the official test set for final evaluation.

**Remaining considerations:**

- Class imbalance and minority-class performance.
- Long issue bodies and input truncation.
- Markdown, code blocks, and structured text preprocessing.
- Exact train/validation overlap prevention.
- Official test overlap audit without modifying the benchmark.
- Near-duplicate detection, label-noise analysis, and repository-specific patterns.
- Interpretation of repeated issue IDs and ID-related conflicts.

**Stage status:** Implementation and validation complete; documentation review, Git commit, and pull request/merge are still pending.

## Next Stage — Evaluation Protocol

**Branch:** `evaluation-protocol` (create after Data Quality merge)

Planned work:

1. Create a reproducible stratified train/validation split from the prepared training data.
2. Verify split integrity and prevent exact-text leakage.
3. Define Micro F1 (primary), Macro F1, Weighted F1, per-class metrics, and confusion matrix reporting.
4. Establish a majority-class baseline.
5. Document seeds, split proportions, and reproducibility.
6. Keep the official test set out of model selection; plan an overlap audit without silently modifying the benchmark.

## Upcoming Stages

- Evaluation Protocol
- Classical NLP Baselines
- Neural NLP
- Transformer Fine-Tuning
- Final Evaluation
- API & Docker
- Portfolio Release

## Notes

- Published benchmark results are not project results.
- The official test set must not be used for model selection or hyperparameter tuning.
- Data quality decisions must be implemented reproducibly and validated before model training.
- Exact-text deduplication does not establish that all remaining labels are semantically correct.
- No modeling results have been produced yet.
