# Project Progress

## Current Status

- **Project:** issue-intent
- **Current stage:** Data Quality (next to implement)
- **Current working branch:** `dataset-feasibility` (documentation finalization)
- **Repository:** `git@github.com:mazen-alasas/issue-intent.git`
- **Local path:** `/home/mazen/issue-intent`
- **Last updated:** 2026-11-02

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
- Class distribution:
  - Bug: 670,951 (52.59%)
  - Feature: 472,216 (37.01%)
  - Question: 76,048 (5.96%)
  - Documentation: 56,666 (4.44%)
- Missing bodies: **4,681**
- Missing titles: **4**
- Unique title–body combinations: **1,222,422**
- Duplicate text occurrences: **53,459**
- Conflicting text groups: **440**
- Rows belonging to conflicting groups: **4,146**
- Median title length: **44 characters**
- Median body length: **490 characters**

**Decisions:**

- Preserve raw datasets unchanged.
- Keep one representative from each exact-text group with consistent labels.
- Exclude entire exact-text groups with conflicting labels from the primary modeling dataset.
- Do not deduplicate based solely on `id` until its semantics are established.
- Retain structured text initially and evaluate preprocessing variants experimentally.
- Exclude `author_association` from initial text-only models.
- Keep the official test set untouched until final evaluation.

**Outcome:** The dataset is suitable for the planned NLP classification experiments, subject to reproducible data preparation and a leakage-aware evaluation protocol.

## Current Work — Data Quality

**Status:** Not started.

Planned work:

1. Implement `src/data/prepare.py`.
2. Handle missing text fields.
3. Remove consistent-label exact duplicates while retaining one representative.
4. Exclude conflicting-label text groups.
5. Generate a preparation report.
6. Add data quality tests.
7. Verify the resulting dataset and document the results.

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
- No modeling results have been produced yet.