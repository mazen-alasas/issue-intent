# Project Progress

**Project:** `issue-intent`  

**Repository:** `git@github.com:mazen-alasas/issue-intent.git`  

**Local path:** `/home/mazen/issue-intent`  

**Last updated:** 2026-11-15

**Current stage:** API serving — inference and FastAPI endpoints committed; benchmarking implemented and measured; benchmark quality gate/commit pending confirmation.

**Last completed branch:** `final-evaluation` (merged into `main`; baseline commit `09672bf`).

**Current branch:** `api-serving` (latest confirmed commit `752dc87`); next: close benchmarking, then Docker and CI.

## Executive Status

Repository foundation, data preparation and quality, evaluation protocol, structured preprocessing, full-scale classical baselines, TextCNN, DistilBERT fine-tuning, and official-test final evaluation are complete. The `final-evaluation` branch was merged into `main`.

The selected DistilBERT checkpoint (epoch 3, chosen by validation Macro F1) achieved **0.886558 Micro F1** and **0.791175 Macro F1** on the 142,320-row official test set. Saved evaluation outputs passed local integrity and source-alignment audits; those audits do not cryptographically prove the provenance of the original Kaggle inference weights.

On `api-serving`, the reusable CPU-first inference engine and FastAPI `/health` and `/predict` endpoints were implemented, tested, exercised against the real local checkpoint, and committed. The most recently reported full automated gate was **310 passing tests**, before the new benchmark tests were added. An HTTP benchmark script and tests were subsequently drafted, and a real local benchmark run produced the results recorded in Section 10; the new test-suite result and benchmark commit have **not yet been confirmed**.

**Next:** verify the benchmark files and complete the quality gate, ignore generated benchmark JSON, commit the benchmark/documentation changes, then proceed to Docker, CI, and portfolio release.

## Environment and Conventions

- WSL2 Ubuntu; Python 3.11.16 managed by `uv` (earlier reported `uv 0.12.19`).

- Earlier verified tools: Git 2.43.0, pytest 9.2.0, Ruff 0.17.3; versions may have changed.

- NVIDIA GeForce GTX 1650 Ti, reported 4 GB VRAM; local memory constraints require conservative training plans.

- `pyarrow==26.0.1` added with `uv add pyarrow` for chunked Parquet generation.

- Flat `src/` layout; exploratory notebooks, production-style scripts, focused tests, and tracked documentation.

- Use `random_state=42` for established sampling/splitting experiments.

- Keep raw datasets, generated CSV/Parquet datasets, experiment outputs, and model artifacts out of Git.

## 1. Repository Foundation — Completed

- Initialized the Git repository, configured the GitHub SSH remote, and created the `uv` Python environment.

- Established the flat `src/` layout, development tooling, initial tests, README, and project documentation.

- Confirmed source imports, Ruff checks, and test execution.

## 2. Dataset Feasibility — Completed

**Notebook:** `notebooks/01_dataset_feasibility.ipynb`  

**Source:** NLBSE 2023 Issue Report Classification competition; official training and test archives downloaded. Exploration was restricted to the official training data.

| Measure | Finding |

|---|---:|

| Original official training rows | 1,275,881 |

| Bug | 670,951 (52.59%) |

| Feature | 472,216 (37.01%) |

| Question | 76,048 (5.96%) |

| Documentation | 56,666 (4.44%) |

| Missing titles | 4 |

| Missing bodies | 4,681 |

| Unique title–body combinations | 1,222,422 |

| Duplicate text occurrences | 53,459 |

| Conflicting text groups | 440 |

| Rows in conflicting text groups | 4,146 |

| Duplicate ID occurrences | 33,353 |

| ID label conflicts | 2,059 |

| ID text conflicts | 1,714 |

| Median title length | 44 characters |

| Median body length | 490 characters |

**Decision:** Keep original data immutable. Deduplicate using exact title–body text rather than `id` alone; remove whole text groups containing conflicting labels; retain structured content for controlled modeling experiments; initially exclude `author_association` from model inputs.

## 3. Data Quality — Completed and Merged

**Implementation:** `src/data/prepare.py`; tests in `tests/test_prepare.py`.

- Two-pass, chunked preparation using SHA-256 title–body hashes and a temporary SQLite index.

- Normalize missing title/body fields to empty strings.

- Keep the first representative of each consistent-label exact-text group and remove all rows of conflicting-label groups.

- Fixed the cross-chunk CSV header bug, including the case where the first chunk retains no rows.

- Generated `data/processed/issues_clean.csv` and `data/processed/preparation_report.json`.

| Measure | Result |

|---|---:

| Source rows | 1,275,881

| Prepared rows | 1,221,982

| Duplicate rows removed | 49,753

| Conflicting-label rows removed| 4,146 |

| Conflicting text groups | 440

| Remaining exact title–body duplicates | 0 |

| Prepared missing titles | 4 |

| Prepared missing bodies | 4,639 |

**Accounting:** `1,221,982 + 49,753 + 4,146 = 1,275,881`.

| Label | Prepared rows |

|---|---:|

| bug | 648,270 |

| feature | 444,745 |

| question | 74,474 |

| documentation | 54,493 |

Exact-text cleaning does not resolve semantic label noise, near duplicates, or all potential repository-related shortcuts.

## 4. Evaluation Protocol — Completed

**Branch:** `evaluation-protocol`  

**Implementation:** `src/data/split.py`, `src/models/evaluation.py`, `src/models/dummy_baseline.py` and corresponding tests.

- Stratified 85%/15% train/validation split with `random_state=42`.

- Preserve original source rows in split files; validate row accounting, class distributions, reproducibility, and zero exact title–body overlap across splits.

- Record SHA-256 hashes and JSON reports.

- Reusable accuracy, Micro/Macro/Weighted F1, per-class precision/recall/F1/support, and ordered confusion matrix.

- DummyClassifier majority-class reference baseline, trained only on training labels.

| Measure | Result |

|---|---:|

| Prepared source rows | 1,221,982 |

| Training rows | 1,038,684 |

| Validation rows | 183,298 |

| Exact train/validation text overlap | 0 |

| Source row multiset preserved | Yes |

| Label | Train | Validation |

|---|---:|---:|

| bug | 551,029 | 97,241 |

| feature | 378,033 | 66,712 |

| question | 63,303 | 11,171 |

| documentation | 46,319 | 8,174 |

**Source and split fingerprints** (`data/processed/split_report.json`):

- Prepared source: `804e63362d1a8096864066191b42e41d3809ca5ff09050622dc5fa7f98d00ef3`

- Train CSV: `41d703815435c2018a93bb9021e69083ce49ebe9b699ecb1aff03f0f7500055d`

- Validation CSV: `7946749613bda611b96cb4871aaa38045195fa980744b1112fa2f9f9bd2f9b23`

| Dummy baseline metric | Validation result |

|---|---:|

| Accuracy / Micro F1 | 0.530508 |

| Macro F1 | 0.173311 |

| Weighted F1 | 0.367771 |

The dummy model predicts `bug` for every validation issue. It has zero F1 for the other three classes; it is a reference, not a useful deployed classifier.

## 5. Classical NLP Baselines — Completed (historical experiment record)

**Existing branch:** `classical-baselines`. The completed structured-preprocessing work was implemented on a separate branch and subsequently merged into `main`.

### Experimental protocol

- All comparable 200K experiments use the same nested stratified 200,000-row training sample, `random_state=42`, and fixed full validation set of 183,298 rows.

- TF-IDF: word unigrams, `max_features=50_000`, `min_df=2`, `dtype=float32`.

- Logistic Regression: `max_iter=1000`, `random_state=42`.

- Fit the vectorizer on training text only; no validation/test fitting.

- Primary metric: Micro F1; always report Macro/Weighted F1 and minority-class performance.

- Report timings as measured, not hardware-independent benchmarks.

### 5.1 Title-only baselines — Completed at sample scale

**Files:** `src/data/sampling.py`, `src/models/train_classical.py`, `src/models/learning_curve.py`.

Initial independently sampled runs (not the nested learning-curve subsets):

| Model | Train rows | Micro F1 | Macro F1 | Weighted F1 |

|---|---:|---:|---:|---:|

| Logistic Regression | 50,000 | 0.777864 | 0.595009 | 0.762348 |

| LinearSVC | 50,000 | 0.766850 | 0.598181 | 0.754801 |

| Logistic Regression | 100,000 | 0.786015 | 0.607988 | 0.771413 |

| LinearSVC | 100,000 | 0.779010 | 0.610171 | 0.766616 |

Nested Logistic Regression learning curve:

| Train rows | Micro F1 | Macro F1 | Weighted F1 | Fit time |

|---|---:|---:|---:|---:|

| 50,000 | 0.776817 | 0.592338 | 0.760995 | 3.155 s |

| 100,000 | 0.785830 | 0.609182 | 0.771420 | 8.634 s |

| 200,000 | 0.792704 | 0.618379 | 0.778851 | 17.749 s |

At 200K, per-class F1: bug `0.849904`, documentation `0.543075`, feature `0.785123`, question `0.295414`. Question recall was only `0.192015`.

### 5.2 Body structure EDA — Completed

**Notebook:** `notebooks/02_body_structure_eda.ipynb`  

**Sample:** 50,000 stratified training issues; official test excluded.

| Label | Median body chars | P95 | P99 | Fenced-code detection | Stack-trace detection |

|---|---:|---:|---:|---:|---:|

| bug | 713 | 5,539.6 | 16,825.48 | 32.67% | 5.60% |

| documentation | 304 | 1,843.6 | 4,213.74 | 11.17% | 0.40% |

| feature | 312 | 1,797.3 | 4,246.03 | 11.85% | 0.27% |

| question | 590 | 4,794.8 | 15,259.16 | 26.88% | 3.45% |

Among issues with detected fenced code, median fenced-code share of body characters was approximately 57.1% for bug, 40.3% for documentation, 37.1% for feature, and 53.1% for question. Manual review found exceptions, stack traces, and diagnostic messages inside some code fences. Regex-based detection is heuristic and not a semantic ground truth.

A hypothetical 1,000-character truncation would affect 37.55% of bug issues and 31.24% of question issues, versus 13.14% of documentation and 13.68% of feature issues in this sample.

### 5.3 Title + raw full body — Completed at 200K

**Implementation:** `src/models/train_title_body.py`; result: `data/processed/title_body_logreg_200k_metrics.json`.

| Metric | Title only | Title + raw body | Change |

|---|---:|---:|---:|

| Micro F1 | 0.792704 | 0.832186 | +0.039482 |

| Macro F1 | 0.618379 | 0.676879 | +0.058500 |

| Weighted F1 | 0.778851 | 0.822626 | +0.043775 |

| Training time | 17.749 s | 85.633 s | \\~4.82× |

Raw-body per-class F1: bug `0.884124`, documentation `0.552732`, feature `0.830018`, question `0.440641`. Question recall improved from `0.192015` (title only) to `0.328440` (raw body), but remained low. Full-validation prediction time: `26.965 s`.

### 5.4 Body truncation ablation — Completed at 200K

The established raw full-body run was compared with character-limited body variants using the same sampling, validation split, TF-IDF configuration, and classifier. The following Micro F1 results were recorded:

| Body representation | Micro F1 |

|---|---:|

| 1,000 characters | 0.831853 |

| 2,000 characters | 0.832513 |

| 4,000 characters | 0.832633 |

| Raw full body | 0.832186 |

**Interpretation:** Truncation is not monotonically harmful or beneficial. The 2K and 4K runs slightly exceeded the raw full-body Micro F1, while 1K was slightly below it. These are validation comparisons, not statistically established improvements. Do not select a truncation limit solely by character count or assume it preserves error context.

### 5.5 Structured preprocessing — Completed and merged

**Branch:** `structured-preprocessing` (merged into `main`, as reported).  

**Files:** `src/preprocessing/structured.py`, `src/models/structured_ablation_experiment.py`, `src/preprocessing/analyze_size.py`, `notebooks/03_structured_text_audit.ipynb`, and focused tests.

Transformations include HTML comment removal, Markdown normalization, URL/path normalization, empty template placeholder removal, optional repeated-line compression, fenced-code handling, and whitespace normalization. A malformed URL parsing regression was fixed by preserving the original URL when `urlsplit` raises `ValueError`.

The controlled ablations use the same 200K training subset and 183,298-row validation set. Preprocessing applies to the **body only**, leaving the original title intact.

| ID | Representation | Micro F1 | Macro F1 | Weighted F1 |

|---|---|---:|---:|---:|

| E0 | Raw title + full body | 0.832186 | 0.676879 | 0.822626 |

| E1 | Minimal preprocessing | 0.831722 | 0.675462 | Not recorded here |

| E2 | Full structured; fenced code → `CODE` | **0.833441** | 0.678902 | **0.824319** |

| E3 | Structured; fenced code retained | 0.830604 | 0.673184 | 0.820802 |

| E4 | Structured; no repeated-line compression | 0.833304 | 0.678782 | 0.824166 |

| E5 | E4 + selective error signals from fenced code | 0.833266 | **0.678985** | 0.824175 |

E5 extracts bounded diagnostic signals (e.g., exception names and selected HTTP/CUDA errors) rather than keeping all code. It improved Macro F1 marginally relative to E4 but did not improve the primary Micro F1. E2 had the highest observed Micro F1; **E4 was selected as the simpler practical classical preprocessing configuration** because the difference was very small and repeated-line compression added little size benefit. This is an engineering trade-off, not a claim that E4 is statistically superior.

Selected E4 configuration:

```python

StructuredPreprocessingConfig(

    remove_html_comments=True,

    normalize_markdown=True,

    normalize_urls_paths=True,

    remove_template_placeholders=True,

    compress_repeated_blocks=False,

    replace_fenced_code=True,

    normalize_whitespace=True,

    selective_code_compression=False,

)

```

**Size analysis** (`src/preprocessing/analyze_size.py`): same 200K training sample, measuring combined title + body character counts.

| Version | Total characters | Reduction vs raw | Mean chars | P99 chars |

|---|---:|---:|---:|---:|

| Raw | 251,265,035 | 0% | 1,256.33 | 12,439 |

| E2 | 129,450,078 | 48.4807% | 647.25 | 3,828.02 |

| E4 | 129,537,189 | 48.4460% | 647.69 | 3,836.01 |

| E5 | 130,897,738 | 47.9045% | 654.49 | 3,839.02 |

E2 removed only 87,111 more characters than E4; E5 retained 1,360,549 more characters than E4. Character reduction is **not** equivalent to token reduction, RAM reduction, or preservation of semantic information. Very long outliers remain (maximum processed combined text: 208,769 characters).

### 5.6 Materialized `structured_v1` dataset — Completed

**Implementation:** `src/preprocessing/build_dataset.py`; tests: `tests/test_build_dataset.py`.  

**Dependency:** `pyarrow==26.0.1`.

- Chunked CSV reads (`10,000` rows/chunk), body-only E4 transformation, and incremental Zstandard-compressed Parquet writes.

- Preserve original `title` and `labels` columns; save processed `body` separately.

- Validate row counts and schema; use temporary files to avoid exposing incomplete Parquet files during a split build.

- Record configuration, source/output hashes, character counts, output sizes, and elapsed times in `manifest.json`.

- Reject empty labels, preserve empty bodies, avoid overwriting existing versioned outputs, and allow temporary test paths outside the repository.

**Local generated artifacts (not tracked by Git):**

```text

data/processed/structured_v1/

    ├── train.parquet

    ├── validation.parquet

    └── manifest.json

```

| Measure | Train | Validation |

|---|---:|---:|

| Rows | 1,038,684 | 183,298 |

| Original body characters | 1,265,833,189 | 226,574,571 |

| Processed body characters | 626,208,289 | 109,328,175 |

| Body character reduction | 50.53% | 51.7474% |

| Parquet file bytes | 283,084,647 | 49,808,518 |

| Build time | 94.940 s | 16.457 s |

**Generated-file SHA-256:**

- Train Parquet: `3a1514f79aab4850e0d1c61885c8ba7da0835b92c03ced0064d5af9a46bfc958`

- Validation Parquet: `bad8a7b29c6719ee97cadf98a4b87220801485f1d759a1e9fa671a751b58ea1c`

The source CSV hashes in the generated manifest match the evaluation-protocol split fingerprints above. A separate post-build integrity check reported:

```text

train: SHA-256, row count, schema OK

validation: SHA-256, row count, schema OK

```

These checks verify artifact integrity and schema/row counts, not semantic equivalence or all possible leakage modes. The official test set was not preprocessed or used for tuning.

### Git closure of structured preprocessing

On `structured-preprocessing`, the following commits were created:

- `1832bb5` — `feat: add structured preprocessing and ablation experiments`

- `5f8f376` — `feat: build versioned structured parquet datasets`

The user confirmed the branch was merged into `main` and that the test suite passed. The Parquet files were confirmed ignored by Git. Exact post-merge HEAD and working-tree status should be checked before the next stage.

## 6. Historical Plan — Full Classical Baseline and Stage Closure (superseded)

1. Verify the current branch, clean working tree, and latest `main`; bring the existing `classical-baselines` branch up to date safely before modifying it. Do not discard divergent work.

2. Commit these updated documentation files on an appropriate branch and merge them as part of stage closure.

3. Review the current classical training scripts and memory budget before loading full train/validation data; avoid duplicating existing code or fitting TF-IDF on validation text.

4. Run a reproducible full-training classical baseline on `structured_v1` (1,038,684 training issues; 183,298 validation issues), recording Micro/Macro/Weighted F1, per-class metrics, runtime, and resource usage where measurable.

5. Decide whether a full-scale LinearSVC comparison is justified based on the existing sampled results and resource budget; preserve the 200K experiments as controlled references.

6. Document the final classical model-selection decision, preprocessing provenance, limitations, and commands. Only then begin the PyTorch neural baseline.

## 7. Original Roadmap (historical; see current status below)

- Neural NLP (`neural-baseline`): TextCNN or BiLSTM in PyTorch, with justified tokenization and sequence-length choices.

- Transformer fine-tuning (`transformer-finetuning`): DistilBERT; optional RoBERTa if compute permits.

- Final evaluation (`final-evaluation`): official test set, error analysis, latency/model size, model card.

- API and Docker (`api-serving`): tested FastAPI inference and packaging.

- Portfolio release (`portfolio-release`): reproducible documentation, CI, architecture/pipeline diagrams (separate from README when appropriate), polished project narrative.

- Optional retrieval/cloud work only if it materially improves the project.

## Open Risks and Interpretation Rules

- Validation metrics are not official competition test results. Never tune on the official test set.

- The selected E4 cleaning pipeline is a **classical TF-IDF preprocessing choice**; neural/transformer pipelines must evaluate their own tokenization and code/log preservation requirements.

- Code-block replacement may remove useful error context; E5 did not clearly resolve that trade-off in classical validation.

- Large body outliers remain, and character counts do not directly predict subword token counts.

- Exact-text deduplication and cross-split checks do not eliminate near-duplicate leakage, repository shortcuts, or noisy labels.

- Minority-class recall, especially for `question` and `documentation`, remains a priority.

- Versioned generated Parquet and JSON artifacts are local and Git-ignored. Reproduction depends on the tracked builder/configuration and access to the original prepared split files; preserve key numeric results in tracked documentation.

- Split-level atomic output does not imply a transaction across both splits; if a build is interrupted between splits, inspect artifacts before rerunning.

## Quality Commands

```bash

uv run ruff check --fix .

uv run ruff format .

uv run ruff check .

uv run ruff format --check .

uv run pytest -q

```

Verify Git state with `git branch --show-current`, `git status --short`, and `git log --oneline -5` before creating the next commit.

## Full-scale classical baselines — November 2026

Dataset: structured_v1 (E4, body-only preprocessing).

Training: 1,038,684 issues. Validation: 183,298 issues.

Features: title + body, TF-IDF unigrams, max_features=50,000.

| Model | Micro F1 | Macro F1 | Weighted F1 | Wall time | Peak RAM |

| --- | ---: | ---: | ---: | ---: | ---: |

| Logistic Regression | 0.842650 | 0.706409 | 0.835740 | 4m 04s | 3.47 GiB |

| LinearSVC | 0.839202 | 0.697714 | 0.831276 | 6m 15s | 3.56 GiB |

Selected classical baseline: **Logistic Regression**.

Compared with the 200K E4 experiment, Micro F1 improved from 0.833304 to 0.842650 and Macro F1 from 0.678782 to 0.706409.

Remaining weaknesses: documentation recall 0.4956 and question recall 0.3928 for Logistic Regression.

The official test split was held out during this classical stage. Subsequent TextCNN and DistilBERT work is recorded below.

## TextCNN Neural Baseline — Completed

See [`TEXTCNN_BASELINE_REPORT.md`](TEXTCNN_BASELINE_REPORT.md) for full validation metrics and error analysis. Best checkpoint: epoch 2, selected by validation Macro F1; Micro F1 `0.85265`, Macro F1 `0.72473`, Weighted F1 `0.84675`. Training stopped after four epochs.

## DistilBERT Fine-Tuning — Completed (Historical Record)

**Historical branch:** `transformer-finetuning`; implementation commit `6986b78`.

- Training: 1,038,684 issues; validation: 183,298 issues; structured E4 body-only preprocessing.

- Model: `distilbert/distilbert-base-uncased`; `[CLS] title [SEP] body [SEP]`.

- Token limit: 256; title budget: 48; body head/tail truncation: 75%/25%.

- Batch size: 8; gradient accumulation: 4; effective batch size: 32; learning rate: `2e-5`; weight decay: `0.01`; warmup ratio: `0.1`; seed: 42.

- Training on Kaggle CUDA; 3 epochs. Final recorded run resumed a prior checkpoint.

- Best checkpoint: epoch 3, selected using **validation Macro F1**, not the project's overall primary reporting metric (Micro F1).

| Epoch | Train loss | Validation Micro F1 | Validation Macro F1 |

|---|---:|---:|---:|

| 1 | 0.4125 | 0.8806 | 0.7734 |

| 2 | 0.3293 | 0.8852 | 0.7845 |

| 3 | 0.2821 | 0.8837 | 0.7884 |

See [`TRANSFORMER_EXPERIMENT.md`](TRANSFORMER_EXPERIMENT.md) for detailed training provenance, model selection, and limitations.

## Final Evaluation — Completed and Merged

**Completed branch:** `final-evaluation`, merged into `main` (confirmed base commit `09672bf`).

### Official Test Preparation and Evaluation

- The official test was processed using structured E4 preprocessing applied to the **body only**, with repeated-block compression disabled.

- **Input configuration:** Maximum length of 256 tokens, title budget of 48 tokens, body head/tail allocation of 75%/25%, evaluation batch size of 16, and random seed of 42.

- **Checkpoint selection:** The selected DistilBERT checkpoint came from epoch 3, chosen using validation Macro F1.

- **Inference environment:** The original full official-test inference was performed on Kaggle. The preserved results were audited locally without rerunning inference or retraining.

- **Label metadata:** The preserved checkpoint's `config.json` explicitly maps class IDs to `bug`, `documentation`, `feature`, and `question`, in that order. This verifies the declared label mapping but does not independently establish the provenance of the weights used during the Kaggle evaluation.

### Official Test Results

| Metric | Official Test Result |

|---|---:|

| Test samples | 142,320 |

| Correct predictions | 126,175 |

| Incorrect predictions | 16,145 |

| Accuracy / Micro F1 | 0.886558 |

| Macro F1 | 0.791175 |

| Weighted F1 | 0.884702 |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |

|---|---:|---:|---:|---:|

| bug | 0.910232 | 0.934649 | 0.922279 | 74,781 |

| documentation | 0.755988 | 0.671465 | 0.711224 | 6,252 |

| feature | 0.891017 | 0.891793 | 0.891405 | 52,797 |

| question | 0.700434 | 0.588810 | 0.639790 | 8,490 |

**Interpretation:** DistilBERT performs strongest on `bug` and `feature` issues. The `question` class has the lowest F1 score and recall, followed by `documentation` in overall F1.

### Implemented Evaluation and Audit Components

The final evaluation stage includes the following scripts and corresponding tests:

| Component | Implementation | Tests |

|---|---|---|

| Official test preparation | `src/preprocessing/prepare_official_test.py` | `tests/test_prepare_official_test.py` |

| Transformer evaluation | `src/models/evaluate_transformer.py` | `tests/test_evaluate_transformer.py` |

| Artifact integrity audit | `src/models/audit_final_evaluation.py` | `tests/test_audit_final_evaluation.py` |

**Evaluation hardening:**

- Explicit checkpoint label validation.

- Temporary-directory publication to avoid exposing incomplete evaluation results.

- Protection against accidentally overwriting existing evaluation outputs.

**Audit coverage:**

- Prediction row order and row counts.

- Valid class labels and probability distributions.

- Argmax, confidence, and correctness consistency.

- Independently recomputed evaluation metrics.

- Confusion matrix and classification report consistency.

- Row-by-row source ID and true-label alignment.

### Preserved Evaluation Artifacts

The final evaluation produced:

- `test_predictions.parquet`

- `test_metrics.json`

- `test_classification_report.csv`

- `test_confusion_matrix.csv`

Model weights, datasets, and generated evaluation artifacts are preserved outside the Git repository.

### Artifact Integrity Audit Results

| Audit Check | Result |

|---|---|

| Artifact integrity | PASSED |

| Source alignment | PASSED |

| Rows checked | 142,320 |

The audit confirms internal consistency and alignment with the saved processed official-test dataset.

**Limitation:** The audit does not cryptographically establish which checkpoint weights were used during the original Kaggle inference.

### Documentation

| Document | Purpose |

|---|---|

| `docs/FINAL_EVALUATION_REPORT.md` | Official-test results, class-level metrics, confusion matrix, and limitations |

| `docs/MODEL_CARD.md` | Intended use, training configuration, checkpoint metadata, evaluation, and deployment considerations |

| `README.md` | Public project overview, benchmark results, and links to detailed documentation |

### Final Evaluation Stage Closure

The final-evaluation documentation, including `docs/FINAL_EVALUATION_REPORT.md`, `docs/MODEL_CARD.md`, and `README.md`, was completed and merged into `main`. The API-serving branch was created from that completed stage. The latest confirmed `main` commit when API serving began was `09672bf`.

## 8. API Serving — Inference Engine Completed and Committed

**Branch:** `api-serving`  
**Commit:** `69fda88` — `feat: add reusable DistilBERT inference engine`

**Files:** `src/api/inference.py`; `tests/test_inference.py`.

- Reusable `IssueClassifier` loads the local DistilBERT tokenizer and sequence-classification checkpoint once, uses `local_files_only=True`, validates the expected label mapping, and switches the model to evaluation mode.
- CPU is the default inference device. The checkpoint is stored outside the Git repository, at `~/issue-intent-artifacts/models/distilbert/best/` in the local environment.
- Applies the selected E4 structured preprocessing **to the body only**, using `EXPERIMENTS["E4"]`, with repeated-block compression disabled.
- Reuses `encode_issue` for `[CLS] title [SEP] body [SEP]`, maximum 256 tokens, title budget 48, and 75%/25% body head/tail allocation.
- Runs under `torch.inference_mode()`, applies softmax, validates output shape and finite logits, and returns label, confidence, and class probabilities.
- Rejects an issue where both title and body are blank.

**Real local CPU checkpoint smoke test:**

```json
{
  "label": "bug",
  "confidence": 0.967095673084259,
  "probabilities": {
    "bug": 0.967095673084259,
    "documentation": 0.0010205748258158565,
    "feature": 0.0060602580197155476,
    "question": 0.025823505595326424
  }
}
```

This is one functional smoke test, not a measure of model accuracy or probability calibration.

## 9. FastAPI Prediction Service — Completed and Committed

**Commit:** `752dc87` — `feat: add FastAPI prediction service and request validation`

**Files:** `src/api/main.py`, `src/api/schemas.py`, `tests/test_api.py`, `tests/test_api_schemas.py`; dependency changes in `pyproject.toml` and `uv.lock`.

- Uses FastAPI lifespan to load one classifier instance at application startup and clear the reference on shutdown.
- `GET /health` reports healthy when the model is loaded and returns 503 when it is unavailable.
- `POST /predict` accepts a validated title/body request and returns label, confidence, and per-class probabilities.
- Pydantic input constraints: title up to 500 characters; optional body up to 20,000 characters; at least one non-whitespace field; unexpected fields forbidden.
- Tests cover schema validation, endpoint behavior, lifespan, unavailable-model behavior, and OpenAPI.
- Most recently reported gate before benchmark tests: **310 tests passed**, Ruff checks passed.
- Real Uvicorn run on `127.0.0.1:8000`: `GET /health` returned 200, `POST /predict` returned 200 with the real checkpoint, and graceful shutdown completed. `GET /` returned 404 as expected because no root endpoint is defined.
- The working tree was confirmed clean immediately after commit `752dc87`.

**Local run:**

```bash
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

**Example HTTP request:**

```bash
curl -sS -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Application crashes on startup",
    "body": "After updating the application, it crashes immediately with RuntimeError when I try to launch it."
  }'
```

## 10. API Performance Benchmark — Implemented and Measured; Closure Pending

**Proposed files:** `scripts/benchmark_api.py`, `scripts/__init__.py`, and `tests/test_benchmark_api.py`.

The user reported successfully executing the benchmark and saving `artifacts/benchmarks/api_baseline.json`. The benchmark performs sequential HTTP requests with short, medium, and long synthetic inputs and bounded concurrency scenarios. It measures end-to-end local HTTP latency, including preprocessing, tokenization, model inference, and response overhead. The first request in each scenario is **not** an independent measurement of model startup or a guaranteed cold inference.

**Environment:** Local WSL2, CPU inference, FastAPI/Uvicorn, one active serving process. Measurements recorded on 2026-11-15 at `11:46:38 UTC`.

### 10.1 Sequential HTTP latency

20 warm requests per payload, plus one initial request per payload.

| Input | Body characters | First request | Warm mean | Warm P50 | Warm P95 | Warm P99 |
|---|---:|---:|---:|---:|---:|---:|
| Short | 57 | 83.62 ms | 23.52 ms | 23.05 ms | 26.88 ms | 32.63 ms |
| Medium | 2,580 | 120.77 ms | 106.79 ms | 112.42 ms | 118.28 ms | 118.69 ms |
| Long | 12,900 | 119.18 ms | 104.72 ms | 110.87 ms | 119.45 ms | 125.74 ms |

The similar medium and long latencies are consistent with the model's 256-token input limit, but this benchmark does not isolate preprocessing or tokenization costs and does not establish causality. The input text is repeated synthetic material, not a representative workload.

### 10.2 Concurrent HTTP requests — Short payload

20 total requests per concurrency level.

| Concurrency | Requests | Elapsed time | Throughput | Mean latency | P50 | P95 | P99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 0.352 s | 56.86 req/s | 17.53 ms | 16.74 ms | 20.65 ms | 20.70 ms |
| 2 | 20 | 0.305 s | 65.59 req/s | 30.36 ms | 29.76 ms | 36.34 ms | 36.75 ms |
| 4 | 20 | 0.288 s | 69.39 req/s | 57.09 ms | 56.72 ms | 61.22 ms | 62.30 ms |

Higher concurrency increased throughput by approximately 22% from level 1 to level 4, while mean per-request latency increased by approximately 3.26 times. These short tests do not establish sustained throughput capacity or production SLAs.

### 10.3 Memory observation

- Active Uvicorn Python process: PID `14147` during observation, RSS `883,300 KiB` (approximately **863 MiB**).
- WSL2 memory snapshot: total `9.7 GiB`, available `5.2 GiB`, swap total `4.0 GiB`, swap used `0 B`.
- The measured RSS includes Python/runtime memory and is **not** the size of model weights alone.
- A separate old Uvicorn parent process and zombie child were observed; the active server on port 8000 was PID `14147`. This is an operational process-management observation, not evidence of multiple active listening servers.

### 10.4 Benchmark limitations and closure checklist

- Results are local, CPU-only, short-duration measurements with 20 samples per scenario.
- No independent model-startup-time benchmark, sustained load test, or production deployment measurement has been completed.
- P95/P99 from 20 observations are descriptive and unstable; avoid interpreting them as robust service-level guarantees.
- The benchmark tests were drafted, but their **full quality-gate result has not been reported**. Do not count them as passing yet.
- `scripts/__init__.py` was recommended for explicit package organization; its creation has not been independently confirmed.
- Confirm `artifacts/benchmarks/api_baseline.json` is ignored by Git. If needed, add `/artifacts/benchmarks/` to `.gitignore`.
- After the quality gate, review the staged diff and commit the benchmark and this documentation. No benchmark commit hash has been reported yet.

**Reproduction command (with the API already running):**

```bash
uv run python scripts/benchmark_api.py \
  --warm-requests 20 \
  --concurrent-requests 20 \
  --concurrency 1 2 4 \
  --output artifacts/benchmarks/api_baseline.json
```

## 11. Next Steps — Docker, CI, and Portfolio Release

1. Confirm `scripts/__init__.py`, the benchmark unit tests, and the ignored output path; run the full quality gate.
2. Commit the reproducible benchmark and updated progress documentation on `api-serving` after successful checks.
3. Add a Dockerfile and `.dockerignore`; keep the local checkpoint outside the image and use a runtime volume mount. Validate `/health` and `/predict` inside the container.
4. Add CI for lint, formatting, and tests that do not require downloading the large private checkpoint.
5. Update README and serving/deployment documentation with verified commands, limitations, and reproducible instructions.
6. Close and merge `api-serving` only after verification, then proceed to `portfolio-release` and separate Excalidraw architecture/pipeline diagrams.

**Quality commands (run in this order):**

```bash
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

**Git verification:**

```bash
git branch --show-current
git status --short
git log --oneline -5
```
