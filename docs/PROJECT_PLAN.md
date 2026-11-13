# issue-intent — Project Plan

**Updated:** 2026-11-13  

**Project:** `issue-intent`  

**Repository:** `git@github.com:mazen-alasas/issue-intent.git`

## 1. Project Overview

`issue-intent` is an NLP/ML engineering project that classifies real GitHub issues into four intent categories: `bug`, `feature`, `question`, and `documentation`. It uses the NLBSE 2023 Issue Report Classification dataset as its benchmark source.

The project emphasizes reproducible data preparation, leakage-aware evaluation, controlled classical/neural/transformer comparisons, resource-aware engineering, and a tested inference API. It is intended to provide a defensible technical portfolio and interview case study, not merely a high validation score.

## 2. Roadmap and Current State

| Stage | Branch | Deliverables | Status |

|---|---|---|---|

| Repository Foundation | `main` | `uv`, flat source layout, tooling, initial docs | Completed |

| Dataset Feasibility | `dataset-feasibility` | Official training-data audit and feasibility findings | Completed |

| Data Quality | `data-quality` | Exact-text deduplication, conflict removal, reports | Completed and merged |

| Evaluation Protocol | `evaluation-protocol` | Stratified split, integrity hashes, reusable metrics, dummy baseline | Completed; incorporated into subsequent development |

| Classical NLP | `classical-baselines` | Title-only, title+body, truncation, full-scale Logistic Regression/LinearSVC | Completed and merged |

| Structured Preprocessing | `structured-preprocessing` | E0–E5 ablations, size audit, `structured_v1` Parquet builder | Completed and merged into `main` |

| Neural NLP | `neural-baseline` | PyTorch TextCNN validation baseline | Completed and merged |

| Transformer Fine-Tuning | `transformer-finetuning` | DistilBERT training and validation | Training and implementation complete; docs/merge pending |

| Final Evaluation | `final-evaluation` | Integrate completed Kaggle official-test evaluation, error analysis, model card | Kaggle inference complete; repository integration pending |

| API & Docker | `api-serving` | FastAPI, inference tests, Docker | Planned |

| Portfolio Release | `portfolio-release` | CI, diagrams, reproducibility, documentation | Planned |

**Optional:** Similar Issue Retrieval and Cloud Deployment, only if justified by measured needs.

**Branch hygiene:** `structured-preprocessing` was merged into `main` after commits `1832bb5` and `5f8f376`. The existing `classical-baselines` branch may be behind `main`; inspect branch history and working-tree state before merging or rebasing. Do not assume a specific current branch from documentation alone.

## 3. Engineering Principles

- Python 3.11, `uv` for environment/dependencies, and a flat `src/` structure.

- English, meaningful docstrings; modular scripts and focused unit tests.

- Notebooks for EDA/audits; scripts for repeatable preparation, training, and evaluation.

- Immutable raw data; generated datasets, metrics JSON, and model artifacts kept outside Git.

- Small, controlled changes with tests, linting, reproducible settings, and descriptive commits.

- Preserve baseline implementations and recorded metrics when introducing new experiments.

- Prefer measurable engineering trade-offs over additional complexity without evidence.

- Avoid unnecessary infrastructure; adapt batch size, memory use, and training scale to available hardware.

- Maintain documentation in parallel with implementation and distinguish observed results from plans.

**Standard checks:**

```bash

uv run ruff check --fix .

uv run ruff format .

uv run ruff check .

uv run ruff format --check .

uv run pytest -q

```

## 4. Data and Evaluation Contract

### Dataset

- Original official NLBSE'23 training: **1,275,881** issues.

- Prepared training pool after exact-text deduplication/conflict removal: **1,221,982** issues.

- Development split: **1,038,684** training and **183,298** validation issues, stratified 85/15 with `random_state=42`.

- Official test: **142,320** issues; reserved for final evaluation.

- Labels: `bug`, `documentation`, `feature`, `question` (use the stored label order for confusion matrices and consistent reporting).

- Initial model inputs: title and body only; do not use `author_association` or `id` as shortcut features.

### Data quality policy

1\. Preserve the original files unchanged.

2\. Normalize missing title/body text to empty strings in the preparation pipeline.

3\. Retain one representative for each exact title–body group with a consistent label.

4\. Exclude all rows of an exact-text group if the group has conflicting labels.

5\. Do not deduplicate on issue `id` alone; its meaning and collisions need separate analysis.

6\. Check exact cross-split title–body overlap; investigate near duplicates and repository shortcuts later.

7\. Keep the official test set unchanged and out of development-time fitting, model selection, and preprocessing decisions.

### Evaluation protocol

- **Primary:** Micro F1.

- **Secondary:** Macro F1, Weighted F1, Accuracy, per-class Precision/Recall/F1/Support, and confusion matrix.

- **Operational:** Fit/preprocessing/prediction time, memory usage where measurable, model size, and serving latency.

- **Sampling:** Fixed stratified, nested subsets for controlled learning-curve studies; do not compare independent random samples as if they were paired.

- **Leakage:** Fit learned vectorizers/tokenizers/statistics on training data only. Deterministic text transformations may be applied to validation without learning from it.

- **Reporting:** Clearly label sample size, input representation, truncation, preprocessing configuration, validation size, seed, and hardware conditions.

- **Final test:** Evaluate selected models only after model selection is complete. Report official-test metrics separately from validation results.

### Reference dummy baseline

The majority-class `bug` predictor achieved validation Micro F1 `0.530508`, Macro F1 `0.173311`, and Weighted F1 `0.367771`. This is the minimum reference for meaningful modeling.

## 5. Completed Work and Evidence

### 5.1 Dataset feasibility and quality

**Implemented:** `notebooks/01_dataset_feasibility.ipynb`, `src/data/prepare.py`, `tests/test_prepare.py`.

| Measure | Verified result |

|---|---:|

| Original train rows | 1,275,881 |

| Unique title–body combinations | 1,222,422 |

| Conflicting text groups | 440 |

| Conflicting-group rows removed | 4,146 |

| Consistent-label duplicates removed | 49,753 |

| Prepared rows | 1,221,982 |

| Remaining exact title–body duplicates | 0 |

Prepared label counts: bug `648,270`; feature `444,745`; question `74,474`; documentation `54,493`. Exact-text preparation does not prove semantic label quality.

### 5.2 Evaluation protocol

**Implemented:** `src/data/split.py`, `src/models/evaluation.py`, `src/models/dummy_baseline.py` and corresponding tests.

- Stratified train/validation split with row preservation and exact-text overlap check (zero overlap).

- `split_report.json` records row counts, class distributions, and SHA-256 source/output hashes.

- `dummy_baseline_metrics.json` records the majority-class reference.

**Train CSV SHA-256:** `41d703815435c2018a93bb9021e69083ce49ebe9b699ecb1aff03f0f7500055d`  

**Validation CSV SHA-256:** `7946749613bda611b96cb4871aaa38045195fa980744b1112fa2f9f9bd2f9b23`

### 5.3 Classical sampled experiments

**Implemented:** `src/data/sampling.py`, `src/models/train_classical.py`, `src/models/learning_curve.py`, `src/models/train_title_body.py`, body structure EDA notebook, and experiment-specific tests/scripts.

**Reference configuration:** TF-IDF word unigrams, `max_features=50_000`, `min_df=2`, `dtype=float32`; Logistic Regression `max_iter=1000`, `random_state=42`; fixed 183,298-row validation set.

| Experiment | Training size | Micro F1 | Macro F1 |

|---|---:|---:|---:|

| Title-only Logistic Regression | 50K nested | 0.776817 | 0.592338 |

| Title-only Logistic Regression | 100K nested | 0.785830 | 0.609182 |

| Title-only Logistic Regression | 200K nested | 0.792704 | 0.618379 |

| Title + raw full body | 200K | 0.832186 | 0.676879 |

| Title + body truncated to 1K chars | 200K | 0.831853 | Not recorded here |

| Title + body truncated to 2K chars | 200K | 0.832513 | Not recorded here |

| Title + body truncated to 4K chars | 200K | 0.832633 | Not recorded here |

Initial independent-sample LinearSVC results: Micro F1 `0.766850` at 50K and `0.779010` at 100K; these should not be conflated with nested sample runs.

**Finding:** Body text substantially improves classification relative to title alone, particularly `question`, but minority-class recall remains a concern. The best truncation choice is not obvious from character length alone. Full-body training at 200K was materially slower than title-only training.

### 5.4 Structured preprocessing and ablations

**Implemented:** `src/preprocessing/structured.py`, `src/models/structured_ablation_experiment.py`, `src/preprocessing/analyze_size.py`, `notebooks/03_structured_text_audit.ipynb`, and tests.

All controlled structured comparisons use 200K training issues and the same full validation split. Preprocessing changes the **body only**.

| ID | Controlled representation | Micro F1 | Macro F1 |

|---|---|---:|---:|

| E0 | Raw title + body | 0.832186 | 0.676879 |

| E1 | Minimal cleaning | 0.831722 | 0.675462 |

| E2 | Full structured, fenced code replaced | **0.833441** | 0.678902 |

| E3 | Structured, fenced code retained | 0.830604 | 0.673184 |

| E4 | E2 without repeated-line compression | 0.833304 | 0.678782 |

| E5 | E4 with selected diagnostic signals retained from fenced code | 0.833266 | **0.678985** |

**Selection:** E4 is the practical classical preprocessing baseline, not the highest-scoring run on Micro F1. E2 scored slightly higher but added repeated-line compression with negligible character savings; E5 retained additional diagnostic content but did not improve Micro F1. Do not present these tiny score differences as statistically significant.

**Size analysis (200K, title + body characters):**

| Input | Characters | Reduction vs raw |

|---|---:|---:|

| Raw | 251,265,035 | 0% |

| E2 | 129,450,078 | 48.4807% |

| E4 | 129,537,189 | 48.4460% |

| E5 | 130,897,738 | 47.9045% |

The chosen E4 configuration is:

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

**Caveat:** These are classical TF-IDF experiments. Transformer/neural tokenization and handling of logs/code should be evaluated independently; removing characters is not proof of semantic preservation or proportional memory savings.

### 5.5 Versioned structured dataset

**Implemented:** `src/preprocessing/build_dataset.py`, `tests/test_build_dataset.py`, `pyarrow==26.0.1`.

The builder reads source CSVs in 10K-row chunks, preserves `title` and `labels`, applies E4 to `body`, writes compressed Parquet incrementally, and generates a manifest with configuration, counts, source/output SHA-256, and size/timing statistics. It rejects empty labels and protects existing outputs from accidental replacement.

**Generated locally (Git-ignored):**

```text

data/processed/structured_v1/

├── train.parquet

├── validation.parquet

└── manifest.json

```

| Measure | Train | Validation |

|---|---:|---:|

| Rows | 1,038,684 | 183,298 |

| Body characters before | 1,265,833,189 | 226,574,571 |

| Body characters after | 626,208,289 | 109,328,175 |

| Body reduction | 50.53% | 51.7474% |

| Output bytes | 283,084,647 | 49,808,518 |

**Output SHA-256:**

- Train: `3a1514f79aab4850e0d1c61885c8ba7da0835b92c03ced0064d5af9a46bfc958`

- Validation: `bad8a7b29c6719ee97cadf98a4b87220801485f1d759a1e9fa671a751b58ea1c`

Post-build SHA-256, row count, and schema verification passed for both files. The official test set was not used.

**Git history:** `1832bb5` (structured preprocessing and ablations) and `5f8f376` (versioned Parquet builder); the branch was reported merged into `main`. All tests were reported passing before the merge. The exact final test count and post-merge HEAD are not independently recorded here.

## 6. Historical Milestone — Classical Baselines (completed; plan retained for provenance)

**Goal:** Establish a reproducible, resource-aware full-training classical reference using the selected `structured_v1` dataset, then make an explicit model-selection decision.

### Implementation plan

1\. Verify `main`, `classical-baselines`, and working-tree state. Safely incorporate completed structured preprocessing changes into the branch used for the next experiment.

2\. Update and commit the tracked documentation so experiment decisions remain available even when local JSON/Parquet files are absent.

3\. Inspect existing training scripts and estimated memory requirements before attempting a 1,038,684-row run. Prefer reuse over rewriting the pipeline.

4\. Read `structured_v1/train.parquet` and `validation.parquet`; construct title + processed body for TF-IDF without re-running E4 on already processed text.

5\. Fit vectorizer/model using training data only. Keep the full fixed validation set; save metrics, confusion matrix, per-class results, timings, and model provenance.

6\. Compare full-scale results with 200K E4 (`Micro F1 0.833304`, `Macro F1 0.678782`). Differences may reflect training size; avoid attributing them solely to preprocessing.

7\. Evaluate whether a full-scale LinearSVC comparison adds enough value to justify compute. If run, hold input representation, split, and evaluation protocol fixed.

8\. Record limitations, final classical selection, and reproducibility commands; then close the classical stage.

### Exit criteria

- No official-test usage or train/validation leakage.

- Full-training run completed or a documented, justified resource-constrained alternative.

- Micro/Macro/Weighted F1, per-class metrics, and confusion matrix recorded.

- Training/prediction cost and input configuration documented.

- Tests and Ruff pass; code and documentation committed.

## 7. Original Neural and Transformer Plans (completed; historical)

### Neural NLP (`neural-baseline`)

- Start with a practical PyTorch TextCNN or BiLSTM, selected for the available GPU/RAM budget.

- Use a training-only vocabulary or tokenizer as appropriate; compare title/body representations and sequence limits.

- Preserve useful error/log context when justified rather than inheriting the classical `CODE` replacement uncritically.

- Report class-level results, training/inference cost, and reproducibility details against the classical reference.

### Transformer Fine-Tuning (`transformer-finetuning`)

- Begin with DistilBERT, using a measured maximum token length, batching/gradient accumulation, and memory-aware settings.

- Optional RoBERTa only if compute and the observed gain justify it.

- Keep official-test data untouched until final selection.

### Final Evaluation (`final-evaluation`)

- Run the selected final model(s) on the official test set once the development protocol is frozen.

- Perform per-class error analysis, confusion analysis, runtime/model-size measurements, and write a model card.

- Clearly separate competition benchmark numbers from project-measured results.

### API and Portfolio (`api-serving`, `portfolio-release`)

- FastAPI inference endpoint, input validation, predictable preprocessing/tokenization, and tests.

- Docker packaging and minimal CI for quality checks.

- Clear README, experiment summary, reproducibility guide, and architecture/data-flow diagrams (Excalidraw or exported images where useful).

- Include honest trade-offs, failures, dataset limitations, and deployment constraints in the portfolio narrative.

## 8. Risks and Decision Rules

- **Imbalanced labels:** Monitor `question` and `documentation` recall/F1, not just Micro F1.

- **Label ambiguity:** Exact conflicts were removed, but semantic errors and near duplicates remain.

- **Data leakage:** Exact cross-split text overlap is zero; repository and near-duplicate leakage need further audit.

- **Long inputs:** Character truncation and token truncation are different; some processed issues remain extremely long.

- **Code and logs:** Fenced blocks may contain diagnostic content. Classical ablation results do not establish the best neural/transformer strategy.

- **Memory:** Use resource-aware profiling; avoid assuming that a 200K model scales linearly to >1M rows.

- **Artifact persistence:** Generated datasets and experiment JSON are Git-ignored. Commit reproducible code, key results, configuration, and provenance to documentation; never commit large Parquet files by accident.

- **Reproducibility:** Use stable seeds, documented package versions, source hashes, and clearly identified sample/split sizes.

- **Scope control:** Do not add retrieval, cloud infrastructure, or complex orchestration without a demonstrated project need.

## 9. Relevant Paths

| Purpose | Path |

|---|---|

| Data preparation | `src/data/prepare.py` |

| Train/validation split | `src/data/split.py` |

| Stratified sampling | `src/data/sampling.py` |

| Metrics | `src/models/evaluation.py` |

| Majority baseline | `src/models/dummy_baseline.py` |

| Classical title-only models | `src/models/train_classical.py` |

| Learning curve | `src/models/learning_curve.py` |

| Title + body baseline | `src/models/train_title_body.py` |

| Structured transformations | `src/preprocessing/structured.py` |

| Structured ablations | `src/models/structured_ablation_experiment.py` |

| Input size analysis | `src/preprocessing/analyze_size.py` |

| Versioned Parquet builder | `src/preprocessing/build_dataset.py` |

| Prepared CSVs | `data/processed/train.csv`, `data/processed/validation.csv` |

| Versioned dataset | `data/processed/structured_v1/` |

| Dataset provenance | `data/processed/structured_v1/manifest.json` |

| Progress log | `docs/PROGRESS.md` |

| Roadmap and policies | `docs/PROJECT_PLAN.md` |

## 10. Commands for Stage Transitions

```bash

# Inspect current Git state before switching or merging.

git branch --show-current

git status --short

git log --oneline -5

# Quality gate.

uv run ruff check --fix .

uv run ruff format .

uv run ruff check .

uv run ruff format --check .

uv run pytest -q

# Existing structured dataset builder (do not rerun if v1 exists).

# uv run python -m src.preprocessing.build_dataset

```

The next phase begins with full-scale classical training and comparison, **not** a repeat of the completed E0–E5 preprocessing ablations.

## Full-scale classical baselines — November 2026

Dataset: structured_v1 (E4, body-only preprocessing).

Training: 1,038,684 issues. Validation: 183,298 issues.

Features: title + body, TF-IDF unigrams, max_features=50,000.

| Model | Micro F1 | Macro F1 | Weighted F1 | Wall time | Peak RAM |

| --- | ---: | ---: | ---: | ---: | ---: |

| Logistic Regression | 0.842650 | 0.706409 | 0.835740 | 4m 04s | 3.47 GiB |

| LinearSVC | 0.839202 | 0.697714 | 0.831276 | 6m 15s | 3.56 GiB |

Selected classical baseline: Logistic Regression.

Compared with the 200K E4 experiment, Micro F1 improved from

0.833304 to 0.842650 and Macro F1 from 0.678782 to 0.706409.

Remaining weaknesses: documentation recall 0.4956 and

question recall 0.3928 for Logistic Regression.

The official test split was held out throughout classical and neural model selection. Subsequent transformer and official-test work is recorded below.
## 11. Current Results and Remaining Roadmap — 2026-11-13

### Completed development-stage comparison (same 183,298-row validation set)

| Model | Validation Micro F1 | Validation Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression (full scale) | 0.842650 | 0.706409 |
| TextCNN (best checkpoint) | 0.852650 | 0.724730 |
| DistilBERT (selected epoch 3) | 0.883665 | 0.788362 |

The project's overall evaluation protocol reports Micro F1 as primary. DistilBERT checkpoint selection specifically used validation Macro F1 to account for class imbalance. This distinction is recorded rather than retroactively changing the protocol.

### Transformer stage closure

- Implementation committed as `6986b78` on `transformer-finetuning`.
- Record training results and configuration in [`TRANSFORMER_EXPERIMENT.md`](TRANSFORMER_EXPERIMENT.md).
- Commit documentation and merge after review; preserve best model checkpoint separately from Git.

### Final evaluation — next branch

Official-test inference has already been performed on Kaggle for 142,320 issues, producing Micro F1 `0.886558`, Macro F1 `0.791175`, and Weighted F1 `0.884702`. These are **official-test** metrics, not validation metrics. Repository integration, metric reproduction from saved predictions, detailed error analysis, model card, and model size/latency reporting remain to be completed. Do not tune on test results.

### Serving and release

- `api-serving`: load frozen model and tokenizer, reproduce E4 body preprocessing and token encoding, build FastAPI endpoints and integration tests, measure serving latency, add Docker packaging.
- `portfolio-release`: CI, README, reproducibility guide, Excalidraw architecture/pipeline diagrams, limitations, and CV/portfolio narrative.

