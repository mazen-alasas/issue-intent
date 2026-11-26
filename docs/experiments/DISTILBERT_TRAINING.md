# DistilBERT Training — Reproducibility Guide

This document describes the data preparation, preprocessing, shuffling, and training workflow for the DistilBERT model used in the Issue Intent classification project.

It provides the commands and configuration required to reproduce the training pipeline from the repository root.

The project uses the NLBSE 2023 Issue Report Classification dataset with four target classes:

- `bug`
- `documentation`
- `feature`
- `question`

---

## 1. Model Overview

The model is based on:

- **Base model:** `distilbert/distilbert-base-uncased`
- **Task:** Multi-class sequence classification
- **Input:** GitHub issue title and body
- **Number of classes:** 4
- **Primary evaluation metric:** Micro F1
- **Additional metrics:** Macro F1, Weighted F1, and per-class metrics

The training implementation is located in:

```text
src/models/train_transformer.py
```

The model uses a title-aware input construction strategy with a configurable title token budget.

---

## 2. Dataset Preparation

The training pipeline expects the prepared dataset to contain the following columns:

| Column | Description |
|---|---|
| `title` | GitHub issue title |
| `body` | GitHub issue description |
| `labels` | Target classification label |

### 2.1 Prepare the dataset

The preparation stage processes the original NLBSE 2023 dataset and applies the project's data-quality and deduplication rules.

Run:

```bash
uv run python -m src.data.prepare
```

The prepared dataset is saved under `data/processed/`.

The preparation process handles duplicate records, conflicting labels, and missing values according to the project's data preparation policy.

### 2.2 Create the train/validation split

Run:

```bash
uv run python -m src.data.split
```

The dataset is split using stratified sampling with a fixed random state.

| Parameter | Value |
|---|---:|
| Training size | 85% |
| Validation size | 15% |
| Random state | 42 |
| Stratification | Enabled |

The resulting datasets are saved as:

```text
data/processed/
├── train.csv
├── validation.csv
└── split_report.json
```

The recorded split contains 1,038,684 training rows and 183,298 validation rows, with no exact text overlap between the two splits.

### 2.3 Structured preprocessing

The transformer training dataset uses the `structured_v1` preprocessing pipeline.

Run:

```bash
python -m src.preprocessing.build_dataset
```

This generates:

```text
data/processed/structured_v1/
├── train.parquet
├── validation.parquet
└── manifest.json
```

The implementation is located in:

```text
src/preprocessing/build_dataset.py
```

The preprocessing implementation and experiment configurations are located in:

```text
src/preprocessing/structured.py
src/preprocessing/experiment_configs.py
```

The selected configuration is **E4**.

#### E4 configuration

| Transformation | Enabled |
|---|---|
| Remove HTML comments | Yes |
| Normalize Markdown | Yes |
| Normalize URLs and absolute paths | Yes |
| Remove empty template placeholders | Yes |
| Compress repeated lines | No |
| Replace fenced code blocks | Yes |
| Normalize whitespace | Yes |
| Selective code compression | No |

Preprocessing is applied to the issue body only. Titles remain unchanged.

Fenced code blocks are replaced with the `CODE` marker under this configuration.

The preprocessing configuration is recorded in `manifest.json`, together with source and output fingerprints.

### Recorded preprocessing statistics

| Dataset | Rows | Body character reduction |
|---|---:|---:|
| Training | 1,038,684 | 50.53% |
| Validation | 183,298 | 51.75% |

These values describe the recorded dataset artifacts, not guaranteed values for a different source dataset or preprocessing implementation.

---

## 3. Training Data Shuffling

The training dataset is shuffled before transformer training using DuckDB.

Implementation:

```text
src/data/shuffle_transformer_train.py
```

Run:

```bash
uv run python -m src.data.shuffle_transformer_train \
    --input data/processed/structured_v1/train.parquet \
    --output data/processed/transformer/train_shuffled.parquet \
    --seed 42 \
    --memory-limit 2GB \
    --row-group-size 10000
```

The shuffle operation uses a seeded ordering to produce a reproducible row sequence for the same input dataset and compatible DuckDB version.

After shuffling, the script verifies data integrity by comparing:

- Total row count.
- Label distribution.
- Aggregate SHA-256 digest sum.
- Aggregate SHA-256 digest XOR.

The output is accepted only when the input and output fingerprints match.

The shuffle report is saved alongside the generated Parquet file:

```text
train_shuffled.shuffle_report.json
```

The report includes the shuffle configuration, DuckDB version, input fingerprint, output fingerprint, and verification status.

---

## 4. Training Configuration

The training implementation is:

```text
src/models/train_transformer.py
```

The following configuration was used for the recorded full training run:

| Parameter | Value |
|---|---:|
| Model | `distilbert/distilbert-base-uncased` |
| Epochs | 3 |
| Batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 32 |
| Maximum sequence length | 256 |
| Title token budget | 48 |
| Learning rate | 0.00002 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Early stopping patience | 2 |
| Random seed | 42 |
| Smoke test | Disabled |

The effective batch size is calculated as:

```text
batch_size × gradient_accumulation
= 8 × 4
= 32
```

The training process evaluates the model on the validation set after each epoch and selects the best checkpoint according to validation Macro F1.

---

## 5. Running Training

Run the following command from the repository root after preparing and shuffling the datasets:

```bash
python -m src.models.train_transformer \
    --train data/processed/transformer/train_shuffled.parquet \
    --validation data/processed/structured_v1/validation.parquet \
    --output-dir data/processed/transformer/distilbert \
    --epochs 3 \
    --batch-size 8 \
    --grad-accum 4 \
    --max-length 256 \
    --title-budget 48 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --patience 2 \
    --seed 42
```

This command starts a fresh training run.

**Important:** Do not add `--resume` when starting a new experiment.

### 5.1 Resume an interrupted run

To resume training from a previously saved checkpoint, use the same output directory and add:

```bash
--resume
```

For example:

```bash
python -m src.models.train_transformer \
    --train data/processed/transformer/train_shuffled.parquet \
    --validation data/processed/structured_v1/validation.parquet \
    --output-dir data/processed/transformer/distilbert \
    --epochs 3 \
    --batch-size 8 \
    --grad-accum 4 \
    --max-length 256 \
    --title-budget 48 \
    --lr 2e-5 \
    --weight-decay 0.01 \
    --warmup-ratio 0.1 \
    --patience 2 \
    --seed 42 \
    --resume
```

A resumed run should use the same training configuration, input datasets, and compatible software environment as the original run.

The `--resume` option restores the training state from `last.pt` in the specified output directory. The best inference checkpoint under `best/` is not used as a training-resume checkpoint.

The training configuration and dataset inputs should remain unchanged when resuming an existing run.

The training checkpoint stores:

- Model state.
- Optimizer state.
- Learning-rate scheduler state.
- Mixed-precision scaler state, when applicable.
- Completed epoch.
- Best validation Macro F1.
- Early stopping state.
- Training history.
- Random number generator states.

The checkpoint is saved after every completed epoch.

### 5.2 Stop after a specific epoch

The optional argument:

```bash
--stop-after-epoch 1
```

can be used to stop after a specified number of completed epochs.

This is useful for controlled experiments and staged training.

---

## 6. Training Artifacts

The training output directory contains the following artifacts:

```text
distilbert/
├── best/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
├── last.pt
├── history.json
└── run_config.json
```

| Artifact | Purpose |
|---|---|
| `best/` | Best validation checkpoint for inference |
| `last.pt` | Training checkpoint used for resuming |
| `history.json` | Epoch-level training and validation metrics |
| `run_config.json` | Training configuration and execution status |

The run configuration records the current run status:

- `running`
- `completed`
- `stopped`
- `early_stopped`
- `failed`

The configuration is updated during training, allowing the execution state to be inspected even if training does not finish normally.

### Important distinction

The `best/` directory is intended for model inference and evaluation.

The `last.pt` checkpoint is intended for training resumption.

Saving the best inference model does not replace the need to retain the training checkpoint.

---

## 7. Recorded Training Results

The recorded full training run achieved its best validation Macro F1 at epoch 3.

### Validation history

| Epoch | Train Loss | Validation Loss | Micro F1 | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.41255 | 0.35825 | 0.88063 | 0.77339 | 0.87710 |
| 2 | 0.32933 | 0.34179 | 0.88515 | 0.78453 | 0.88201 |
| 3 | 0.28209 | 0.35377 | 0.88367 | **0.78836** | 0.88184 |

### Best validation result

| Metric | Score |
|---|---:|
| Micro F1 | 88.37% |
| Macro F1 | **78.84%** |
| Weighted F1 | 88.18% |

The best checkpoint was selected using validation Macro F1.

### Official evaluation

The recorded official evaluation results were:

| Metric | Score |
|---|---:|
| Micro F1 | 88.66% |
| Macro F1 | 79.12% |

The official evaluation results are separate from the validation metrics and should not be interpreted as validation performance.

---

## 8. Artifact Storage

Large datasets and model checkpoints are not tracked in Git.

The following artifacts should be retained outside the repository:

```text
data/
├── processed/
│   ├── train.csv
│   ├── validation.csv
│   ├── split_report.json
│   ├── preparation_report.json
│   ├── structured_v1/
│   │   ├── train.parquet
│   │   ├── validation.parquet
│   │   └── manifest.json
│   └── transformer/
│       ├── train_shuffled.parquet
│       ├── train_shuffled.shuffle_report.json
│       └── distilbert/
│           ├── best/
│           ├── last.pt
│           ├── history.json
│           └── run_config.json
```

The exact storage location may differ between local development and cloud training environments.

The repository should contain the code and documentation required to regenerate these artifacts, not the large generated files themselves.

---

## 9. Reproducibility Considerations

The training pipeline uses explicit random seeds for dataset splitting, data shuffling, model initialization, and transformer data loading.

Training checkpoints preserve the model, optimizer, scheduler, and random number generator states to support resuming interrupted training runs.

For reproducible experiments, keep the following information:

- Git commit hash.
- Python and dependency versions.
- CUDA and hardware information.
- Dataset fingerprints.
- Training configuration.
- Training and validation metrics.

Dataset manifests and shuffle reports provide fingerprints for verifying the generated datasets.

Exact numerical reproducibility may vary across hardware, CUDA versions, library versions, and execution environments.

---

## 10. Software Environment

The project uses `uv` for dependency and environment management.

Install the project dependencies:

```bash
uv sync
```

Run the pipeline using:

```bash
uv run python -m src.models.train_transformer
```

The training pipeline uses PyTorch and supports CUDA acceleration when a compatible NVIDIA GPU environment is available.

Dependency versions are maintained in `pyproject.toml` and `uv.lock`.

---

## 11. End-to-End Workflow

The complete workflow is:

```text
NLBSE Raw Dataset
        |
        v
Dataset Preparation
        |
        v
Train / Validation Split
        |
        v
Structured Preprocessing (E4)
        |
        v
Parquet Datasets
        |
        v
Shuffle Training Dataset
        |
        v
Fingerprint Verification
        |
        v
DistilBERT Training
        |
        v
Validation Evaluation
        |
        v
Best Model + Training Checkpoints
```

Run the stages in order:

1. Prepare the raw dataset.
2. Generate the train/validation split.
3. Build the `structured_v1` datasets.
4. Shuffle the training Parquet dataset.
5. Verify the shuffle report.
6. Train DistilBERT.
7. Retain the best model and training checkpoints.
8. Evaluate the model using the appropriate evaluation dataset.

This workflow provides a documented path from the prepared source data to the final trained DistilBERT model.