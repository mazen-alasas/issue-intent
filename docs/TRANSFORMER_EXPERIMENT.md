# DistilBERT Fine-Tuning — Validation Report

## Experiment

- Dataset: NLBSE 2023 GitHub Issue Classification
- Training rows: 1,038,684; validation rows: 183,298
- Classes: bug, documentation, feature, question
- Model: `distilbert/distilbert-base-uncased`
- Input: original title + E4-processed body (`structured_v1`)
- Encoding: `[CLS] title [SEP] body [SEP]`
- Max sequence length: 256 tokens; title budget: 48 tokens
- Long-body token truncation: 75% head / 25% tail
- Training data: globally shuffled Parquet; validation data: unshuffled Parquet
- Hardware: Kaggle CUDA (exact GPU configuration not recorded in the supplied run metadata)

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 8 |
| Gradient accumulation | 4 |
| Effective batch size | 32 (except possible final partial batch) |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Warmup ratio | `0.1` |
| Early stopping patience | 2 |
| Random seed | 42 |
| Device | CUDA |
| Model selection | Highest validation Macro F1 |

The final `run_config.json` records `resume=true` and `stop_after_epoch=3`; training was resumed from a previous checkpoint. It does not by itself establish a single uninterrupted training run.

## Validation Results

| Epoch | Training Loss | Micro F1 | Macro F1 |
|---|---:|---:|---:|
| 1 | 0.4125 | 0.8806 | 0.7734 |
| 2 | 0.3293 | **0.8852** | 0.7845 |
| 3 | **0.2821** | 0.8837 | **0.7884** |

**Selected checkpoint: epoch 3**, highest validation Macro F1 (`0.7883622793213235`). Epoch 2 achieved the highest Micro F1. The overall project protocol uses Micro F1 as its primary reporting metric, while the transformer checkpoint was explicitly selected using Macro F1; both facts should remain visible.

## Validation Comparison

| Model | Micro F1 | Macro F1 |
|---|---:|---:|
| Full-scale TF-IDF + Logistic Regression | 0.842650 | 0.706409 |
| TextCNN | 0.852650 | 0.724730 |
| DistilBERT (selected epoch 3) | **0.883665** | **0.788362** |

## Interpretation and Limitations

Training loss decreased across all three epochs, while validation Micro F1 peaked at epoch 2. This alone does not prove overfitting. Minority-class performance and semantic label ambiguity remain important concerns. Token truncation and E4 fenced-code replacement may discard useful diagnostic context. Exact-text train/validation overlap checks do not rule out near-duplicate or repository-related leakage.

## Implementation and Reproducibility

- `src/models/train_transformer.py`
- `src/data/transformer_dataset.py`
- `src/data/transformer_streaming.py`
- `src/data/shuffle_transformer_train.py`
- `src/preprocessing/analyze_transformer_lengths.py`
- `src/preprocessing/structured.py`

Training configuration and per-epoch metrics were preserved in `run_config.json` and `history.json` outside Git. The source implementation was committed as `6986b78` on `transformer-finetuning`. The trained best checkpoint and tokenizer must be separately preserved for API serving; they are not included in the Kaggle evaluation backup ZIP.

## Evaluation Boundary

This report covers development and validation. The official 142,320-row test set was evaluated only after selecting the final checkpoint. Official-test results belong in the separate `final-evaluation` report, not this validation report. No further test-driven tuning is planned.
