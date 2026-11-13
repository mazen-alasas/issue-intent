# TextCNN Neural Baseline — Validation Report

## Experiment

- Dataset: NLBSE 2023 GitHub Issue Classification
- Training rows: 1,038,684
- Validation rows: 183,298
- Classes: bug, documentation, feature, question
- Model: TextCNN
- Vocabulary size: 50,000
- Maximum sequence length: 256 tokens
- Title budget: 32 tokens
- Body truncation: 75% head / 25% tail
- Batch size: 64
- Learning rate: 0.001
- Maximum epochs: 8
- Early stopping patience: 2
- Random seed: 42
- Device: CUDA

## Best Checkpoint

Selected using validation Macro F1. Training stopped after four epochs.

| Metric | Result |
|---|---:|
| Best epoch | 2 |
| Micro F1 | 0.85265 |
| Macro F1 | 0.72473 |
| Weighted F1 | 0.84675 |

## Classical Baseline Comparison

| Metric | Logistic Regression | TextCNN |
|---|---:|---:|
| Micro F1 | 0.84265 | 0.85265 |
| Macro F1 | 0.70641 | 0.72473 |

## Per-Class Performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| bug | 0.8684 | 0.9317 | 0.8990 |
| documentation | 0.7652 | 0.4972 | 0.6027 |
| feature | 0.8545 | 0.8472 | 0.8508 |
| question | 0.6793 | 0.4570 | 0.5464 |

## Error Analysis

The independent evaluation produced 27,007 misclassified examples across 183,298 validation rows.

Largest confusion pairs:

- feature → bug: 8,254
- bug → feature: 5,271
- question → bug: 4,069
- documentation → feature: 2,445
- question → feature: 1,905
- documentation → bug: 1,407

Manual inspection of sampled errors suggests semantic overlap, possible label ambiguity, and sensitivity to surface-level wording. These are hypotheses rather than established causal explanations.

## Truncation Analysis

Actual tokenization and encoding were used.

| Class | Any Truncation Rate |
|---|---:|
| bug | 14.02% |
| documentation | 5.52% |
| feature | 5.68% |
| question | 13.40% |
| Overall | 10.57% |

Title truncation overall: 0.37%. Body truncation overall: 10.31%. Truncation rate among 100 sampled errors: 8.00%.

The sampled comparison does not establish whether truncation causes classification errors.

## Conclusions

TextCNN improves validation Micro F1 and Macro F1 relative to the classical Logistic Regression baseline. Minority-class recall remains a major limitation, particularly for question and documentation. No additional TextCNN hyperparameter tuning is planned at this stage.

## Evaluation Policy

All results in this report are from the validation split. The official test set was held out during TextCNN development and was subsequently evaluated only for the selected DistilBERT model.
