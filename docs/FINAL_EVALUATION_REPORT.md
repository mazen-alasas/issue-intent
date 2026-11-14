# Final Evaluation Report

## 1. Overview

**Project:** issue-intent
**Task:** Multi-class GitHub Issue Intent Classification
**Dataset:** NLBSE 2023 Issue Report Classification
**Final Model:** Fine-tuned DistilBERT
**Evaluation Split:** Official held-out test set
**Test Samples:** 142,320

The final model classifies GitHub issues into four categories:

- `bug`
- `documentation`
- `feature`
- `question`

The official test set was reserved for final evaluation.
Model selection was based on validation performance.

## 2. Evaluation Protocol

### Dataset

The dataset contains 1,418,201 GitHub issues:

| Split | Samples |
|---|---:|
| Original training set | 1,275,881 |
| Official test set | 142,320 |
| Total | 1,418,201 |

Training data underwent quality checks, exact-text
deduplication, conflicting-label filtering, and a
stratified training/validation split.

The official test set was not used for model selection.

### Model Selection

The final checkpoint was selected using validation
Macro F1.

The selected DistilBERT checkpoint was obtained
at epoch 3.

### Input Processing

The final evaluation used:

| Parameter | Value |
|---|---|
| Structured preprocessing | E4 |
| Preprocessed field | Body only |
| Repeated-block compression | Disabled |
| Maximum token length | 256 |
| Title token budget | 48 |
| Body allocation | Head 75%, tail 25% |
| Evaluation batch size | 16 |
| Random seed | 42 |

The same tokenization and input construction logic
used during training was reused for evaluation.

### Evaluation Metrics

The primary project metric is Micro F1.

Macro F1 is reported alongside Micro F1 to capture
performance across classes with different frequencies.

Additional metrics include:

- Accuracy
- Weighted F1
- Per-class precision, recall, and F1
- Confusion matrix

## 3. Official Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.886558 |
| Micro F1 | 0.886558 |
| Macro F1 | 0.791175 |
| Weighted F1 | 0.884702 |

| Outcome | Samples |
|---|---:|
| Correct predictions | 126,175 |
| Incorrect predictions | 16,145 |
| Total | 142,320 |

The reported scores are from the saved Kaggle
official-test evaluation artifacts.

They were independently recomputed from saved
predictions during the local artifact audit.

## 4. Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| bug | 0.910232 | 0.934649 | 0.922279 | 74,781 |
| documentation | 0.755988 | 0.671465 | 0.711224 | 6,252 |
| feature | 0.891017 | 0.891793 | 0.891405 | 52,797 |
| question | 0.700434 | 0.588810 | 0.639790 | 8,490 |

The model performs best on bug and feature issues.

Question classification is the weakest category
by F1 score, followed by documentation.

## 5. Confusion Matrix

Rows represent true labels.
Columns represent predicted labels.

| True / Predicted | bug | documentation | feature | question |
|---|---:|---:|---:|---:|
| bug | 69,894 | 391 | 3,437 | 1,059 |
| documentation | 725 | 4,198 | 1,101 | 228 |
| feature | 4,005 | 857 | 47,084 | 851 |
| question | 2,163 | 107 | 1,221 | 4,999 |

Notable patterns:

- Bug and feature issues are frequently confused.
- Some question issues are predicted as bug or feature.
- Documentation issues are sometimes classified as feature.

These observations describe error frequencies,
not confirmed root causes.

## 6. Validation Comparison

The following results were measured on validation data.

| Model | Micro F1 | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.842650 | 0.706409 |
| TextCNN | 0.852650 | 0.724730 |
| DistilBERT | 0.883665 | 0.788362 |

DistilBERT achieved the strongest validation
performance among these reported experiments.

The differences should not be interpreted as
controlled architecture-only effects unless the
underlying preprocessing and evaluation settings
are confirmed to be comparable.

### Official Test Performance

The final DistilBERT checkpoint achieved:

- Micro F1: 0.886558
- Macro F1: 0.791175

The validation and official test scores are
reported separately because they were measured
on different evaluation splits.

## 7. Artifact Integrity Audit

The saved final evaluation artifacts include:

- `test_predictions.parquet`
- `test_metrics.json`
- `test_classification_report.csv`
- `test_confusion_matrix.csv`

The local audit verified:

1. Prediction row order and row count.
2. Valid class labels.
3. Probability ranges and probability sums.
4. Predicted-label and argmax consistency.
5. Confidence consistency.
6. Correctness flags.
7. Recomputed evaluation metrics.
8. Saved confusion matrix.
9. Saved classification report.
10. Row-by-row source ID and true-label alignment.

**Audit status:** PASSED
**Rows checked:** 142,320
**Source alignment:** PASSED

The source alignment used the saved
`official_test_structured_v1.parquet` artifact.

The audit establishes internal artifact consistency
and alignment with that saved test dataset.

It does not independently establish the identity
or cryptographic provenance of the checkpoint
used during the original Kaggle inference run.

## 8. Limitations

### Class Imbalance

Bug and feature issues account for most samples.
Question and documentation have substantially
smaller support.

Overall accuracy and Micro F1 therefore do not
fully describe minority-class performance.

### Semantic Ambiguity

GitHub issues may combine multiple intents.

For example, an issue may report a bug while
also requesting documentation or a feature.

The evaluation uses the dataset's supplied labels.

### Input Truncation

The model processes a maximum of 256 tokens.

Long issue bodies may lose relevant information
outside the retained head and tail portions.

### Domain Generalization

The evaluation is based on GitHub issues from
the NLBSE 2023 dataset.

Performance on private repositories, different
organizations, new projects, or other issue
trackers has not been established.

### Confidence Calibration

Saved class probabilities and confidence values
were checked for numerical consistency.

Their calibration and reliability as probability
estimates have not yet been established.

## 9. Reproducibility

The repository includes Python scripts for:

- Official test preprocessing.
- Transformer evaluation.
- Final evaluation artifact auditing.

The original Kaggle evaluation artifacts were
preserved outside the Git repository.

The local audit was performed without retraining
or rerunning full-model inference.

### Audit Command

```bash
uv run python -m src.models.audit_final_evaluation \
  --artifacts-dir "$HOME/issue-intent-artifacts/kaggle-backup/extracted/distilbert_final_test" \
  --source "$HOME/issue-intent-artifacts/kaggle-backup/extracted/official_test_structured_v1.parquet"
```

## 10. Conclusion

The final fine-tuned DistilBERT model achieved
88.66% Micro F1 and 79.12% Macro F1 on the
official held-out test set.

The saved evaluation artifacts passed local
integrity and source-alignment checks.

The results demonstrate strong overall
classification performance, with remaining
weaknesses in question and documentation
classification.
