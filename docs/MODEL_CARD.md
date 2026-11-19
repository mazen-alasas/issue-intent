# Model Card — issue-intent DistilBERT

## 1. Model Details

| Property | Description |
|---|---|
| Model name | issue-intent DistilBERT |
| Base checkpoint | `distilbert/distilbert-base-uncased` |
| Architecture | `DistilBertForSequenceClassification` |
| Task | GitHub issue intent classification |
| Classification type | Single-label, four-class classification |
| Language | English |
| Framework | PyTorch and Hugging Face Transformers |
| Dataset | NLBSE 2023 Issue Report Classification |
| Selection criterion | Validation Macro F1 |
| Selected epoch | 3 |
| Checkpoint label metadata | Verified locally against the project label order |
| Evaluation status | Official held-out test evaluated; saved artifacts audited |
| Deployment status | Public browser demo serves an INT8 ONNX export on [Cloudflare Pages](https://issue-intent.pages.dev/); PyTorch checkpoint is not hosted as a public server |

The model predicts one intent category from the title and body of a GitHub issue.

### Label Mapping and Checkpoint Verification

| Class ID | Label |
|---:|---|
| 0 | `bug` |
| 1 | `documentation` |
| 2 | `feature` |
| 3 | `question` |

The preserved checkpoint's `config.json` was inspected locally. Its explicit `id2label` and `label2id` fields match the project mapping above. The checkpoint identifies its `model_type` as `distilbert` and its architecture as `DistilBertForSequenceClassification`.

The raw JSON does not contain a separate `num_labels` field; the four entries in `id2label` establish the declared class count. The checkpoint therefore does **not** require the evaluation script's `--allow-generic-labels` override.

This check validates the checkpoint's declared label metadata, not the provenance of the model weights used in the original Kaggle evaluation.

## 2. Intended Use

The model is intended for GitHub issue categorization, human-assisted issue triage, software engineering NLP research, and comparisons of issue classification methods.

Predictions should be treated as suggested labels rather than definitive judgments. Maintainers can review and correct labels before using them in automated workflows. Confidence thresholds or escalation policies should be evaluated and calibrated before operational use.

### Out-of-Scope Use

The model should not be used to determine issue severity, security risk, technical validity, developer responsibility, or contributor trustworthiness. It is not a general-purpose document classifier and should not be assumed reliable for non-English reports or unrelated ticketing systems.

It produces exactly one of the four dataset labels; it does not represent multiple simultaneous intents.

## 3. Training Data

### Original Dataset

| Original split | Rows |
|---|---:|
| Training | 1,275,881 |
| Official held-out test | 142,320 |
| Total | 1,418,201 |

The NLBSE 2023 Issue Report Classification dataset consists of real GitHub issue reports. The model uses `title` and `body`; fields such as `id` and `author_association` are not classification inputs.

### Prepared Training and Validation Data

The project's training-data preparation deduplicates exact title–body text with consistent labels and excludes entire exact-text groups with conflicting labels. Missing title and body values are normalized to empty strings. The raw source data is preserved separately.

| Prepared dataset | Rows |
|---|---:|
| Training | 1,038,684 |
| Validation | 183,298 |
| Total | 1,221,982 |

The prepared dataset was split using stratification with `random_state=42`. Exact title–body overlap between the prepared training and validation splits was checked and reported as zero. This check does not eliminate near duplicates, semantic overlap, or repository-specific shortcuts.

The **official test set was excluded from training, model selection, and hyperparameter tuning**.

## 4. Input and Preprocessing

| Setting | Configuration |
|---|---|
| Input | Issue title and body |
| Base tokenizer | Fast tokenizer for `distilbert/distilbert-base-uncased` |
| Structured preprocessing | E4 configuration, applied to the body only |
| Repeated-block compression | Disabled |
| Maximum sequence length | 256 tokens |
| Title budget | 48 tokens |
| Body truncation | Retain head and tail, with a 75% / 25% allocation |
| Padding | Dynamic batch padding |

The input encoding pipeline is implemented in `src/data/transformer_dataset.py`, including `TransformerInputConfig` and `encode_issue()`. The streaming dataset implementation is in `src/data/transformer_streaming.py`.

The structured preprocessing pipeline removes or normalizes selected formatting and template noise. Its selected E4 configuration replaces fenced code blocks and does not enable selective code-signal extraction. Consequently, some useful diagnostics inside fenced code can be lost; the preprocessing is a measured engineering choice rather than a guarantee of semantic preservation.

Long bodies may lose information outside the retained head and tail segments. Character-length reductions from preprocessing should not be interpreted as equivalent token or compute reductions.

## 5. Training Configuration

The repository training implementation is `src/models/train_transformer.py`. The recorded training setup and corresponding code defaults are:

| Parameter | Value |
|---|---|
| Base model | `distilbert/distilbert-base-uncased` |
| Optimizer | PyTorch `AdamW` |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Training batch size | 8 |
| Gradient accumulation | 4 steps |
| Nominal effective batch size | 32 examples |
| Training epochs | 3 |
| Warmup ratio | 0.10 of planned optimizer steps |
| Learning-rate scheduler | Linear warmup, then linear decay |
| Gradient clipping | Maximum norm 1.0 |
| Mixed precision | CUDA `float16` autocast with `GradScaler` |
| Early-stopping patience | 2 epochs without validation Macro F1 improvement |
| Seed | 42 |
| Model selection | Highest validation Macro F1 |
| Training environment | Kaggle with CUDA |

The final recorded training run resumed from a saved training state. The implementation saves the model and tokenizer under a `best/` directory whenever validation Macro F1 improves. It also stores a resumable training state (`last.pt`), training history, and a run configuration.

### Recorded Validation History

| Epoch | Training loss | Validation Micro F1 | Validation Macro F1 |
|---:|---:|---:|---:|
| 1 | 0.4125 | 0.8806 | 0.7734 |
| 2 | 0.3293 | 0.8852 | 0.7845 |
| 3 | 0.2821 | 0.8837 | 0.7884 |

Epoch 3 was selected because it had the highest validation **Macro F1** in the recorded run, even though epoch 2 had a slightly higher validation Micro F1. The project's primary reported metric remains Micro F1; checkpoint selection used Macro F1 to account more directly for minority-class performance.

The more precise selected-checkpoint validation results were Micro F1 `0.8836648518` and Macro F1 `0.7883622793`.

See `docs/TRANSFORMER_EXPERIMENT.md` for the training experiment record and provenance.

## 6. Evaluation

### Official Held-Out Test

The selected checkpoint was evaluated on the 142,320-row official NLBSE 2023 test split on Kaggle. Inference used the selected input preprocessing and encoding configuration, an evaluation batch size of 16, and saved per-row predictions and class probabilities.

| Metric | Official test result |
|---|---:|
| Accuracy | 0.886558 |
| Micro F1 | 0.886558 |
| Macro F1 | 0.791175 |
| Weighted F1 | 0.884702 |
| Correct predictions | 126,175 |
| Incorrect predictions | 16,145 |

### Per-Class Results

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `bug` | 0.910232 | 0.934649 | 0.922279 | 74,781 |
| `documentation` | 0.755988 | 0.671465 | 0.711224 | 6,252 |
| `feature` | 0.891017 | 0.891793 | 0.891405 | 52,797 |
| `question` | 0.700434 | 0.588810 | 0.639790 | 8,490 |

The model performs best on `bug` and `feature`. The lower recall for `question` and `documentation` highlights the importance of reporting per-class results alongside aggregate performance.

### Recorded Validation Baselines

| Model | Validation Micro F1 | Validation Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 0.842650 | 0.706409 |
| TextCNN | 0.852650 | 0.724730 |
| DistilBERT | 0.883665 | 0.788362 |

These are **validation** scores, not official test scores for all models. They summarize experiments with different model families and do not isolate architecture as the only factor behind the observed differences.

Full final-test results and the ordered confusion matrix are documented in `docs/FINAL_EVALUATION_REPORT.md`.

## 7. Evaluation Artifact Integrity

The preserved Kaggle test evaluation produced:

- `test_predictions.parquet`
- `test_metrics.json`
- `test_classification_report.csv`
- `test_confusion_matrix.csv`

The local audit in `src/models/audit_final_evaluation.py` passed on all 142,320 saved predictions, including checks of row order, allowed labels, class probabilities, predicted-label/argmax consistency, confidence, correctness flags, recomputed metrics, saved reports, and confusion matrix.

The saved predictions also passed row-by-row `id` and true-label alignment against the preserved `official_test_structured_v1.parquet` source artifact.

**Audit result:** Passed. **Source alignment:** Passed. **Rows checked:** 142,320.

The audit verifies the consistency of the saved prediction artifacts with one another and with that saved source dataset. It does **not** independently prove that the original Kaggle inference used the exact local model weights or establish end-to-end checkpoint provenance.

## 8. Limitations and Risks

### Class Imbalance

The official test has much larger `bug` and `feature` classes than `documentation` and `question`. Aggregate accuracy and Micro F1 can hide weaker minority-class outcomes.

### Ambiguous and Noisy Labels

A single GitHub issue may combine a bug report, question, feature request, or documentation concern. The model learns from the supplied single-label annotations, which may be ambiguous or imperfect.

### Truncation and Code Context

A 256-token limit and head/tail selection can exclude important information. E4 preprocessing also transforms fenced code blocks, which may remove useful diagnostic details.

### Domain Generalization

Performance was not established for new repositories, private enterprise trackers, non-English content, or materially different issue conventions.

### Confidence and Calibration

The audit checks softmax probability consistency, not probability calibration. No validated confidence threshold or selective-prediction policy is established.

### Privacy and Security

Issue text may contain personal information, credentials, logs, or proprietary content. Integrations should implement appropriate filtering, storage controls, access restrictions, and logging practices. This model is not designed to detect secrets or security incidents.

## 9. Reproducibility and Artifacts

| Component | Repository path |
|---|---|
| Training | `src/models/train_transformer.py` |
| Input encoding | `src/data/transformer_dataset.py` |
| Streaming dataset | `src/data/transformer_streaming.py` |
| Official test preparation | `src/preprocessing/prepare_official_test.py` |
| Official test inference/evaluation | `src/models/evaluate_transformer.py` |
| Final artifact integrity audit | `src/models/audit_final_evaluation.py` |
| Detailed test report | `docs/FINAL_EVALUATION_REPORT.md` |

The selected model/tokenizer checkpoint is retained locally outside Git under `~/issue-intent-artifacts/models/distilbert/best/`. Preserved test artifacts are stored separately under `~/issue-intent-artifacts/kaggle-backup/extracted/`.

These local paths describe the recorded development environment; they are not hosted model download URLs. Reproduction requires obtaining the underlying dataset and checkpoint and using compatible code and library versions. Exact numerical identity across environments is not guaranteed.

## 10. Deployment Considerations

The model is **not deployed**. Before production use, the project should:

1. Load and verify the intended checkpoint and matching tokenizer.
2. Reuse the same body-only E4 preprocessing and input encoding.
3. Validate and safely handle missing, unusually long, and sensitive inputs.
4. Evaluate prediction confidence calibration and establish a human-review policy.
5. Benchmark inference latency, throughput, and memory on the target hardware.
6. Test API responses, failure handling, packaging, and dependency compatibility.
7. Monitor drift and per-class quality after deployment.

Planned downstream work includes a tested REST API, Docker packaging, CI, and portfolio-ready release documentation. These features should not be described as implemented until verified.

## 11. Status

**Model lifecycle:** Fine-tuned and evaluated; not deployed.

**Evidence:** Saved official-test predictions, metrics, classification report, confusion matrix, a successful local artifact audit, and source-row alignment.

**Remaining work:** Serving, operational validation, confidence calibration, and generalization studies.
