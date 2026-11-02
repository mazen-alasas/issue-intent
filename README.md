# issue-intent

**NLP-based GitHub Issue Type Classification**

issue-intent is an NLP and machine learning engineering project focused on classifying real-world GitHub issues into four categories:

- **Bug:** Reports describing incorrect or unexpected software behavior.
- **Feature:** Requests for new functionality or improvements.
- **Question:** Requests for information, guidance, or clarification.
- **Documentation:** Reports or requests related to documentation.

The project compares classical NLP, neural networks, and transformer-based models, with an emphasis on reproducibility, reliable evaluation, and practical deployment.

## Dataset

The project uses the [NLBSE 2023 Issue Report Classification Competition](https://github.com/nlbse2023/issue-report-classification) dataset.

The official dataset contains approximately **1.42 million labeled GitHub issues**, divided into:

- **Training set:** 1,275,881 issues
- **Test set:** 142,320 issues

The training dataset has been downloaded and audited. The official test set is reserved for final evaluation and will not be used during model development or hyperparameter tuning.

The initial feasibility audit identified:

- Class imbalance, with Bug and Feature representing approximately 90% of training examples.
- Missing issue titles and bodies.
- Long issue descriptions containing Markdown, code blocks, links, and other structured text.
- 53,459 duplicate title–body occurrences.
- 440 unique title–body groups with conflicting labels, involving 4,146 rows.

These findings inform the project's data preparation and evaluation strategy.

## Project Goals

- Audit and validate real-world issue data.
- Investigate class imbalance, label noise, and data leakage.
- Establish reproducible data preparation, training, and evaluation pipelines.
- Compare TF-IDF-based models, a neural baseline, and transformer-based models.
- Evaluate classification performance, inference latency, and resource requirements.
- Serve the selected model through a tested REST API.
- Package the application with Docker and establish CI.

## Technology Stack

**Current development tools:**

- Python 3.11
- uv
- pandas
- pytest
- Ruff

**Planned tools:**

- scikit-learn
- PyTorch
- Hugging Face Transformers
- MLflow
- FastAPI
- Docker
- GitHub Actions

Additional tools will be introduced only when justified by project requirements.

## Development Setup

Clone the repository:

```bash
git clone git@github.com:mazen-alasas/issue-intent.git
cd issue-intent
```

Install the environment:

```bash
uv sync
```

Run code quality checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

Run tests when implemented:

```bash
uv run pytest
```

## Project Status

| Stage | Status |
|---|---|
| Repository Foundation | Completed |
| Dataset Feasibility | Completed |
| Data Quality | Next |
| Evaluation Protocol | Planned |
| Classical NLP | Planned |
| Neural NLP | Planned |
| Transformer Fine-Tuning | Planned |
| Final Evaluation | Planned |
| API & Docker | Planned |
| Portfolio Release | Planned |

The initial dataset feasibility audit is complete. The next stage focuses on implementing reproducible data preparation, including missing-value handling, exact-text deduplication, and conflicting-label exclusion.

No model training or final benchmark evaluation has been performed yet.

For further details, see the [Project Plan](docs/PROJECT_PLAN.md), [Progress](docs/PROGRESS.md), and [Dataset Feasibility Notebook](notebooks/01_dataset_feasibility.ipynb).

## Evaluation Principles

The official test set is reserved for final evaluation. Model development and hyperparameter selection will use validation data derived exclusively from the official training set.

**Primary metric:** Micro F1.

**Additional metrics:**

- Macro F1 and Weighted F1
- Per-class Precision, Recall, and F1
- Confusion Matrix
- Inference Latency
- Model Size

All modeling approaches will follow a consistent evaluation protocol to support fair comparisons.

## License

A code license has not yet been selected. Dataset usage and redistribution conditions require further review before distributing dataset-derived artifacts.