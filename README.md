# issue-intent

**NLP-based GitHub Issue Type Classification**

issue-intent is a machine learning project focused on classifying real-world GitHub issues into four categories:

- **Bug:** Reports describing incorrect or unexpected software behavior.
- **Feature:** Requests for new functionality or improvements.
- **Question:** Requests for information, guidance, or clarification.
- **Documentation:** Reports or requests related to documentation.

The project explores traditional NLP, neural networks, and transformer-based models, with an emphasis on reproducibility, evaluation quality, and practical deployment.

## Dataset

The planned dataset is from the [NLBSE 2023 Issue Report Classification Competition](https://github.com/nlbse2023/issue-report-classification).

The published dataset contains approximately 1.42 million labeled GitHub issues, split into official training and test sets.

Dataset accessibility, schema, licensing, and integrity have not yet been verified locally.

## Project Goals

- Audit and validate real-world issue data.
- Investigate class imbalance, label noise, and data leakage.
- Establish reproducible training and evaluation pipelines.
- Compare TF-IDF-based models, a neural baseline, and transformers.
- Evaluate classification performance, latency, and resource requirements.
- Serve the selected model through a tested REST API.
- Package the application with Docker and establish CI.

## Technology Stack

Current development tools:

- Python 3.11
- uv
- pytest
- Ruff

Planned tools include Polars, scikit-learn, PyTorch, Hugging Face Transformers, MLflow, FastAPI, Docker, and GitHub Actions.

Tools will be introduced incrementally as needed.

## Development Setup

Clone the repository:

```bash
git clone git@github.com:mazen-alasas/issue-intent.git
cd issue-intent
````

Install the environment:

```
uv sync
```

Run quality checks:

```
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Project Status

Stage: Repository Foundation

The project is being initialized. No dataset analysis, model training, benchmark evaluation, or deployment has been completed yet.

See [Project Plan](https://docs/PROJECT_PLAN.md?utm_source=chatgpt.com) and [Progress](https://docs/PROGRESS.md?utm_source=chatgpt.com).

## Evaluation Principles

The official test set will be reserved for final evaluation. Model development and hyperparameter selection will use training-derived validation data.

Primary metric: Micro F1.

Additional metrics: Macro F1, per-class Precision/Recall/F1, confusion matrix, and inference latency.

## License

A code license has not been selected yet. Dataset usage and redistribution conditions will be reviewed during dataset feasibility analysis.