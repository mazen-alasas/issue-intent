import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.sampling import load_stratified_sample
from src.models.evaluation import compute_classification_metrics

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/processed/train.csv"
VALIDATION_PATH = ROOT / "data/processed/validation.csv"
RESULTS_PATH = ROOT / "data/processed/title_logreg_50k_metrics.json"

LABELS = ["bug", "documentation", "feature", "question"]
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42


def run_experiment(
    model_name: str = "logreg",
    sample_size: int = 50_000,
) -> dict:
    """Train and evaluate a title-only TF-IDF Logistic Regression model.

    Returns:
        dict: Experiment configuration, metrics, and timing results.
    """
    train = load_stratified_sample(
        csv_path=TRAIN_PATH,
        sample_size=sample_size,
        random_state=RANDOM_STATE,
    )

    validation = pd.read_csv(
        VALIDATION_PATH,
        usecols=["title", "labels"],
        dtype=str,
        keep_default_na=False,
    )

    pipeline = build_title_pipeline(model_name=model_name)

    start = time.perf_counter()

    pipeline.fit(train["title"], train["labels"])

    training_seconds = time.perf_counter() - start

    start = time.perf_counter()

    predictions = pipeline.predict(validation["title"])

    prediction_seconds = time.perf_counter() - start

    classifier_config = pipeline.named_steps["classifier"].get_params()

    metrics = compute_classification_metrics(
        y_true=validation["labels"],
        y_pred=predictions,
        labels=LABELS,
    )

    return {
        "experiment": f"title_tfidf_{model_name}_{sample_size // 1000}k",
        "config": {
            "sample_size": sample_size,
            "random_state": RANDOM_STATE,
            "input": "title",
            "ngram_range": [1, 1],
            "max_features": 50_000,
            "min_df": 2,
            "max_iter": classifier_config["max_iter"],
            "model": model_name,
        },
        "timing": {
            "training_seconds": round(training_seconds, 3),
            "prediction_seconds": round(prediction_seconds, 3),
        },
        "metrics": metrics,
    }


def build_title_pipeline(model_name: str = "logreg") -> Pipeline:
    """Build a title-only TF-IDF classification pipeline.

    Args:
        model_name (str): Classifier name, either logreg or linear_svc.

    Returns:
        Pipeline: Unfitted text classification pipeline.

    Raises:
        ValueError: If the classifier name is unsupported.
    """
    if model_name == "logreg":
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    elif model_name == "linear_svc":
        classifier = LinearSVC(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 1),
                    max_features=50_000,
                    min_df=2,
                    dtype=np.float32,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    """Run and save a selected classical NLP experiment."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a title-only classifier."
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "linear_svc"],
        default="logreg",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50_000,
        help="Number of stratified training samples.",
    )

    args = parser.parse_args()

    results = run_experiment(
        model_name=args.model,
        sample_size=args.sample_size,
    )

    results_path = ROOT / "data/processed" / f"{results['experiment']}_metrics.json"

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = results["metrics"]

    print(f"Experiment:    {results['experiment']}")
    print(f"Training time: {results['timing']['training_seconds']:.2f}s")
    print(f"Prediction time: {results['timing']['prediction_seconds']:.2f}s")
    print(f"Micro F1:      {metrics['micro_f1']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:   {metrics['weighted_f1']:.4f}")
    print(f"Results:       {results_path}")


if __name__ == "__main__":
    main()
