import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.sampling import load_stratified_sample
from src.models.evaluation import compute_classification_metrics

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/processed/train.csv"
VALIDATION_PATH = ROOT / "data/processed/validation.csv"
RESULTS_PATH = ROOT / "data/processed/title_logreg_50k_metrics.json"

LABELS = ["bug", "documentation", "feature", "question"]
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42


def run_experiment() -> dict:
    """Train and evaluate a title-only TF-IDF Logistic Regression model.

    Returns:
        dict: Experiment configuration, metrics, and timing results.
    """
    train = load_stratified_sample(
        csv_path=TRAIN_PATH,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    validation = pd.read_csv(
        VALIDATION_PATH,
        usecols=["title", "labels"],
        dtype=str,
        keep_default_na=False,
    )

    pipeline = build_title_logreg_pipeline()

    start = time.perf_counter()

    pipeline.fit(train["title"], train["labels"])

    training_seconds = time.perf_counter() - start

    start = time.perf_counter()

    predictions = pipeline.predict(validation["title"])

    prediction_seconds = time.perf_counter() - start

    metrics = compute_classification_metrics(
        y_true=validation["labels"],
        y_pred=predictions,
        labels=LABELS,
    )

    return {
        "experiment": "title_tfidf_logreg_50k",
        "config": {
            "sample_size": SAMPLE_SIZE,
            "random_state": RANDOM_STATE,
            "input": "title",
            "ngram_range": [1, 1],
            "max_features": 50_000,
            "min_df": 2,
            "max_iter": 1000,
        },
        "timing": {
            "training_seconds": round(training_seconds, 3),
            "prediction_seconds": round(prediction_seconds, 3),
        },
        "metrics": metrics,
    }


def build_title_logreg_pipeline() -> Pipeline:
    """Build a TF-IDF Logistic Regression classification pipeline.

    Returns:
        Pipeline: Unfitted title classification pipeline.
    """
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
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    """Run the experiment and save its results."""
    results = run_experiment()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = results["metrics"]

    print("Title-only Logistic Regression experiment completed.")
    print(f"Training time: {results['timing']['training_seconds']:.2f}s")
    print(f"Prediction time: {results['timing']['prediction_seconds']:.2f}s")
    print(f"Micro F1:       {metrics['micro_f1']:.4f}")
    print(f"Macro F1:       {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:    {metrics['weighted_f1']:.4f}")
    print(f"Results:        {RESULTS_PATH}")


if __name__ == "__main__":
    main()
