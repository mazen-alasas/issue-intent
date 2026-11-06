import json
import time

import pandas as pd

from src.data.sampling import load_nested_stratified_samples
from src.models.evaluation import compute_classification_metrics
from src.models.train_classical import (
    LABELS,
    RANDOM_STATE,
    ROOT,
    TRAIN_PATH,
    VALIDATION_PATH,
    build_title_pipeline,
)

SAMPLE_SIZE = 200_000
RESULTS_PATH = ROOT / "data/processed/title_body_logreg_200k_metrics.json"


def combine_title_body(data: pd.DataFrame) -> pd.Series:
    """Combine issue titles and raw bodies into a single text field.

    Args:
        data: DataFrame containing title and body columns.

    Returns:
        pd.Series: Combined issue text with missing values handled.
    """
    title = data["title"].fillna("")
    body = data["body"].fillna("")

    return title + "\n\n" + body


def run_experiment() -> dict:
    """Train and evaluate Logistic Regression using title and raw body.

    Returns:
        dict: Experiment configuration, timing, and evaluation metrics.
    """
    samples = load_nested_stratified_samples(
        csv_path=TRAIN_PATH,
        sample_sizes=[SAMPLE_SIZE],
        random_state=RANDOM_STATE,
        columns=["title", "body", "labels"],
    )
    train = samples[SAMPLE_SIZE]

    validation = pd.read_csv(
        VALIDATION_PATH,
        usecols=["title", "body", "labels"],
        dtype=str,
        keep_default_na=False,
    )

    train_text = combine_title_body(train)
    validation_text = combine_title_body(validation)

    pipeline = build_title_pipeline("logreg")

    start = time.perf_counter()
    pipeline.fit(train_text, train["labels"])
    training_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = pipeline.predict(validation_text)
    prediction_seconds = time.perf_counter() - start

    metrics = compute_classification_metrics(
        y_true=validation["labels"],
        y_pred=predictions,
        labels=LABELS,
    )

    return {
        "experiment": "title_body_raw_logreg",
        "sample_size": SAMPLE_SIZE,
        "random_state": RANDOM_STATE,
        "model": "logreg",
        "input": "title_raw_body",
        "preprocessing": "none",
        "truncation": None,
        "sampling": "stratified",
        "training_seconds": round(training_seconds, 3),
        "prediction_seconds": round(prediction_seconds, 3),
        "metrics": metrics,
    }


def main() -> None:
    """Run and save the title-body experiment."""
    results = run_experiment()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = results["metrics"]

    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Training: {results['training_seconds']:.2f}s")
    print(f"Prediction: {results['prediction_seconds']:.2f}s")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
