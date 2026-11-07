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
from src.models.train_title_body import combine_title_body

SAMPLE_SIZE = 200_000
BODY_LIMITS = [1_000, 2_000, 4_000]
RESULTS_PATH = ROOT / "data/processed/truncation_ablation_metrics.json"


def truncate_body(data: pd.DataFrame, max_chars: int) -> pd.DataFrame:
    """Truncate issue bodies while preserving their complete titles.

    Args:
        data: DataFrame containing title and body columns.
        max_chars: Maximum number of body characters to retain.

    Returns:
        pd.DataFrame: Copy of the input with truncated body text.

    Raises:
        ValueError: If max_chars is not positive.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    truncated = data.copy()
    truncated["body"] = truncated["body"].fillna("").str.slice(0, max_chars)
    return truncated


def run_experiment() -> dict:
    """Evaluate several body truncation limits on the same data split.

    Returns:
        dict: Configuration, timing, and metrics for each truncation limit.
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

    results = {
        "experiment": "body_truncation_ablation",
        "sample_size": SAMPLE_SIZE,
        "random_state": RANDOM_STATE,
        "model": "logreg",
        "body_limits": BODY_LIMITS,
        "results": [],
    }

    for max_chars in BODY_LIMITS:
        print(f"\nRunning body limit: {max_chars:,} characters")

        train_truncated = truncate_body(train, max_chars)
        validation_truncated = truncate_body(validation, max_chars)

        train_text = combine_title_body(train_truncated)
        validation_text = combine_title_body(validation_truncated)

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

        result = {
            "max_body_chars": max_chars,
            "training_seconds": round(training_seconds, 3),
            "prediction_seconds": round(prediction_seconds, 3),
            "metrics": metrics,
        }
        results["results"].append(result)

        print(f"Micro F1: {metrics['micro_f1']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Training: {training_seconds:.2f}s")
        print(f"Prediction: {prediction_seconds:.2f}s")

    return results


def main() -> None:
    """Run truncation experiments and save their metrics."""
    results = run_experiment()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
