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

SAMPLE_SIZES = [50_000, 100_000, 200_000]

RESULTS_PATH = ROOT / "data/processed/title_logreg_learning_curve.json"


def run_learning_curve() -> dict:
    """Evaluate Logistic Regression across nested training sizes.

    Returns:
        dict: Experiment configuration and results by sample size.
    """
    samples = load_nested_stratified_samples(
        csv_path=TRAIN_PATH,
        sample_sizes=SAMPLE_SIZES,
        random_state=RANDOM_STATE,
        columns=["title", "body"],
    )

    validation = pd.read_csv(
        VALIDATION_PATH,
        usecols=["title", "labels"],
        dtype=str,
        keep_default_na=False,
    )

    results = []

    for sample_size in SAMPLE_SIZES:
        train = samples[sample_size]
        pipeline = build_title_pipeline("logreg")

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

        results.append(
            {
                "sample_size": sample_size,
                "training_seconds": round(training_seconds, 3),
                "prediction_seconds": round(prediction_seconds, 3),
                "metrics": metrics,
            }
        )

        print(
            f"{sample_size:,} samples | "
            f"Micro F1: {metrics['micro_f1']:.4f} | "
            f"Macro F1: {metrics['macro_f1']:.4f} | "
            f"Training: {training_seconds:.2f}s"
        )

    return {
        "experiment": "title_logreg_nested_learning_curve",
        "sample_sizes": SAMPLE_SIZES,
        "random_state": RANDOM_STATE,
        "model": "logreg",
        "input": "title",
        "sampling": "nested_stratified",
        "results": results,
    }


def main() -> None:
    """Run and save the nested learning curve experiment."""
    results = run_learning_curve()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
