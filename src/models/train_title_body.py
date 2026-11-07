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
from src.preprocessing.text import preprocess_text

SAMPLE_SIZE = 200_000
RESULTS_PATH = ROOT / "data/processed/title_body_logreg_200k_metrics.json"


def combine_title_body(
    data: pd.DataFrame,
    preprocessing: str = "none",
) -> pd.Series:
    """Combine issue titles and bodies with optional preprocessing.

    Args:
        data: DataFrame containing title and body columns.
        preprocessing: Body preprocessing strategy, either "none" or "minimal".

    Returns:
        pd.Series: Combined issue text.

    Raises:
        ValueError: If the preprocessing strategy is unsupported.
    """
    if preprocessing not in {"none", "minimal"}:
        raise ValueError(f"Unsupported preprocessing strategy: {preprocessing}")

    title = data["title"].fillna("")
    body = data["body"].fillna("")

    if preprocessing == "minimal":
        body = body.map(preprocess_text)

    return title + "\n\n" + body


def run_experiment(
    preprocessing: str = "none",
) -> dict:
    """Train and evaluate Logistic Regression with optional body cleaning.

    Args:
        preprocessing: Body preprocessing strategy.

    Returns:
        dict: Experiment configuration, timing, and evaluation metrics.
    """
    if preprocessing not in {"none", "minimal"}:
        raise ValueError(f"Unsupported preprocessing strategy: {preprocessing}")

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

    train_text = combine_title_body(
        train,
        preprocessing=preprocessing,
    )
    validation_text = combine_title_body(
        validation,
        preprocessing=preprocessing,
    )

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
        "experiment": f"title_body_{'raw' if preprocessing == 'none' else preprocessing}_logreg",
        "sample_size": SAMPLE_SIZE,
        "random_state": RANDOM_STATE,
        "model": "logreg",
        "input": "title_body",
        "preprocessing": preprocessing,
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
