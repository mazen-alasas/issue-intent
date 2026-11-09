import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

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
from src.preprocessing.structured import (
    StructuredPreprocessingConfig,
    preprocess_structured,
)

SAMPLE_SIZE = 200_000
RESULTS_DIR = ROOT / "data/processed"

EXPERIMENTS = {
    "E2": StructuredPreprocessingConfig(),
    "E3": StructuredPreprocessingConfig(
        replace_fenced_code=False,
    ),
    "E4": StructuredPreprocessingConfig(
        compress_repeated_blocks=False,
    ),
    "E5": StructuredPreprocessingConfig(
        compress_repeated_blocks=False,
        selective_code_compression=True,
    ),
}


def build_experiment_text(
    data: pd.DataFrame,
    config: StructuredPreprocessingConfig,
) -> pd.Series:
    """Combine unchanged titles with structured-preprocessed bodies."""
    title = data["title"].fillna("")
    body = data["body"].fillna("").map(lambda text: preprocess_structured(text, config))

    return title + "\n\n" + body


def result_path(experiment_id: str) -> Path:
    """Return the output path for an experiment."""
    return RESULTS_DIR / f"structured_{experiment_id.lower()}_200k_metrics.json"


def save_results(results: dict, path: Path) -> None:
    """Save metrics as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_experiment(experiment_id: str) -> dict:
    """Train and evaluate one structured preprocessing configuration."""
    if experiment_id not in EXPERIMENTS:
        raise ValueError(f"Unsupported experiment: {experiment_id}")

    config = EXPERIMENTS[experiment_id]

    samples = load_nested_stratified_samples(
        csv_path=TRAIN_PATH,
        sample_sizes=[SAMPLE_SIZE],
        random_state=RANDOM_STATE,
        columns=["title", "body", "labels"],
    )
    train = samples[SAMPLE_SIZE]
    del samples
    gc.collect()

    validation = pd.read_csv(
        VALIDATION_PATH,
        usecols=["title", "body", "labels"],
        dtype=str,
        keep_default_na=False,
    )

    start = time.perf_counter()

    train_text = build_experiment_text(train, config)
    validation_text = build_experiment_text(validation, config)

    preprocessing_seconds = time.perf_counter() - start

    pipeline = build_title_pipeline("logreg")

    start = time.perf_counter()
    pipeline.fit(train_text, train["labels"])
    training_seconds = time.perf_counter() - start

    del train_text
    gc.collect()

    start = time.perf_counter()
    predictions = pipeline.predict(validation_text)
    prediction_seconds = time.perf_counter() - start

    metrics = compute_classification_metrics(
        y_true=validation["labels"],
        y_pred=predictions,
        labels=LABELS,
    )

    return {
        "experiment": experiment_id,
        "sample_size": SAMPLE_SIZE,
        "validation_size": len(validation),
        "random_state": RANDOM_STATE,
        "model": "logreg",
        "input": "title_body",
        "preprocessing": "structured",
        "preprocessing_scope": "body_only",
        "truncation": None,
        "sampling": "nested_stratified",
        "config": asdict(config),
        "preprocessing_seconds": round(preprocessing_seconds, 3),
        "training_seconds": round(training_seconds, 3),
        "prediction_seconds": round(prediction_seconds, 3),
        "metrics": metrics,
    }


def main() -> None:
    """Run one ablation and save its results."""
    parser = argparse.ArgumentParser(
        description="Run a structured preprocessing ablation."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=sorted(EXPERIMENTS),
    )
    args = parser.parse_args()

    experiment_id = args.experiment
    output = result_path(experiment_id)

    if output.exists():
        parser.error(
            f"Results already exist: {output}. "
            "Move or remove the file before rerunning."
        )

    print(f"Running {experiment_id}", flush=True)

    results = run_experiment(experiment_id)
    save_results(results, output)

    metrics = results["metrics"]

    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Preprocessing: {results['preprocessing_seconds']:.2f}s")
    print(f"Training: {results['training_seconds']:.2f}s")
    print(f"Prediction: {results['prediction_seconds']:.2f}s")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
