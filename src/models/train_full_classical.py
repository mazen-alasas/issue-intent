import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd

from src.models.evaluation import compute_classification_metrics
from src.models.train_classical import (
    LABELS,
    ROOT,
    build_title_pipeline,
)
from src.models.train_title_body import combine_title_body

DATA_DIR = ROOT / "data/processed/structured_v1"
TRAIN_PATH = DATA_DIR / "train.parquet"
VALIDATION_PATH = DATA_DIR / "validation.parquet"
RESULTS_DIR = ROOT / "data/processed"

EXPECTED_TRAIN_ROWS = 1_038_684
EXPECTED_VALIDATION_ROWS = 183_298
COLUMNS = ["title", "body", "labels"]


def load_dataset(path: Path, expected_rows: int) -> pd.DataFrame:
    """Load and validate a structured Parquet dataset."""
    if not path.is_file():
        raise FileNotFoundError(path)

    data = pd.read_parquet(path, columns=COLUMNS)

    if len(data) != expected_rows:
        raise ValueError(
            f"Unexpected row count for {path}: "
            f"expected {expected_rows}, got {len(data)}"
        )

    if data[COLUMNS].isna().any().any():
        raise ValueError(f"Null values found in {path}")

    if not data["labels"].isin(LABELS).all():
        raise ValueError(f"Invalid labels found in {path}")

    return data


def run_experiment(
    model_name: str = "logreg",
    train_path: Path = TRAIN_PATH,
    validation_path: Path = VALIDATION_PATH,
    expected_train_rows: int = EXPECTED_TRAIN_ROWS,
    expected_validation_rows: int = EXPECTED_VALIDATION_ROWS,
) -> dict:
    """Train and evaluate a full-scale structured classical model."""
    pipeline = build_title_pipeline(model_name)

    start = time.perf_counter()
    train = load_dataset(train_path, expected_train_rows)
    train_text = combine_title_body(train, preprocessing="none")
    train_labels = train["labels"].copy()
    train_loading_seconds = time.perf_counter() - start

    del train
    gc.collect()

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    start = time.perf_counter()
    train_features = vectorizer.fit_transform(train_text)
    vectorization_seconds = time.perf_counter() - start

    del train_text
    gc.collect()

    start = time.perf_counter()
    classifier.fit(train_features, train_labels)
    classifier_fit_seconds = time.perf_counter() - start

    feature_count = train_features.shape[1]
    train_nonzero = int(train_features.nnz)

    del train_features, train_labels
    gc.collect()

    start = time.perf_counter()
    validation = load_dataset(
        validation_path,
        expected_validation_rows,
    )
    validation_text = combine_title_body(
        validation,
        preprocessing="none",
    )
    validation_labels = validation["labels"].copy()
    validation_loading_seconds = time.perf_counter() - start

    del validation
    gc.collect()

    start = time.perf_counter()
    validation_features = vectorizer.transform(validation_text)
    validation_vectorization_seconds = time.perf_counter() - start

    del validation_text
    gc.collect()

    start = time.perf_counter()
    predictions = classifier.predict(validation_features)
    prediction_seconds = time.perf_counter() - start

    del validation_features
    gc.collect()

    metrics = compute_classification_metrics(
        y_true=validation_labels,
        y_pred=predictions,
        labels=LABELS,
    )

    return {
        "experiment": f"structured_v1_full_{model_name}",
        "dataset": "structured_v1",
        "preprocessing": "E4",
        "preprocessing_scope": "body_only",
        "model": model_name,
        "input": "title_body",
        "train_rows": expected_train_rows,
        "validation_rows": expected_validation_rows,
        "config": {
            "tfidf": {
                "ngram_range": [1, 1],
                "max_features": 50_000,
                "min_df": 2,
                "dtype": "float32",
            },
            "classifier": {
                key: value
                for key, value in classifier.get_params().items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        },
        "features": {
            "vocabulary_size": feature_count,
            "train_nonzero": train_nonzero,
        },
        "timing": {
            "train_loading_seconds": round(train_loading_seconds, 3),
            "vectorization_seconds": round(vectorization_seconds, 3),
            "classifier_fit_seconds": round(classifier_fit_seconds, 3),
            "validation_loading_seconds": round(validation_loading_seconds, 3),
            "validation_vectorization_seconds": round(
                validation_vectorization_seconds, 3
            ),
            "prediction_seconds": round(prediction_seconds, 3),
        },
        "metrics": metrics,
    }


def main() -> None:
    """Run and save the full-scale classical experiment."""
    parser = argparse.ArgumentParser(
        description="Train a full-scale structured classical model."
    )
    parser.add_argument(
        "--model",
        choices=["logreg", "linear_svc"],
        default="logreg",
    )
    args = parser.parse_args()

    output = RESULTS_DIR / (f"structured_v1_full_{args.model}_metrics.json")

    if output.exists():
        parser.error(
            f"Results already exist: {output}. "
            "Move or remove the file before rerunning."
        )

    print(f"Training full structured_v1 with {args.model}...", flush=True)

    results = run_experiment(args.model)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = results["metrics"]

    print(f"Micro F1: {metrics['micro_f1']:.6f}")
    print(f"Macro F1: {metrics['macro_f1']:.6f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.6f}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
