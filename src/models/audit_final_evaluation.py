import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.data.neural_dataset import LABEL_TO_ID

LABEL_NAMES = [
    label for label, _ in sorted(LABEL_TO_ID.items(), key=lambda item: item[1])
]
LABEL_IDS = list(range(len(LABEL_NAMES)))
PROBABILITY_COLUMNS = [f"prob_{label}" for label in LABEL_NAMES]

REQUIRED_COLUMNS = [
    "row_index",
    "id",
    "true_label",
    "predicted_label",
    "confidence",
    "correct",
    *PROBABILITY_COLUMNS,
]

METRIC_KEYS = ("accuracy", "micro_f1", "macro_f1", "weighted_f1")
TOLERANCE = 1e-5


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_number(actual, expected, name, tolerance=TOLERANCE):
    require(
        np.isfinite(actual)
        and np.isfinite(expected)
        and np.isclose(actual, expected, rtol=0, atol=tolerance),
        f"{name} mismatch: actual={actual}, expected={expected}",
    )


def read_batches(path, columns, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    parquet = pq.ParquetFile(path)
    missing = set(columns) - set(parquet.schema_arrow.names)
    require(not missing, f"Missing columns: {sorted(missing)}")

    yield from parquet.iter_batches(
        batch_size=batch_size,
        columns=columns,
    )


def calculate_from_confusion(matrix):
    total = int(matrix.sum())
    require(total > 0, "Predictions contain no rows")

    predicted_parts = []

    # Reconstruct label pairs from the small confusion matrix.
    actual_parts = []
    for true_index in LABEL_IDS:
        for predicted_index in LABEL_IDS:
            count = int(matrix[true_index, predicted_index])
            if count:
                actual_parts.append(np.full(count, true_index, dtype=np.int8))
                predicted_parts.append(np.full(count, predicted_index, dtype=np.int8))

    y_true = np.concatenate(actual_parts)
    y_pred = np.concatenate(predicted_parts)

    require(
        len(y_true) == total,
        "Internal count mismatch",
    )

    metrics = {
        "total_samples": total,
        "correct": int(np.trace(matrix)),
        "incorrect": int(total - np.trace(matrix)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_IDS,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_IDS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_IDS,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return metrics, report


def validate_predictions(predictions_path, batch_size=10_000):
    matrix = np.zeros(
        (len(LABEL_NAMES), len(LABEL_NAMES)),
        dtype=np.int64,
    )
    row_offset = 0

    for batch in read_batches(
        predictions_path,
        REQUIRED_COLUMNS,
        batch_size,
    ):
        frame = batch.to_pandas()
        count = len(frame)

        expected_indices = np.arange(
            row_offset,
            row_offset + count,
        )
        require(
            np.array_equal(
                frame["row_index"].to_numpy(),
                expected_indices,
            ),
            f"row_index mismatch near row {row_offset}",
        )

        require(
            frame["id"].notna().all(),
            f"Missing id near row {row_offset}",
        )
        require(
            frame["true_label"].isin(LABEL_TO_ID).all(),
            f"Invalid true_label near row {row_offset}",
        )
        require(
            frame["predicted_label"].isin(LABEL_TO_ID).all(),
            f"Invalid predicted_label near row {row_offset}",
        )

        probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64)
        require(
            np.isfinite(probabilities).all(),
            f"Non-finite probability near row {row_offset}",
        )
        require(
            ((probabilities >= 0) & (probabilities <= 1)).all(),
            f"Probability out of range near row {row_offset}",
        )
        require(
            np.allclose(
                probabilities.sum(axis=1),
                1.0,
                atol=TOLERANCE,
                rtol=0,
            ),
            f"Probability sum mismatch near row {row_offset}",
        )

        true_ids = frame["true_label"].map(LABEL_TO_ID).to_numpy()
        predicted_ids = frame["predicted_label"].map(LABEL_TO_ID).to_numpy()
        argmax_ids = probabilities.argmax(axis=1)

        require(
            np.array_equal(predicted_ids, argmax_ids),
            f"argmax mismatch near row {row_offset}",
        )

        confidence = frame["confidence"].to_numpy(dtype=np.float64)
        require(
            np.isfinite(confidence).all()
            and np.allclose(
                confidence,
                probabilities.max(axis=1),
                atol=TOLERANCE,
                rtol=0,
            ),
            f"Confidence mismatch near row {row_offset}",
        )

        correct = frame["correct"]
        require(
            correct.notna().all()
            and correct.map(lambda value: isinstance(value, (bool, np.bool_))).all(),
            f"Invalid correct flag near row {row_offset}",
        )
        require(
            np.array_equal(
                correct.to_numpy(dtype=bool),
                true_ids == predicted_ids,
            ),
            f"Correct flag mismatch near row {row_offset}",
        )

        matrix += confusion_matrix(
            true_ids,
            predicted_ids,
            labels=LABEL_IDS,
        )
        row_offset += count

    require(row_offset > 0, "Predictions contain no rows")
    return matrix, row_offset


def validate_source_alignment(
    predictions_path,
    source_path,
    batch_size=10_000,
):
    prediction_reader = pq.ParquetFile(predictions_path)
    source_reader = pq.ParquetFile(source_path)

    prediction_columns = {"id", "true_label"}
    source_columns = {"id", "labels"}

    require(
        prediction_columns.issubset(prediction_reader.schema_arrow.names),
        "Predictions missing alignment columns",
    )
    require(
        source_columns.issubset(source_reader.schema_arrow.names),
        "Source missing alignment columns",
    )

    require(
        prediction_reader.metadata.num_rows == source_reader.metadata.num_rows,
        "Source row count mismatch",
    )

    predictions = prediction_reader.iter_batches(
        batch_size=batch_size,
        columns=["id", "true_label"],
    )
    source = source_reader.iter_batches(
        batch_size=batch_size,
        columns=["id", "labels"],
    )

    offset = 0
    for prediction_batch, source_batch in zip(
        predictions,
        source,
        strict=True,
    ):
        pred = prediction_batch.to_pydict()
        raw = source_batch.to_pydict()

        require(
            pred["id"] == raw["id"],
            f"Source id mismatch near row {offset}",
        )
        require(
            pred["true_label"] == raw["labels"],
            f"Source label mismatch near row {offset}",
        )
        offset += len(pred["id"])

    return offset


def validate_saved_metrics(metrics_path, calculated):
    saved = json.loads(metrics_path.read_text(encoding="utf-8"))

    for key in ("total_samples", "correct", "incorrect"):
        require(
            key in saved and saved[key] == calculated[key],
            f"Saved {key} mismatch",
        )

    for key in METRIC_KEYS:
        require(key in saved, f"Missing saved metric: {key}")
        compare_number(
            calculated[key],
            saved[key],
            key,
            tolerance=1e-9,
        )

    return saved


def validate_saved_matrix(path, calculated):
    saved = pd.read_csv(path, index_col=0)

    require(
        saved.index.tolist() == LABEL_NAMES,
        "Confusion matrix row labels mismatch",
    )
    require(
        saved.columns.tolist() == LABEL_NAMES,
        "Confusion matrix column labels mismatch",
    )
    require(
        np.array_equal(saved.to_numpy(), calculated),
        "Saved confusion matrix mismatch",
    )


def validate_saved_report(path, calculated):
    saved = pd.read_csv(path, index_col=0)

    for label in LABEL_NAMES:
        require(label in saved.index, f"Missing report class: {label}")

        for metric in ("precision", "recall", "f1-score", "support"):
            require(metric in saved.columns, f"Missing report metric: {metric}")
            compare_number(
                float(saved.loc[label, metric]),
                float(calculated[label][metric]),
                f"Report {label}/{metric}",
            )

    for average in ("macro avg", "weighted avg"):
        require(average in saved.index, f"Missing report row: {average}")

        for metric in ("precision", "recall", "f1-score", "support"):
            compare_number(
                float(saved.loc[average, metric]),
                float(calculated[average][metric]),
                f"Report {average}/{metric}",
            )

    require("accuracy" in saved.index, "Missing report accuracy")
    compare_number(
        float(saved.loc["accuracy", "precision"]),
        float(calculated["accuracy"]),
        "Report accuracy",
    )


def audit_final_evaluation(
    artifacts_dir,
    source_path=None,
    *,
    batch_size=10_000,
):
    artifacts_dir = Path(artifacts_dir)

    paths = {
        "predictions": artifacts_dir / "test_predictions.parquet",
        "metrics": artifacts_dir / "test_metrics.json",
        "matrix": artifacts_dir / "test_confusion_matrix.csv",
        "report": artifacts_dir / "test_classification_report.csv",
    }

    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    matrix, rows = validate_predictions(
        paths["predictions"],
        batch_size=batch_size,
    )
    calculated, report = calculate_from_confusion(matrix)

    require(
        calculated["total_samples"] == rows,
        "Internal row count mismatch",
    )

    validate_saved_metrics(paths["metrics"], calculated)
    validate_saved_matrix(paths["matrix"], matrix)
    validate_saved_report(paths["report"], report)

    source_checked = False
    if source_path is not None:
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        validate_source_alignment(
            paths["predictions"],
            source_path,
            batch_size=batch_size,
        )
        source_checked = True

    return {
        "status": "passed",
        "rows_checked": rows,
        "source_alignment_checked": source_checked,
        "label_order": LABEL_NAMES,
        "metrics": calculated,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit saved Transformer evaluation artifacts."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Original official test Parquet for row alignment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
    )
    args = parser.parse_args()

    result = audit_final_evaluation(
        artifacts_dir=args.artifacts_dir,
        source_path=args.source,
        batch_size=args.batch_size,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
