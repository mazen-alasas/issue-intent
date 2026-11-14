import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from src.models import audit_final_evaluation as audit

LABELS = ["bug", "documentation", "feature", "question"]


@pytest.fixture
def sample_artifacts(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    true_labels = ["bug", "feature", "question", "documentation"]
    predicted_labels = ["bug", "feature", "feature", "documentation"]

    probabilities = np.array(
        [
            [0.9, 0.03, 0.04, 0.03],
            [0.05, 0.05, 0.85, 0.05],
            [0.1, 0.1, 0.7, 0.1],
            [0.05, 0.8, 0.1, 0.05],
        ],
        dtype=np.float32,
    )

    frame = pd.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "id": [101, 102, 101, 104],
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "confidence": probabilities.max(axis=1),
            "correct": [True, True, False, True],
        }
    )

    for index, label in enumerate(LABELS):
        frame[f"prob_{label}"] = probabilities[:, index]

    frame.to_parquet(
        artifacts / "test_predictions.parquet",
        index=False,
    )

    source = pd.DataFrame(
        {
            "id": [101, 102, 101, 104],
            "labels": true_labels,
            "title": ["A", "B", "C", "D"],
            "body": ["one", "two", "three", "four"],
        }
    )
    source_path = tmp_path / "official_test.parquet"
    source.to_parquet(source_path, index=False)

    true_ids = np.array([LABELS.index(value) for value in true_labels])
    predicted_ids = np.array([LABELS.index(value) for value in predicted_labels])

    matrix = confusion_matrix(
        true_ids,
        predicted_ids,
        labels=[0, 1, 2, 3],
    )
    calculated, report = audit.calculate_from_confusion(matrix)

    (artifacts / "test_metrics.json").write_text(
        json.dumps(calculated),
        encoding="utf-8",
    )

    pd.DataFrame(
        matrix,
        index=LABELS,
        columns=LABELS,
    ).to_csv(artifacts / "test_confusion_matrix.csv")

    pd.DataFrame(report).T.to_csv(artifacts / "test_classification_report.csv")

    return artifacts, source_path


def change_prediction(artifacts, column, row, value):
    path = artifacts / "test_predictions.parquet"
    frame = pd.read_parquet(path)
    frame.loc[row, column] = value
    frame.to_parquet(path, index=False)


def test_complete_audit_passes(sample_artifacts):
    artifacts, source = sample_artifacts

    result = audit.audit_final_evaluation(
        artifacts,
        source,
        batch_size=2,
    )

    assert result["status"] == "passed"
    assert result["rows_checked"] == 4
    assert result["source_alignment_checked"] is True
    assert result["metrics"]["correct"] == 3
    assert result["metrics"]["micro_f1"] == pytest.approx(0.75)


def test_audit_without_source(sample_artifacts):
    artifacts, _ = sample_artifacts

    result = audit.audit_final_evaluation(artifacts)

    assert result["source_alignment_checked"] is False


def test_rejects_missing_artifact(sample_artifacts):
    artifacts, _ = sample_artifacts
    (artifacts / "test_metrics.json").unlink()

    with pytest.raises(FileNotFoundError):
        audit.audit_final_evaluation(artifacts)


def test_rejects_wrong_row_index(sample_artifacts):
    artifacts, _ = sample_artifacts
    change_prediction(artifacts, "row_index", 2, 9)

    with pytest.raises(ValueError, match="row_index"):
        audit.audit_final_evaluation(artifacts, batch_size=2)


def test_rejects_unknown_label(sample_artifacts):
    artifacts, _ = sample_artifacts
    change_prediction(artifacts, "true_label", 0, "unknown")

    with pytest.raises(ValueError, match="Invalid true_label"):
        audit.audit_final_evaluation(artifacts)


def test_rejects_invalid_probability_sum(sample_artifacts):
    artifacts, _ = sample_artifacts
    change_prediction(artifacts, "prob_bug", 0, 0.5)

    with pytest.raises(ValueError, match="Probability sum"):
        audit.audit_final_evaluation(artifacts)


def test_rejects_nan_probability(sample_artifacts):
    artifacts, _ = sample_artifacts
    change_prediction(artifacts, "prob_bug", 0, np.nan)

    with pytest.raises(ValueError, match="Non-finite"):
        audit.audit_final_evaluation(artifacts)


def test_rejects_argmax_mismatch(sample_artifacts):
    artifacts, _ = sample_artifacts
    change_prediction(artifacts, "predicted_label", 0, "feature")

    with pytest.raises(ValueError, match="argmax"):
        audit.audit_final_evaluation(artifacts)


def test_rejects_confidence_mismatch(sample_artifacts):
    artifacts, _ = sample_artifacts
    change_prediction(artifacts, "confidence", 0, 0.4)

    with pytest.raises(ValueError, match="Confidence"):
        audit.audit_final_evaluation(artifacts)


def test_rejects_correct_flag_mismatch(sample_artifacts):
    artifacts, _ = sample_artifacts
    change_prediction(artifacts, "correct", 0, False)

    with pytest.raises(ValueError, match="Correct flag"):
        audit.audit_final_evaluation(artifacts)


def test_rejects_saved_metric_mismatch(sample_artifacts):
    artifacts, _ = sample_artifacts
    path = artifacts / "test_metrics.json"

    metrics = json.loads(path.read_text())
    metrics["macro_f1"] = 0.99
    path.write_text(json.dumps(metrics))

    with pytest.raises(ValueError, match="macro_f1"):
        audit.audit_final_evaluation(artifacts)


def test_rejects_saved_confusion_matrix_mismatch(sample_artifacts):
    artifacts, _ = sample_artifacts
    path = artifacts / "test_confusion_matrix.csv"

    matrix = pd.read_csv(path, index_col=0)
    matrix.loc["bug", "bug"] += 1
    matrix.to_csv(path)

    with pytest.raises(ValueError, match="confusion matrix mismatch"):
        audit.audit_final_evaluation(artifacts)


def test_rejects_saved_report_mismatch(sample_artifacts):
    artifacts, _ = sample_artifacts
    path = artifacts / "test_classification_report.csv"

    report = pd.read_csv(path, index_col=0)
    report.loc["bug", "recall"] = 0.25
    report.to_csv(path)

    with pytest.raises(ValueError, match="Report bug/recall"):
        audit.audit_final_evaluation(artifacts)


def test_rejects_source_id_mismatch(sample_artifacts):
    artifacts, source = sample_artifacts

    frame = pd.read_parquet(source)
    frame.loc[2, "id"] = 999
    frame.to_parquet(source, index=False)

    with pytest.raises(ValueError, match="Source id mismatch"):
        audit.audit_final_evaluation(artifacts, source, batch_size=2)


def test_rejects_source_label_mismatch(sample_artifacts):
    artifacts, source = sample_artifacts

    frame = pd.read_parquet(source)
    frame.loc[2, "labels"] = "bug"
    frame.to_parquet(source, index=False)

    with pytest.raises(ValueError, match="Source label mismatch"):
        audit.audit_final_evaluation(artifacts, source)


def test_rejects_source_row_count_mismatch(sample_artifacts):
    artifacts, source = sample_artifacts

    frame = pd.read_parquet(source).iloc[:3]
    frame.to_parquet(source, index=False)

    with pytest.raises(ValueError, match="Source row count"):
        audit.audit_final_evaluation(artifacts, source)


def test_rejects_invalid_batch_size(sample_artifacts):
    artifacts, _ = sample_artifacts

    with pytest.raises(ValueError, match="batch_size"):
        audit.audit_final_evaluation(artifacts, batch_size=0)


def test_rejects_empty_predictions(sample_artifacts):
    artifacts, _ = sample_artifacts
    path = artifacts / "test_predictions.parquet"

    frame = pd.read_parquet(path).iloc[:0]
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="no rows"):
        audit.audit_final_evaluation(artifacts)


def test_calculate_metrics_from_matrix():
    matrix = np.array(
        [
            [2, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ]
    )

    metrics, report = audit.calculate_from_confusion(matrix)

    assert metrics["total_samples"] == 5
    assert metrics["correct"] == 4
    assert metrics["incorrect"] == 1
    assert metrics["micro_f1"] == pytest.approx(0.8)
    assert report["question"]["recall"] == 0
