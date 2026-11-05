import pytest

from src.models.evaluation import compute_classification_metrics


def test_compute_classification_metrics_perfect_predictions() -> None:
    """Verify perfect predictions produce perfect metrics."""
    y_true = ["bug", "feature", "question", "documentation"]
    y_pred = ["bug", "feature", "question", "documentation"]

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels=["bug", "feature", "question", "documentation"],
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["micro_f1"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["weighted_f1"] == pytest.approx(1.0)

    for class_metrics in metrics["per_class"].values():
        assert class_metrics["f1"] == pytest.approx(1.0)
        assert class_metrics["precision"] == pytest.approx(1.0)
        assert class_metrics["recall"] == pytest.approx(1.0)
        assert class_metrics["support"] == 1


def test_compute_classification_metrics_imbalanced_predictions() -> None:
    """Verify metrics expose poor minority-class performance."""
    labels = ["bug", "feature", "question", "documentation"]

    y_true = [
        "bug",
        "bug",
        "bug",
        "bug",
        "feature",
        "feature",
        "question",
        "documentation",
    ]

    y_pred = [
        "bug",
        "bug",
        "bug",
        "bug",
        "bug",
        "feature",
        "bug",
        "bug",
    ]

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )

    assert metrics["accuracy"] == pytest.approx(5 / 8)
    assert metrics["micro_f1"] == pytest.approx(5 / 8)

    assert metrics["macro_f1"] < metrics["micro_f1"]

    assert metrics["per_class"]["question"]["f1"] == 0.0
    assert metrics["per_class"]["documentation"]["f1"] == 0.0

    assert metrics["per_class"]["question"]["support"] == 1
    assert metrics["per_class"]["documentation"]["support"] == 1

    assert metrics["labels"] == labels

    assert metrics["confusion_matrix"] == [
        [4, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ]
