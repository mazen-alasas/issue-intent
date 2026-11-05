import pandas as pd
import pytest

from src.models.dummy_baseline import evaluate_dummy_baseline


def test_dummy_baseline_predicts_majority_class(tmp_path) -> None:
    """Verify the baseline predicts the training majority class."""
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    pd.DataFrame({"labels": ["bug", "bug", "bug", "feature", "question"]}).to_csv(
        train_path, index=False
    )

    pd.DataFrame({"labels": ["bug", "feature", "question", "documentation"]}).to_csv(
        validation_path, index=False
    )

    metrics = evaluate_dummy_baseline(
        train_path=str(train_path),
        validation_path=str(validation_path),
    )

    assert metrics["accuracy"] == pytest.approx(0.25)
    assert metrics["micro_f1"] == pytest.approx(0.25)
    assert metrics["per_class"]["bug"]["recall"] == pytest.approx(1.0)

    assert metrics["confusion_matrix"] == [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
    ]
