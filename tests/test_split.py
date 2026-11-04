from collections import Counter

import numpy as np
import pandas as pd

from src.data.split import create_split_indices, load_labels


def test_load_labels(tmp_path):
    """Verify that labels are loaded in their original order."""
    input_path = tmp_path / "issues.csv"

    pd.DataFrame(
        {
            "labels": ["bug", "feature", "question"],
            "title": ["Issue A", "Issue B", "Issue C"],
        }
    ).to_csv(input_path, index=False)

    labels = load_labels(input_path)

    assert labels.tolist() == ["bug", "feature", "question"]
    assert labels.name == "labels"


def test_create_split_indices_preserves_stratification():
    """Verify that stratified splitting preserves class proportions."""
    labels = pd.Series(["bug"] * 60 + ["feature"] * 30 + ["question"] * 10)

    train_indices, validation_indices = create_split_indices(labels)

    assert len(train_indices) == 85
    assert len(validation_indices) == 15

    train_counts = Counter(labels.iloc[train_indices])
    validation_counts = Counter(labels.iloc[validation_indices])

    for label in labels.unique():
        original_ratio = (labels == label).mean()
        train_ratio = train_counts[label] / len(train_indices)
        validation_ratio = validation_counts[label] / len(validation_indices)

        assert abs(train_ratio - original_ratio) <= 0.05
        assert abs(validation_ratio - original_ratio) <= 0.05


def test_create_split_indices_has_no_overlap():
    """Verify that training and validation indices are disjoint."""
    labels = pd.Series(["bug"] * 50 + ["feature"] * 50)

    train_indices, validation_indices = create_split_indices(labels)

    assert len(set(train_indices) & set(validation_indices)) == 0

    assert len(train_indices) + len(validation_indices) == len(labels)


def test_create_split_indices_is_reproducible():
    """Verify that the same seed produces identical split indices."""
    labels = pd.Series(["bug"] * 50 + ["feature"] * 50)

    first_train, first_validation = create_split_indices(labels)
    second_train, second_validation = create_split_indices(labels)

    np.testing.assert_array_equal(first_train, second_train)
    np.testing.assert_array_equal(first_validation, second_validation)
