from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.data.split import create_split_indices, load_labels, write_split_files


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


def test_write_split_files_preserves_rows(tmp_path):
    """Verify correct row assignment and original source order."""
    input_path = tmp_path / "issues.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    data = pd.DataFrame(
        {
            "id": [10, 11, 12, 13, 14, 15],
            "labels": [
                "bug",
                "feature",
                "bug",
                "question",
                "feature",
                "question",
            ],
            "title": ["A", "B", "C", "D", "E", "F"],
            "body": ["a", "b", "c", "d", "e", "f"],
        }
    )
    data.to_csv(input_path, index=False)

    counts = write_split_files(
        input_path=input_path,
        train_path=train_path,
        validation_path=validation_path,
        train_indices=np.array([4, 0, 2]),
        validation_indices=np.array([5, 1, 3]),
        chunk_size=2,
    )

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)

    assert counts == {"train_rows": 3, "validation_rows": 3}
    assert train["id"].tolist() == [10, 12, 14]
    assert validation["id"].tolist() == [11, 13, 15]

    assert set(train["id"]).isdisjoint(set(validation["id"]))
    assert len(train) + len(validation) == len(data)


def test_write_split_files_handles_empty_first_chunk(tmp_path):
    """Verify CSV headers when the first chunk has no training rows."""
    input_path = tmp_path / "issues.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "labels": ["bug", "bug", "feature", "feature"],
        }
    ).to_csv(input_path, index=False)

    write_split_files(
        input_path=input_path,
        train_path=train_path,
        validation_path=validation_path,
        train_indices=np.array([2, 3]),
        validation_indices=np.array([0, 1]),
        chunk_size=2,
    )

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)

    assert train["id"].tolist() == [3, 4]
    assert validation["id"].tolist() == [1, 2]
    assert train.columns.tolist() == ["id", "labels"]


def test_write_split_files_rejects_overlapping_indices(tmp_path):
    """Verify that overlapping split indices raise an error."""
    with pytest.raises(ValueError, match="complete partition"):
        write_split_files(
            input_path=tmp_path / "issues.csv",
            train_path=tmp_path / "train.csv",
            validation_path=tmp_path / "validation.csv",
            train_indices=np.array([0, 1]),
            validation_indices=np.array([1, 2]),
        )


def test_write_split_files_rejects_row_count_mismatch(tmp_path):
    """Verify that source row count must match split indices."""
    input_path = tmp_path / "issues.csv"

    pd.DataFrame({"labels": ["bug", "feature", "question"]}).to_csv(
        input_path, index=False
    )

    with pytest.raises(ValueError, match="row count"):
        write_split_files(
            input_path=input_path,
            train_path=tmp_path / "train.csv",
            validation_path=tmp_path / "validation.csv",
            train_indices=np.array([0]),
            validation_indices=np.array([1]),
            chunk_size=2,
        )


def test_write_split_files_preserves_outputs_on_failure(tmp_path):
    """Verify that failed splitting does not replace existing files."""
    input_path = tmp_path / "issues.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    pd.DataFrame(
        {
            "labels": ["bug", "feature", "question"],
        }
    ).to_csv(input_path, index=False)

    train_path.write_text("original train", encoding="utf-8")
    validation_path.write_text("original validation", encoding="utf-8")

    with pytest.raises(ValueError, match="row count"):
        write_split_files(
            input_path=input_path,
            train_path=train_path,
            validation_path=validation_path,
            train_indices=np.array([0]),
            validation_indices=np.array([1]),
            chunk_size=2,
        )

    assert train_path.read_text(encoding="utf-8") == "original train"
    assert validation_path.read_text(encoding="utf-8") == "original validation"
