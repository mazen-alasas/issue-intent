import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.data.split import (
    build_split_report,
    compute_class_distribution,
    compute_file_sha256,
    count_cross_split_text_overlap,
    create_split_indices,
    load_labels,
    run_split_pipeline,
    save_split_report,
    validate_source_row_integrity,
    write_split_files,
)


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


def test_compute_file_sha256_matches_known_digest(tmp_path):
    """Verify SHA-256 against a known digest."""
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"abc")

    digest = compute_file_sha256(file_path, block_size=2)

    assert digest == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_compute_file_sha256_rejects_invalid_block_size(tmp_path):
    """Verify that the hash function rejects invalid block sizes."""
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"test")

    with pytest.raises(ValueError, match="block_size"):
        compute_file_sha256(file_path, block_size=0)


def test_count_cross_split_text_overlap_detects_matches(tmp_path):
    """Verify exact text overlap is detected across splits."""
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    pd.DataFrame(
        {
            "title": ["Login fails", "Add dark mode"],
            "body": ["Error 500", "Feature request"],
        }
    ).to_csv(train_path, index=False)

    pd.DataFrame(
        {
            "title": ["Login fails", "Login fails", "Other issue"],
            "body": ["Error 500", "Different error", "Some text"],
        }
    ).to_csv(validation_path, index=False)

    overlap = count_cross_split_text_overlap(
        train_path,
        validation_path,
        chunk_size=1,
    )

    assert overlap == 1


def test_count_cross_split_text_overlap_handles_empty_text(tmp_path):
    """Verify that empty text fields are compared consistently."""
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    pd.DataFrame(
        {
            "title": ["Missing body", ""],
            "body": ["", "Missing title"],
        }
    ).to_csv(train_path, index=False)

    pd.DataFrame(
        {
            "title": ["Missing body", "Different title"],
            "body": ["", ""],
        }
    ).to_csv(validation_path, index=False)

    overlap = count_cross_split_text_overlap(
        train_path,
        validation_path,
        chunk_size=1,
    )

    assert overlap == 1


def test_count_cross_split_text_overlap_returns_zero(tmp_path):
    """Verify disjoint title-body pairs produce zero overlap."""
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    pd.DataFrame({"title": ["A"], "body": ["B"]}).to_csv(train_path, index=False)

    pd.DataFrame({"title": ["A"], "body": ["C"]}).to_csv(validation_path, index=False)

    assert (
        count_cross_split_text_overlap(
            train_path,
            validation_path,
        )
        == 0
    )


def test_compute_class_distribution(tmp_path):
    """Verify class counts and proportions across multiple chunks."""
    csv_path = tmp_path / "issues.csv"

    pd.DataFrame(
        {
            "labels": [
                "bug",
                "bug",
                "feature",
                "bug",
                "question",
                "feature",
            ]
        }
    ).to_csv(csv_path, index=False)

    distribution = compute_class_distribution(
        csv_path,
        chunk_size=2,
    )

    assert distribution["bug"]["count"] == 3
    assert distribution["bug"]["proportion"] == pytest.approx(0.5)

    assert distribution["feature"]["count"] == 2
    assert distribution["feature"]["proportion"] == pytest.approx(2 / 6)

    assert distribution["question"]["count"] == 1
    assert distribution["question"]["proportion"] == pytest.approx(1 / 6)

    assert sum(item["count"] for item in distribution.values()) == 6


def test_compute_class_distribution_rejects_invalid_chunk_size(tmp_path):
    """Verify that invalid chunk sizes raise an error."""
    with pytest.raises(ValueError, match="chunk_size"):
        compute_class_distribution(
            tmp_path / "issues.csv",
            chunk_size=0,
        )


def test_build_split_report_success(tmp_path):
    """Verify a valid split produces complete report metadata."""
    source_path = tmp_path / "source.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    source = pd.DataFrame(
        {
            "labels": ["bug", "bug", "feature", "question"],
            "title": ["A", "B", "C", "D"],
            "body": ["a", "b", "c", "d"],
        }
    )
    source.to_csv(source_path, index=False)
    source.iloc[:3].to_csv(train_path, index=False)
    source.iloc[3:].to_csv(validation_path, index=False)

    report = build_split_report(
        source_path,
        train_path,
        validation_path,
        source_rows=4,
        train_rows=3,
        validation_rows=1,
        chunk_size=2,
    )

    assert report["row_counts"] == {
        "source": 4,
        "train": 3,
        "validation": 1,
    }
    assert report["integrity"]["row_counts_match"] is True
    assert report["integrity"]["cross_split_text_overlap"] == 0
    assert len(report["sha256"]["train"]) == 64
    assert report["class_distribution"]["source"]["bug"]["count"] == 2


def test_build_split_report_rejects_invalid_counts(tmp_path):
    """Verify the report rejects incorrect expected row counts."""
    source_path = tmp_path / "source.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    data = pd.DataFrame(
        {
            "labels": ["bug", "feature"],
            "title": ["A", "B"],
            "body": ["a", "b"],
        }
    )
    data.to_csv(source_path, index=False)
    data.iloc[:1].to_csv(train_path, index=False)
    data.iloc[1:].to_csv(validation_path, index=False)

    with pytest.raises(ValueError, match="row counts"):
        build_split_report(
            source_path,
            train_path,
            validation_path,
            source_rows=3,
            train_rows=1,
            validation_rows=1,
        )


def test_build_split_report_rejects_text_overlap(tmp_path):
    """Verify the report rejects identical text across splits."""
    source_path = tmp_path / "source.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    data = pd.DataFrame(
        {
            "labels": ["bug", "bug"],
            "title": ["Same", "Same"],
            "body": ["Text", "Text"],
        }
    )
    data.to_csv(source_path, index=False)
    data.iloc[:1].to_csv(train_path, index=False)
    data.iloc[1:].to_csv(validation_path, index=False)

    with pytest.raises(ValueError, match="overlap"):
        build_split_report(
            source_path,
            train_path,
            validation_path,
            source_rows=2,
            train_rows=1,
            validation_rows=1,
        )


def test_build_split_report_rejects_class_count_mismatch(tmp_path):
    """Verify that class counts remain consistent across splits."""
    source_path = tmp_path / "source.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    source = pd.DataFrame(
        {
            "labels": ["bug", "bug", "feature", "question"],
            "title": ["A", "B", "C", "D"],
            "body": ["a", "b", "c", "d"],
        }
    )

    train = source.iloc[:3].copy()
    validation = source.iloc[3:].copy()

    train.loc[train.index[0], "labels"] = "feature"

    source.to_csv(source_path, index=False)
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)

    with pytest.raises(ValueError, match="Class count mismatch"):
        build_split_report(
            source_path,
            train_path,
            validation_path,
            source_rows=4,
            train_rows=3,
            validation_rows=1,
        )


def test_save_split_report_creates_valid_json(tmp_path):
    """Verify the report is saved as readable JSON."""
    report_path = tmp_path / "reports" / "split_report.json"

    report = {
        "split_config": {
            "validation_size": 0.15,
            "random_state": 42,
        },
        "integrity": {
            "cross_split_text_overlap": 0,
        },
    }

    save_split_report(report, report_path)

    saved = json.loads(report_path.read_text(encoding="utf-8"))

    assert saved == report


def test_save_split_report_replaces_existing_report(tmp_path):
    """Verify an existing report is replaced successfully."""
    report_path = tmp_path / "split_report.json"
    report_path.write_text('{"old": true}', encoding="utf-8")

    save_split_report({"new": True}, report_path)

    saved = json.loads(report_path.read_text(encoding="utf-8"))

    assert saved == {"new": True}


def test_run_split_pipeline_end_to_end(tmp_path):
    """Verify the complete pipeline creates valid split artifacts."""
    input_path = tmp_path / "issues.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    report_path = tmp_path / "split_report.json"

    data = pd.DataFrame(
        {
            "id": range(40),
            "labels": ["bug"] * 20 + ["feature"] * 20,
            "title": [f"Issue {i}" for i in range(40)],
            "body": [f"Description {i}" for i in range(40)],
        }
    )
    data.to_csv(input_path, index=False)

    report = run_split_pipeline(
        input_path=input_path,
        train_path=train_path,
        validation_path=validation_path,
        report_path=report_path,
        validation_size=0.25,
        random_state=42,
        chunk_size=7,
    )

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert len(train) == 30
    assert len(validation) == 10

    assert report == saved_report
    assert report["integrity"]["cross_split_text_overlap"] == 0

    assert set(train["id"]).isdisjoint(set(validation["id"]))
    assert set(train["id"]) | set(validation["id"]) == set(data["id"])

    assert report["sha256"]["train"] == compute_file_sha256(train_path)
    assert report["sha256"]["validation"] == compute_file_sha256(validation_path)


def test_run_split_pipeline_does_not_publish_report_on_failure(
    tmp_path,
    monkeypatch,
):
    """Verify failed validation leaves no published success report."""
    import src.data.split as split_module

    input_path = tmp_path / "issues.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    report_path = tmp_path / "split_report.json"

    pd.DataFrame(
        {
            "labels": ["bug"] * 10 + ["feature"] * 10,
            "title": [f"Issue {i}" for i in range(20)],
            "body": [f"Body {i}" for i in range(20)],
        }
    ).to_csv(input_path, index=False)

    report_path.write_text('{"old": true}', encoding="utf-8")

    def fail_validation(*args, **kwargs):
        """Simulate an integrity validation failure.

        Args:
            *args: Unused positional arguments.
            **kwargs: Unused keyword arguments.

        Raises:
            ValueError: Always, to simulate failed validation.
        """
        raise ValueError("Simulated integrity failure")

    monkeypatch.setattr(
        split_module,
        "build_split_report",
        fail_validation,
    )

    with pytest.raises(ValueError, match="Simulated integrity failure"):
        run_split_pipeline(
            input_path=input_path,
            train_path=train_path,
            validation_path=validation_path,
            report_path=report_path,
            validation_size=0.2,
            chunk_size=5,
        )

    assert not report_path.exists()


def test_run_split_pipeline_is_reproducible(tmp_path):
    """Verify identical inputs and seeds produce identical split files."""
    input_path = tmp_path / "issues.csv"

    data = pd.DataFrame(
        {
            "id": range(40),
            "labels": ["bug"] * 20 + ["feature"] * 20,
            "title": [f"Issue {i}" for i in range(40)],
            "body": [f"Description {i}" for i in range(40)],
        }
    )
    data.to_csv(input_path, index=False)

    reports = []

    for run_number in (1, 2):
        output_dir = tmp_path / f"run_{run_number}"

        report = run_split_pipeline(
            input_path=input_path,
            train_path=output_dir / "train.csv",
            validation_path=output_dir / "validation.csv",
            report_path=output_dir / "split_report.json",
            validation_size=0.25,
            random_state=42,
            chunk_size=7,
        )
        reports.append(report)

    assert reports[0]["sha256"] == reports[1]["sha256"]
    assert reports[0]["row_counts"] == reports[1]["row_counts"]
    assert reports[0]["class_distribution"] == reports[1]["class_distribution"]


def test_split_files_preserve_exact_source_rows(tmp_path):
    """Verify split outputs contain every original row exactly once."""
    input_path = tmp_path / "issues.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    source = pd.DataFrame(
        {
            "id": range(12),
            "labels": ["bug"] * 6 + ["feature"] * 6,
            "title": [f"Title {i}" for i in range(12)],
            "body": [f"Body {i}" for i in range(12)],
        }
    )
    source.to_csv(input_path, index=False)

    train_indices = np.array([8, 0, 4, 2, 10, 6])
    validation_indices = np.array([11, 1, 9, 3, 7, 5])

    write_split_files(
        input_path=input_path,
        train_path=train_path,
        validation_path=validation_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        chunk_size=3,
    )

    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)

    train["original_position"] = np.sort(train_indices)
    validation["original_position"] = np.sort(validation_indices)

    reconstructed = (
        pd.concat([train, validation], ignore_index=True)
        .sort_values("original_position")
        .drop(columns="original_position")
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(
        reconstructed,
        source,
        check_dtype=True,
    )


def test_write_split_files_preserves_special_text_values(tmp_path):
    """Verify literal NA values and empty strings survive splitting."""
    input_path = tmp_path / "issues.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    source = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "labels": ["bug", "feature", "bug", "feature"],
            "title": ["NA", "", "NULL", "Normal issue"],
            "body": ["", "NA", "Some text", ""],
        }
    )
    source.to_csv(input_path, index=False)

    write_split_files(
        input_path=input_path,
        train_path=train_path,
        validation_path=validation_path,
        train_indices=np.array([0, 2]),
        validation_indices=np.array([1, 3]),
        chunk_size=2,
    )

    train = pd.read_csv(train_path, keep_default_na=False)
    validation = pd.read_csv(
        validation_path,
        keep_default_na=False,
    )

    assert train["title"].tolist() == ["NA", "NULL"]
    assert train["body"].tolist() == ["", "Some text"]

    assert validation["title"].tolist() == ["", "Normal issue"]
    assert validation["body"].tolist() == ["NA", ""]


def test_validate_source_row_integrity_success(tmp_path):
    """Verify valid splits preserve all source rows."""
    source_path = tmp_path / "source.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    source = pd.DataFrame(
        {
            "labels": ["bug", "bug", "feature", "question"],
            "title": ["A", "B", "C", "D"],
            "body": ["a", "b", "c", "d"],
        }
    )

    source.to_csv(source_path, index=False)
    source.iloc[:3].to_csv(train_path, index=False)
    source.iloc[3:].to_csv(validation_path, index=False)

    validate_source_row_integrity(
        source_path,
        train_path,
        validation_path,
        chunk_size=2,
    )


def test_validate_source_row_integrity_rejects_modified_row(tmp_path):
    """Verify row-content changes are detected despite equal counts."""
    source_path = tmp_path / "source.csv"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"

    source = pd.DataFrame(
        {
            "labels": ["bug", "bug", "feature", "question"],
            "title": ["A", "B", "C", "D"],
            "body": ["a", "b", "c", "d"],
        }
    )

    train = source.iloc[:3].copy()
    validation = source.iloc[3:].copy()

    train.loc[train.index[0], "body"] = "modified"

    source.to_csv(source_path, index=False)
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)

    with pytest.raises(ValueError, match="Source row integrity"):
        validate_source_row_integrity(
            source_path,
            train_path,
            validation_path,
            chunk_size=2,
        )
