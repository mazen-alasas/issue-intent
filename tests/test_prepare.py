import json

import pandas as pd

from src.data.prepare import prepare_dataset


def test_prepare_dataset(tmp_path):
    """Verify deduplication, conflict removal, and report generation.

    Create a small dataset containing consistent duplicates,
    conflicting labels, unique issues, and a missing body.

    Run the preparation pipeline with multiple chunks and verify
    that the correct rows are retained, removal counts are accurate,
    and the saved JSON report matches the returned statistics.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary
            directory for input and output files.
    """
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "clean.csv"
    report_path = tmp_path / "report.json"

    data = pd.DataFrame(
        [
            [1, "bug", "Crash", "App closes", "NONE"],
            [2, "bug", "Crash", "App closes", "OWNER"],
            [3, "feature", "Dark mode", "Add theme", "NONE"],
            [4, "bug", "Login", "Cannot log in", "NONE"],
            [5, "question", "Login", "Cannot log in", "NONE"],
            [6, "documentation", "Docs", None, "NONE"],
        ],
        columns=[
            "id",
            "labels",
            "title",
            "body",
            "author_association",
        ],
    )
    data.to_csv(raw_path, index=False)

    report = prepare_dataset(
        raw_path,
        output_path,
        report_path,
        chunk_size=2,
    )

    cleaned = pd.read_csv(output_path)

    assert report["total_rows"] == 6
    assert report["kept_rows"] == 3
    assert report["duplicate_rows_removed"] == 1
    assert report["conflicting_rows_removed"] == 2
    assert report["conflicting_groups"] == 1
    assert report["missing_body_rows"] == 1

    assert set(cleaned["id"]) == {1, 3, 6}
    assert not cleaned.duplicated(["title", "body"]).any()

    saved_report = json.loads(report_path.read_text())
    assert saved_report == report


def test_missing_title_is_preserved(tmp_path):
    """Verify that issues with missing titles are not discarded.

    Create a dataset containing an issue with a missing title but
    a valid body, then run the preparation pipeline with a
    single-row chunk size.

    Check that both issues are retained, the missing title is
    counted correctly, and the available body text is preserved.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary
            directory for test files.

    """
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "clean.csv"
    report_path = tmp_path / "report.json"

    data = pd.DataFrame(
        [
            [1, "bug", None, "Unexpected error", "NONE"],
            [2, "feature", "New option", "Add a toggle", "OWNER"],
        ],
        columns=[
            "id",
            "labels",
            "title",
            "body",
            "author_association",
        ],
    )
    data.to_csv(raw_path, index=False)

    report = prepare_dataset(
        raw_path,
        output_path,
        report_path,
        chunk_size=1,
    )

    cleaned = pd.read_csv(output_path).fillna("")

    assert report["kept_rows"] == 2
    assert report["missing_title_rows"] == 1
    assert "Unexpected error" in cleaned["body"].values


def test_conflicting_labels_across_chunks(tmp_path):
    """Verify that conflicting labels are detected across CSV chunks.

    Create two issues with identical title-body text but different
    labels, separated by another unique issue.

    Process one row per chunk to ensure conflicting records occur
    in different chunks, and verify that all conflicting rows
    are excluded while the unrelated issue is retained.

    Args:
        tmp_path (Path): Pytest fixture providing a temporary
            directory for test files.

    Returns:
        None: Assertions validate cross-chunk conflict handling.
    """
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "clean.csv"
    report_path = tmp_path / "report.json"

    data = pd.DataFrame(
        [
            [1, "bug", "Same issue", "Same description", "NONE"],
            [2, "feature", "Different issue", "Different body", "NONE"],
            [3, "question", "Same issue", "Same description", "NONE"],
        ],
        columns=[
            "id",
            "labels",
            "title",
            "body",
            "author_association",
        ],
    )
    data.to_csv(raw_path, index=False)

    report = prepare_dataset(
        raw_path,
        output_path,
        report_path,
        chunk_size=1,
    )

    cleaned = pd.read_csv(output_path)

    assert report["total_rows"] == 3
    assert report["conflicting_groups"] == 1
    assert report["conflicting_rows_removed"] == 2
    assert report["duplicate_rows_removed"] == 0
    assert report["kept_rows"] == 1
    assert cleaned["title"].tolist() == ["Different issue"]
