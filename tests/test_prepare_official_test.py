import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.models.structured_ablation_experiment import EXPERIMENTS
from src.preprocessing.prepare_official_test import (
    COLUMNS,
    SCHEMA,
    prepare_official_test,
)
from src.preprocessing.structured import preprocess_structured


@pytest.fixture
def sample_rows():
    return [
        {
            "id": 101,
            "labels": "bug",
            "title": "Login fails",
            "body": "## Error\nLogin fails with `ValueError`.",
            "author_association": "NONE",
        },
        {
            "id": 102,
            "labels": "feature",
            "title": "Add dark mode",
            "body": "",
            "author_association": "CONTRIBUTOR",
        },
        {
            "id": 101,
            "labels": "question",
            "title": "How does login work?",
            "body": "See https://example.com/docs?token=123",
            "author_association": "MEMBER",
        },
    ]


def write_csv(path, rows):
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)


def test_preserves_metadata_and_row_order(tmp_path, sample_rows):
    source = tmp_path / "test.csv"
    output = tmp_path / "test.parquet"
    write_csv(source, sample_rows)

    rows = prepare_official_test(source, output, chunk_size=2)
    actual = pq.read_table(output).to_pandas()

    assert rows == len(sample_rows)
    assert actual["id"].tolist() == [101, 102, 101]
    assert actual["labels"].tolist() == [
        "bug",
        "feature",
        "question",
    ]
    assert actual["title"].tolist() == [
        "Login fails",
        "Add dark mode",
        "How does login work?",
    ]
    assert actual["author_association"].tolist() == [
        "NONE",
        "CONTRIBUTOR",
        "MEMBER",
    ]


def test_preprocesses_only_body_with_e4(tmp_path, sample_rows):
    source = tmp_path / "test.csv"
    output = tmp_path / "test.parquet"
    write_csv(source, sample_rows)

    prepare_official_test(source, output, chunk_size=2)
    actual = pq.read_table(output).to_pandas()

    config = EXPERIMENTS["E4"]

    expected = [preprocess_structured(row["body"], config) for row in sample_rows]

    assert actual["body"].tolist() == expected
    assert actual["title"].tolist() == [row["title"] for row in sample_rows]


def test_preserves_empty_body(tmp_path, sample_rows):
    source = tmp_path / "test.csv"
    output = tmp_path / "test.parquet"
    write_csv(source, sample_rows)

    prepare_official_test(source, output)

    actual = pq.read_table(output).to_pandas()

    assert actual.loc[1, "body"] == ""


def test_output_schema_and_row_count(tmp_path, sample_rows):
    source = tmp_path / "test.csv"
    output = tmp_path / "test.parquet"
    write_csv(source, sample_rows)

    count = prepare_official_test(source, output, chunk_size=1)

    metadata = pq.read_metadata(output)

    assert count == 3
    assert metadata.num_rows == 3
    assert metadata.schema.to_arrow_schema() == SCHEMA


def test_rejects_existing_output(tmp_path, sample_rows):
    source = tmp_path / "test.csv"
    output = tmp_path / "test.parquet"
    write_csv(source, sample_rows)
    output.write_bytes(b"original")

    with pytest.raises(FileExistsError):
        prepare_official_test(source, output)

    assert output.read_bytes() == b"original"


def test_rejects_invalid_chunk_size(tmp_path):
    with pytest.raises(ValueError, match="chunk_size"):
        prepare_official_test(
            tmp_path / "missing.csv",
            tmp_path / "test.parquet",
            chunk_size=0,
        )


def test_rejects_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare_official_test(
            tmp_path / "missing.csv",
            tmp_path / "test.parquet",
        )


def test_rejects_empty_labels_and_cleans_temporary_file(tmp_path, sample_rows):
    source = tmp_path / "test.csv"
    output = tmp_path / "test.parquet"

    sample_rows[2]["labels"] = ""
    write_csv(source, sample_rows)

    with pytest.raises(ValueError, match="Empty labels"):
        prepare_official_test(source, output, chunk_size=2)

    assert not output.exists()
    assert not (tmp_path / "test.parquet.tmp").exists()


def test_rejects_empty_source(tmp_path):
    source = tmp_path / "test.csv"
    output = tmp_path / "test.parquet"

    pd.DataFrame(columns=COLUMNS).to_csv(source, index=False)

    with pytest.raises(ValueError, match="no rows"):
        prepare_official_test(source, output)

    assert not output.exists()
    assert not (tmp_path / "test.parquet.tmp").exists()


def test_rejects_existing_temporary_file(tmp_path, sample_rows):
    source = tmp_path / "test.csv"
    output = tmp_path / "test.parquet"
    temporary = tmp_path / "test.parquet.tmp"

    write_csv(source, sample_rows)
    temporary.write_bytes(b"existing temporary file")

    with pytest.raises(FileExistsError):
        prepare_official_test(source, output)

    assert temporary.read_bytes() == b"existing temporary file"
    assert not output.exists()
