import hashlib

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.preprocessing.build_dataset import (
    SCHEMA,
    build_split,
    display_path,
    process_chunk,
    sha256_file,
)


def test_process_chunk_preserves_title_and_labels():
    data = pd.DataFrame(
        {
            "title": ["Bug report", "Documentation issue"],
            "body": [
                "# Error\nSomething failed",
                "Read the **documentation**",
            ],
            "labels": ["bug", "documentation"],
        }
    )

    result = process_chunk(data)

    assert result["title"].tolist() == data["title"].tolist()
    assert result["labels"].tolist() == data["labels"].tolist()
    assert result["body"].tolist() == [
        "Error\nSomething failed",
        "Read the documentation",
    ]


def test_sha256_file(tmp_path):
    path = tmp_path / "sample.txt"
    content = b"hello world"
    path.write_bytes(content)

    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


def test_build_split_round_trip(tmp_path):
    source = tmp_path / "train.csv"
    destination = tmp_path / "train.parquet"

    data = pd.DataFrame(
        {
            "title": ["First issue", "Second issue", "Third"],
            "body": [
                "# Heading\nA problem",
                "Some **bold** text",
                "",
            ],
            "labels": ["bug", "feature", "question"],
        }
    )

    data.to_csv(source, index=False)

    result = build_split(
        source,
        destination,
        chunk_size=2,
    )

    saved = pd.read_parquet(destination)

    assert result["rows"] == 3
    assert len(saved) == 3
    assert saved["title"].tolist() == data["title"].tolist()
    assert saved["labels"].tolist() == data["labels"].tolist()
    assert saved["body"].tolist() == [
        "Heading\nA problem",
        "Some bold text",
        "",
    ]
    assert pq.read_schema(destination) == SCHEMA
    assert result["output_sha256"] == sha256_file(destination)


def test_build_split_rejects_invalid_chunk_size(tmp_path):
    with pytest.raises(ValueError, match="chunk_size"):
        build_split(
            tmp_path / "input.csv",
            tmp_path / "output.parquet",
            chunk_size=0,
        )


def test_build_split_rejects_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_split(
            tmp_path / "missing.csv",
            tmp_path / "output.parquet",
        )


def test_build_split_does_not_overwrite_existing_output(tmp_path):
    source = tmp_path / "train.csv"
    destination = tmp_path / "train.parquet"

    pd.DataFrame(
        {
            "title": ["Example"],
            "body": ["Some text"],
            "labels": ["bug"],
        }
    ).to_csv(source, index=False)

    destination.write_bytes(b"existing data")

    with pytest.raises(FileExistsError):
        build_split(source, destination, chunk_size=1)

    assert destination.read_bytes() == b"existing data"


def test_build_split_cleans_temporary_file_on_failure(tmp_path):
    source = tmp_path / "train.csv"
    destination = tmp_path / "train.parquet"

    source.write_text(
        "title,body,labels\nFirst,valid,bug\nSecond,valid,feature\nThird,valid\n",
        encoding="utf-8",
    )

    # A malformed row has an empty label; test the actual
    # policy separately if empty labels are permitted.
    with pytest.raises(ValueError, match="Empty label"):
        build_split(source, destination, chunk_size=2)

    assert not destination.exists()
    assert not destination.with_suffix(".parquet.tmp").exists()


def test_display_path_outside_project(tmp_path):
    path = tmp_path / "train.csv"

    assert display_path(path) == str(path)
