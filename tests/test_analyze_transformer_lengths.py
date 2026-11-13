import pandas as pd
import pytest

from src.preprocessing.analyze_transformer_lengths import (
    sample_parquet,
)


@pytest.fixture
def parquet_path(tmp_path):
    path = tmp_path / "issues.parquet"

    pd.DataFrame(
        {
            "title": [f"title {i}" for i in range(100)],
            "body": [f"body {i}" for i in range(100)],
            "labels": ["bug", "feature"] * 50,
        }
    ).to_parquet(path, index=False)

    return path


def test_sample_size(parquet_path):
    sample = sample_parquet(parquet_path, sample_size=20)

    assert len(sample) == 20
    assert sample["title"].is_unique


def test_reproducibility(parquet_path):
    first = sample_parquet(parquet_path, 20, seed=42)
    second = sample_parquet(parquet_path, 20, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_sample_larger_than_dataset(parquet_path):
    sample = sample_parquet(parquet_path, 200)

    assert len(sample) == 100


def test_invalid_sample_size(parquet_path):
    with pytest.raises(ValueError):
        sample_parquet(parquet_path, 0)


def test_invalid_batch_size(parquet_path):
    with pytest.raises(ValueError):
        sample_parquet(parquet_path, 10, batch_size=0)


def test_empty_parquet(tmp_path):
    path = tmp_path / "empty.parquet"

    pd.DataFrame(
        {
            "title": pd.Series(dtype="str"),
            "body": pd.Series(dtype="str"),
            "labels": pd.Series(dtype="str"),
        }
    ).to_parquet(path)

    with pytest.raises(ValueError):
        sample_parquet(path, 10)
