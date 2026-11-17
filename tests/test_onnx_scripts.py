import numpy as np
import pandas as pd
import pytest

from scripts.benchmark_onnx import make_batch, sample_data, timing
from scripts.validate_onnx import make_batch as make_validation_batch


def test_batch_padding_and_attention_mask():
    cases = [
        {"input_ids": [101, 10, 102], "attention_mask": [1, 1, 1]},
        {"input_ids": [101, 20, 30, 40, 102], "attention_mask": [1] * 5},
    ]
    for batch_fn in (make_batch, make_validation_batch):
        batch = batch_fn(cases, pad_id=0)
        assert batch["input_ids"].shape == (2, 5)
        assert batch["input_ids"].dtype == np.int64
        assert batch["attention_mask"].dtype == np.int64
        assert batch["input_ids"][0].tolist() == [101, 10, 102, 0, 0]
        assert batch["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
        assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1, 1]


def test_batch_size_one_and_variable_length():
    short = [{"input_ids": [101, 102], "attention_mask": [1, 1]}]
    long = [{"input_ids": [101] + [5] * 254 + [102], "attention_mask": [1] * 256}]
    assert make_batch(short, 0)["input_ids"].shape == (1, 2)
    assert make_batch(long, 0)["input_ids"].shape == (1, 256)


def test_stratified_sample_is_reproducible_and_normalizes_null(tmp_path):
    rows = []
    for label, n in (
        ("bug", 20),
        ("feature", 12),
        ("question", 4),
        ("documentation", 4),
    ):
        for i in range(n):
            rows.append(
                {
                    "id": len(rows),
                    "labels": label,
                    "title": f"Issue {i}",
                    "body": None if i == 0 else "Details",
                }
            )
    path = tmp_path / "test.parquet"
    pd.DataFrame(rows).to_parquet(path)
    first = sample_data(path, 20, 42)
    second = sample_data(path, 20, 42)
    assert first.equals(second)
    assert len(first) == 20
    assert first["labels"].value_counts().to_dict() == {
        "bug": 10,
        "feature": 6,
        "question": 2,
        "documentation": 2,
    }
    assert first["body"].notna().all()
    assert all(isinstance(value, str) for value in first["body"])


def test_sample_rejects_oversize(tmp_path):
    path = tmp_path / "test.parquet"
    pd.DataFrame(
        [{"id": 1, "labels": "bug", "title": "Example", "body": ""}]
    ).to_parquet(path)
    with pytest.raises(ValueError, match="exceeds dataset size"):
        sample_data(path, 2, 42)


def test_sample_rejects_unknown_label(tmp_path):
    path = tmp_path / "test.parquet"
    pd.DataFrame(
        [{"id": 1, "labels": "unknown", "title": "Example", "body": ""}]
    ).to_parquet(path)
    with pytest.raises(ValueError, match="Unexpected label"):
        sample_data(path, 1, 42)


def test_timing_counts_calls_and_issues():
    batches = [
        {"input_ids": np.ones((2, 3), dtype=np.int64)},
        {"input_ids": np.ones((1, 4), dtype=np.int64)},
    ]
    calls = []

    def forward(batch):
        calls.append(len(batch["input_ids"]))

    result = timing(forward, batches, warmup=1, repeats=2, label="Test")
    assert calls == [2, 1, 2, 1, 2, 1]
    assert result["batch_calls"] == 4
    assert result["median_batch_ms"] >= 0
    assert result["p95_batch_ms"] >= 0
    assert result["throughput_issues_per_second"] > 0
