import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from torch.utils.data import DataLoader

from src.data.neural_dataset import LABEL_TO_ID
from src.data.transformer_dataset import TransformerInputConfig
from src.data.transformer_streaming import TransformerIssueDataset


class DummyTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        **kwargs,
    ):
        token_ids = [ord(char) + 1000 for char in str(text)]

        return {"input_ids": token_ids}


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
    ):
        assert add_special_tokens is False
        assert truncation is False

        tokens = text.split()
        return {"input_ids": list(range(10, 10 + len(tokens)))}


def make_dataset(tmp_path, shuffle, seed=42, buffer_size=4):
    path = tmp_path / "issues.parquet"

    table = pa.table(
        {
            "title": [f"title-{i}" for i in range(20)],
            "body": [f"body-{i}" for i in range(20)],
            "labels": ["bug"] * 10 + ["feature"] * 10,
        }
    )

    pq.write_table(
        table,
        path,
        row_group_size=5,
    )

    dataset = TransformerIssueDataset(
        path=path,
        tokenizer=DummyTokenizer(),
        config=TransformerInputConfig(
            max_length=32,
            title_budget=12,
        ),
        shuffle=shuffle,
        seed=seed,
        shuffle_buffer_size=buffer_size,
    )

    return dataset


def collect_examples(dataset):
    return [(tuple(row["input_ids"]), row["labels"]) for row in dataset]


def test_shuffle_preserves_all_examples(tmp_path):
    dataset = make_dataset(
        tmp_path,
        shuffle=True,
    )

    shuffled = collect_examples(dataset)

    dataset.shuffle = False
    original = collect_examples(dataset)

    assert len(shuffled) == len(original) == 20
    assert sorted(shuffled) == sorted(original)


def test_shuffle_is_reproducible(tmp_path):
    dataset = make_dataset(
        tmp_path,
        shuffle=True,
    )

    dataset.set_epoch(0)
    first = collect_examples(dataset)

    dataset.set_epoch(0)
    second = collect_examples(dataset)

    assert first == second


def test_shuffle_changes_across_epochs(tmp_path):
    dataset = make_dataset(
        tmp_path,
        shuffle=True,
    )

    dataset.set_epoch(0)
    first = collect_examples(dataset)

    dataset.set_epoch(1)
    second = collect_examples(dataset)

    assert first != second
    assert sorted(first) == sorted(second)


def test_invalid_shuffle_buffer_size(tmp_path):
    with pytest.raises(ValueError, match="shuffle_buffer_size"):
        make_dataset(
            tmp_path,
            shuffle=True,
            buffer_size=0,
        )


def test_invalid_epoch(tmp_path):
    dataset = make_dataset(
        tmp_path,
        shuffle=True,
    )

    with pytest.raises(ValueError, match="epoch"):
        dataset.set_epoch(-1)


def test_streaming_reads_all_rows(tmp_path):
    path = tmp_path / "issues.parquet"

    pd.DataFrame(
        {
            "title": ["first issue", "second issue"],
            "body": ["first body", "second body"],
            "labels": ["bug", "feature"],
        }
    ).to_parquet(path)

    dataset = TransformerIssueDataset(
        path=path,
        tokenizer=FakeTokenizer(),
        config=TransformerInputConfig(),
    )

    rows = list(dataset)

    assert len(rows) == 2
    assert [row["labels"] for row in rows] == [
        LABEL_TO_ID["bug"],
        LABEL_TO_ID["feature"],
    ]
    assert all(len(row["input_ids"]) <= 256 for row in rows)


def test_streaming_reproducible(tmp_path):
    path = tmp_path / "issues.parquet"

    pd.DataFrame(
        {
            "title": [f"title {i}" for i in range(20)],
            "body": [f"body {i}" for i in range(20)],
            "labels": ["bug"] * 20,
        }
    ).to_parquet(path, row_group_size=5)

    dataset = TransformerIssueDataset(
        path=path,
        tokenizer=FakeTokenizer(),
        config=TransformerInputConfig(),
        shuffle=True,
        seed=42,
    )

    first = list(dataset)
    second = list(dataset)

    assert first == second


def test_multiple_workers_no_duplicates(tmp_path):
    path = tmp_path / "issues.parquet"

    pd.DataFrame(
        {
            "title": [f"title {i}" for i in range(20)],
            "body": ["body"] * 20,
            "labels": ["bug"] * 20,
        }
    ).to_parquet(path, row_group_size=5)

    dataset = TransformerIssueDataset(
        path=path,
        tokenizer=FakeTokenizer(),
        config=TransformerInputConfig(),
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
    )

    rows = list(loader)

    assert len(rows) == 20
