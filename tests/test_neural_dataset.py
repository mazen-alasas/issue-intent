import pandas as pd
import pytest

from src.data.neural_dataset import (
    ParquetIssueDataset,
    build_vocab,
    collate_issues,
    encode,
    tokenize,
)


def test_tokenize_and_encode():
    vocab = {"<pad>": 0, "<unk>": 1, "login": 2, "error": 3}
    assert tokenize("Login ERROR!") == ["login", "error", "!"]
    assert encode("Login", "error unknown", vocab, max_length=5) == [2, 3, 1]
    assert len(encode("Login", "error " * 50, vocab, max_length=8)) == 8


def test_vocab_is_train_only(tmp_path):
    path = tmp_path / "train.parquet"
    pd.DataFrame(
        {
            "title": ["trainword", "trainword"],
            "body": ["bodyword", "bodyword"],
            "labels": ["bug", "feature"],
        }
    ).to_parquet(path)
    vocab = build_vocab(path)
    assert "trainword" in vocab and "bodyword" in vocab
    assert "validationonly" not in vocab
    dataset = ParquetIssueDataset(path, vocab)
    rows = list(dataset)
    assert len(rows) == 2
    assert {label for _, label in rows} == {0, 2}
    x, y = collate_issues(rows)
    assert x.shape[0] == y.shape[0] == 2
    assert x.shape[1] >= 5


def test_empty_vocab_config(tmp_path):
    with pytest.raises(ValueError):
        build_vocab(tmp_path / "missing", max_size=1)
