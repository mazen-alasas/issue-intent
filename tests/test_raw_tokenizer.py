from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from src.api.raw_tokenizer import RawTokenizerAdapter, load_onnx_tokenizer


def test_raw_tokenizer_loads_saved_graph(tmp_path: Path):
    vocabulary = {"[UNK]": 0, "[CLS]": 101, "[SEP]": 102, "hello": 103}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    loaded = load_onnx_tokenizer(tmp_path)
    assert loaded.cls_token_id == 101
    assert loaded.sep_token_id == 102
    assert loaded("hello", add_special_tokens=False, truncation=False) == {
        "input_ids": [103]
    }


def test_raw_tokenizer_requires_fast_graph(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        load_onnx_tokenizer(tmp_path)


def test_raw_tokenizer_does_not_allow_implicit_special_tokens():
    tokenizer = Tokenizer(
        models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}, unk_token="[UNK]")
    )
    adapter = RawTokenizerAdapter(tokenizer)
    with pytest.raises(ValueError, match="raw"):
        adapter("hi", add_special_tokens=True, truncation=False)
