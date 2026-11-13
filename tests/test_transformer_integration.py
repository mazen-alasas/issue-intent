import pytest
from transformers import AutoTokenizer

from src.data.transformer_dataset import (
    TransformerInputConfig,
    encode_issue,
)


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return AutoTokenizer.from_pretrained(
            "distilbert/distilbert-base-uncased",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("Tokenizer not available in local cache")


def test_real_tokenizer_short_issue(tokenizer):
    result = encode_issue(
        "Database connection failed",
        "Unable to connect to PostgreSQL server.",
        tokenizer,
        TransformerInputConfig(),
    )

    ids = result["input_ids"]

    assert ids[0] == tokenizer.cls_token_id
    assert ids.count(tokenizer.sep_token_id) == 2
    assert len(ids) <= 256
    assert len(result["attention_mask"]) == len(ids)


def test_real_tokenizer_long_issue(tokenizer):
    result = encode_issue(
        "Database connection failed",
        " ".join(["connection timeout"] * 1000),
        tokenizer,
        TransformerInputConfig(
            max_length=256,
            title_budget=48,
        ),
    )

    assert len(result["input_ids"]) == 256
    assert len(result["attention_mask"]) == 256
