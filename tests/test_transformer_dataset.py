from src.data.transformer_dataset import (
    TransformerInputConfig,
    encode_issue,
)


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


def test_short_issue_preserves_all_tokens():
    result = encode_issue(
        "short title",
        "short body text",
        FakeTokenizer(),
        TransformerInputConfig(),
    )

    assert len(result["input_ids"]) == 8
    assert len(result["attention_mask"]) == 8


def test_long_issue_respects_max_length():
    result = encode_issue(
        " ".join(["title"] * 100),
        " ".join(["body"] * 1000),
        FakeTokenizer(),
        TransformerInputConfig(max_length=256, title_budget=48),
    )

    assert len(result["input_ids"]) == 256


def test_invalid_config():
    import pytest

    with pytest.raises(ValueError):
        TransformerInputConfig(max_length=8, title_budget=8)


def test_invalid_text():
    import pytest

    with pytest.raises(TypeError):
        encode_issue(
            None,
            "body",
            FakeTokenizer(),
            TransformerInputConfig(),
        )
