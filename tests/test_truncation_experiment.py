import pandas as pd
import pytest

from src.models.truncation_experiment import truncate_body


def test_truncate_body_preserves_title() -> None:
    data = pd.DataFrame(
        {
            "title": ["Application crashes"],
            "body": ["abcdefghij"],
            "labels": ["bug"],
        }
    )

    result = truncate_body(data, max_chars=5)

    assert result["title"].iloc[0] == "Application crashes"
    assert result["body"].iloc[0] == "abcde"
    assert result["labels"].iloc[0] == "bug"


def test_truncate_body_does_not_modify_original() -> None:
    data = pd.DataFrame(
        {
            "title": ["Issue"],
            "body": ["abcdefghij"],
        }
    )

    truncate_body(data, max_chars=4)

    assert data["body"].iloc[0] == "abcdefghij"


def test_truncate_body_handles_missing_body() -> None:
    data = pd.DataFrame(
        {
            "title": ["Missing description"],
            "body": [None],
        }
    )

    result = truncate_body(data, max_chars=10)

    assert result["body"].iloc[0] == ""


@pytest.mark.parametrize("max_chars", [0, -1, -100])
def test_truncate_body_rejects_invalid_limit(max_chars: int) -> None:
    data = pd.DataFrame({"title": ["Issue"], "body": ["Text"]})

    with pytest.raises(ValueError, match="max_chars must be positive"):
        truncate_body(data, max_chars=max_chars)
