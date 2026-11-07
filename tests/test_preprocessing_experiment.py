import pandas as pd
import pytest

from src.models.train_title_body import combine_title_body


def test_combine_title_body_without_preprocessing() -> None:
    data = pd.DataFrame(
        {
            "title": ["Bug report"],
            "body": ["### Actual Behavior\nSomething failed."],
        }
    )

    result = combine_title_body(data, preprocessing="none")

    assert result.iloc[0] == ("Bug report\n\n### Actual Behavior\nSomething failed.")


def test_combine_title_body_with_minimal_preprocessing() -> None:
    data = pd.DataFrame(
        {
            "title": ["Bug report"],
            "body": ["<!-- Template instruction -->\nActual bug."],
        }
    )

    result = combine_title_body(data, preprocessing="minimal")

    assert "Template instruction" not in result.iloc[0]
    assert "Actual bug." in result.iloc[0]
    assert result.iloc[0].startswith("Bug report\n\n")


def test_combine_title_body_rejects_invalid_strategy() -> None:
    data = pd.DataFrame(
        {
            "title": ["Bug report"],
            "body": ["Something failed."],
        }
    )

    with pytest.raises(ValueError, match="Unsupported preprocessing"):
        combine_title_body(data, preprocessing="unknown")


def test_combine_title_body_preserves_missing_values() -> None:
    data = pd.DataFrame(
        {
            "title": [None],
            "body": [None],
        }
    )

    result = combine_title_body(data, preprocessing="minimal")

    assert result.iloc[0] == "\n\n"
