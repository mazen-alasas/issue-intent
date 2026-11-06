import pandas as pd
import pytest

from src.models.train_title_body import combine_title_body


def test_combine_title_body() -> None:
    """Combine titles and bodies without modifying their content."""
    data = pd.DataFrame(
        {
            "title": ["Application crashes", "Feature request"],
            "body": ["```python\nprint('error')\n```", "Add dark mode"],
        }
    )

    result = combine_title_body(data)

    assert result.tolist() == [
        "Application crashes\n\n```python\nprint('error')\n```",
        "Feature request\n\nAdd dark mode",
    ]


@pytest.mark.parametrize(
    ("title", "body", "expected"),
    [
        ("Bug report", "", "Bug report\n\n"),
        ("Feature", None, "Feature\n\n"),
        (None, "Some description", "\n\nSome description"),
    ],
)
def test_combine_title_body_missing_values(
    title: str | None,
    body: str | None,
    expected: str,
) -> None:
    """Handle missing titles and bodies without errors."""
    data = pd.DataFrame({"title": [title], "body": [body]})

    result = combine_title_body(data)

    assert result.iloc[0] == expected
