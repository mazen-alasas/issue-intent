import pytest

from src.preprocessing.text import preprocess_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Hello\r\nWorld", "Hello\nWorld"),
        ("Hello\n\n\n\nWorld", "Hello\n\nWorld"),
        (
            "<!-- Template instruction -->\nActual bug",
            " \nActual bug",
        ),
        ("### Expected Behavior", "### Expected Behavior"),
        ("Use `LogisticRegression`", "Use `LogisticRegression`"),
        ("[Docs](https://example.com)", "[Docs](https://example.com)"),
        (None, ""),
        ("", ""),
    ],
)
def test_preprocess_text(raw: str | None, expected: str) -> None:
    assert preprocess_text(raw) == expected


def test_preserves_indentation_and_code() -> None:
    raw = "```python\n    value = 42\n```"

    assert preprocess_text(raw) == raw


def test_preserves_diagnostic_information() -> None:
    raw = (
        "### Actual Behavior\n"
        "```text\n"
        "Traceback (most recent call last):\n"
        "    File 'app.py', line 10\n"
        "ValueError: invalid input\n"
        "```"
    )

    assert preprocess_text(raw) == raw


def test_does_not_truncate_long_text() -> None:
    raw = "Important diagnostic information. " * 10_000

    assert preprocess_text(raw) == raw
