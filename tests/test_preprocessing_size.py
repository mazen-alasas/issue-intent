import pandas as pd
import pytest

from src.preprocessing.analyze_size import (
    compare_text_versions,
    summarize_lengths,
)


def test_summarize_lengths():
    text = pd.Series(["abc", "hello", ""])

    result = summarize_lengths(text)

    assert result["total_characters"] == 8
    assert result["mean"] == pytest.approx(8 / 3, abs=0.01)
    assert result["median"] == 3
    assert result["max"] == 5


def test_compare_text_versions():
    raw = pd.Series(["hello world", "abc", "test"])
    processed = pd.Series(["hello", "abc", "testing"])

    result = compare_text_versions(raw, processed)

    assert result["changed_issues"] == 2
    assert result["shortened_issues"] == 1
    assert result["expanded_issues"] == 1
    assert result["unchanged_length_issues"] == 1
    assert result["total_characters_removed"] == 3


def test_identical_text_has_no_reduction():
    text = pd.Series(["hello", "world"])

    result = compare_text_versions(text, text)

    assert result["changed_issues"] == 0
    assert result["reduction_pct"] == 0


def test_empty_text_has_no_division_error():
    raw = pd.Series(["", ""])
    processed = pd.Series(["", ""])

    result = compare_text_versions(raw, processed)

    assert result["reduction_pct"] == 0


def test_different_lengths_are_rejected():
    raw = pd.Series(["one", "two"])
    processed = pd.Series(["one"])

    with pytest.raises(ValueError, match="equal lengths"):
        compare_text_versions(raw, processed)
