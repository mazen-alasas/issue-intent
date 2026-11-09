"""Tests for configurable structured preprocessing of GitHub issues.

Organized by transformation, followed by code-fence and integration checks.
"""

import pytest

from src.preprocessing.structured import (
    MAX_CODE_SIGNAL_CHARS,
    MAX_CODE_SIGNAL_LINES,
    StructuredPreprocessingConfig,
    extract_code_signals,
    preprocess_structured,
)

BACKTICK = chr(96)
STAR = chr(42)
FENCE = BACKTICK * 3


# ---------------------------------------------------------------------------
# Configuration and input validation
# ---------------------------------------------------------------------------


def test_default_configuration_enables_transformations():
    config = StructuredPreprocessingConfig()

    assert config.remove_html_comments
    assert config.normalize_markdown
    assert config.normalize_urls_paths
    assert config.remove_template_placeholders
    assert config.compress_repeated_blocks
    assert config.replace_fenced_code
    assert config.normalize_whitespace


def test_preprocessing_rejects_non_string_input():
    with pytest.raises(TypeError, match="text must be a string"):
        preprocess_structured(None)


# ---------------------------------------------------------------------------
# HTML comments
# ---------------------------------------------------------------------------


def test_removes_html_comments():
    text = "Before <!-- template instruction --> After"
    assert preprocess_structured(text) == "Before   After"


def test_preserves_text_outside_html_comments():
    text = "Expected behavior\n<!-- instruction -->\nApp should work."
    result = preprocess_structured(text)

    assert "Expected behavior" in result
    assert "App should work." in result
    assert "instruction" not in result


def test_prevents_word_merging():
    text = "hello<!-- comment -->world"
    assert preprocess_structured(text) == "hello world"


def test_can_disable_html_comment_removal():
    text = "hello<!-- comment -->world"
    config = StructuredPreprocessingConfig(remove_html_comments=False)
    assert preprocess_structured(text, config) == text


def test_preserves_unclosed_html_comment():
    text = "Before <!-- incomplete comment"
    assert preprocess_structured(text) == text


# ---------------------------------------------------------------------------
# Markdown normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("## Expected Behavior", "Expected Behavior"),
        (f"{STAR * 2}Application crashes{STAR * 2}", "Application crashes"),
        (f"{STAR}Important note{STAR}", "Important note"),
        ("[Documentation](https://example.com)", "Documentation"),
        (f"{BACKTICK}ValueError{BACKTICK}", "ValueError"),
        ("- First step", "First step"),
        ("1. Open settings", "1. Open settings"),
    ],
)
def test_normalizes_common_markdown(text, expected):
    assert preprocess_structured(text) == expected


def test_can_disable_markdown_normalization():
    text = f"## Expected Behavior\n{STAR * 2}Application crashes{STAR * 2}"
    config = StructuredPreprocessingConfig(
        normalize_markdown=False,
        replace_fenced_code=False,
    )
    assert preprocess_structured(text, config) == text


def test_preserves_inline_exception_names():
    text = f"The application raises {BACKTICK}ValueError{BACKTICK}."
    assert preprocess_structured(text) == "The application raises ValueError."


def test_preserves_template_headings():
    text = "## Steps to reproduce\n1. Open the application"
    assert preprocess_structured(text) == "Steps to reproduce\n1. Open the application"


# ---------------------------------------------------------------------------
# URL and absolute-path normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://example.com/api?token=abc123", "https://example.com/api"),
        (
            "https://github.com/pytorch/pytorch/issues/123",
            "https://github.com/pytorch/pytorch/issues/123",
        ),
        ("/home/user/project/config.yaml", "PATH/config.yaml"),
        ("src/models/train.py", "src/models/train.py"),
        ("config.yaml", "config.yaml"),
        ("ValueError", "ValueError"),
    ],
)
def test_normalizes_urls_and_paths(text, expected):
    assert preprocess_structured(text) == expected


def test_preserves_url_paths_without_corruption():
    text = (
        "See https://github.com/pytorch/pytorch/issues/123 "
        "and /home/user/project/config.yaml"
    )
    assert preprocess_structured(text) == (
        "See https://github.com/pytorch/pytorch/issues/123 and PATH/config.yaml"
    )


def test_can_disable_url_path_normalization():
    text = "https://example.com/api?token=abc123"
    config = StructuredPreprocessingConfig(normalize_urls_paths=False)
    assert preprocess_structured(text, config) == text


@pytest.mark.parametrize(
    "text",
    [
        "https://[invalid",
        "https://[::1",
        "https://example.com/[broken",
    ],
)
def test_preserves_malformed_urls_without_crashing(text):
    assert preprocess_structured(text) == text


def test_continues_processing_after_malformed_url():
    text = "Broken URL: https://[invalid\nValid URL: https://example.com/api?token=123"

    result = preprocess_structured(text)

    assert "https://[invalid" in result
    assert "https://example.com/api" in result
    assert "token=123" not in result


# ---------------------------------------------------------------------------
# Empty template placeholders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["- [ ]\n", "* [ ]\n", "+ [ ]\n", "  - [ ]\n", "- [  ]\n"],
)
def test_removes_empty_checkboxes(text):
    assert preprocess_structured(text) == ""


def test_preserves_checkboxes_with_content():
    text = "- [ ] Add unit tests"
    assert "[ ] Add unit tests" in preprocess_structured(text)


def test_preserves_checked_checkboxes():
    text = "- [x] Tests completed"
    assert "[x] Tests completed" in preprocess_structured(text)


def test_removes_empty_checkbox_between_content():
    text = "Before\n- [ ]\nAfter"
    assert preprocess_structured(text) == "Before\nAfter"


def test_can_disable_placeholder_removal():
    text = "- [ ]"
    config = StructuredPreprocessingConfig(
        remove_template_placeholders=False,
        normalize_markdown=False,
    )
    assert preprocess_structured(text, config) == text


# ---------------------------------------------------------------------------
# Repeated-line compression
# ---------------------------------------------------------------------------


def test_compresses_long_runs_of_identical_lines():
    text = "Connection timeout\n" * 6
    expected = "Connection timeout\nConnection timeout\n[REPEATED_LINES]\n"
    assert preprocess_structured(text) == expected


def test_preserves_short_repeated_runs():
    text = "Connection timeout\n" * 4
    assert preprocess_structured(text) == text


def test_preserves_distinct_error_messages():
    text = (
        "ValueError: invalid input\n"
        "TypeError: unexpected argument\n"
        "RuntimeError: connection failed\n"
    )
    assert preprocess_structured(text) == text


def test_can_disable_repeated_line_compression():
    text = "Connection timeout\n" * 6
    config = StructuredPreprocessingConfig(compress_repeated_blocks=False)
    assert preprocess_structured(text, config) == text


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------


def test_normalizes_line_endings():
    text = "First line\r\nSecond line\rThird line"
    assert preprocess_structured(text) == "First line\nSecond line\nThird line"


def test_removes_trailing_whitespace():
    text = "First line   \nSecond line\t"
    assert preprocess_structured(text) == "First line\nSecond line"


def test_collapses_excessive_blank_lines():
    text = "First paragraph\n\n\n\nSecond paragraph"
    assert preprocess_structured(text) == "First paragraph\n\nSecond paragraph"


def test_removes_surrounding_whitespace():
    text = "\n\n  Expected behavior  \n\n"
    assert preprocess_structured(text) == "Expected behavior\n"


def test_can_disable_whitespace_normalization():
    text = "First line   \n\n\nSecond line  "
    config = StructuredPreprocessingConfig(normalize_whitespace=False)
    assert preprocess_structured(text, config) == text


def test_whitespace_normalization_is_idempotent():
    text = "First line   \n\n\nSecond line  "
    once = preprocess_structured(text)
    twice = preprocess_structured(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Fenced code: replacement, preservation, and isolation from prose cleaning
# ---------------------------------------------------------------------------


def test_replaces_fenced_code_with_token():
    text = (
        "The application crashes.\n"
        f"{FENCE}python\n"
        "raise ValueError('Invalid configuration')\n"
        f"{FENCE}\n"
        "Please investigate."
    )
    assert preprocess_structured(text) == (
        "The application crashes.\nCODE\nPlease investigate."
    )


def test_preserves_fenced_code_when_replacement_disabled():
    text = f"## Error\n{FENCE}python\n# This is Python code\nprint('hello')\n{FENCE}\n"
    config = StructuredPreprocessingConfig(replace_fenced_code=False)
    assert preprocess_structured(text, config) == (
        f"Error\n{FENCE}python\n# This is Python code\nprint('hello')\n{FENCE}\n"
    )


def test_preserves_unclosed_code_fence():
    text = f"Before\n{FENCE}python\nprint('hello')"
    config = StructuredPreprocessingConfig(normalize_markdown=False)
    assert preprocess_structured(text, config) == text


def test_supports_tilde_code_fences():
    text = "Before\n~~~python\nprint('hello')\n~~~\nAfter"
    assert preprocess_structured(text) == "Before\nCODE\nAfter"


def test_preserves_markdown_inside_code_when_replacement_disabled():
    text = (
        f"{FENCE}python\n"
        "# Heading inside code\n"
        f"value = '{STAR * 2}important{STAR * 2}'\n"
        f"{FENCE}\n"
    )
    config = StructuredPreprocessingConfig(replace_fenced_code=False)
    assert preprocess_structured(text, config) == text


def test_preserves_html_comments_inside_retained_code():
    text = (
        f"{FENCE}html\n<!-- Important HTML comment -->\n<div>Content</div>\n{FENCE}\n"
    )
    config = StructuredPreprocessingConfig(replace_fenced_code=False)
    assert preprocess_structured(text, config) == text


def test_removes_html_comments_outside_code_only():
    text = f"<!-- Remove this -->\n{FENCE}html\n<!-- Preserve this -->\n{FENCE}\n"
    config = StructuredPreprocessingConfig(replace_fenced_code=False)
    result = preprocess_structured(text, config)
    assert "Remove this" not in result
    assert "<!-- Preserve this -->" in result


def test_preserves_paths_inside_retained_code():
    text = f"{FENCE}python\npath = '/home/user/project/config.yaml'\n{FENCE}\n"
    config = StructuredPreprocessingConfig(replace_fenced_code=False)
    assert preprocess_structured(text, config) == text


def test_preserves_empty_checkboxes_inside_retained_code():
    text = f"{FENCE}markdown\n- [ ]\n{FENCE}\n"
    config = StructuredPreprocessingConfig(replace_fenced_code=False)
    assert preprocess_structured(text, config) == text


def test_preserves_repeated_lines_inside_retained_code():
    text = f"{FENCE}text\n" + "Connection timeout\n" * 6 + f"{FENCE}\n"
    config = StructuredPreprocessingConfig(replace_fenced_code=False)
    assert preprocess_structured(text, config) == text


# ---------------------------------------------------------------------------
# Integration: configuration interaction and combined transformations
# ---------------------------------------------------------------------------


def test_all_implemented_transformations_can_be_disabled():
    text = f"## Heading\n<!-- Comment -->\n{FENCE}python\nprint('hello')\n{FENCE}\n"
    config = StructuredPreprocessingConfig(
        remove_html_comments=False,
        normalize_markdown=False,
        replace_fenced_code=False,
    )
    assert preprocess_structured(text, config) == text


def test_combined_structured_preprocessing():
    text = (
        "## Bug Report\n"
        "<!-- Fill in the details -->\n"
        "The application raises `ValueError`.\n"
        "See https://example.com/docs?session=123\n"
        "- [ ]\n"
        + "Connection timeout\n" * 6
        + f"{FENCE}python\nraise ValueError('failed')\n{FENCE}\n"
    )
    result = preprocess_structured(text)

    assert "Bug Report" in result
    assert "Fill in the details" not in result
    assert "ValueError" in result
    assert "https://example.com/docs" in result
    assert "session=123" not in result
    assert "- [ ]" not in result
    assert "[REPEATED_LINES]" in result
    assert "CODE" in result
    assert "raise ValueError('failed')" not in result


def test_extract_code_signals_preserves_errors():
    code = (
        "model.predict(data)\n"
        "ValueError: expected 2D array\n"
        "print('debugging')\n"
        "CUDA out of memory\n"
    )

    result = extract_code_signals(code)

    assert result.startswith("CODE\n")
    assert "ValueError: expected 2D array" in result
    assert "CUDA out of memory" in result
    assert "model.predict(data)" not in result
    assert "print('debugging')" not in result


def test_extract_code_signals_without_errors():
    code = "x = 10\nprint(x)\n"

    assert extract_code_signals(code) == "CODE\n"


def test_extract_code_signals_deduplicates():
    code = "ValueError: failed\n" * 10

    result = extract_code_signals(code)

    assert result.count("ValueError: failed") == 1


def test_extract_code_signals_respects_limits():
    code = "\n".join(f"ValueError: {'x' * 300}{i}" for i in range(10))

    result = extract_code_signals(code)
    signal_text = result.removeprefix("CODE\n").rstrip("\n")

    assert len(signal_text) <= MAX_CODE_SIGNAL_CHARS
    assert len(signal_text.splitlines()) <= MAX_CODE_SIGNAL_LINES


def test_selective_code_compression_integration():
    fence = "`" * 3
    text = (
        f"{fence}python\nmodel.predict(data)\nValueError: expected 2D array\n{fence}\n"
    )

    config = StructuredPreprocessingConfig(
        compress_repeated_blocks=False,
        selective_code_compression=True,
    )

    result = preprocess_structured(text, config)

    assert "CODE" in result
    assert "ValueError: expected 2D array" in result
    assert "model.predict(data)" not in result


def test_selective_code_requires_replacement():
    config = StructuredPreprocessingConfig(
        replace_fenced_code=False,
        selective_code_compression=True,
    )

    with pytest.raises(
        ValueError,
        match="requires replace_fenced_code",
    ):
        preprocess_structured("text", config)
