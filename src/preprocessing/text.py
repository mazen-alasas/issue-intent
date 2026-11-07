import re

HTML_COMMENT_PATTERN = re.compile(r"<!--[\s\S]*?-->")
BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


def preprocess_text(text: str | None) -> str:
    """Apply conservative normalization without truncating issue text.

    Args:
        text: Raw issue text or None.

    Returns:
        str: Normalized text with code, links, and diagnostics retained.
    """
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = HTML_COMMENT_PATTERN.sub(" ", cleaned)
    cleaned = BLANK_LINES_PATTERN.sub("\n\n", cleaned)

    return cleaned
