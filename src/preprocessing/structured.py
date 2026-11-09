import re
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class StructuredPreprocessingConfig:
    """Control individual text preprocessing transformations."""

    remove_html_comments: bool = True
    normalize_markdown: bool = True
    normalize_urls_paths: bool = True
    remove_template_placeholders: bool = True
    compress_repeated_blocks: bool = True
    replace_fenced_code: bool = True
    normalize_whitespace: bool = True
    selective_code_compression: bool = False


HTML_COMMENT_PATTERN = re.compile(r"<!--[\s\S]*?-->")

MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")
MARKDOWN_BOLD_PATTERN = re.compile(r"\*\*([^*\n]+)\*\*")
MARKDOWN_ITALIC_PATTERN = re.compile(r"(?<!\w)\*([^*\n]+)\*(?!\w)")
MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^)]+\)")
INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
MARKDOWN_BULLET_PATTERN = re.compile(r"(?m)^[ \t]*[-*+][ \t]+")

FENCE_OPEN_PATTERN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[^\n]*$")
FENCE_CLOSE_PATTERN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*$")

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+")

ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![\w/])/(?:[\w.-]+/)+[\w.-]+")

EMPTY_CHECKBOX_PATTERN = re.compile(
    r"(?m)^[ \t]*[-*+][ \t]+\[[ \t]*\][ \t]*(?:\r?\n|$)"
)


def remove_html_comments(text: str) -> str:
    """Remove complete HTML comments without merging adjacent words."""
    return HTML_COMMENT_PATTERN.sub(" ", text)


def normalize_markdown(text: str) -> str:
    """Normalize common Markdown syntax in non-code text."""
    text = MARKDOWN_HEADING_PATTERN.sub("", text)
    text = MARKDOWN_LINK_PATTERN.sub(r"\1", text)
    text = MARKDOWN_BOLD_PATTERN.sub(r"\1", text)
    text = MARKDOWN_ITALIC_PATTERN.sub(r"\1", text)
    text = INLINE_CODE_PATTERN.sub(r"\1", text)
    text = MARKDOWN_BULLET_PATTERN.sub("", text)
    return text


def normalize_url(url: str) -> str:
    """Remove URL query parameters and fragments when parsing is valid.

    Preserve malformed URLs unchanged instead of failing preprocessing.
    """
    try:
        parsed = urlsplit(url)
    except ValueError:
        return url

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            "",
            "",
        )
    )


def normalize_absolute_path(path: str) -> str:
    """Normalize absolute paths while preserving filenames."""
    filename = path.rsplit("/", maxsplit=1)[-1]
    return f"PATH/{filename}"


def normalize_urls_paths(text: str) -> str:
    """Normalize URLs and absolute paths without corrupting URLs."""
    result = []
    last_end = 0

    for match in URL_PATTERN.finditer(text):
        # Normalize paths only in text outside URLs.
        before_url = text[last_end : match.start()]
        before_url = ABSOLUTE_PATH_PATTERN.sub(
            lambda path_match: normalize_absolute_path(path_match.group(0)),
            before_url,
        )
        result.append(before_url)

        url = match.group(0)
        clean_url = url.rstrip(".,;!")
        punctuation = url[len(clean_url) :]

        result.append(normalize_url(clean_url) + punctuation)
        last_end = match.end()

    remaining = text[last_end:]
    remaining = ABSOLUTE_PATH_PATTERN.sub(
        lambda path_match: normalize_absolute_path(path_match.group(0)),
        remaining,
    )
    result.append(remaining)

    return "".join(result)


def remove_template_placeholders(text: str) -> str:
    """Remove standalone empty Markdown checklist items."""
    return EMPTY_CHECKBOX_PATTERN.sub("", text)


def compress_repeated_lines(
    text: str,
    *,
    min_repetitions: int = 5,
    keep_first: int = 2,
) -> str:
    """Compress long runs of identical consecutive lines."""
    if min_repetitions <= keep_first:
        raise ValueError("min_repetitions must exceed keep_first")

    lines = text.splitlines(keepends=True)

    if not lines:
        return text

    result = []
    index = 0

    while index < len(lines):
        current = lines[index]
        current_value = current.rstrip("\r\n")

        end = index + 1

        while end < len(lines) and lines[end].rstrip("\r\n") == current_value:
            end += 1

        count = end - index

        if current_value.strip() and count >= min_repetitions:
            result.extend(lines[index : index + keep_first])
            result.append("[REPEATED_LINES]\n")
        else:
            result.extend(lines[index:end])

        index = end

    return "".join(result)


ERROR_SIGNAL_PATTERN = re.compile(
    r"\b(?:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)"
    r"|(?:HTTP\s*)?[45]\d{2}\s+"
    r"(?:Not Found|Forbidden|Unauthorized|Bad Request|"
    r"Internal Server Error|Service Unavailable)"
    r"|(?:CUDA|GPU)\s+out of memory"
    r"|outofmemoryerror"
    r")\b",
    flags=re.IGNORECASE,
)

MAX_CODE_SIGNAL_LINES = 3
MAX_CODE_SIGNAL_CHARS = 240


def extract_code_signals(code: str) -> str:
    """Extract a bounded set of diagnostic lines from fenced code."""
    signals = []
    seen = set()
    total_chars = 0

    for line in code.splitlines():
        line = line.strip()

        if not line or not ERROR_SIGNAL_PATTERN.search(line):
            continue

        # Avoid retaining very long log or source-code lines.
        line = line[:MAX_CODE_SIGNAL_CHARS]

        if line in seen:
            continue

        remaining = MAX_CODE_SIGNAL_CHARS - total_chars

        if remaining <= 0:
            break

        line = line[:remaining].rstrip()

        if not line:
            break

        signals.append(line)
        seen.add(line)
        total_chars += len(line)

        if len(signals) >= MAX_CODE_SIGNAL_LINES:
            break

    if not signals:
        return "CODE\n"

    return "CODE\n" + "\n".join(signals) + "\n"


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving a final newline."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    has_final_newline = text.endswith("\n")

    lines = [line.rstrip(" \t") for line in text.split("\n")]
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    if text and has_final_newline:
        text += "\n"

    return text


def process_fenced_blocks(
    text: str,
    *,
    replace_code: bool,
    clean_urls_paths: bool,
    clean_markdown: bool,
    clean_template_placeholders: bool,
    compress_repeats: bool,
    clean_html_comments: bool,
    selective_code_compression: bool = False,
) -> str:
    """Process prose separately from complete fenced code blocks."""
    lines = text.splitlines(keepends=True)
    result = []
    prose_lines = []
    index = 0

    def flush_prose() -> None:
        if prose_lines:
            prose = "".join(prose_lines)

            if clean_html_comments:
                prose = remove_html_comments(prose)

            if clean_template_placeholders:
                prose = remove_template_placeholders(prose)

            if clean_markdown:
                prose = normalize_markdown(prose)

            if clean_urls_paths:
                prose = normalize_urls_paths(prose)

            if compress_repeats:
                prose = compress_repeated_lines(prose)

            result.append(prose)
            prose_lines.clear()

    while index < len(lines):
        opening = FENCE_OPEN_PATTERN.fullmatch(lines[index].rstrip("\r\n"))

        if opening is None:
            prose_lines.append(lines[index])
            index += 1
            continue

        marker = opening.group(1)
        fence_char = marker[0]
        fence_length = len(marker)

        closing_index = None

        for candidate_index in range(index + 1, len(lines)):
            closing = FENCE_CLOSE_PATTERN.fullmatch(
                lines[candidate_index].rstrip("\r\n")
            )

            if closing is None:
                continue

            closing_marker = closing.group(1)

            if closing_marker[0] == fence_char and len(closing_marker) >= fence_length:
                closing_index = candidate_index
                break

        if closing_index is None:
            prose_lines.extend(lines[index:])
            break

        flush_prose()

        if replace_code:
            if selective_code_compression:
                code_content = "".join(lines[index + 1 : closing_index])
                result.append(extract_code_signals(code_content))
            else:
                result.append("CODE\n")
        else:
            result.extend(lines[index : closing_index + 1])

        index = closing_index + 1

    flush_prose()
    return "".join(result)


def preprocess_structured(
    text: str,
    config: StructuredPreprocessingConfig | None = None,
) -> str:
    """Apply implemented transformations to GitHub issue text.

    Unimplemented configuration options are reserved for later stages.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if config is None:
        config = StructuredPreprocessingConfig()

    if config.selective_code_compression and not config.replace_fenced_code:
        raise ValueError("selective_code_compression requires replace_fenced_code=True")

    result = process_fenced_blocks(
        text,
        replace_code=config.replace_fenced_code,
        clean_markdown=config.normalize_markdown,
        clean_html_comments=config.remove_html_comments,
        clean_urls_paths=config.normalize_urls_paths,
        clean_template_placeholders=config.remove_template_placeholders,
        compress_repeats=config.compress_repeated_blocks,
        selective_code_compression=config.selective_code_compression,
    )

    if config.normalize_whitespace:
        result = normalize_whitespace(result)

    return result
