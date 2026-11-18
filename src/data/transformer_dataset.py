from dataclasses import dataclass
from typing import Protocol


class IssueTokenizer(Protocol):
    cls_token_id: int | None
    sep_token_id: int | None

    def __call__(
        self, text: str, *, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[int]]: ...


@dataclass(frozen=True)
class TransformerInputConfig:
    max_length: int = 256
    title_budget: int = 48

    def __post_init__(self):
        if self.max_length < 8:
            raise ValueError("max_length must be >= 8")
        if not 1 <= self.title_budget <= self.max_length - 3:
            raise ValueError("Invalid title_budget")


def encode_issue(
    title: str,
    body: str,
    tokenizer: IssueTokenizer,
    config: TransformerInputConfig,
) -> dict[str, list[int]]:
    if not isinstance(title, str) or not isinstance(body, str):
        raise TypeError("title and body must be strings")

    title_ids = tokenizer(
        title,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"][: config.title_budget]

    body_ids = tokenizer(
        body,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    # DistilBERT/BERT pair:
    # [CLS] title [SEP] body [SEP]
    available = config.max_length - len(title_ids) - 3

    if available < 0:
        raise ValueError("Insufficient sequence budget")

    if len(body_ids) > available:
        head = (available * 3) // 4
        tail = available - head

        body_ids = body_ids[:head] + body_ids[-tail:] if tail > 0 else body_ids[:head]

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    if cls_id is None or sep_id is None:
        raise ValueError("Tokenizer requires CLS and SEP tokens")

    input_ids = [cls_id] + title_ids + [sep_id] + body_ids + [sep_id]

    if len(input_ids) > config.max_length:
        raise RuntimeError("Encoded sequence exceeds max_length")

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
    }
