"""Load the checkpoint's exact fast-tokenizer graph without importing Transformers."""

from pathlib import Path

from tokenizers import Tokenizer


class RawTokenizerAdapter:
    """Expose the small tokenizer interface used by ``encode_issue``."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self.cls_token_id = tokenizer.token_to_id("[CLS]")
        self.sep_token_id = tokenizer.token_to_id("[SEP]")
        if self.cls_token_id is None or self.sep_token_id is None:
            raise ValueError("Checkpoint tokenizer must contain [CLS] and [SEP]")

    def __call__(
        self, text: str, *, add_special_tokens: bool, truncation: bool
    ) -> dict[str, list[int]]:
        if add_special_tokens or truncation:
            raise ValueError("Only raw, untruncated tokenization is supported")
        return {"input_ids": self._tokenizer.encode(text, add_special_tokens=False).ids}


def load_onnx_tokenizer(checkpoint: Path) -> RawTokenizerAdapter:
    """Use the saved tokenizer.json from the evaluated DistilBERT checkpoint."""
    tokenizer_path = checkpoint / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Fast tokenizer not found: {tokenizer_path}. "
            "ONNX-only serving requires tokenizer.json."
        )
    return RawTokenizerAdapter(Tokenizer.from_file(str(tokenizer_path)))
