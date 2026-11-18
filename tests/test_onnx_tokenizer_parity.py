"""Compare raw checkpoint tokenization against the training-time HF pipeline."""

import os
from pathlib import Path

import pytest

from src.api.raw_tokenizer import load_onnx_tokenizer
from src.data.transformer_dataset import TransformerInputConfig, encode_issue
from src.preprocessing.experiment_configs import EXPERIMENTS
from src.preprocessing.structured import preprocess_structured


def test_raw_tokenizer_exact_parity_with_training_checkpoint():
    transformers = pytest.importorskip("transformers")
    checkpoint = Path(
        os.environ.get(
            "ISSUE_INTENT_CHECKPOINT",
            Path.home() / "issue-intent-artifacts/models/distilbert/best",
        )
    ).expanduser()
    if not (checkpoint / "tokenizer.json").is_file():
        pytest.skip("Original evaluated tokenizer.json unavailable")
    old = transformers.AutoTokenizer.from_pretrained(
        checkpoint, use_fast=True, local_files_only=True
    )
    new = load_onnx_tokenizer(checkpoint)
    samples = [
        ("Application crashes on startup", "RuntimeError after updating."),
        ("", "Question about docs and installation"),
        ("Title only", ""),
        ("Emoji 🎉 and unicode العربية", "Whitespace\n\t and special chars !?"),
        ("Duplicate title " * 30, "Error stack trace\n" * 200),
        ("A short title", "Body with URL https://example.org/path?key=value"),
        ("[CLS] [SEP]", "A <unk> [PAD] <mask> B"),
    ]
    for title, body in samples:
        cleaned = preprocess_structured(body, config=EXPERIMENTS["E4"])
        for config in (
            TransformerInputConfig(max_length=256, title_budget=48),
            TransformerInputConfig(max_length=64, title_budget=16),
        ):
            actual = encode_issue(title, cleaned, new, config)
            expected = encode_issue(title, cleaned, old, config)
            assert actual == expected, (title[:40], config)
