import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.transformer_dataset import TransformerInputConfig, encode_issue
from src.models.transformer_utils import validate_model_labels


def build_cases(tokenizer):
    config = TransformerInputConfig()
    examples = [
        ("App crashes", "RuntimeError on startup"),
        ("How do I configure authentication?", ""),
        ("Add support for exporting reports", "Please add CSV export."),
        (
            "Update installation documentation",
            "The installation steps are outdated. " * 12,
        ),
        ("Cannot start server", "Traceback: ValueError invalid configuration. " * 150),
        ("Feature request " * 20, "Need better filtering and search. " * 120),
    ]
    return [encode_issue(title, body, tokenizer, config) for title, body in examples]


def make_batch(cases, pad_id):
    length = max(len(case["input_ids"]) for case in cases)
    ids = np.full((len(cases), length), pad_id, dtype=np.int64)
    mask = np.zeros((len(cases), length), dtype=np.int64)
    for row, case in enumerate(cases):
        count = len(case["input_ids"])
        ids[row, :count] = case["input_ids"]
        mask[row, :count] = case["attention_mask"]
    return {"input_ids": ids, "attention_mask": mask}


def validate(checkpoint: Path, model_path: Path, atol: float, rtol: float) -> None:
    checkpoint = checkpoint.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, use_fast=True, local_files_only=True
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.float32
        )
        .cpu()
        .eval()
    )
    validate_model_labels(model)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    assert {item.name for item in session.get_inputs()} == {
        "input_ids",
        "attention_mask",
    }
    assert [item.name for item in session.get_outputs()] == ["logits"]
    cases = build_cases(tokenizer)
    batches = [[case] for case in cases] + [cases[:3], cases[3:], cases]
    max_abs = 0.0
    max_rel = 0.0
    for index, batch_cases in enumerate(batches, start=1):
        batch = make_batch(batch_cases, tokenizer.pad_token_id)
        with torch.inference_mode():
            expected = model(
                **{key: torch.from_numpy(value) for key, value in batch.items()}
            ).logits.numpy()
        actual = session.run(["logits"], batch)[0]
        if actual.shape != expected.shape:
            raise AssertionError(
                f"Batch {index}: shape mismatch {actual.shape} vs {expected.shape}"
            )
        difference = np.abs(actual - expected)
        max_abs = max(max_abs, float(difference.max()))
        max_rel = max(
            max_rel, float((difference / np.maximum(np.abs(expected), 1e-8)).max())
        )
        agree = np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1))
        close = np.allclose(actual, expected, atol=atol, rtol=rtol)
        print(
            f"Batch {index}: size={len(batch_cases)}, seq={batch['input_ids'].shape[1]}, "
            f"max_abs={difference.max():.8f}, predictions_agree={agree}, logits_close={close}"
        )
        if not agree or not close:
            raise AssertionError(f"Parity failed for batch {index}")
    print(f"PASS: {len(batches)} batches; max_abs={max_abs:.8f}; max_rel={max_rel:.8f}")
    print(
        "Note: Synthetic cases verify numerical parity, not dataset-level accuracy or browser parity."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "issue-intent-artifacts/models/distilbert/best",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home()
        / "issue-intent-artifacts/models/distilbert/onnx/model_fp32.onnx",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    validate(args.checkpoint, args.model, args.atol, args.rtol)


if __name__ == "__main__":
    main()
