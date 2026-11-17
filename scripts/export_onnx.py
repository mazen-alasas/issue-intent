import argparse
from pathlib import Path

import onnx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.models.transformer_utils import validate_model_labels


class LogitsWrapper(torch.nn.Module):
    """Expose classifier logits as the only ONNX output."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def export(checkpoint: Path, output: Path, opset: int = 17) -> None:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

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

    inputs = tokenizer(
        "Application crashes", "RuntimeError on startup", return_tensors="pt"
    )
    wrapper = LogitsWrapper(model).eval()

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (inputs["input_ids"], inputs["attention_mask"]),
            str(output),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    print(f"Exported and checked: {output}")
    print(f"Size: {output.stat().st_size / (1024**2):.2f} MiB")
    print(f"Inputs: {[value.name for value in onnx_model.graph.input]}")
    print(f"Outputs: {[value.name for value in onnx_model.graph.output]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home() / "issue-intent-artifacts/models/distilbert/best",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home()
        / "issue-intent-artifacts/models/distilbert/onnx/model_fp32.onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    export(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
