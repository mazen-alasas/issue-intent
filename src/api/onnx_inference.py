"""CPU ONNX inference with the same text pipeline and response contract as PyTorch."""

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from src.api.inference import (
    LABEL_NAMES,
    Prediction,
    resolve_checkpoint_path,
)
from src.data.transformer_dataset import TransformerInputConfig, encode_issue
from src.preprocessing.experiment_configs import EXPERIMENTS
from src.preprocessing.structured import preprocess_structured

DEFAULT_ONNX_MODEL = (
    Path.home() / "issue-intent-artifacts/models/distilbert/onnx/model_int8.onnx"
)


class OnnxIssueClassifier:
    """Run INT8 ONNX inference on CPU, loading tokenizer and graph once."""

    def __init__(
        self,
        checkpoint: Path | None = None,
        model_path: Path | None = None,
        input_config: TransformerInputConfig | None = None,
        threads: int = 2,
    ) -> None:
        self.checkpoint = resolve_checkpoint_path(checkpoint)
        if not self.checkpoint.is_dir():
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found: {self.checkpoint}"
            )
        configured = os.environ.get("ISSUE_INTENT_ONNX_MODEL")
        self.model_path = Path(
            model_path or configured or DEFAULT_ONNX_MODEL
        ).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if threads < 1:
            raise ValueError("threads must be positive")
        self.input_config = input_config or TransformerInputConfig(
            max_length=256, title_budget=48
        )
        self.preprocessing_config = EXPERIMENTS["E4"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, use_fast=True, local_files_only=True
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        if "CPUExecutionProvider" not in self.session.get_providers():
            raise RuntimeError("CPUExecutionProvider is not active")
        if {x.name for x in self.session.get_inputs()} != {
            "input_ids",
            "attention_mask",
        }:
            raise ValueError("Unexpected ONNX input names")
        if [x.name for x in self.session.get_outputs()] != ["logits"]:
            raise ValueError("Unexpected ONNX output names")

    def predict(self, title: str, body: str) -> Prediction:
        if not isinstance(title, str):
            raise TypeError("title must be a string")
        if not isinstance(body, str):
            raise TypeError("body must be a string")
        if not title.strip() and not body.strip():
            raise ValueError("title and body cannot both be empty")
        processed_body = preprocess_structured(body, config=self.preprocessing_config)
        encoded = encode_issue(
            title=title,
            body=processed_body,
            tokenizer=self.tokenizer,
            config=self.input_config,
        )
        inputs = {
            name: np.asarray([encoded[name]], dtype=np.int64)
            for name in ("input_ids", "attention_mask")
        }
        logits = np.asarray(self.session.run(["logits"], inputs)[0])
        if logits.shape != (1, len(LABEL_NAMES)):
            raise ValueError(f"Unexpected model output shape: {logits.shape}")
        if not np.isfinite(logits).all():
            raise ValueError("Model returned non-finite logits")
        shifted = logits[0].astype(np.float64) - float(logits[0].max())
        exp = np.exp(shifted)
        values = exp / exp.sum()
        predicted_id = int(values.argmax())
        return Prediction(
            label=LABEL_NAMES[predicted_id],
            confidence=float(values[predicted_id]),
            probabilities={
                label: float(probability)
                for label, probability in zip(LABEL_NAMES, values, strict=True)
            },
        )
