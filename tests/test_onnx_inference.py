from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.api import onnx_inference


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def __call__(self, text, *, add_special_tokens, truncation):
        return {"input_ids": [ord(c) % 100 + 1 for c in text]}


class FakeSession:
    def __init__(self, logits):
        self.logits = logits
        self.calls = []

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in ("input_ids", "attention_mask")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, outputs, inputs):
        self.calls.append((outputs, inputs))
        return [self.logits]


@pytest.fixture
def make_classifier(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best"
    checkpoint.mkdir()
    graph = tmp_path / "model_int8.onnx"
    graph.write_bytes(b"test")
    session = FakeSession(np.array([[4, 1, 0, -1]], dtype=np.float32))
    monkeypatch.setattr(
        onnx_inference,
        "load_onnx_tokenizer",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        onnx_inference.ort,
        "InferenceSession",
        lambda *args, **kwargs: session,
    )
    classifier = onnx_inference.OnnxIssueClassifier(
        checkpoint=checkpoint, model_path=graph
    )
    return classifier, session


def test_prediction_and_numpy_inputs(make_classifier):
    classifier, session = make_classifier
    result = classifier.predict("Application crashes", "RuntimeError")
    assert result.label == "bug"
    assert sum(result.probabilities.values()) == pytest.approx(1)
    assert result.confidence == pytest.approx(result.probabilities["bug"])
    outputs, inputs = session.calls[0]
    assert outputs == ["logits"]
    assert inputs["input_ids"].dtype == np.int64
    assert inputs["input_ids"].shape[0] == 1
    assert inputs["attention_mask"].shape == inputs["input_ids"].shape


def test_empty_body_allowed(make_classifier):
    classifier, _ = make_classifier
    assert classifier.predict("Title", "").label == "bug"


@pytest.mark.parametrize("title,body", [("", ""), ("  ", "\n")])
def test_empty_issue_rejected(make_classifier, title, body):
    classifier, _ = make_classifier
    with pytest.raises(ValueError, match="cannot both be empty"):
        classifier.predict(title, body)


def test_non_finite_logits_rejected(make_classifier):
    classifier, session = make_classifier
    session.logits = np.array([[np.nan, 1, 2, 3]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        classifier.predict("Title", "Body")


def test_wrong_output_shape_rejected(make_classifier):
    classifier, session = make_classifier
    session.logits = np.zeros((1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="output shape"):
        classifier.predict("Title", "Body")


def test_uses_e4_body_preprocessing(make_classifier, monkeypatch):
    classifier, _ = make_classifier
    captured = {}

    def fake_preprocess(body, config):
        captured["body"] = body
        captured["config"] = config
        return "processed"

    monkeypatch.setattr(onnx_inference, "preprocess_structured", fake_preprocess)
    classifier.predict("Title", "Original")
    assert captured == {"body": "Original", "config": onnx_inference.EXPERIMENTS["E4"]}


def test_missing_model_rejected(tmp_path):
    checkpoint = tmp_path / "best"
    checkpoint.mkdir()
    with pytest.raises(FileNotFoundError, match="ONNX model not found"):
        onnx_inference.OnnxIssueClassifier(
            checkpoint=checkpoint, model_path=Path(tmp_path / "missing.onnx")
        )
