"""Tests for shared Transformer utilities."""

from types import SimpleNamespace

import pytest
import torch

from src.models import evaluate_transformer
from src.models.transformer_utils import resolve_device, validate_model_labels


def make_model(labels):
    return SimpleNamespace(config=SimpleNamespace(id2label=labels))


def test_resolve_cpu_device():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_device():
    device = resolve_device("auto")
    assert device.type in {"cpu", "cuda"}


def test_reject_unavailable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA requested"):
        resolve_device("cuda")


def test_accept_correct_label_mapping():
    model = make_model(
        {
            0: "bug",
            1: "documentation",
            2: "feature",
            3: "question",
        }
    )

    validate_model_labels(model)


def test_reject_wrong_label_mapping():
    model = make_model(
        {
            0: "feature",
            1: "documentation",
            2: "bug",
            3: "question",
        }
    )

    with pytest.raises(ValueError, match="mapping mismatch"):
        validate_model_labels(model)


def test_generic_labels_require_explicit_permission():
    model = make_model({index: f"LABEL_{index}" for index in range(4)})

    with pytest.raises(ValueError, match="generic LABEL_N"):
        validate_model_labels(model)

    validate_model_labels(model, allow_generic_labels=True)


def test_legacy_imports_remain_compatible():
    assert evaluate_transformer.resolve_device is resolve_device
    assert evaluate_transformer.validate_model_labels is validate_model_labels
