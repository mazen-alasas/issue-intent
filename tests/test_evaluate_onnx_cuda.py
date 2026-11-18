from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_onnx_int8 import make_session, verify_saved_fp32


def test_cpu_provider_selection(tmp_path):
    session = MagicMock()
    session.get_inputs.return_value = [
        SimpleNamespace(name="input_ids"),
        SimpleNamespace(name="attention_mask"),
    ]
    session.get_outputs.return_value = [SimpleNamespace(name="logits")]
    with patch(
        "scripts.evaluate_onnx_int8.ort.InferenceSession", return_value=session
    ) as factory:
        assert make_session(tmp_path / "model.onnx", 2, "cpu") is session
    assert factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]


def test_invalid_provider(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        make_session(tmp_path / "model.onnx", 2, "invalid")


def test_cuda_fallback_rejected(tmp_path):
    session = MagicMock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.init"),
        patch("scripts.evaluate_onnx_int8.ort.preload_dlls"),
        patch(
            "scripts.evaluate_onnx_int8.ort.get_available_providers",
            return_value=["CUDAExecutionProvider"],
        ),
        patch("scripts.evaluate_onnx_int8.ort.InferenceSession", return_value=session),
        pytest.raises(RuntimeError, match="fell back"),
    ):
        make_session(tmp_path / "model.onnx", 2, "cuda")


def test_resume_verification_matches_saved_logits():
    df = pd.DataFrame({"title": ["a"] * 8, "body": [""] * 8})
    saved = np.tile(np.array([[2.0, 0.0, -1.0, -2.0]], dtype=np.float32), (8, 1))
    session = MagicMock()
    session.run.return_value = [saved[:4] + 1e-6]
    with patch(
        "scripts.evaluate_onnx_int8.make_batch",
        return_value={"input_ids": np.zeros((4, 2), dtype=np.int64)},
    ):
        result = verify_saved_fp32(df, 8, 4, None, None, saved, session, 2)
    assert result["checked_rows"] == 8
    assert session.run.call_count == 2


def test_resume_verification_rejects_mismatch():
    df = pd.DataFrame({"title": ["a"] * 4, "body": [""] * 4})
    saved = np.tile(np.array([[2.0, 0.0, -1.0, -2.0]], dtype=np.float32), (4, 1))
    session = MagicMock()
    session.run.return_value = [-saved]
    with (
        patch(
            "scripts.evaluate_onnx_int8.make_batch",
            return_value={"input_ids": np.zeros((4, 2), dtype=np.int64)},
        ),
        pytest.raises(RuntimeError, match="validation failed"),
    ):
        verify_saved_fp32(df, 4, 4, None, None, saved, session, 1)
