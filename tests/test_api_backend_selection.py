from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


def test_onnx_backend_selection(monkeypatch):
    from src.api import onnx_inference

    class FakeOnnxClassifier:
        def predict(self, title, body):
            return SimpleNamespace(
                label="bug",
                confidence=1.0,
                probabilities={
                    "bug": 1.0,
                    "documentation": 0.0,
                    "feature": 0.0,
                    "question": 0.0,
                },
            )

    monkeypatch.setenv("ISSUE_INTENT_BACKEND", "onnx-int8")
    monkeypatch.setattr(onnx_inference, "OnnxIssueClassifier", FakeOnnxClassifier)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json={"title": "Crash", "body": ""})
        assert response.status_code == 200
        assert response.json()["label"] == "bug"


def test_unknown_backend_rejected(monkeypatch):
    monkeypatch.setenv("ISSUE_INTENT_BACKEND", "unknown")
    with pytest.raises(ValueError, match="Unsupported ISSUE_INTENT_BACKEND"):
        create_app()
