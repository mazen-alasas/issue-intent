"""Unit tests for API smoke-test response validation."""

from unittest.mock import patch

import pytest

from scripts.smoke_test_api import run_smoke_test


def test_smoke_test_accepts_valid_responses(capsys):
    responses = [
        {"status": "healthy", "model_loaded": True},
        {
            "label": "bug",
            "confidence": 0.9,
            "probabilities": {
                "bug": 0.9,
                "documentation": 0.05,
                "feature": 0.03,
                "question": 0.02,
            },
        },
    ]

    with patch(
        "scripts.smoke_test_api.request_json",
        side_effect=responses,
    ) as mock_request:
        run_smoke_test("http://localhost:8001/")

    assert mock_request.call_count == 2
    assert "All API smoke tests passed." in capsys.readouterr().out


def test_smoke_test_rejects_unhealthy_service():
    with patch(
        "scripts.smoke_test_api.request_json",
        return_value={"status": "unhealthy", "model_loaded": False},
    ):
        with pytest.raises(AssertionError, match="Unexpected health"):
            run_smoke_test("http://localhost:8001")


def test_smoke_test_rejects_invalid_probabilities():
    responses = [
        {"status": "healthy", "model_loaded": True},
        {
            "label": "bug",
            "confidence": 0.9,
            "probabilities": {
                "bug": 0.9,
                "documentation": 0.05,
                "feature": 0.03,
                "question": 0.5,
            },
        },
    ]

    with patch(
        "scripts.smoke_test_api.request_json",
        side_effect=responses,
    ):
        with pytest.raises(AssertionError):
            run_smoke_test("http://localhost:8001")
