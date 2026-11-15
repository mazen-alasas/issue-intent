import pytest
from pydantic import ValidationError

from src.api.schemas import (
    MAX_BODY_LENGTH,
    MAX_TITLE_LENGTH,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


def test_valid_request():
    request = PredictRequest(
        title="Application crashes",
        body="RuntimeError on startup",
    )

    assert request.title == "Application crashes"
    assert request.body == "RuntimeError on startup"


def test_body_defaults_to_empty():
    request = PredictRequest(title="Application crashes")

    assert request.body == ""


def test_body_only_is_valid():
    request = PredictRequest(
        title="",
        body="Application crashes on startup",
    )

    assert request.title == ""


def test_preserves_original_text():
    request = PredictRequest(
        title="  Application crashes  ",
        body="\nRuntimeError\n",
    )

    assert request.title == "  Application crashes  "
    assert request.body == "\nRuntimeError\n"


@pytest.mark.parametrize(
    ("title", "body"),
    [
        ("", ""),
        ("   ", "\n\t"),
    ],
)
def test_rejects_empty_content(title, body):
    with pytest.raises(
        ValidationError,
        match="cannot both be empty",
    ):
        PredictRequest(title=title, body=body)


def test_rejects_missing_title():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({"body": "Some issue"})


@pytest.mark.parametrize(
    "payload",
    [
        {"title": 123, "body": "Body"},
        {"title": "Title", "body": 123},
        {"title": None, "body": "Body"},
        {"title": "Title", "body": None},
    ],
)
def test_rejects_non_string_values(payload):
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(payload)


def test_rejects_long_title():
    with pytest.raises(ValidationError):
        PredictRequest(
            title="a" * (MAX_TITLE_LENGTH + 1),
            body="Body",
        )


def test_rejects_long_body():
    with pytest.raises(ValidationError):
        PredictRequest(
            title="Title",
            body="a" * (MAX_BODY_LENGTH + 1),
        )


def test_accepts_boundary_lengths():
    request = PredictRequest(
        title="a" * MAX_TITLE_LENGTH,
        body="b" * MAX_BODY_LENGTH,
    )

    assert len(request.title) == MAX_TITLE_LENGTH
    assert len(request.body) == MAX_BODY_LENGTH


def test_rejects_extra_request_fields():
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(
            {
                "title": "Title",
                "body": "Body",
                "labels": "bug",
            }
        )


def test_valid_response():
    response = PredictResponse(
        label="bug",
        confidence=0.9,
        probabilities={
            "bug": 0.9,
            "documentation": 0.02,
            "feature": 0.05,
            "question": 0.03,
        },
    )

    assert response.label == "bug"
    assert response.confidence == 0.9


def test_rejects_invalid_response_label():
    with pytest.raises(ValidationError):
        PredictResponse(
            label="unknown",
            confidence=0.9,
            probabilities={"bug": 0.9},
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_rejects_invalid_confidence(confidence):
    with pytest.raises(ValidationError):
        PredictResponse(
            label="bug",
            confidence=confidence,
            probabilities={"bug": 0.9},
        )


def test_rejects_unknown_probability_label():
    with pytest.raises(ValidationError):
        PredictResponse(
            label="bug",
            confidence=0.9,
            probabilities={"unknown": 0.9},
        )


def test_health_response():
    response = HealthResponse(
        status="healthy",
        model_loaded=True,
    )

    assert response.model_loaded is True
