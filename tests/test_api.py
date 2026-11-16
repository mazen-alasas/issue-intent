from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


class FakeClassifier:
    def __init__(self):
        self.calls = []

    def predict(self, title, body):
        self.calls.append(
            {
                "title": title,
                "body": body,
            }
        )

        return SimpleNamespace(
            label="bug",
            confidence=0.9,
            probabilities={
                "bug": 0.9,
                "documentation": 0.02,
                "feature": 0.05,
                "question": 0.03,
            },
        )


@pytest.fixture
def api():
    classifier = FakeClassifier()
    load_count = 0

    def factory():
        nonlocal load_count
        load_count += 1
        return classifier

    app = create_app(classifier_factory=factory)

    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client,
            classifier=classifier,
            app=app,
            get_load_count=lambda: load_count,
        )


def test_health_endpoint(api):
    response = api.client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_predict_endpoint(api):
    response = api.client.post(
        "/predict",
        json={
            "title": "Application crashes",
            "body": "RuntimeError on startup",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["label"] == "bug"
    assert data["confidence"] == pytest.approx(0.9)
    assert data["probabilities"]["bug"] == pytest.approx(0.9)

    assert api.classifier.calls == [
        {
            "title": "Application crashes",
            "body": "RuntimeError on startup",
        }
    ]


def test_predict_without_body(api):
    response = api.client.post(
        "/predict",
        json={"title": "Application crashes"},
    )

    assert response.status_code == 200
    assert api.classifier.calls[-1]["body"] == ""


def test_predict_preserves_whitespace(api):
    response = api.client.post(
        "/predict",
        json={
            "title": "  Application crashes  ",
            "body": "\nRuntimeError\n",
        },
    )

    assert response.status_code == 200
    assert api.classifier.calls[-1] == {
        "title": "  Application crashes  ",
        "body": "\nRuntimeError\n",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "", "body": ""},
        {"title": "   ", "body": " \n "},
        {"body": "Missing title"},
        {"title": 123, "body": "Body"},
        {"title": "Title", "body": None},
        {"title": "Title", "body": "Body", "extra": True},
        {"title": "x" * 501, "body": "Body"},
        {"title": "Title", "body": "x" * 20_001},
    ],
)
def test_invalid_requests_return_422(api, payload):
    response = api.client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422
    assert api.classifier.calls == []


def test_malformed_json_returns_422(api):
    response = api.client.post(
        "/predict",
        content='{"title":',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert api.classifier.calls == []


def test_classifier_loaded_once(api):
    assert api.get_load_count() == 1

    api.client.post(
        "/predict",
        json={"title": "First issue"},
    )

    api.client.post(
        "/predict",
        json={"title": "Second issue"},
    )

    assert api.get_load_count() == 1
    assert len(api.classifier.calls) == 2


def test_model_unavailable_returns_503():
    app = create_app(classifier_factory=lambda: FakeClassifier())

    # No lifespan startup: classifier is not initialized.
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Model is not loaded",
    }


def test_model_loading_failure():
    def failing_factory():
        raise RuntimeError("Checkpoint loading failed")

    app = create_app(classifier_factory=failing_factory)

    with pytest.raises(
        RuntimeError,
        match="Checkpoint loading failed",
    ):
        with TestClient(app):
            pass


def test_openapi_schema(api):
    response = api.client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/health" in paths
    assert "/predict" in paths
    assert "post" in paths["/predict"]


def test_unknown_endpoint_returns_404(api):
    response = api.client.get("/unknown")

    assert response.status_code == 404
