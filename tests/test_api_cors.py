import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app

LOCAL_ORIGIN = "http://localhost:5173"
PRIVATE_NETWORK_HEADER = "Access-Control-Request-Private-Network"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("ISSUE_INTENT_CORS_ORIGINS", raising=False)

    app = create_app(classifier_factory=lambda: object())

    with TestClient(app) as test_client:
        yield test_client


def preflight(client, origin, *, private_network=False):
    headers = {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    }

    if private_network:
        headers[PRIVATE_NETWORK_HEADER] = "true"

    return client.options("/predict", headers=headers)


def test_allowed_origin_preflight(client):
    response = preflight(client, LOCAL_ORIGIN)

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == LOCAL_ORIGIN
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"].lower()


def test_private_network_preflight(client):
    response = preflight(
        client,
        LOCAL_ORIGIN,
        private_network=True,
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == LOCAL_ORIGIN
    assert response.headers["access-control-allow-private-network"] == "true"


def test_disallowed_origin_preflight(client):
    response = preflight(
        client,
        "https://untrusted.example",
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_disallowed_origin_with_private_network(client):
    response = preflight(
        client,
        "https://untrusted.example",
        private_network=True,
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_custom_allowed_origin(monkeypatch):
    custom_origin = "https://issue-intent.example.com"

    monkeypatch.setenv(
        "ISSUE_INTENT_CORS_ORIGINS",
        custom_origin,
    )

    app = create_app(classifier_factory=lambda: object())

    with TestClient(app) as client:
        response = preflight(client, custom_origin)

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == custom_origin

        default_origin_response = preflight(client, LOCAL_ORIGIN)

        assert default_origin_response.status_code == 400


def test_multiple_configured_origins(monkeypatch):
    origins = [
        "http://localhost:5173",
        "https://issue-intent.example.com",
    ]

    monkeypatch.setenv(
        "ISSUE_INTENT_CORS_ORIGINS",
        ", ".join(origins),
    )

    app = create_app(classifier_factory=lambda: object())

    with TestClient(app) as client:
        for origin in origins:
            response = preflight(client, origin)

            assert response.status_code == 200
            assert response.headers["access-control-allow-origin"] == origin
