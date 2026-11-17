import argparse
import json
import urllib.error
import urllib.request

EXPECTED_LABELS = {"bug", "documentation", "feature", "question"}


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise AssertionError(f"Unexpected HTTP status: {response.status}")

        return json.load(response)


def run_smoke_test(base_url: str) -> None:
    base_url = base_url.rstrip("/")

    health = request_json(f"{base_url}/health")

    assert health == {
        "status": "healthy",
        "model_loaded": True,
    }, f"Unexpected health response: {health}"

    print("PASS: /health")

    prediction = request_json(
        f"{base_url}/predict",
        {
            "title": "Application crashes on startup",
            "body": "The application crashes with RuntimeError after updating.",
        },
    )

    assert prediction["label"] == "bug", prediction
    assert set(prediction["probabilities"]) == EXPECTED_LABELS
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-5

    print("PASS: /predict")
    print(f"Predicted label: {prediction['label']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print("All API smoke tests passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a running API.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8001",
        help="Base URL of the running API.",
    )
    args = parser.parse_args()

    run_smoke_test(args.base_url)


if __name__ == "__main__":
    main()
