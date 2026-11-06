import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.models.train_classical import build_title_pipeline


def test_title_logreg_pipeline_predicts_valid_labels() -> None:
    """Verify the pipeline can fit and predict valid class labels."""
    titles = [
        "application crashes on startup",
        "application crashes after login",
        "request new export feature",
        "request additional export feature",
        "how to configure the server",
        "how to configure the client",
        "update installation documentation",
        "improve installation documentation",
    ]

    labels = [
        "bug",
        "bug",
        "feature",
        "feature",
        "question",
        "question",
        "documentation",
        "documentation",
    ]

    pipeline = build_title_pipeline("logreg")
    pipeline.fit(titles, labels)

    predictions = pipeline.predict(
        [
            "application crashes",
            "update documentation",
        ]
    )

    assert len(predictions) == 2
    assert set(predictions).issubset(set(labels))


@pytest.mark.parametrize(
    ("model_name", "expected_type"),
    [
        ("logreg", LogisticRegression),
        ("linear_svc", LinearSVC),
    ],
)
def test_build_title_pipeline_selects_classifier(
    model_name: str,
    expected_type: type,
) -> None:
    """Verify the pipeline selects the requested classifier."""
    pipeline = build_title_pipeline(model_name)

    assert isinstance(
        pipeline.named_steps["classifier"],
        expected_type,
    )


def test_build_title_pipeline_rejects_unknown_model() -> None:
    """Verify unsupported classifier names are rejected."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_title_pipeline("unknown")
