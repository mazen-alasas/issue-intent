from src.models.train_classical import build_title_logreg_pipeline


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

    pipeline = build_title_logreg_pipeline()
    pipeline.fit(titles, labels)

    predictions = pipeline.predict(
        [
            "application crashes",
            "update documentation",
        ]
    )

    assert len(predictions) == 2
    assert set(predictions).issubset(set(labels))
