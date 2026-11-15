from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request

from src.api.inference import IssueClassifier
from src.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


def create_app(classifier_factory=None) -> FastAPI:
    """Create the API with an injectable classifier factory."""

    if classifier_factory is None:
        classifier_factory = IssueClassifier

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = classifier_factory()
        try:
            yield
        finally:
            app.state.classifier = None

    app = FastAPI(
        title="Issue Intent API",
        description="Classify GitHub issues using DistilBERT.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def get_classifier(request: Request) -> IssueClassifier:
        classifier = getattr(
            request.app.state,
            "classifier",
            None,
        )

        if classifier is None:
            raise HTTPException(
                status_code=503,
                detail="Model is not loaded",
            )

        return classifier

    ClassifierDependency = Annotated[
        IssueClassifier,
        Depends(get_classifier),
    ]

    @app.get(
        "/health",
        response_model=HealthResponse,
    )
    def health(
        classifier: ClassifierDependency,
    ) -> HealthResponse:
        return HealthResponse(
            status="healthy",
            model_loaded=True,
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
    )
    def predict(
        payload: PredictRequest,
        classifier: ClassifierDependency,
    ) -> PredictResponse:
        prediction = classifier.predict(
            title=payload.title,
            body=payload.body,
        )

        return PredictResponse(
            label=prediction.label,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
        )

    return app


app = create_app()
