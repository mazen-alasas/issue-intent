from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

IssueLabel = Literal["bug", "documentation", "feature", "question"]

MAX_TITLE_LENGTH = 500
MAX_BODY_LENGTH = 20_000


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        min_length=0,
        max_length=MAX_TITLE_LENGTH,
        strict=True,
    )
    body: str = Field(
        default="",
        max_length=MAX_BODY_LENGTH,
        strict=True,
    )

    @model_validator(mode="after")
    def validate_content(self):
        if not self.title.strip() and not self.body.strip():
            raise ValueError("title and body cannot both be empty")
        return self


class PredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: IssueLabel
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[IssueLabel, float]


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model_loaded: Literal[True]
