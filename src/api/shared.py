"""Backend-neutral inference response and checkpoint configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.data.labels import LABEL_TO_ID

LABEL_NAMES = tuple(
    label for label, _ in sorted(LABEL_TO_ID.items(), key=lambda item: item[1])
)
DEFAULT_CHECKPOINT = Path.home() / "issue-intent-artifacts/models/distilbert/best"


def resolve_checkpoint_path(checkpoint: Path | None = None) -> Path:
    """Resolve explicit, environment, or default checkpoint path."""
    if checkpoint is not None:
        return Path(checkpoint).expanduser()
    configured = os.environ.get("ISSUE_INTENT_CHECKPOINT")
    return Path(configured).expanduser() if configured else DEFAULT_CHECKPOINT


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class Classifier(Protocol):
    def predict(self, title: str, body: str) -> Prediction: ...
