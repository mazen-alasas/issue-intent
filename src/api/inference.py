from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.data.neural_dataset import LABEL_TO_ID
from src.data.transformer_dataset import TransformerInputConfig, encode_issue
from src.models.evaluate_transformer import (
    resolve_device,
    validate_model_labels,
)
from src.models.structured_ablation_experiment import EXPERIMENTS
from src.preprocessing.structured import preprocess_structured

LABEL_NAMES = tuple(
    label
    for label, _ in sorted(
        LABEL_TO_ID.items(),
        key=lambda item: item[1],
    )
)

DEFAULT_CHECKPOINT = Path.home() / "issue-intent-artifacts/models/distilbert/best"


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class IssueClassifier:
    """Run inference with the evaluated DistilBERT pipeline."""

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        device: str = "cpu",
        input_config: TransformerInputConfig | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser()

        if not self.checkpoint.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory not found: {self.checkpoint}"
            )

        self.device = resolve_device(device)
        self.input_config = (
            input_config
            if input_config is not None
            else TransformerInputConfig(
                max_length=256,
                title_budget=48,
            )
        )

        self.preprocessing_config = EXPERIMENTS["E4"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint,
            use_fast=True,
            local_files_only=True,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
            local_files_only=True,
        ).to(self.device)

        validate_model_labels(self.model)
        self.model.eval()

    def predict(self, title: str, body: str) -> Prediction:
        """Classify one issue using training-compatible processing."""

        if not isinstance(title, str):
            raise TypeError("title must be a string")

        if not isinstance(body, str):
            raise TypeError("body must be a string")

        if not title.strip() and not body.strip():
            raise ValueError("title and body cannot both be empty")

        processed_body = preprocess_structured(
            body,
            config=self.preprocessing_config,
        )

        encoded = encode_issue(
            title=title,
            body=processed_body,
            tokenizer=self.tokenizer,
            config=self.input_config,
        )

        inputs = {
            key: torch.tensor(
                [value],
                dtype=torch.long,
                device=self.device,
            )
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            logits = self.model(**inputs).logits

            if logits.shape != (1, len(LABEL_NAMES)):
                raise ValueError(
                    f"Unexpected model output shape: {tuple(logits.shape)}"
                )

            if not torch.isfinite(logits).all().item():
                raise ValueError("Model returned non-finite logits")

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )[0]

        values = probabilities.cpu().tolist()
        predicted_id = int(probabilities.argmax().item())

        return Prediction(
            label=LABEL_NAMES[predicted_id],
            confidence=float(values[predicted_id]),
            probabilities={
                label: float(probability)
                for label, probability in zip(
                    LABEL_NAMES,
                    values,
                    strict=True,
                )
            },
        )
