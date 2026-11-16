from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.api import inference
from src.api.inference import DEFAULT_CHECKPOINT, resolve_checkpoint_path
from src.data.transformer_dataset import TransformerInputConfig


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        truncation,
    ):
        self.calls.append(text)
        return {"input_ids": [ord(char) % 100 + 1 for char in text]}


class FakeModel:
    def __init__(self, logits=None, label_mapping=None):
        if logits is None:
            logits = [[4.0, 1.0, 0.0, -1.0]]

        if label_mapping is None:
            label_mapping = {
                index: label for index, label in enumerate(inference.LABEL_NAMES)
            }

        self.logits = torch.tensor(
            logits,
            dtype=torch.float32,
        )
        self.config = SimpleNamespace(
            id2label=label_mapping,
        )
        self.device = None
        self.eval_called = False
        self.calls = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **inputs):
        self.calls.append(
            {
                "inputs": inputs,
                "inference_mode": torch.is_inference_mode_enabled(),
            }
        )
        return SimpleNamespace(logits=self.logits)


@pytest.fixture
def classifier_factory(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best"
    checkpoint.mkdir()

    def build(*, logits=None, label_mapping=None):
        tokenizer = FakeTokenizer()
        model = FakeModel(
            logits=logits,
            label_mapping=label_mapping,
        )

        tokenizer_loads = []
        model_loads = []

        def load_tokenizer(path, **kwargs):
            tokenizer_loads.append((path, kwargs))
            return tokenizer

        def load_model(path, **kwargs):
            model_loads.append((path, kwargs))
            return model

        monkeypatch.setattr(
            inference.AutoTokenizer,
            "from_pretrained",
            load_tokenizer,
        )
        monkeypatch.setattr(
            inference.AutoModelForSequenceClassification,
            "from_pretrained",
            load_model,
        )

        classifier = inference.IssueClassifier(
            checkpoint=checkpoint,
            device="cpu",
        )

        return SimpleNamespace(
            classifier=classifier,
            tokenizer=tokenizer,
            model=model,
            tokenizer_loads=tokenizer_loads,
            model_loads=model_loads,
            checkpoint=checkpoint,
        )

    return build


def test_loads_checkpoint_once(classifier_factory):
    context = classifier_factory()

    context.classifier.predict("First issue", "First body")
    context.classifier.predict("Second issue", "Second body")

    assert len(context.tokenizer_loads) == 1
    assert len(context.model_loads) == 1

    assert context.tokenizer_loads[0] == (
        context.checkpoint,
        {
            "use_fast": True,
            "local_files_only": True,
        },
    )

    assert context.model_loads[0] == (
        context.checkpoint,
        {
            "local_files_only": True,
        },
    )

    assert context.model.eval_called


def test_default_input_configuration(classifier_factory):
    context = classifier_factory()

    assert context.classifier.input_config == TransformerInputConfig(
        max_length=256,
        title_budget=48,
    )


def test_uses_e4_preprocessing(classifier_factory, monkeypatch):
    context = classifier_factory()
    captured = {}

    def fake_preprocess(body, config):
        captured["body"] = body
        captured["config"] = config
        return "PROCESSED BODY"

    monkeypatch.setattr(
        inference,
        "preprocess_structured",
        fake_preprocess,
    )

    context.classifier.predict(
        "Original title",
        "Original body",
    )

    assert captured["body"] == "Original body"
    assert captured["config"] is inference.EXPERIMENTS["E4"]

    assert context.tokenizer.calls == [
        "Original title",
        "PROCESSED BODY",
    ]


def test_returns_prediction_probabilities(classifier_factory):
    context = classifier_factory()

    prediction = context.classifier.predict(
        "Application crashes",
        "RuntimeError on startup",
    )

    assert prediction.label == "bug"
    assert 0.0 <= prediction.confidence <= 1.0
    assert set(prediction.probabilities) == set(inference.LABEL_NAMES)

    assert sum(prediction.probabilities.values()) == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert prediction.confidence == pytest.approx(prediction.probabilities["bug"])


def test_uses_inference_mode(classifier_factory):
    context = classifier_factory()

    context.classifier.predict(
        "Application crashes",
        "RuntimeError",
    )

    assert len(context.model.calls) == 1
    assert context.model.calls[0]["inference_mode"] is True


def test_encoded_inputs_have_batch_dimension(classifier_factory):
    context = classifier_factory()

    context.classifier.predict("Title", "Body")

    inputs = context.model.calls[0]["inputs"]

    assert inputs["input_ids"].ndim == 2
    assert inputs["attention_mask"].ndim == 2
    assert inputs["input_ids"].shape[0] == 1
    assert inputs["input_ids"].shape == inputs["attention_mask"].shape
    assert inputs["input_ids"].dtype == torch.long


@pytest.mark.parametrize(
    ("title", "body"),
    [
        ("", ""),
        ("   ", "\n\t"),
    ],
)
def test_rejects_completely_empty_issue(
    classifier_factory,
    title,
    body,
):
    context = classifier_factory()

    with pytest.raises(ValueError, match="cannot both be empty"):
        context.classifier.predict(title, body)


@pytest.mark.parametrize(
    ("title", "body"),
    [
        (None, "Body"),
        (123, "Body"),
        ("Title", None),
        ("Title", 123),
    ],
)
def test_rejects_non_string_inputs(
    classifier_factory,
    title,
    body,
):
    context = classifier_factory()

    with pytest.raises(TypeError):
        context.classifier.predict(title, body)


def test_allows_empty_body_with_title(classifier_factory):
    context = classifier_factory()

    prediction = context.classifier.predict(
        "Application crashes",
        "",
    )

    assert prediction.label == "bug"


def test_rejects_wrong_label_mapping(classifier_factory):
    incorrect_mapping = {
        0: "feature",
        1: "documentation",
        2: "bug",
        3: "question",
    }

    with pytest.raises(ValueError, match="mapping mismatch"):
        classifier_factory(
            label_mapping=incorrect_mapping,
        )


def test_rejects_generic_label_mapping(classifier_factory):
    generic_mapping = {index: f"LABEL_{index}" for index in range(4)}

    with pytest.raises(ValueError, match="generic LABEL_N"):
        classifier_factory(
            label_mapping=generic_mapping,
        )


def test_rejects_unexpected_output_shape(classifier_factory):
    context = classifier_factory(
        logits=[[1.0, 2.0, 3.0]],
    )

    with pytest.raises(ValueError, match="output shape"):
        context.classifier.predict("Title", "Body")


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_rejects_non_finite_logits(
    classifier_factory,
    invalid_value,
):
    context = classifier_factory(
        logits=[[invalid_value, 1.0, 0.0, -1.0]],
    )

    with pytest.raises(ValueError, match="non-finite logits"):
        context.classifier.predict("Title", "Body")


def test_rejects_missing_checkpoint(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(
        FileNotFoundError,
        match="Checkpoint directory not found",
    ):
        inference.IssueClassifier(checkpoint=missing)


def test_rejects_unavailable_cuda(classifier_factory, monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: False,
    )

    with pytest.raises(
        RuntimeError,
        match="CUDA requested but unavailable",
    ):
        inference.IssueClassifier(
            checkpoint=classifier_factory().checkpoint,
            device="cuda",
        )


def test_uses_real_e4_preprocessing(classifier_factory):
    context = classifier_factory()

    context.classifier.predict(
        "Original title",
        "See **important** details.",
    )

    assert context.tokenizer.calls[0] == "Original title"
    assert context.tokenizer.calls[1] == "See important details."


def test_checkpoint_path_uses_default(monkeypatch):
    monkeypatch.delenv("ISSUE_INTENT_CHECKPOINT", raising=False)

    assert resolve_checkpoint_path() == DEFAULT_CHECKPOINT


def test_checkpoint_path_uses_environment(monkeypatch):
    monkeypatch.setenv(
        "ISSUE_INTENT_CHECKPOINT",
        "/models/distilbert",
    )

    assert resolve_checkpoint_path() == Path("/models/distilbert")


def test_explicit_checkpoint_overrides_environment(monkeypatch):
    monkeypatch.setenv(
        "ISSUE_INTENT_CHECKPOINT",
        "/models/environment",
    )

    assert resolve_checkpoint_path(Path("/models/explicit")) == Path("/models/explicit")


def test_empty_environment_uses_default(monkeypatch):
    monkeypatch.setenv("ISSUE_INTENT_CHECKPOINT", "")

    assert resolve_checkpoint_path() == DEFAULT_CHECKPOINT
