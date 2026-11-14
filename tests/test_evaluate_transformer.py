from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.models import evaluate_transformer as module


class FakeModel:
    def __init__(self, labels=None):
        if labels is None:
            labels = {index: label for index, label in enumerate(module.LABEL_NAMES)}
        self.config = SimpleNamespace(id2label=labels)

    def to(self, device):
        return self

    def eval(self):
        return self


@pytest.fixture
def test_data(tmp_path):
    path = tmp_path / "official_test.parquet"

    frame = pd.DataFrame(
        {
            "id": [101, 102, 101, 104],
            "labels": ["bug", "feature", "question", "documentation"],
            "title": ["A", "B", "C", "D"],
            "body": ["body A", "", "body C", "body D"],
            "author_association": ["NONE"] * 4,
        }
    )

    frame.to_parquet(path, index=False)
    return path


@pytest.fixture
def fake_checkpoint(tmp_path):
    path = tmp_path / "best"
    path.mkdir()
    return path


@pytest.fixture
def mocked_inference(monkeypatch):
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )

    monkeypatch.setattr(
        module.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )

    probabilities_by_title = {
        "A": [0.9, 0.03, 0.04, 0.03],
        "B": [0.05, 0.05, 0.85, 0.05],
        "C": [0.1, 0.1, 0.7, 0.1],
        "D": [0.05, 0.8, 0.1, 0.05],
    }

    def fake_predict(model, tokenizer, config, batch, device):
        return np.asarray(
            [probabilities_by_title[title] for title in batch["title"]],
            dtype=np.float32,
        )

    monkeypatch.setattr(module, "predict_batch", fake_predict)


def test_label_order():
    assert module.LABEL_NAMES == [
        "bug",
        "documentation",
        "feature",
        "question",
    ]


def test_validate_model_labels_accepts_correct_mapping():
    module.validate_model_labels(FakeModel())


def test_validate_model_labels_rejects_wrong_mapping():
    model = FakeModel(
        {
            0: "feature",
            1: "documentation",
            2: "bug",
            3: "question",
        }
    )

    with pytest.raises(ValueError, match="mapping mismatch"):
        module.validate_model_labels(model)


def test_iter_batches_preserves_order_and_duplicates(test_data):
    batches = list(module.iter_test_batches(test_data, batch_size=2))

    assert len(batches) == 2
    assert batches[0]["id"] == [101, 102]
    assert batches[1]["id"] == [101, 104]
    assert batches[1]["labels"] == ["question", "documentation"]


def test_iter_batches_rejects_missing_columns(tmp_path):
    path = tmp_path / "invalid.parquet"
    pd.DataFrame({"id": [1], "labels": ["bug"]}).to_parquet(path)

    with pytest.raises(ValueError, match="Missing test columns"):
        list(module.iter_test_batches(path, batch_size=2))


def test_iter_batches_rejects_invalid_batch_size(test_data):
    with pytest.raises(ValueError, match="batch_size"):
        list(module.iter_test_batches(test_data, batch_size=0))


def test_build_prediction_frame():
    batch = {
        "id": [5, 5],
        "labels": ["bug", "question"],
    }
    probabilities = np.array(
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.1, 0.7, 0.1],
        ],
        dtype=np.float32,
    )

    frame, y_true, y_pred = module.build_prediction_frame(
        batch, probabilities, start_index=10
    )

    assert frame["row_index"].tolist() == [10, 11]
    assert frame["id"].tolist() == [5, 5]
    assert frame["predicted_label"].tolist() == ["bug", "feature"]
    assert frame["correct"].tolist() == [True, False]
    assert frame["confidence"].tolist() == pytest.approx([0.8, 0.7])
    assert y_true.tolist() == [0, 3]
    assert y_pred.tolist() == [0, 2]


def test_build_prediction_frame_rejects_invalid_shape():
    batch = {"id": [1], "labels": ["bug"]}

    with pytest.raises(ValueError, match="shape"):
        module.build_prediction_frame(
            batch,
            np.array([[0.5, 0.5]]),
            start_index=0,
        )


def test_build_prediction_frame_rejects_nan():
    batch = {"id": [1], "labels": ["bug"]}

    with pytest.raises(ValueError, match="Non-finite"):
        module.build_prediction_frame(
            batch,
            np.array([[np.nan, 0.2, 0.3, 0.5]]),
            start_index=0,
        )


def test_calculate_metrics():
    y_true = np.array([0, 2, 3, 1])
    y_pred = np.array([0, 2, 2, 1])

    args = SimpleNamespace(
        checkpoint="best",
        max_length=256,
        title_budget=48,
        batch_size=16,
        seed=42,
    )

    metrics, report, matrix = module.calculate_metrics(y_true, y_pred, 1.25, args)

    assert metrics["total_samples"] == 4
    assert metrics["correct"] == 3
    assert metrics["incorrect"] == 1
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["micro_f1"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(2 / 3)
    assert metrics["weighted_f1"] == pytest.approx(2 / 3)
    assert report["question"]["recall"] == 0
    assert matrix.shape == (4, 4)
    assert matrix[3, 2] == 1


def test_calculate_metrics_rejects_empty():
    args = SimpleNamespace(checkpoint="best")

    with pytest.raises(ValueError, match="no rows"):
        module.calculate_metrics(
            np.array([]),
            np.array([]),
            0,
            args,
        )


def test_full_pipeline_with_mocked_model(
    test_data,
    fake_checkpoint,
    mocked_inference,
    tmp_path,
):
    output = tmp_path / "results"

    metrics = module.evaluate_transformer(
        test_path=test_data,
        checkpoint=fake_checkpoint,
        output_dir=output,
        batch_size=2,
        device_name="cpu",
    )

    assert metrics["total_samples"] == 4
    assert metrics["correct"] == 3
    assert metrics["incorrect"] == 1
    assert metrics["accuracy"] == pytest.approx(0.75)

    predictions = pq.read_table(output / "test_predictions.parquet").to_pandas()

    assert predictions["row_index"].tolist() == [0, 1, 2, 3]
    assert predictions["id"].tolist() == [101, 102, 101, 104]
    assert predictions["correct"].tolist() == [True, True, False, True]

    assert (
        pq.read_metadata(output / "test_predictions.parquet").schema.to_arrow_schema()
        == module.PREDICTION_SCHEMA
    )

    assert (output / "test_metrics.json").is_file()

    assert (output / "test_classification_report.csv").is_file()

    assert (output / "test_confusion_matrix.csv").is_file()

    matrix = pd.read_csv(
        output / "test_confusion_matrix.csv",
        index_col=0,
    )

    assert matrix.loc["question", "feature"] == 1


def test_rejects_existing_output(test_data, fake_checkpoint, tmp_path):
    output = tmp_path / "results"
    output.mkdir()
    (output / "existing.txt").write_text("keep")

    with pytest.raises(FileExistsError):
        module.evaluate_transformer(
            test_data,
            fake_checkpoint,
            output,
        )

    assert (output / "existing.txt").read_text() == "keep"


def test_rejects_missing_test(fake_checkpoint, tmp_path):
    with pytest.raises(FileNotFoundError):
        module.evaluate_transformer(
            tmp_path / "missing.parquet",
            fake_checkpoint,
            tmp_path / "results",
        )


def test_rejects_missing_checkpoint(test_data, tmp_path):
    with pytest.raises(FileNotFoundError):
        module.evaluate_transformer(
            test_data,
            tmp_path / "missing",
            tmp_path / "results",
        )


def test_rejects_invalid_batch_size(test_data, fake_checkpoint, tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        module.evaluate_transformer(
            test_data,
            fake_checkpoint,
            tmp_path / "results",
            batch_size=0,
        )


def test_empty_test_cleans_predictions(tmp_path, fake_checkpoint, mocked_inference):
    test_path = tmp_path / "empty.parquet"
    output = tmp_path / "results"

    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("labels", pa.string()),
            pa.field("title", pa.string()),
            pa.field("body", pa.string()),
        ]
    )

    pq.write_table(
        pa.Table.from_pylist([], schema=schema),
        test_path,
    )

    with pytest.raises(ValueError, match="no rows"):
        module.evaluate_transformer(
            test_path,
            fake_checkpoint,
            output,
            device_name="cpu",
        )

    assert not (output / "test_predictions.parquet").exists()


def test_rejects_generic_checkpoint_labels_by_default():
    model = FakeModel(
        {
            0: "LABEL_0",
            1: "LABEL_1",
            2: "LABEL_2",
            3: "LABEL_3",
        }
    )

    with pytest.raises(ValueError, match="generic LABEL_N"):
        module.validate_model_labels(model)


def test_accepts_generic_labels_with_explicit_confirmation():
    model = FakeModel(
        {
            0: "LABEL_0",
            1: "LABEL_1",
            2: "LABEL_2",
            3: "LABEL_3",
        }
    )

    module.validate_model_labels(
        model,
        allow_generic_labels=True,
    )


def test_rejects_incorrect_mapping_even_with_override():
    model = FakeModel(
        {
            0: "feature",
            1: "documentation",
            2: "bug",
            3: "question",
        }
    )

    with pytest.raises(ValueError, match="mapping mismatch"):
        module.validate_model_labels(
            model,
            allow_generic_labels=True,
        )


def test_failed_inference_does_not_publish_partial_outputs(
    test_data,
    fake_checkpoint,
    mocked_inference,
    monkeypatch,
    tmp_path,
):
    output = tmp_path / "results"

    original_predict = module.predict_batch
    calls = 0

    def fail_on_second_batch(*args, **kwargs):
        nonlocal calls
        calls += 1

        if calls == 2:
            raise RuntimeError("Simulated inference failure")

        return original_predict(*args, **kwargs)

    monkeypatch.setattr(
        module,
        "predict_batch",
        fail_on_second_batch,
    )

    with pytest.raises(
        RuntimeError,
        match="Simulated inference failure",
    ):
        module.evaluate_transformer(
            test_path=test_data,
            checkpoint=fake_checkpoint,
            output_dir=output,
            batch_size=2,
            device_name="cpu",
        )

    assert not output.exists()
    assert list(tmp_path.glob(".results.tmp-*")) == []


def test_successful_evaluation_cleans_temporary_directory(
    test_data,
    fake_checkpoint,
    mocked_inference,
    tmp_path,
):
    output = tmp_path / "results"

    module.evaluate_transformer(
        test_path=test_data,
        checkpoint=fake_checkpoint,
        output_dir=output,
        batch_size=2,
        device_name="cpu",
    )

    assert output.is_dir()
    assert (output / "test_predictions.parquet").is_file()
    assert (output / "test_metrics.json").is_file()

    assert list(tmp_path.glob(".results.tmp-*")) == []


def test_rejects_existing_empty_output_directory(
    test_data,
    fake_checkpoint,
    tmp_path,
):
    output = tmp_path / "results"
    output.mkdir()

    with pytest.raises(FileExistsError):
        module.evaluate_transformer(
            test_path=test_data,
            checkpoint=fake_checkpoint,
            output_dir=output,
            device_name="cpu",
        )

    assert output.is_dir()
