import json

import pandas as pd
import pytest

from src.models import train_full_classical as module


@pytest.fixture
def valid_dataset(tmp_path):
    """Create a small valid structured Parquet dataset."""
    path = tmp_path / "train.parquet"

    data = pd.DataFrame(
        {
            "title": [
                "Login error",
                "Add dark mode",
                "Update documentation",
                "How to install",
            ],
            "body": [
                "Authentication failed",
                "Please add theme support",
                "Improve the installation guide",
                "How can I install this package",
            ],
            "labels": [
                "bug",
                "feature",
                "documentation",
                "question",
            ],
        }
    )

    data.to_parquet(path, index=False)
    return path


def test_load_dataset_valid(valid_dataset):
    data = module.load_dataset(valid_dataset, expected_rows=4)

    assert len(data) == 4
    assert list(data.columns) == module.COLUMNS
    assert set(data["labels"]) == set(module.LABELS)


def test_load_dataset_missing_file(tmp_path):
    path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError):
        module.load_dataset(path, expected_rows=4)


def test_load_dataset_wrong_row_count(valid_dataset):
    with pytest.raises(ValueError, match="Unexpected row count"):
        module.load_dataset(valid_dataset, expected_rows=5)


def test_load_dataset_invalid_labels(tmp_path):
    path = tmp_path / "invalid.parquet"

    pd.DataFrame(
        {
            "title": ["Unknown issue"],
            "body": ["Unknown description"],
            "labels": ["invalid_label"],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Invalid labels"):
        module.load_dataset(path, expected_rows=1)


def test_load_dataset_null_values(tmp_path):
    path = tmp_path / "null.parquet"

    pd.DataFrame(
        {
            "title": ["Valid title"],
            "body": [None],
            "labels": ["bug"],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Null values"):
        module.load_dataset(path, expected_rows=1)


def test_load_dataset_empty_body_allowed(tmp_path):
    path = tmp_path / "empty_body.parquet"

    pd.DataFrame(
        {
            "title": ["Login error"],
            "body": [""],
            "labels": ["bug"],
        }
    ).to_parquet(path, index=False)

    data = module.load_dataset(path, expected_rows=1)

    assert data.loc[0, "body"] == ""


def test_run_experiment_smoke_test(tmp_path, monkeypatch):
    """Train and evaluate on small, isolated datasets."""
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"

    examples = {
        "bug": [
            "Login crashes with authentication error",
            "Application fails when opening settings",
            "Database connection throws exception",
            "Server crashes during startup",
        ],
        "documentation": [
            "Improve the installation documentation",
            "Update the configuration guide",
            "Add examples to the README",
            "Correct the API documentation",
        ],
        "feature": [
            "Add support for dark mode",
            "Implement a new search filter",
            "Allow users to export reports",
            "Introduce notification preferences",
        ],
        "question": [
            "How do I configure authentication",
            "Where can I find the API key",
            "Can I install this on Linux",
            "Why does the server need a token",
        ],
    }

    train_rows = []
    validation_rows = []

    for label, titles in examples.items():
        for title in titles[:3]:
            train_rows.append(
                {
                    "title": title,
                    "body": f"Details about {title.lower()}",
                    "labels": label,
                }
            )

        validation_rows.append(
            {
                "title": titles[3],
                "body": f"Details about {titles[3].lower()}",
                "labels": label,
            }
        )

    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(validation_rows).to_parquet(validation_path, index=False)

    monkeypatch.setattr(module, "TRAIN_PATH", train_path)
    monkeypatch.setattr(module, "VALIDATION_PATH", validation_path)
    monkeypatch.setattr(module, "EXPECTED_TRAIN_ROWS", 12)
    monkeypatch.setattr(module, "EXPECTED_VALIDATION_ROWS", 4)

    results = module.run_experiment(
        model_name="logreg",
        train_path=train_path,
        validation_path=validation_path,
        expected_train_rows=12,
        expected_validation_rows=4,
    )

    assert results["train_rows"] == 12
    assert results["validation_rows"] == 4
    assert results["features"]["vocabulary_size"] > 0
    assert results["features"]["train_nonzero"] > 0
    assert 0 <= results["metrics"]["micro_f1"] <= 1
    assert 0 <= results["metrics"]["macro_f1"] <= 1
    assert set(results["metrics"]["per_class"]) == set(module.LABELS)

    assert all(value >= 0 for value in results["timing"].values())

    # The experiment result must be JSON serializable.
    serialized = json.dumps(results)
    assert json.loads(serialized)["experiment"] == ("structured_v1_full_logreg")


def test_validation_does_not_fit_vectorizer(tmp_path, monkeypatch):
    """Ensure validation-only words never enter the TF-IDF vocabulary."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"

    labels = module.LABELS

    train = pd.DataFrame(
        {
            "title": [
                f"common training issue {i} {label}"
                for i, label in enumerate(labels * 3)
            ],
            "body": [f"training context for {label}" for label in labels * 3],
            "labels": labels * 3,
        }
    )

    validation = pd.DataFrame(
        {
            "title": [f"validationonlytoken issue {label}" for label in labels],
            "body": ["validationonlytoken"] * 4,
            "labels": labels,
        }
    )

    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)

    monkeypatch.setattr(module, "TRAIN_PATH", train_path)
    monkeypatch.setattr(module, "VALIDATION_PATH", validation_path)
    monkeypatch.setattr(module, "EXPECTED_TRAIN_ROWS", 12)
    monkeypatch.setattr(module, "EXPECTED_VALIDATION_ROWS", 4)

    original_fit_transform = TfidfVectorizer.fit_transform
    captured_vocabulary = {}

    def tracked_fit_transform(self, raw_documents, *args, **kwargs):
        result = original_fit_transform(self, raw_documents, *args, **kwargs)
        captured_vocabulary.update(self.vocabulary_)
        return result

    monkeypatch.setattr(
        TfidfVectorizer,
        "fit_transform",
        tracked_fit_transform,
    )

    module.run_experiment(
        model_name="logreg",
        train_path=train_path,
        validation_path=validation_path,
        expected_train_rows=12,
        expected_validation_rows=4,
    )

    assert captured_vocabulary
    assert "validationonlytoken" not in captured_vocabulary
