import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier

from src.models.evaluation import compute_classification_metrics

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data/processed/train.csv"
VALIDATION_PATH = ROOT / "data/processed/validation.csv"
RESULTS_PATH = ROOT / "data/processed/dummy_baseline_metrics.json"

LABELS = ["bug", "documentation", "feature", "question"]


def evaluate_dummy_baseline(
    train_path: str,
    validation_path: str,
) -> dict:
    """Evaluate a majority-class baseline on the validation split.

    Args:
        train_path (str): Path to the training CSV.
        validation_path (str): Path to the validation CSV.

    Returns:
        dict: Classification metrics for the dummy baseline.
    """
    train = pd.read_csv(train_path, usecols=["labels"])
    validation = pd.read_csv(validation_path, usecols=["labels"])

    classifier = DummyClassifier(strategy="most_frequent")

    # DummyClassifier ignores the actual feature values.
    x_train = pd.DataFrame({"constant": [0] * len(train)})
    x_validation = pd.DataFrame({"constant": [0] * len(validation)})

    classifier.fit(x_train, train["labels"])
    predictions = classifier.predict(x_validation)

    return compute_classification_metrics(
        y_true=validation["labels"],
        y_pred=predictions,
        labels=LABELS,
    )


def main() -> None:
    """Evaluate and save the majority-class baseline metrics."""
    metrics = evaluate_dummy_baseline(
        train_path=str(TRAIN_PATH),
        validation_path=str(VALIDATION_PATH),
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Majority-class baseline evaluation completed.")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Micro F1:    {metrics['micro_f1']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Results:     {RESULTS_PATH}")


if __name__ == "__main__":
    main()
