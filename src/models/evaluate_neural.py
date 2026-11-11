import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.data.neural_dataset import LABEL_TO_ID, LABELS, collate_issues, encode
from src.models.textcnn import TextCNN
from src.models.train_classical import ROOT

DEFAULT_DATA = ROOT / "data/processed/structured_v1/validation.parquet"
DEFAULT_ARTIFACTS = ROOT / "data/processed/neural_textcnn"


def load_model(checkpoint_path: Path, vocab: dict[str, int], device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["vocab_size"] != len(vocab):
        raise ValueError("Checkpoint and vocabulary sizes do not match")
    if tuple(checkpoint["labels"]) != LABELS:
        raise ValueError("Checkpoint label ordering differs from the dataset")
    model = TextCNN(len(vocab))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def evaluate_checkpoint(
    checkpoint_path: Path,
    vocab_path: Path,
    validation_path: Path,
    output_dir: Path,
    batch_size: int = 128,
    sample_size: int = 100,
    seed: int = 42,
    cpu: bool = False,
):
    """
    Evaluate a trained model checkpoint on a validation dataset and save the results.

    Args:
        checkpoint_path: Path to the model checkpoint.
        vocab_path: Path to the vocabulary file.
        validation_path: Path to the validation dataset.
        output_dir: Directory to save the evaluation results.
        batch_size: Batch size for evaluation.
        sample_size: Number of errors to sample for detailed analysis.
        seed: Random seed for reproducibility.
        cpu: Whether to use CPU instead of GPU.

    Returns:
        A summary dictionary containing evaluation metrics and error analysis.

    Raises:
        ValueError: If batch_size or sample_size are invalid, or if the validation dataset is empty.
        RuntimeError: If there is an error loading the model or processing the data.
    """
    if batch_size < 1 or sample_size < 0:
        raise ValueError("Invalid batch_size or sample_size")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    model, checkpoint = load_model(checkpoint_path, vocab, device)
    max_length = int(checkpoint["max_length"])
    title_budget = int(checkpoint["title_budget"])
    if max_length < 5 or title_budget < 1:
        raise ValueError("Invalid checkpoint sequence settings")
    y_true, y_pred = [], []
    errors = []
    rng = random.Random(seed)
    error_count = 0
    confusions = Counter()
    total = 0
    parquet = pq.ParquetFile(validation_path)
    with torch.inference_mode():
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=["title", "body", "labels"]
        ):
            rows = batch.to_pydict()
            examples = []
            for title, body, label in zip(
                rows["title"], rows["body"], rows["labels"], strict=True
            ):
                if (
                    label not in LABEL_TO_ID
                    or not isinstance(title, str)
                    or not isinstance(body, str)
                ):
                    raise ValueError("Invalid validation row")
                examples.append(
                    (
                        encode(title, body, vocab, max_length, title_budget),
                        LABEL_TO_ID[label],
                    )
                )
            if not examples:
                continue
            features, targets = collate_issues(examples)
            preds = model(features.to(device)).argmax(dim=1).cpu().tolist()
            actual = targets.tolist()
            y_true.extend(actual)
            y_pred.extend(preds)
            for idx, (truth, prediction) in enumerate(zip(actual, preds, strict=True)):
                if truth != prediction:
                    error_count += 1
                    confusions[(LABELS[truth], LABELS[prediction])] += 1
                    item = {
                        "row_index": total + idx,
                        "true_label": LABELS[truth],
                        "predicted_label": LABELS[prediction],
                        "title": rows["title"][idx],
                        "body_excerpt": rows["body"][idx][:800],
                    }
                    if len(errors) < sample_size:
                        errors.append(item)
                    elif sample_size:
                        j = rng.randrange(error_count)
                        if j < sample_size:
                            errors[j] = item
            total += len(actual)
    if not total:
        raise ValueError("Empty validation dataset")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    summary = {
        "total_rows": total,
        "total_errors": error_count,
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "labels": list(LABELS),
        "top_confusions": [
            {"actual": actual, "predicted": predicted, "count": count}
            for (actual, predicted), count in confusions.most_common()
        ],
        "checkpoint": str(checkpoint_path),
        "validation_path": str(validation_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "error_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *LABELS])
        for label, counts in zip(LABELS, matrix, strict=True):
            writer.writerow([label, *[int(value) for value in counts]])
    with (output_dir / "error_samples.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "row_index",
                "true_label",
                "predicted_label",
                "title",
                "body_excerpt",
            ],
        )
        writer.writeheader()
        writer.writerows(errors)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_ARTIFACTS / "best.pt"
    )
    parser.add_argument("--vocab", type=Path, default=DEFAULT_ARTIFACTS / "vocab.json")
    parser.add_argument("--validation", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ARTIFACTS / "evaluation"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    summary = evaluate_checkpoint(
        args.checkpoint,
        args.vocab,
        args.validation,
        args.output_dir,
        args.batch_size,
        args.sample_size,
        args.seed,
        args.cpu,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
