import csv
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.data.neural_dataset import LABELS
from src.models.evaluate_neural import evaluate_checkpoint, load_model
from src.models.textcnn import TextCNN


def test_evaluate_checkpoint_creates_reports(tmp_path):
    vocab = {"<pad>": 0, "<unk>": 1, "bug": 2, "feature": 3, "docs": 4, "help": 5}
    vocab_path = tmp_path / "vocab.json"
    vocab_path.write_text(json.dumps(vocab), encoding="utf-8")
    model = TextCNN(len(vocab))
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": len(vocab),
            "labels": list(LABELS),
            "max_length": 16,
            "title_budget": 4,
        },
        checkpoint_path,
    )
    table = pa.table(
        {
            "title": ["bug", "feature", "docs", "help"],
            "body": ["help"] * 4,
            "labels": list(LABELS),
        }
    )
    validation_path = tmp_path / "validation.parquet"
    pq.write_table(table, validation_path, row_group_size=2)
    output_dir = tmp_path / "evaluation"
    summary = evaluate_checkpoint(
        checkpoint_path,
        vocab_path,
        validation_path,
        output_dir,
        batch_size=2,
        sample_size=3,
        cpu=True,
    )
    assert summary["total_rows"] == 4
    assert (output_dir / "classification_report.json").exists()
    assert (output_dir / "error_summary.json").exists()
    with (output_dir / "confusion_matrix.csv").open(newline="") as file:
        rows = list(csv.reader(file))
    assert len(rows) == 5
    assert sum(int(n) for row in rows[1:] for n in row[1:]) == 4
    with (output_dir / "error_samples.csv").open(newline="") as file:
        assert len(list(csv.DictReader(file))) <= 3


def test_reject_mismatched_vocab(tmp_path):
    checkpoint_path = tmp_path / "bad.pt"
    torch.save({"vocab_size": 9, "labels": list(LABELS)}, checkpoint_path)
    with pytest.raises(ValueError, match="vocabulary sizes"):
        load_model(checkpoint_path, {"<pad>": 0, "<unk>": 1}, torch.device("cpu"))


def test_reject_invalid_arguments(tmp_path):
    with pytest.raises(ValueError, match="Invalid"):
        evaluate_checkpoint(
            tmp_path / "missing.pt",
            tmp_path / "missing.json",
            tmp_path / "missing.parquet",
            tmp_path / "out",
            batch_size=0,
        )
