import pandas as pd
import pytest

from src.preprocessing.analyze_neural_lengths import analyze_lengths


def test_analyze_lengths_valid_dataset(tmp_path):
    path = tmp_path / "train.parquet"

    data = pd.DataFrame(
        {
            "title": ["Login failed", "Add dark mode", "How to install"],
            "body": [
                "Authentication error occurred",
                "Please support dark themes",
                "Need help with installation",
            ],
        }
    )
    data.to_parquet(path, index=False)

    report = analyze_lengths(path)

    assert report["rows"] == 3
    assert report["length_unit"] == "whitespace_tokens"
    assert report["mean_length"] == pytest.approx(6.33)
    assert report["coverage"]["128"] == 1.0
    assert report["title"]["max"] == 3
    assert report["body"]["max"] == 4


def test_analyze_lengths_empty_dataset(tmp_path):
    path = tmp_path / "empty.parquet"

    pd.DataFrame({"title": [], "body": []}).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="empty"):
        analyze_lengths(path)


def test_analyze_lengths_null_text(tmp_path):
    path = tmp_path / "null.parquet"

    pd.DataFrame(
        {
            "title": ["Valid title"],
            "body": [None],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Null text"):
        analyze_lengths(path)


def test_analyze_lengths_multiple_batches(tmp_path, monkeypatch):
    from src.preprocessing import analyze_neural_lengths as module

    path = tmp_path / "multiple_batches.parquet"

    data = pd.DataFrame(
        {
            "title": ["Login error", "Add feature", "Fix docs"],
            "body": ["Something failed", "New capability", "Update guide"],
        }
    )
    data.to_parquet(path, index=False)

    monkeypatch.setattr(module, "BATCH_SIZE", 2)

    report = module.analyze_lengths(path)

    assert report["rows"] == 3
    assert report["mean_length"] == pytest.approx(4.0)
    assert report["coverage"]["128"] == 1.0
