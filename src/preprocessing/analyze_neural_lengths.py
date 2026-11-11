import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.models.train_classical import ROOT

TRAIN_PATH = ROOT / "data/processed/structured_v1/train.parquet"
OUTPUT_PATH = ROOT / "data/processed/structured_v1/neural_length_report.json"

BATCH_SIZE = 10_000
PERCENTILES = [50, 75, 90, 95, 99, 99.5, 100]
LENGTH_LIMITS = [128, 256, 384, 512, 768, 1024]


def count_words(text: str) -> int:
    """Count whitespace-separated tokens without creating a word list."""
    count = 0
    inside_word = False

    for char in text:
        if char.isspace():
            inside_word = False
        elif not inside_word:
            count += 1
            inside_word = True

    return count


def analyze_lengths(path: Path = TRAIN_PATH) -> dict:
    """Analyze text lengths with bounded Parquet batch reads."""
    parquet = pq.ParquetFile(path)

    title_chunks = []
    body_chunks = []
    combined_chunks = []

    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["title", "body"],
    ):
        data = batch.to_pydict()

        title_lengths = []
        body_lengths = []
        combined_lengths = []

        for title, body in zip(
            data["title"],
            data["body"],
            strict=True,
        ):
            if title is None or body is None:
                raise ValueError("Null text found in training dataset")

            title_count = count_words(title)
            body_count = count_words(body)

            title_lengths.append(title_count)
            body_lengths.append(body_count)
            combined_lengths.append(title_count + body_count)

        title_chunks.append(np.asarray(title_lengths, dtype=np.int32))
        body_chunks.append(np.asarray(body_lengths, dtype=np.int32))
        combined_chunks.append(np.asarray(combined_lengths, dtype=np.int32))

    if not combined_chunks:
        raise ValueError("Training dataset is empty")

    title_lengths = np.concatenate(title_chunks)
    body_lengths = np.concatenate(body_chunks)
    combined_lengths = np.concatenate(combined_chunks)

    if combined_lengths.size == 0:
        raise ValueError("Training dataset is empty")

    return {
        "dataset": "structured_v1",
        "split": "train",
        "length_unit": "whitespace_tokens",
        "rows": int(combined_lengths.size),
        "mean_length": round(float(combined_lengths.mean()), 2),
        "percentiles": {
            str(p): round(float(np.percentile(combined_lengths, p)), 2)
            for p in PERCENTILES
        },
        "coverage": {
            str(limit): round(float(np.mean(combined_lengths <= limit)), 6)
            for limit in LENGTH_LIMITS
        },
        "title": {
            "p95": float(np.percentile(title_lengths, 95)),
            "p99": float(np.percentile(title_lengths, 99)),
            "max": int(title_lengths.max()),
        },
        "body": {
            "p95": float(np.percentile(body_lengths, 95)),
            "p99": float(np.percentile(body_lengths, 99)),
            "max": int(body_lengths.max()),
        },
    }


def main() -> None:
    report = analyze_lengths()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"Report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
