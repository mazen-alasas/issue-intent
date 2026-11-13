import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

CONFIGS = [
    (128, 32),
    (256, 32),
    (256, 48),
    (384, 48),
]


def sample_parquet(
    path: Path,
    sample_size: int,
    seed: int = 42,
    batch_size: int = 2048,
) -> pd.DataFrame:
    if sample_size < 1 or batch_size < 1:
        raise ValueError("Sample size and batch size must be positive")

    rng = random.Random(seed)
    reservoir = []
    seen = 0

    parquet = pq.ParquetFile(path)

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["title", "body", "labels"],
    ):
        columns = batch.to_pydict()

        for title, body, label in zip(
            columns["title"],
            columns["body"],
            columns["labels"],
            strict=True,
        ):
            row = {
                "title": title,
                "body": body,
                "labels": label,
            }

            seen += 1

            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                index = rng.randrange(seen)

                if index < sample_size:
                    reservoir[index] = row

    if not reservoir:
        raise ValueError("Input dataset is empty")

    return pd.DataFrame(reservoir)


def analyze_sample(
    frame: pd.DataFrame,
    tokenizer,
) -> list[dict]:
    title_lengths = np.array(
        [
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in frame["title"]
        ],
        dtype=np.int32,
    )
    body_lengths = np.array(
        [
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in frame["body"]
        ],
        dtype=np.int32,
    )

    results = []

    for max_length, title_budget in CONFIGS:
        kept_title = np.minimum(title_lengths, title_budget)
        available_body = max_length - kept_title - 3
        kept_body = np.minimum(body_lengths, available_body)

        dropped_title = title_lengths - kept_title
        dropped_body = body_lengths - kept_body
        dropped_total = dropped_title + dropped_body

        encoded_lengths = kept_title + kept_body + 3

        results.append(
            {
                "max_length": max_length,
                "title_budget": title_budget,
                "rows": len(frame),
                "title_truncation_rate": float(np.mean(dropped_title > 0)),
                "body_truncation_rate": float(np.mean(dropped_body > 0)),
                "any_truncation_rate": float(np.mean(dropped_total > 0)),
                "mean_encoded_length": float(np.mean(encoded_lengths)),
                "p95_encoded_length": float(np.quantile(encoded_lengths, 0.95)),
                "mean_dropped_tokens": float(np.mean(dropped_total)),
                "p95_dropped_tokens": float(np.quantile(dropped_total, 0.95)),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/structured_v1/train.parquet"),
    )
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/transformer/length_analysis.json"),
    )
    args = parser.parse_args()

    if args.sample_size < 1:
        raise ValueError("sample-size must be positive")

    frame = sample_parquet(
        path=args.input,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased",
        use_fast=True,
    )

    results = analyze_sample(frame, tokenizer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
