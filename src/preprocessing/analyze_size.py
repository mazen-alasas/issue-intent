import json
import time

import pandas as pd

from src.data.sampling import load_nested_stratified_samples
from src.models.structured_ablation_experiment import (
    EXPERIMENTS,
    build_experiment_text,
)
from src.models.train_classical import (
    RANDOM_STATE,
    ROOT,
    TRAIN_PATH,
)
from src.models.train_title_body import combine_title_body

SAMPLE_SIZE = 200_000
RESULTS_PATH = ROOT / "data/processed/preprocessing_size_analysis.json"


def summarize_lengths(text: pd.Series) -> dict:
    """Summarize character lengths of a text series."""
    lengths = text.str.len()

    return {
        "total_characters": int(lengths.sum()),
        "mean": round(float(lengths.mean()), 2),
        "median": float(lengths.median()),
        "p90": float(lengths.quantile(0.90)),
        "p95": float(lengths.quantile(0.95)),
        "p99": float(lengths.quantile(0.99)),
        "max": int(lengths.max()),
    }


def compare_text_versions(
    raw: pd.Series,
    processed: pd.Series,
) -> dict:
    """Compare processed text against its original version."""
    if len(raw) != len(processed):
        raise ValueError("Text series must have equal lengths.")

    raw_lengths = raw.str.len()
    processed_lengths = processed.str.len()

    difference = raw_lengths - processed_lengths

    total_raw = int(raw_lengths.sum())
    total_processed = int(processed_lengths.sum())

    reduction_pct = (
        100 * (total_raw - total_processed) / total_raw if total_raw else 0.0
    )

    return {
        "changed_issues": int((raw.to_numpy() != processed.to_numpy()).sum()),
        "shortened_issues": int((difference > 0).sum()),
        "expanded_issues": int((difference < 0).sum()),
        "unchanged_length_issues": int((difference == 0).sum()),
        "total_characters_removed": total_raw - total_processed,
        "reduction_pct": round(reduction_pct, 4),
        "average_characters_removed": round(float(difference.mean()), 2),
    }


def run_analysis() -> dict:
    """Compare Raw, E2, and E4 on one fixed training sample."""
    samples = load_nested_stratified_samples(
        csv_path=TRAIN_PATH,
        sample_sizes=[SAMPLE_SIZE],
        random_state=RANDOM_STATE,
        columns=["title", "body", "labels"],
    )
    train = samples[SAMPLE_SIZE]

    raw = combine_title_body(train)

    results = {
        "experiment": "preprocessing_size_analysis",
        "sample_size": SAMPLE_SIZE,
        "random_state": RANDOM_STATE,
        "versions": {
            "raw": {
                "lengths": summarize_lengths(raw),
            },
        },
    }

    for experiment_id in ("E2", "E4", "E5"):
        print(f"Analyzing {experiment_id}...", flush=True)

        start = time.perf_counter()

        processed = build_experiment_text(
            train,
            EXPERIMENTS[experiment_id],
        )

        elapsed = time.perf_counter() - start

        results["versions"][experiment_id] = {
            "lengths": summarize_lengths(processed),
            "comparison_to_raw": compare_text_versions(raw, processed),
            "processing_seconds": round(elapsed, 3),
        }

        del processed

    return results


def main() -> None:
    """Run size analysis and save JSON results."""
    results = run_analysis()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nSize analysis summary")
    print("-" * 55)

    raw_total = results["versions"]["raw"]["lengths"]["total_characters"]

    for name, version in results["versions"].items():
        total = version["lengths"]["total_characters"]
        reduction = 100 * (raw_total - total) / raw_total if raw_total else 0

        print(f"{name:>4}: {total:,} characters | reduction: {reduction:.2f}%")

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
