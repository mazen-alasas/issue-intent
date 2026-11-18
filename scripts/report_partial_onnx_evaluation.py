import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data.labels import LABELS


def metrics(truth, predictions):
    ids = np.arange(len(LABELS))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=ids, zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(truth, predictions, labels=ids).tolist(),
    }
    for average in ("micro", "macro", "weighted"):
        result[f"{average}_f1"] = float(
            precision_recall_fscore_support(
                truth, predictions, labels=ids, average=average, zero_division=0
            )[2]
        )
    return result


def main():
    root = Path.home() / "issue-intent-artifacts/models/distilbert/onnx"
    parser = argparse.ArgumentParser(
        description="Report saved partial FP32/INT8 ONNX quality"
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/raw/nlbse23-test.parquet")
    )
    parser.add_argument(
        "--progress", type=Path, default=root / "int8_full_evaluation.progress.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "int8_partial_evaluation.json"
    )
    args = parser.parse_args()

    with np.load(args.progress, allow_pickle=False) as saved:
        completed = int(saved["completed"].item())
        fingerprint = str(saved["fingerprint"].item())
        fp32 = saved["fp32"].copy()
        int8 = saved["int8"].copy()

    if (
        completed <= 0
        or fp32.shape != (completed, len(LABELS))
        or int8.shape != fp32.shape
    ):
        raise ValueError("Invalid checkpoint count or logits shape")
    if not np.isfinite(fp32).all() or not np.isfinite(int8).all():
        raise ValueError("Checkpoint contains non-finite logits")

    df = pd.read_parquet(args.data, columns=["id", "labels"])
    if completed > len(df):
        raise ValueError("Checkpoint has more rows than dataset")
    if df["labels"].isna().any() or not set(df["labels"].unique()).issubset(
        set(LABELS)
    ):
        raise ValueError("Missing or unknown labels")

    label_to_id = {label: index for index, label in enumerate(LABELS)}
    truth = df["labels"].iloc[:completed].map(label_to_id).to_numpy(dtype=np.int64)
    fp32_pred = fp32.argmax(axis=1)
    int8_pred = int8.argmax(axis=1)
    changed = np.flatnonzero(fp32_pred != int8_pred)
    full_counts = df["labels"].value_counts()
    partial_counts = df["labels"].iloc[:completed].value_counts()

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source": "Saved logits from the first N rows of the official test dataset",
        "limitations": "Partial prefix, not a random/stratified sample; not the full official test result",
        "checkpoint_fingerprint_sha256": fingerprint,
        "dataset_path": str(args.data.expanduser().resolve()),
        "completed_issues": completed,
        "total_test_issues": len(df),
        "coverage": completed / len(df),
        "label_distribution": {
            label: {
                "partial_count": int(partial_counts.get(label, 0)),
                "partial_fraction": float(partial_counts.get(label, 0) / completed),
                "full_count": int(full_counts.get(label, 0)),
                "full_fraction": float(full_counts.get(label, 0) / len(df)),
            }
            for label in LABELS
        },
        "quality": {
            "onnx_fp32": metrics(truth, fp32_pred),
            "onnx_int8": metrics(truth, int8_pred),
        },
        "comparison": {
            "prediction_agreement": float((fp32_pred == int8_pred).mean()),
            "changed_predictions": int(len(changed)),
            "fp32_correct_int8_wrong": int(
                ((fp32_pred == truth) & (int8_pred != truth)).sum()
            ),
            "fp32_wrong_int8_correct": int(
                ((fp32_pred != truth) & (int8_pred == truth)).sum()
            ),
            "both_wrong_different": int(
                (
                    (fp32_pred != truth)
                    & (int8_pred != truth)
                    & (fp32_pred != int8_pred)
                ).sum()
            ),
            "max_absolute_logit_difference": float(np.max(np.abs(fp32 - int8))),
            "changed_issues": [
                {
                    "sample_index": int(i),
                    "issue_id": str(df.iloc[int(i)]["id"]),
                    "truth": LABELS[int(truth[i])],
                    "fp32": LABELS[int(fp32_pred[i])],
                    "int8": LABELS[int(int8_pred[i])],
                }
                for i in changed
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, args.output)

    print(f"Evaluated: {completed:,}/{len(df):,} ({completed / len(df):.2%})")
    for model in ("onnx_fp32", "onnx_int8"):
        quality = report["quality"][model]
        print(
            f"{model}: micro F1={quality['micro_f1']:.6f}; macro F1={quality['macro_f1']:.6f}"
        )
        for label, values in quality["per_class"].items():
            print(f"  {label}: F1={values['f1']:.6f}; support={values['support']:,}")
    comparison = report["comparison"]
    print(
        f"Agreement: {comparison['prediction_agreement']:.4%}; changed={comparison['changed_predictions']:,}"
    )
    print(f"FP32 correct -> INT8 wrong: {comparison['fp32_correct_int8_wrong']:,}")
    print(f"FP32 wrong -> INT8 correct: {comparison['fp32_wrong_int8_correct']:,}")
    print("Label distribution (partial vs full):")
    for label, values in report["label_distribution"].items():
        print(
            f"  {label}: {values['partial_fraction']:.2%} vs {values['full_fraction']:.2%}"
        )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
