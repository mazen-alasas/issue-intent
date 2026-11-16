import argparse
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src.data.neural_dataset import LABEL_TO_ID
from src.data.transformer_dataset import TransformerInputConfig, encode_issue
from src.models.transformer_utils import (
    resolve_device as resolve_device,
)
from src.models.transformer_utils import (
    validate_model_labels as validate_model_labels,
)

LABEL_NAMES = [
    label for label, _ in sorted(LABEL_TO_ID.items(), key=lambda item: item[1])
]
LABEL_IDS = list(range(len(LABEL_NAMES)))

PREDICTION_SCHEMA = pa.schema(
    [
        pa.field("row_index", pa.int64()),
        pa.field("id", pa.int64()),
        pa.field("true_label", pa.string()),
        pa.field("predicted_label", pa.string()),
        pa.field("confidence", pa.float32()),
        pa.field("correct", pa.bool_()),
        *[pa.field(f"prob_{label}", pa.float32()) for label in LABEL_NAMES],
    ]
)

INPUT_COLUMNS = ["id", "labels", "title", "body"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen Transformer checkpoint."
    )
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--title-budget", type=int, default=48)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--allow-generic-labels",
        action="store_true",
        help=(
            "Accept LABEL_N checkpoint names only after independently "
            "verifying the training label order."
        ),
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_test_batches(path, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    parquet = pq.ParquetFile(path)

    missing = set(INPUT_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing test columns: {sorted(missing)}")

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=INPUT_COLUMNS,
    ):
        yield batch.to_pydict()


def predict_batch(model, tokenizer, config, batch, device):
    labels = batch["labels"]

    if any(label not in LABEL_TO_ID for label in labels):
        raise ValueError("Unknown label in official test")

    if any(not isinstance(value, str) for value in batch["title"]):
        raise ValueError("Invalid title in official test")

    if any(not isinstance(value, str) for value in batch["body"]):
        raise ValueError("Invalid body in official test")

    encoded = [
        encode_issue(title, body, tokenizer, config)
        for title, body in zip(batch["title"], batch["body"], strict=True)
    ]

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in collator(encoded).items()}

    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)

    if probabilities.shape != (len(labels), len(LABEL_NAMES)):
        raise ValueError("Unexpected model output shape")

    return probabilities.cpu().numpy()


def build_prediction_frame(batch, probabilities, start_index):
    count = len(batch["labels"])

    if probabilities.shape != (count, len(LABEL_NAMES)):
        raise ValueError("Invalid probability matrix shape")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite probabilities")

    predicted_ids = probabilities.argmax(axis=1)
    true_ids = np.array([LABEL_TO_ID[label] for label in batch["labels"]])

    frame = pd.DataFrame(
        {
            "row_index": np.arange(
                start_index,
                start_index + count,
                dtype=np.int64,
            ),
            "id": batch["id"],
            "true_label": batch["labels"],
            "predicted_label": [LABEL_NAMES[index] for index in predicted_ids],
            "confidence": probabilities.max(axis=1),
            "correct": true_ids == predicted_ids,
        }
    )

    for index, label in enumerate(LABEL_NAMES):
        frame[f"prob_{label}"] = probabilities[:, index]

    return frame, true_ids, predicted_ids


def calculate_metrics(y_true, y_pred, elapsed, args):
    if len(y_true) == 0:
        raise ValueError("Official test contains no rows")

    correct = int(np.sum(y_true == y_pred))

    metrics = {
        "dataset": "NLBSE23 official test",
        "preprocessing": "structured_v1 E4 body_only",
        "checkpoint": str(args.checkpoint),
        "total_samples": int(len(y_true)),
        "correct": correct,
        "incorrect": int(len(y_true) - correct),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_IDS,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_IDS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_IDS,
                average="weighted",
                zero_division=0,
            )
        ),
        "inference_seconds": round(elapsed, 2),
        "max_length": args.max_length,
        "title_budget": args.title_budget,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_IDS,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABEL_IDS,
    )

    return metrics, report, matrix


def evaluate_transformer(
    test_path,
    checkpoint,
    output_dir,
    *,
    batch_size=16,
    max_length=256,
    title_budget=48,
    seed=42,
    device_name="auto",
    allow_generic_labels=False,
):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    config = TransformerInputConfig(
        max_length=max_length,
        title_budget=title_budget,
    )

    test_path = Path(test_path)
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)

    if not test_path.is_file():
        raise FileNotFoundError(test_path)

    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    set_seed(seed)
    device = resolve_device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        local_files_only=True,
    ).to(device)

    validate_model_labels(
        model,
        allow_generic_labels=allow_generic_labels,
    )
    model.eval()

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )

    writer = None

    try:
        predictions_path = temporary_dir / "test_predictions.parquet"

        true_parts = []
        predicted_parts = []
        row_count = 0

        start = time.perf_counter()

        for batch in iter_test_batches(test_path, batch_size):
            probabilities = predict_batch(model, tokenizer, config, batch, device)

            frame, true_ids, predicted_ids = build_prediction_frame(
                batch,
                probabilities,
                row_count,
            )

            table = pa.Table.from_pandas(
                frame,
                schema=PREDICTION_SCHEMA,
                preserve_index=False,
                safe=True,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    predictions_path,
                    PREDICTION_SCHEMA,
                    compression="zstd",
                )

            writer.write_table(table)

            true_parts.append(true_ids)
            predicted_parts.append(predicted_ids)
            row_count += len(frame)

        if row_count == 0:
            raise ValueError("Official test contains no rows")

        writer.close()
        writer = None

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start

        y_true = np.concatenate(true_parts)
        y_pred = np.concatenate(predicted_parts)

        args = argparse.Namespace(
            checkpoint=checkpoint,
            max_length=max_length,
            title_budget=title_budget,
            batch_size=batch_size,
            seed=seed,
        )

        metrics, report, matrix = calculate_metrics(y_true, y_pred, elapsed, args)

        (temporary_dir / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )

        pd.DataFrame(report).T.to_csv(temporary_dir / "test_classification_report.csv")

        pd.DataFrame(
            matrix,
            index=LABEL_NAMES,
            columns=LABEL_NAMES,
        ).to_csv(temporary_dir / "test_confusion_matrix.csv")

        # Publish only after all artifacts are complete.
        os.rename(temporary_dir, output_dir)

        return metrics

    finally:
        if writer is not None:
            writer.close()

        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def main():
    args = parse_args()

    metrics = evaluate_transformer(
        test_path=args.test,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        title_budget=args.title_budget,
        seed=args.seed,
        device_name=args.device,
        allow_generic_labels=args.allow_generic_labels,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
