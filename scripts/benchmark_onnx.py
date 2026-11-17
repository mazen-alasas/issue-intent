import argparse
import hashlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.labels import LABELS
from src.data.transformer_dataset import TransformerInputConfig, encode_issue
from src.models.transformer_utils import validate_model_labels
from src.preprocessing.experiment_configs import EXPERIMENTS
from src.preprocessing.structured import preprocess_structured


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path.home() / "issue-intent-artifacts/models/distilbert"
    parser.add_argument(
        "--data", type=Path, default=Path("data/raw/nlbse23-test.parquet")
    )
    parser.add_argument("--checkpoint", type=Path, default=root / "best")
    parser.add_argument("--onnx", type=Path, default=root / "onnx/model_fp32.onnx")
    parser.add_argument(
        "--output", type=Path, default=root / "onnx/fp32_benchmark.json"
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    for name in ("sample_size", "batch_size", "warmup", "repeats", "threads"):
        value = getattr(args, name)
        if value < (0 if name == "warmup" else 1):
            parser.error(
                f"--{name.replace('_', '-')} must be positive (warmup may be zero)"
            )
    return args


def sample_data(path, count, seed):
    df = pd.read_parquet(path, columns=["id", "labels", "title", "body"])
    if count > len(df):
        raise ValueError(f"Sample size {count} exceeds dataset size {len(df)}")
    if df["labels"].isna().any() or df["title"].isna().any():
        raise ValueError("Missing title or label in test data")
    if not set(df["labels"].unique()).issubset(set(LABELS)):
        raise ValueError("Unexpected label in test data")
    # Allocate proportional class quotas with largest-remainder rounding.
    counts = df["labels"].value_counts().sort_index()
    exact = counts / len(df) * count
    quotas = np.floor(exact).astype(int)
    remainder = count - int(quotas.sum())
    for label in (
        (exact - quotas).sort_values(ascending=False, kind="stable").index[:remainder]
    ):
        quotas.loc[label] += 1
    if (quotas > counts).any():
        raise ValueError("Requested sample cannot satisfy class quotas")
    pieces = [
        df.loc[df["labels"] == label].sample(n=int(n), random_state=seed)
        for label, n in quotas.items()
        if n
    ]
    result = pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)
    result["body"] = result["body"].fillna("")
    if not result["body"].map(lambda x: isinstance(x, str)).all():
        raise ValueError("Non-string body after null normalization")
    return result


def encode_rows(df, tokenizer):
    config = TransformerInputConfig(max_length=256, title_budget=48)
    for row in df.itertuples(index=False):
        body = preprocess_structured(row.body, config=EXPERIMENTS["E4"])
        yield encode_issue(row.title, body, tokenizer, config)


def make_batch(cases, pad_id):
    length = max(len(case["input_ids"]) for case in cases)
    ids = np.full((len(cases), length), pad_id, dtype=np.int64)
    mask = np.zeros((len(cases), length), dtype=np.int64)
    for i, case in enumerate(cases):
        n = len(case["input_ids"])
        ids[i, :n] = case["input_ids"]
        mask[i, :n] = case["attention_mask"]
    return {"input_ids": ids, "attention_mask": mask}


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q))


def timing(fn, batches, warmup, repeats, *, label="Model"):
    # Same pre-tokenized batches, excludes preprocessing, tokenization, and model load.
    print(
        f"{label}: warmup {warmup} round(s), then {repeats} measured round(s)",
        flush=True,
    )
    for _ in range(warmup):
        for batch in batches:
            fn(batch)
    measurements = []
    for round_index in range(repeats):
        for batch in batches:
            start = time.perf_counter()
            fn(batch)
            measurements.append((time.perf_counter() - start) * 1000)
        print(
            f"{label}: completed {round_index + 1}/{repeats} measured rounds",
            flush=True,
        )
    total_seconds = sum(measurements) / 1000
    return {
        "batch_calls": len(measurements),
        "median_batch_ms": float(statistics.median(measurements)),
        "p95_batch_ms": percentile(measurements, 95),
        "mean_batch_ms": float(statistics.mean(measurements)),
        "throughput_issues_per_second": float(
            repeats * sum(len(batch["input_ids"]) for batch in batches) / total_seconds
        ),
    }


def run(args):
    torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint.expanduser().resolve()
    onnx_path = args.onnx.expanduser().resolve()
    df = sample_data(args.data, args.sample_size, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, use_fast=True, local_files_only=True
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.float32
        )
        .cpu()
        .eval()
    )
    validate_model_labels(model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if {x.name for x in session.get_inputs()} != {"input_ids", "attention_mask"}:
        raise ValueError("Unexpected ONNX inputs")
    if [x.name for x in session.get_outputs()] != ["logits"]:
        raise ValueError("Unexpected ONNX outputs")
    cases = list(encode_rows(df, tokenizer))
    batches = [
        make_batch(cases[i : i + args.batch_size], tokenizer.pad_token_id)
        for i in range(0, len(cases), args.batch_size)
    ]

    def torch_forward(batch):
        with torch.inference_mode():
            return model(
                **{k: torch.from_numpy(v) for k, v in batch.items()}
            ).logits.numpy()

    def onnx_forward(batch):
        return session.run(["logits"], batch)[0]

    max_abs = 0.0
    max_rel = 0.0
    matches = 0
    close_count = 0
    truth = df["labels"].tolist()
    pytorch_correct = 0
    onnx_correct = 0
    offset = 0
    disagreements = []
    for batch in batches:
        expected = torch_forward(batch)
        actual = onnx_forward(batch)
        if expected.shape != actual.shape or expected.shape[1] != len(LABELS):
            raise AssertionError("Unexpected or mismatched logits shape")
        if not np.isfinite(expected).all() or not np.isfinite(actual).all():
            raise AssertionError("Non-finite logits")
        diff = np.abs(expected - actual)
        max_abs = max(max_abs, float(diff.max()))
        max_rel = max(max_rel, float((diff / np.maximum(np.abs(expected), 1e-8)).max()))
        close_count += int(
            np.all(
                np.isclose(expected, actual, atol=args.atol, rtol=args.rtol), axis=1
            ).sum()
        )
        py_ids = expected.argmax(axis=1)
        ox_ids = actual.argmax(axis=1)
        matches += int((py_ids == ox_ids).sum())
        for j, (p, o) in enumerate(zip(py_ids, ox_ids, strict=True)):
            label = truth[offset + j]
            pytorch_correct += int(LABELS[int(p)] == label)
            onnx_correct += int(LABELS[int(o)] == label)
            if p != o:
                disagreements.append(
                    {
                        "sample_index": offset + j,
                        "issue_id": str(df.iloc[offset + j]["id"]),
                        "pytorch": LABELS[int(p)],
                        "onnx": LABELS[int(o)],
                    }
                )
        offset += len(py_ids)
    sample_ids = [str(x) for x in df["id"].tolist()]
    digest = hashlib.sha256(
        json.dumps(sample_ids, separators=(",", ":")).encode()
    ).hexdigest()
    print(
        f"Sample: {len(df)} issues; distribution={df['labels'].value_counts().sort_index().to_dict()}"
    )
    print(
        f"Parity: agreement={matches}/{len(df)}; close={close_count}/{len(df)}; "
        f"max_abs={max_abs:.8f}; max_rel={max_rel:.8f}"
    )
    print("Benchmarking CPU model-only latency; this may take a few minutes...")
    torch_timing = timing(
        torch_forward, batches, args.warmup, args.repeats, label="PyTorch"
    )
    onnx_timing = timing(onnx_forward, batches, args.warmup, args.repeats, label="ONNX")
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": str(args.data),
        "sample_size": len(df),
        "seed": args.seed,
        "sample_issue_ids": sample_ids,
        "sample_ids_sha256": digest,
        "class_counts": {
            str(k): int(v) for k, v in df["labels"].value_counts().sort_index().items()
        },
        "null_bodies_in_sample": int((df["body"] == "").sum()),
        "preprocessing": "E4 structured body only",
        "max_length": 256,
        "title_budget": 48,
        "checkpoint": str(checkpoint),
        "onnx_model": str(onnx_path),
        "onnx_size_mib": onnx_path.stat().st_size / 1024**2,
        "checkpoint_weights_size_mib": (checkpoint / "model.safetensors").stat().st_size
        / 1024**2,
        "parity": {
            "prediction_agreement": matches / len(df),
            "close_logits_fraction": close_count / len(df),
            "max_absolute_logit_difference": max_abs,
            "max_relative_logit_difference": max_rel,
            "atol": args.atol,
            "rtol": args.rtol,
            "disagreements": disagreements,
            "pytorch_sample_accuracy": pytorch_correct / len(df),
            "onnx_sample_accuracy": onnx_correct / len(df),
        },
        "benchmark": {
            "device": "cpu",
            "threads": args.threads,
            "batch_size": args.batch_size,
            "warmup_rounds": args.warmup,
            "measurement_rounds": args.repeats,
            "excludes": ["model_load", "preprocessing", "tokenization"],
            "pytorch": torch_timing,
            "onnx": onnx_timing,
            "onnx_speedup_vs_pytorch": onnx_timing["throughput_issues_per_second"]
            / torch_timing["throughput_issues_per_second"],
        },
    }
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"PyTorch: {torch_timing}")
    print(f"ONNX: {onnx_timing}")
    print(f"Report: {args.output.expanduser()}")
    if matches != len(df) or close_count != len(df):
        raise AssertionError(
            "Real-data parity failed; inspect report before proceeding"
        )
    print("PASS: real-data FP32 parity on sampled issues")


if __name__ == "__main__":
    run(parse_args())
