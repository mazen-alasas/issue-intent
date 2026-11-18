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
from sklearn.metrics import accuracy_score, f1_score
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
    parser.add_argument("--int8", type=Path, default=root / "onnx/model_int8.onnx")
    parser.add_argument(
        "--output", type=Path, default=root / "onnx/int8_comparison.json"
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--benchmark-sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    for name in (
        "sample_size",
        "benchmark_sample_size",
        "batch_size",
        "warmup",
        "repeats",
        "threads",
    ):
        value = getattr(args, name)
        if value < (0 if name == "warmup" else 1):
            parser.error(
                f"--{name.replace('_', '-')} must be positive (warmup may be zero)"
            )
    if args.benchmark_sample_size > args.sample_size:
        parser.error("--benchmark-sample-size cannot exceed --sample-size")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be nonnegative")
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


def make_session(path, threads):
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if {item.name for item in session.get_inputs()} != {"input_ids", "attention_mask"}:
        raise ValueError(f"Unexpected ONNX inputs: {path}")
    if [item.name for item in session.get_outputs()] != ["logits"]:
        raise ValueError(f"Unexpected ONNX outputs: {path}")
    return session


def evaluate_logits(reference, candidate, truth, *, atol, rtol, issue_ids):
    if reference.shape != candidate.shape or reference.shape != (
        len(truth),
        len(LABELS),
    ):
        raise AssertionError("Unexpected or mismatched logits shape")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise AssertionError("Non-finite logits")

    ref_ids = reference.argmax(axis=1)
    cand_ids = candidate.argmax(axis=1)
    close = np.all(np.isclose(reference, candidate, atol=atol, rtol=rtol), axis=1)
    diff = np.abs(reference - candidate)
    relative = diff / np.maximum(np.abs(reference), 1e-8)
    disagreements = [
        {
            "sample_index": i,
            "issue_id": issue_ids[i],
            "reference": LABELS[int(ref_ids[i])],
            "candidate": LABELS[int(cand_ids[i])],
        }
        for i in range(len(truth))
        if ref_ids[i] != cand_ids[i]
    ]
    return {
        "prediction_agreement": float(np.mean(ref_ids == cand_ids)),
        "close_logits_fraction": float(np.mean(close)),
        "max_absolute_logit_difference": float(diff.max()),
        "max_relative_logit_difference": float(relative.max()),
        "disagreements": disagreements,
    }


def classification_metrics(logits, truth):
    predictions = [LABELS[int(i)] for i in logits.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "micro_f1": float(
            f1_score(
                truth, predictions, labels=LABELS, average="micro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                truth, predictions, labels=LABELS, average="macro", zero_division=0
            )
        ),
        "per_class_f1": {
            label: float(value)
            for label, value in zip(
                LABELS,
                f1_score(
                    truth, predictions, labels=LABELS, average=None, zero_division=0
                ),
                strict=True,
            )
        },
    }


def probabilities(logits):
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def run(args):
    torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint.expanduser().resolve()
    fp32_path = args.onnx.expanduser().resolve()
    int8_path = args.int8.expanduser().resolve()
    if fp32_path == int8_path:
        raise ValueError("FP32 and INT8 paths must differ")

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
    fp32_session = make_session(fp32_path, args.threads)
    int8_session = make_session(int8_path, args.threads)
    cases = list(encode_rows(df, tokenizer))
    quality_batches = [
        make_batch(cases[i : i + args.batch_size], tokenizer.pad_token_id)
        for i in range(0, len(cases), args.batch_size)
    ]

    def torch_forward(batch):
        with torch.inference_mode():
            return model(
                **{k: torch.from_numpy(v) for k, v in batch.items()}
            ).logits.numpy()

    def fp32_forward(batch):
        return fp32_session.run(["logits"], batch)[0]

    def int8_forward(batch):
        return int8_session.run(["logits"], batch)[0]

    print(f"Evaluating {len(df)} real issues across three runtimes...", flush=True)
    torch_logits = np.concatenate([torch_forward(batch) for batch in quality_batches])
    fp32_logits = np.concatenate([fp32_forward(batch) for batch in quality_batches])
    int8_logits = np.concatenate([int8_forward(batch) for batch in quality_batches])
    truth = df["labels"].tolist()
    ids = [str(value) for value in df["id"]]
    fp32_parity = evaluate_logits(
        torch_logits,
        fp32_logits,
        truth,
        atol=args.atol,
        rtol=args.rtol,
        issue_ids=ids,
    )
    int8_parity = evaluate_logits(
        fp32_logits,
        int8_logits,
        truth,
        atol=args.atol,
        rtol=args.rtol,
        issue_ids=ids,
    )
    fp32_prob = probabilities(fp32_logits)
    int8_prob = probabilities(int8_logits)
    int8_parity["max_absolute_probability_difference"] = float(
        np.max(np.abs(fp32_prob - int8_prob))
    )
    int8_parity["mean_absolute_probability_difference"] = float(
        np.mean(np.abs(fp32_prob - int8_prob))
    )
    print(
        f"FP32 vs PyTorch: agreement={fp32_parity['prediction_agreement']:.2%}, "
        f"close={fp32_parity['close_logits_fraction']:.2%}",
        flush=True,
    )
    print(
        f"INT8 vs FP32: agreement={int8_parity['prediction_agreement']:.2%}, "
        f"changed={len(int8_parity['disagreements'])}, "
        f"max_abs_logit={int8_parity['max_absolute_logit_difference']:.6f}",
        flush=True,
    )

    # Use a deterministic subset of the quality sample for timing, avoiding another
    # dataset read and keeping the quality and timing populations related.
    timing_df = sample_frame(df, args.benchmark_sample_size, args.seed)
    by_id = {
        str(issue_id): case for issue_id, case in zip(df["id"], cases, strict=True)
    }
    timing_cases = [by_id[str(issue_id)] for issue_id in timing_df["id"]]
    timing_batches = [
        make_batch(timing_cases[i : i + args.batch_size], tokenizer.pad_token_id)
        for i in range(0, len(timing_cases), args.batch_size)
    ]
    print(
        f"Timing {len(timing_df)} issues: {len(timing_batches)} batches/round; "
        f"warmup={args.warmup}, repeats={args.repeats}",
        flush=True,
    )
    timings = {
        "pytorch": timing(
            torch_forward, timing_batches, args.warmup, args.repeats, label="PyTorch"
        ),
        "onnx_fp32": timing(
            fp32_forward, timing_batches, args.warmup, args.repeats, label="ONNX FP32"
        ),
        "onnx_int8": timing(
            int8_forward, timing_batches, args.warmup, args.repeats, label="ONNX INT8"
        ),
    }
    fp32_size = fp32_path.stat().st_size / 1024**2
    int8_size = int8_path.stat().st_size / 1024**2
    sample_digest = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "dataset": str(args.data),
        "sample_size": len(df),
        "seed": args.seed,
        "sample_issue_ids": ids,
        "sample_ids_sha256": sample_digest,
        "class_counts": {
            str(k): int(v) for k, v in df["labels"].value_counts().sort_index().items()
        },
        "null_bodies_in_sample": int((df["body"] == "").sum()),
        "preprocessing": "E4 structured body only",
        "max_length": 256,
        "title_budget": 48,
        "checkpoint": str(checkpoint),
        "onnx_fp32_model": str(fp32_path),
        "onnx_int8_model": str(int8_path),
        "model_sizes_mib": {
            "onnx_fp32": fp32_size,
            "onnx_int8": int8_size,
            "size_reduction_percent": 100 * (1 - int8_size / fp32_size),
        },
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
        "parity": {"pytorch_vs_fp32": fp32_parity, "fp32_vs_int8": int8_parity},
        "quality": {
            "pytorch": classification_metrics(torch_logits, truth),
            "onnx_fp32": classification_metrics(fp32_logits, truth),
            "onnx_int8": classification_metrics(int8_logits, truth),
        },
        "benchmark": {
            "device": "cpu",
            "threads": args.threads,
            "batch_size": args.batch_size,
            "warmup_rounds": args.warmup,
            "measurement_rounds": args.repeats,
            "timing_sample_size": len(timing_df),
            "timing_issue_ids": [str(x) for x in timing_df["id"]],
            "excludes": ["model_load", "preprocessing", "tokenization"],
            "measurements": timings,
            "int8_speedup_vs_fp32": (
                timings["onnx_int8"]["throughput_issues_per_second"]
                / timings["onnx_fp32"]["throughput_issues_per_second"]
            ),
        },
    }
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for name, values in timings.items():
        print(f"{name}: {values}")
    for name, values in report["quality"].items():
        print(
            f"{name}: macro_f1={values['macro_f1']:.6f}, micro_f1={values['micro_f1']:.6f}"
        )
    print(f"Report: {output}")
    if (
        fp32_parity["prediction_agreement"] != 1.0
        or fp32_parity["close_logits_fraction"] != 1.0
    ):
        raise AssertionError("PyTorch/FP32 regression detected; inspect saved report")
    print(
        "PASS: FP32 parity verified; INT8 quality and performance reported (not gated)"
    )


def sample_frame(df, count, seed):
    if count > len(df) or count < 1:
        raise ValueError("Invalid benchmark sample size")
    counts = df["labels"].value_counts().sort_index()
    exact = counts / len(df) * count
    quotas = np.floor(exact).astype(int)
    remainder = count - int(quotas.sum())
    for label in (
        (exact - quotas).sort_values(ascending=False, kind="stable").index[:remainder]
    ):
        quotas.loc[label] += 1
    pieces = [
        df.loc[df["labels"] == label].sample(n=int(n), random_state=seed)
        for label, n in quotas.items()
        if n
    ]
    return pd.concat(pieces).sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    run(parse_args())
