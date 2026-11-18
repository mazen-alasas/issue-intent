import argparse
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoTokenizer

from src.data.labels import LABELS
from src.data.transformer_dataset import TransformerInputConfig, encode_issue
from src.preprocessing.experiment_configs import EXPERIMENTS
from src.preprocessing.structured import preprocess_structured


def parse_args():
    root = Path.home() / "issue-intent-artifacts/models/distilbert"
    parser = argparse.ArgumentParser(
        description="Evaluate ONNX FP32 and INT8 on official test issues"
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/raw/nlbse23-test.parquet")
    )
    parser.add_argument("--checkpoint", type=Path, default=root / "best")
    parser.add_argument("--fp32", type=Path, default=root / "onnx/model_fp32.onnx")
    parser.add_argument("--int8", type=Path, default=root / "onnx/model_int8.onnx")
    parser.add_argument(
        "--output", type=Path, default=root / "onnx/int8_full_evaluation.json"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fp32-provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume-check-batches", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    for name in ("batch_size", "threads", "save_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.resume_check_batches < 1:
        parser.error("--resume-check-batches must be positive")
    if args.max_rows is not None and args.max_rows < 1:
        parser.error("--max-rows must be positive")
    if args.fp32.resolve() == args.int8.resolve():
        parser.error("FP32 and INT8 must be different model files")
    return args


def file_identity(path):
    path = path.expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def configuration(args, total_rows):
    identity = {
        "data": file_identity(args.data),
        "checkpoint_config": file_identity(args.checkpoint / "config.json"),
        "tokenizer": file_identity(args.checkpoint / "tokenizer.json"),
        "fp32": file_identity(args.fp32),
        "int8": file_identity(args.int8),
        "rows": total_rows,
        "batch_size": args.batch_size,
        "preprocessing": "E4 structured body only",
        "max_length": 256,
        "title_budget": 48,
        "labels": list(LABELS),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return identity, digest


def make_session(path, threads, provider="cpu"):
    if provider not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported execution provider: {provider}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    if provider == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA unavailable; cannot initialize ONNX CUDA")
        torch.cuda.init()
        ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime CUDAExecutionProvider unavailable")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(
        str(path.expanduser().resolve()),
        sess_options=options,
        providers=providers,
    )
    if provider == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            "ONNX Runtime silently fell back to CPU; refusing to continue"
        )
    if {item.name for item in session.get_inputs()} != {"input_ids", "attention_mask"}:
        raise ValueError(f"Unexpected model inputs: {path}")
    if [item.name for item in session.get_outputs()] != ["logits"]:
        raise ValueError(f"Unexpected model outputs: {path}")
    return session


def verify_saved_fp32(
    df,
    completed,
    batch_size,
    tokenizer,
    input_config,
    fp32_logits,
    session,
    check_batches,
    *,
    atol=1e-4,
    rtol=1e-4,
):
    if completed == 0:
        return {"checked_batches": 0, "checked_rows": 0, "max_abs_difference": 0.0}
    if completed % batch_size:
        raise ValueError("Resume requires a full-batch checkpoint boundary")
    total_batches = completed // batch_size
    indices = np.unique(
        np.linspace(0, total_batches - 1, min(check_batches, total_batches), dtype=int)
    )
    checked = 0
    max_abs = 0.0
    for batch_index in indices:
        start = int(batch_index) * batch_size
        end = start + batch_size
        feed = make_batch(df.iloc[start:end], tokenizer, input_config)
        actual = session.run(["logits"], feed)[0]
        saved = fp32_logits[start:end]
        if actual.shape != saved.shape or not np.isfinite(actual).all():
            raise RuntimeError(
                f"Resume CUDA validation failed at rows {start}:{end}: invalid logits"
            )
        difference = float(np.max(np.abs(actual - saved)))
        max_abs = max(max_abs, difference)
        if not np.allclose(actual, saved, atol=atol, rtol=rtol):
            raise RuntimeError(
                f"Resume CUDA validation failed at rows {start}:{end}: "
                f"max_abs={difference:.8g}; checkpoint unchanged"
            )
        if not np.array_equal(actual.argmax(axis=1), saved.argmax(axis=1)):
            raise RuntimeError(
                f"Resume CUDA prediction mismatch at rows {start}:{end}; checkpoint unchanged"
            )
        checked += len(saved)
    return {
        "checked_batches": len(indices),
        "checked_rows": checked,
        "max_abs_difference": max_abs,
    }


def make_batch(rows, tokenizer, config):
    encoded = []
    for row in rows.itertuples(index=False):
        body = preprocess_structured(row.body, config=EXPERIMENTS["E4"])
        encoded.append(encode_issue(row.title, body, tokenizer, config))
    length = max(len(item["input_ids"]) for item in encoded)
    ids = np.full((len(encoded), length), tokenizer.pad_token_id, dtype=np.int64)
    mask = np.zeros_like(ids)
    for index, item in enumerate(encoded):
        n = len(item["input_ids"])
        ids[index, :n] = item["input_ids"]
        mask[index, :n] = item["attention_mask"]
    return {"input_ids": ids, "attention_mask": mask}


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def save_progress(path, fingerprint, completed, fp32_logits, int8_logits):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            fingerprint=np.asarray(fingerprint),
            completed=np.asarray(completed, dtype=np.int64),
            fp32=fp32_logits[:completed],
            int8=int8_logits[:completed],
        )
    os.replace(temporary, path)


def restore_progress(path, fingerprint, fp32_logits, int8_logits, batch_size):
    if not path.exists():
        return 0
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["fingerprint"].item()) != fingerprint:
            raise ValueError(
                "Saved progress belongs to different data/models/config; use --restart"
            )
        completed = int(saved["completed"].item())
        if not 0 <= completed <= len(fp32_logits):
            raise ValueError("Invalid progress row count")
        if completed != len(fp32_logits) and completed % batch_size:
            raise ValueError("Progress must end on a batch boundary")
        expected_shape = (completed, len(LABELS))
        if (
            saved["fp32"].shape != expected_shape
            or saved["int8"].shape != expected_shape
        ):
            raise ValueError("Saved logits have unexpected dimensions")
        fp32_logits[:completed] = saved["fp32"]
        int8_logits[:completed] = saved["int8"]
    return completed


def metrics(y_true, y_pred):
    ids = np.arange(len(LABELS))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, zero_division=0
    )
    _, _, micro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, average="micro", zero_division=0
    )
    _, _, macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, average="macro", zero_division=0
    )
    _, _, weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=ids, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(micro),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=ids).tolist(),
    }


def probabilities(logits):
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def run(args):
    started = time.perf_counter()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output.with_name(output.stem + ".progress.npz")
    if args.restart and progress_path.exists():
        progress_path.unlink()

    print("Loading official test dataset...", flush=True)
    df = pd.read_parquet(args.data, columns=["id", "labels", "title", "body"])
    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].copy()
    if df.empty or df["labels"].isna().any() or df["title"].isna().any():
        raise ValueError("Empty dataset or missing labels/titles")
    if not set(df["labels"].unique()).issubset(set(LABELS)):
        raise ValueError("Unknown labels in test dataset")
    df["body"] = df["body"].fillna("")
    if not df["body"].map(lambda x: isinstance(x, str)).all():
        raise ValueError("Non-string body values")
    df = df.reset_index(drop=True)
    n = len(df)
    config, fingerprint = configuration(args, n)
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    truth = np.asarray([label_to_id[label] for label in df["labels"]], dtype=np.int64)
    fp32_logits = np.empty((n, len(LABELS)), dtype=np.float32)
    int8_logits = np.empty_like(fp32_logits)
    completed = restore_progress(
        progress_path, fingerprint, fp32_logits, int8_logits, args.batch_size
    )
    print(f"Rows: {n:,}; resuming at {completed:,}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, use_fast=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer is missing pad_token_id")
    input_config = TransformerInputConfig(max_length=256, title_budget=48)
    fp32_session = make_session(args.fp32, args.threads, args.fp32_provider)
    int8_session = make_session(args.int8, args.threads, "cpu")
    resume_check = None
    if completed and args.fp32_provider == "cuda":
        resume_check = verify_saved_fp32(
            df,
            completed,
            args.batch_size,
            tokenizer,
            input_config,
            fp32_logits,
            fp32_session,
            args.resume_check_batches,
        )
        print(f"CUDA resume validation passed: {resume_check}", flush=True)
    print(
        f"FP32 providers: {fp32_session.get_providers()}; "
        f"INT8 providers: {int8_session.get_providers()}",
        flush=True,
    )
    batches_since_save = 0
    try:
        for start in range(completed, n, args.batch_size):
            end = min(start + args.batch_size, n)
            feed = make_batch(df.iloc[start:end], tokenizer, input_config)
            reference = fp32_session.run(["logits"], feed)[0]
            candidate = int8_session.run(["logits"], feed)[0]
            expected = (end - start, len(LABELS))
            if reference.shape != expected or candidate.shape != expected:
                raise ValueError(f"Unexpected logits shape at rows {start}:{end}")
            if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
                raise ValueError(f"Non-finite logits at rows {start}:{end}")
            fp32_logits[start:end] = reference
            int8_logits[start:end] = candidate
            completed = end
            batches_since_save += 1
            if batches_since_save >= args.save_every or completed == n:
                save_progress(
                    progress_path, fingerprint, completed, fp32_logits, int8_logits
                )
                elapsed = time.perf_counter() - started
                print(
                    f"Completed {completed:,}/{n:,} ({completed / n:.1%}); elapsed {elapsed / 60:.1f} min",
                    flush=True,
                )
                batches_since_save = 0
    except KeyboardInterrupt:
        save_progress(progress_path, fingerprint, completed, fp32_logits, int8_logits)
        print(
            f"Interrupted; saved {completed:,} rows. Re-run the same command to resume.",
            flush=True,
        )
        raise SystemExit(130) from None

    fp32_pred = fp32_logits.argmax(axis=1)
    int8_pred = int8_logits.argmax(axis=1)
    different = np.flatnonzero(fp32_pred != int8_pred)
    fp32_probs = probabilities(fp32_logits)
    int8_probs = probabilities(int8_logits)
    abs_prob_diff = np.abs(fp32_probs - int8_probs)
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "configuration": config,
        "fingerprint_sha256": fingerprint,
        "execution": {
            "fp32_requested_provider": args.fp32_provider,
            "fp32_active_providers": fp32_session.get_providers(),
            "int8_active_providers": int8_session.get_providers(),
            "resume_validation": resume_check,
        },
        "total_issues": n,
        "class_counts": {
            label: int((truth == i).sum()) for i, label in enumerate(LABELS)
        },
        "label_order": list(LABELS),
        "quality": {
            "onnx_fp32": metrics(truth, fp32_pred),
            "onnx_int8": metrics(truth, int8_pred),
        },
        "comparison": {
            "prediction_agreement": float((fp32_pred == int8_pred).mean()),
            "changed_predictions": int(len(different)),
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
            "max_absolute_logit_difference": float(
                np.max(np.abs(fp32_logits - int8_logits))
            ),
            "mean_absolute_probability_difference": float(abs_prob_diff.mean()),
            "max_absolute_probability_difference": float(abs_prob_diff.max()),
            "changed_issues": [
                {
                    "sample_index": int(i),
                    "issue_id": str(df.at[int(i), "id"]),
                    "truth": LABELS[int(truth[i])],
                    "fp32": LABELS[int(fp32_pred[i])],
                    "int8": LABELS[int(int8_pred[i])],
                    "fp32_confidence": float(fp32_probs[i, fp32_pred[i]]),
                    "int8_confidence": float(int8_probs[i, int8_pred[i]]),
                }
                for i in different
            ],
        },
        "evaluation_seconds": float(time.perf_counter() - started),
        "note": (
            "Model quality evaluation only; no timing benchmark. Checkpoint stores "
            "inference logits. Existing CPU FP32 logits may be reused after sampled "
            "CUDA parity verification."
        ),
    }
    atomic_json(output, report)
    print(
        f"FP32: micro={report['quality']['onnx_fp32']['micro_f1']:.6f}, macro={report['quality']['onnx_fp32']['macro_f1']:.6f}"
    )
    print(
        f"INT8: micro={report['quality']['onnx_int8']['micro_f1']:.6f}, macro={report['quality']['onnx_int8']['macro_f1']:.6f}"
    )
    print(
        f"Agreement: {report['comparison']['prediction_agreement']:.4%}; changed={len(different):,}"
    )
    print(f"Report: {output}")
    print(f"Progress checkpoint retained: {progress_path}")


if __name__ == "__main__":
    run(parse_args())
