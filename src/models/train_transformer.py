import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from src.data.neural_dataset import LABEL_TO_ID
from src.data.transformer_dataset import TransformerInputConfig
from src.data.transformer_streaming import TransformerIssueDataset

MODEL_NAME = "distilbert/distilbert-base-uncased"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/processed/structured_v1/train.parquet"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/processed/structured_v1/validation.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/transformer/distilbert"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last completed epoch.",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--title-budget", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
    )
    parser.add_argument("--smoke-test", action="store_true")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_rows(path):
    return pq.ParquetFile(path).metadata.num_rows


def make_loader(path, tokenizer, config, batch_size, shuffle, seed):
    dataset = TransformerIssueDataset(
        path=path,
        tokenizer=tokenizer,
        config=config,
        shuffle=shuffle,
        seed=seed,
    )

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collator,
    )

    return dataset, loader


def save_run_config(
    output_dir,
    args,
    device,
    status,
    completed_epochs,
    best_macro_f1,
    error=None,
):
    training_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    metadata = {
        "model": MODEL_NAME,
        "device": str(device),
        "smoke_test": args.smoke_test,
        "status": status,
        "completed_epochs": completed_epochs,
        "best_macro_f1": best_macro_f1,
        "training_config": training_config,
    }

    if error is not None:
        metadata["error"] = error

    (output_dir / "run_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, loader, device, max_batches=None):
    model.eval()

    predictions = []
    targets = []
    total_loss = 0.0
    total_rows = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = move_batch(batch, device)

            outputs = model(**batch)
            batch_size = batch["labels"].size(0)

            total_loss += outputs.loss.item() * batch_size
            total_rows += batch_size

            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            targets.extend(batch["labels"].cpu().tolist())

    if total_rows == 0:
        raise RuntimeError("Validation produced no examples")

    label_ids = list(range(len(LABEL_TO_ID)))

    return {
        "loss": total_loss / total_rows,
        "micro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=label_ids,
                average="micro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=label_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                targets,
                predictions,
                labels=label_ids,
                average="weighted",
                zero_division=0,
            )
        ),
        "rows": total_rows,
    }


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    grad_accum,
    max_batches,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_rows = 0
    pending = 0
    optimizer_steps = 0

    def optimizer_step():
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        # A decrease means GradScaler detected invalid gradients
        # and skipped the optimizer update.
        did_update = not scaler.is_enabled() or scale_after >= scale_before

        if did_update:
            scheduler.step()

        optimizer.zero_grad(set_to_none=True)
        return did_update

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        batch = move_batch(batch, device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model(**batch)
            loss = outputs.loss

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_rows += batch_size

        scaler.scale(loss / grad_accum).backward()
        pending += 1

        if pending == grad_accum:
            if optimizer_step():
                optimizer_steps += 1
            pending = 0

    if pending:
        # Correct the scale of the incomplete final accumulation.
        correction = grad_accum / pending

        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)

        if optimizer_step():
            optimizer_steps += 1

    if total_rows == 0:
        raise RuntimeError("Training produced no examples")

    return {
        "loss": total_loss / total_rows,
        "rows": total_rows,
        "optimizer_steps": optimizer_steps,
    }


def save_training_state(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_macro_f1,
    stale_epochs,
    history,
    total_steps,
    steps_per_epoch,
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_macro_f1": best_macro_f1,
        "stale_epochs": stale_epochs,
        "history": history,
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }

    temporary_path = path.with_name(path.name + ".tmp")

    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_training_state(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    total_steps,
    steps_per_epoch,
):
    state = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if (
        state["total_steps"] != total_steps
        or state["steps_per_epoch"] != steps_per_epoch
    ):
        raise ValueError("Resume requires the original training schedule")

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])

    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch_rng = state["torch_rng"]

    if not isinstance(torch_rng, torch.Tensor):
        torch_rng = torch.as_tensor(torch_rng, dtype=torch.uint8)

    torch.set_rng_state(torch_rng.to(device="cpu", dtype=torch.uint8))

    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        cuda_rng_states = [
            torch.as_tensor(rng, dtype=torch.uint8, device="cpu")
            for rng in state["cuda_rng"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_states)

    return {
        "start_epoch": state["epoch"],
        "best_macro_f1": state["best_macro_f1"],
        "stale_epochs": state["stale_epochs"],
        "history": state["history"],
    }


def main():
    args = parse_args()

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.grad_accum < 1
        or args.lr <= 0
        or args.patience < 1
        or not 0 <= args.warmup_ratio < 1
    ):
        raise ValueError("Invalid training configuration")

    for limit in (args.max_train_batches, args.max_val_batches):
        if limit is not None and limit < 1:
            raise ValueError("Batch limits must be positive")

    if args.stop_after_epoch is not None and not (
        1 <= args.stop_after_epoch <= args.epochs
    ):
        raise ValueError("Invalid stop-after-epoch value")

    if not args.smoke_test and args.max_val_batches is not None:
        raise ValueError("Partial validation is allowed only in smoke-test mode")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.smoke_test and device.type != "cuda":
        raise RuntimeError("Full fine-tuning requires CUDA in this pipeline")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    config = TransformerInputConfig(
        max_length=args.max_length,
        title_budget=args.title_budget,
    )

    train_dataset, train_loader = make_loader(
        args.train,
        tokenizer,
        config,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )

    _, val_loader = make_loader(
        args.validation,
        tokenizer,
        config,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    train_rows = count_rows(args.train)
    total_batches = math.ceil(train_rows / args.batch_size)

    if args.max_train_batches is not None:
        total_batches = min(
            total_batches,
            args.max_train_batches,
        )

    steps_per_epoch = math.ceil(total_batches / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    id_to_label = {index: label for label, index in LABEL_TO_ID.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_TO_ID),
        id2label=id_to_label,
        label2id=LABEL_TO_ID,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    output_dir = args.output_dir

    if args.smoke_test:
        output_dir = output_dir / "smoke"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = []
    best_macro_f1 = -1.0
    stale_epochs = 0
    start_epoch = 0

    checkpoint_path = output_dir / "last.pt"

    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

        state = load_training_state(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            total_steps=total_steps,
            steps_per_epoch=steps_per_epoch,
        )

        start_epoch = state["start_epoch"]
        best_macro_f1 = state["best_macro_f1"]
        stale_epochs = state["stale_epochs"]
        history = state["history"]

        if start_epoch >= args.epochs:
            raise ValueError("All configured epochs are completed")

        if stale_epochs >= args.patience:
            raise ValueError("Early stopping already triggered")

        print(
            f"Resuming from epoch {start_epoch + 1}",
            flush=True,
        )

    status = "running"
    completed_epochs = len(history)

    save_run_config(
        output_dir=output_dir,
        args=args,
        device=device,
        status=status,
        completed_epochs=completed_epochs,
        best_macro_f1=best_macro_f1,
    )

    try:
        for epoch in range(start_epoch, args.epochs):
            train_dataset.set_epoch(epoch)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            epoch_start = time.perf_counter()

            # -------------------------
            # Training
            # -------------------------

            train_metrics = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                grad_accum=args.grad_accum,
                max_batches=args.max_train_batches,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            training_seconds = time.perf_counter() - epoch_start

            # -------------------------
            # Validation
            # -------------------------

            validation_start = time.perf_counter()

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                max_batches=args.max_val_batches,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            validation_seconds = time.perf_counter() - validation_start

            epoch_seconds = time.perf_counter() - epoch_start

            peak_vram_gib = (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else None
            )

            record = {
                "epoch": epoch + 1,
                "train": train_metrics,
                "validation": val_metrics,
                "performance": {
                    "training_seconds": round(training_seconds, 2),
                    "validation_seconds": round(validation_seconds, 2),
                    "epoch_seconds": round(epoch_seconds, 2),
                    "peak_vram_gib": (
                        round(peak_vram_gib, 3) if peak_vram_gib is not None else None
                    ),
                },
            }

            history.append(record)

            print(
                json.dumps(record, indent=2),
                flush=True,
            )

            # -------------------------
            # Best model checkpoint
            # -------------------------

            if val_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = val_metrics["macro_f1"]
                stale_epochs = 0

                checkpoint_dir = output_dir / "best"

                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
            else:
                stale_epochs += 1

            # -------------------------
            # Training history
            # -------------------------

            (output_dir / "history.json").write_text(
                json.dumps(history, indent=2) + "\n",
                encoding="utf-8",
            )

            # -------------------------
            # Resume checkpoint
            # -------------------------

            save_training_state(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                best_macro_f1=best_macro_f1,
                stale_epochs=stale_epochs,
                history=history,
                total_steps=total_steps,
                steps_per_epoch=steps_per_epoch,
            )

            completed_epochs = epoch + 1

            if (
                args.stop_after_epoch is not None
                and completed_epochs >= args.stop_after_epoch
            ):
                if completed_epochs < args.epochs:
                    status = "stopped"
                break

            if stale_epochs >= args.patience and completed_epochs < args.epochs:
                status = "early_stopped"
                break

            save_run_config(
                output_dir=output_dir,
                args=args,
                device=device,
                status=status,
                completed_epochs=completed_epochs,
                best_macro_f1=best_macro_f1,
            )

        if status == "running":
            status = "completed"

        save_run_config(
            output_dir=output_dir,
            args=args,
            device=device,
            status=status,
            completed_epochs=completed_epochs,
            best_macro_f1=best_macro_f1,
        )

    except Exception as exc:
        save_run_config(
            output_dir=output_dir,
            args=args,
            device=device,
            status="failed",
            completed_epochs=completed_epochs,
            best_macro_f1=best_macro_f1,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    print(
        f"Training run {status}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
