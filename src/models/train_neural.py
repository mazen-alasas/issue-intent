"""Train TextCNN without loading the full corpus into memory."""

import argparse
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.neural_dataset import (
    LABELS,
    ParquetIssueDataset,
    build_vocab,
    collate_issues,
    save_vocab,
)
from src.models.textcnn import TextCNN
from src.models.train_classical import ROOT

DATA_DIR = ROOT / "data/processed/structured_v1"
OUTPUT_DIR = ROOT / "data/processed/neural_textcnn"


def evaluate(model, loader, device):
    model.eval()
    truth, predicted = [], []
    total_loss, total_rows = 0.0, 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.inference_mode():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            total_loss += loss_fn(logits, labels).item()
            total_rows += len(labels)
            truth.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    if not total_rows:
        raise ValueError("Empty validation dataset")
    return {
        "loss": total_loss / total_rows,
        "micro_f1": float(
            f1_score(
                truth,
                predicted,
                average="micro",
                labels=list(range(4)),
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                truth,
                predicted,
                average="macro",
                labels=list(range(4)),
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                truth,
                predicted,
                average="weighted",
                labels=list(range(4)),
                zero_division=0,
            )
        ),
        "per_class": classification_report(
            truth,
            predicted,
            labels=list(range(4)),
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    train_path = DATA_DIR / "train.parquet"
    validation_path = DATA_DIR / "validation.parquet"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = OUTPUT_DIR / "vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    else:
        print("Building vocabulary from training split...", flush=True)
        vocab = build_vocab(train_path, max_size=args.vocab_size)
        save_vocab(vocab, vocab_path)
    print(f"Vocabulary={len(vocab)} device={device}", flush=True)
    train_set = ParquetIssueDataset(
        train_path,
        vocab,
        args.max_length,
        args.title_budget,
        shuffle=True,
        seed=args.seed,
    )
    val_set = ParquetIssueDataset(
        validation_path, vocab, args.max_length, args.title_budget
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, collate_fn=collate_issues, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, collate_fn=collate_issues, num_workers=0
    )
    model = TextCNN(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_macro = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_set.set_epoch(epoch)
        model.train()
        loss_sum, rows = 0.0, 0
        for step, (features, labels) in enumerate(train_loader, start=1):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            rows += len(labels)
            loss_sum += loss.item() * len(labels)
            if args.max_train_batches and step >= args.max_train_batches:
                break
        metrics = evaluate(model, val_loader, device)
        entry = {
            "epoch": epoch,
            "train_loss": loss_sum / rows,
            "train_rows": rows,
            "validation": metrics,
            "seconds": round(time.perf_counter() - start, 2),
        }
        history.append(entry)
        print(
            f"Epoch {epoch}: train_loss={entry['train_loss']:.4f} "
            f"val_micro={metrics['micro_f1']:.4f} val_macro={metrics['macro_f1']:.4f} "
            f"seconds={entry['seconds']}",
            flush=True,
        )
        if metrics["macro_f1"] > best_macro:
            best_macro = metrics["macro_f1"]
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "vocab_size": len(vocab),
                    "labels": list(LABELS),
                    "max_length": args.max_length,
                    "title_budget": args.title_budget,
                },
                OUTPUT_DIR / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping", flush=True)
                break
    report = {
        "config": vars(args),
        "device": str(device),
        "vocab_size": len(vocab),
        "best_validation_macro_f1": best_macro,
        "history": history,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--title-budget", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="Debug only; never use for final metrics",
    )
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.epochs < 1
        or args.max_length < 5
        or args.patience < 1
    ):
        parser.error("Invalid training configuration")
    train(args)


if __name__ == "__main__":
    main()
