import json
import random
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
LABELS = ("bug", "documentation", "feature", "question")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
PAD, UNK = "<pad>", "<unk>"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(path: Path, max_size: int = 50000, min_freq: int = 2) -> dict[str, int]:
    if max_size < 2 or min_freq < 1:
        raise ValueError("Invalid vocabulary configuration")
    counts = Counter()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=2048, columns=["title", "body"]):
        for title, body in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
        ):
            if not isinstance(title, str) or not isinstance(body, str):
                raise ValueError("Null or non-string text")
            counts.update(tokenize(title))
            counts.update(tokenize(body))
    vocab = {PAD: 0, UNK: 1}
    for token, freq in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if freq < min_freq or len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def encode(
    title: str,
    body: str,
    vocab: dict[str, int],
    max_length: int = 256,
    title_budget: int = 32,
) -> list[int]:
    if max_length < 5 or title_budget < 1:
        raise ValueError("Invalid sequence budget")
    title_tokens = tokenize(title)[: min(title_budget, max_length - 1)]
    body_tokens = tokenize(body)
    available = max_length - len(title_tokens)
    if len(body_tokens) > available:
        head = (available * 3) // 4
        body_tokens = body_tokens[:head] + body_tokens[-(available - head) :]
    tokens = title_tokens + body_tokens
    return [vocab.get(token, 1) for token in tokens] or [1]


class ParquetIssueDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        vocab: dict[str, int],
        max_length: int = 256,
        title_budget: int = 32,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.path = Path(path)
        self.vocab = vocab
        self.max_length = max_length
        self.title_budget = title_budget
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        parquet = pq.ParquetFile(self.path)
        groups = list(range(parquet.num_row_groups))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(groups)
        for group in groups:
            batch = parquet.read_row_group(
                group, columns=["title", "body", "labels"]
            ).to_pydict()
            indices = list(range(len(batch["labels"])))
            if self.shuffle:
                rng.shuffle(indices)
            for i in indices:
                label = batch["labels"][i]
                if label not in LABEL_TO_ID:
                    raise ValueError(f"Invalid label: {label}")
                yield (
                    encode(
                        batch["title"][i],
                        batch["body"][i],
                        self.vocab,
                        self.max_length,
                        self.title_budget,
                    ),
                    LABEL_TO_ID[label],
                )


def collate_issues(batch):
    max_len = max(5, max(len(ids) for ids, _ in batch))
    features = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.tensor([label for _, label in batch], dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        features[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    return features, labels


def save_vocab(vocab: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, ensure_ascii=False) + "\n", encoding="utf-8")
