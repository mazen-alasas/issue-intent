import random
from pathlib import Path

import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

from src.data.neural_dataset import LABEL_TO_ID
from src.data.transformer_dataset import (
    TransformerInputConfig,
    encode_issue,
)


class TransformerIssueDataset(IterableDataset):
    def __init__(
        self,
        path: Path,
        tokenizer,
        config: TransformerInputConfig,
        shuffle: bool = False,
        seed: int = 42,
        shuffle_buffer_size: int = 4096,
    ):
        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be positive")

        self.path = Path(path)
        self.tokenizer = tokenizer
        self.config = config
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer_size = shuffle_buffer_size
        self.epoch = 0

    def set_epoch(self, epoch: int):
        if epoch < 0:
            raise ValueError("epoch must be non-negative")

        self.epoch = epoch

    def _iter_rows(self, rng, worker):
        parquet = pq.ParquetFile(self.path)
        groups = list(range(parquet.num_row_groups))

        if self.shuffle:
            rng.shuffle(groups)

        if worker is not None:
            groups = groups[worker.id :: worker.num_workers]

        for group in groups:
            data = parquet.read_row_group(
                group,
                columns=["title", "body", "labels"],
            ).to_pydict()

            indices = list(range(len(data["labels"])))

            if self.shuffle:
                rng.shuffle(indices)

            for index in indices:
                yield (
                    data["title"][index],
                    data["body"][index],
                    data["labels"][index],
                )

    def _iter_shuffled_rows(self, rows, rng):
        buffer = []

        for row in rows:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(row)
                continue

            index = rng.randrange(len(buffer))
            selected = buffer[index]
            buffer[index] = row

            yield selected

        rng.shuffle(buffer)
        yield from buffer

    def __iter__(self):
        worker = get_worker_info()

        worker_id = worker.id if worker is not None else 0

        rng = random.Random(self.seed + self.epoch * 1_000_003 + worker_id)

        rows = self._iter_rows(rng, worker)

        if self.shuffle:
            rows = self._iter_shuffled_rows(rows, rng)

        for title, body, label in rows:
            if label not in LABEL_TO_ID:
                raise ValueError(f"Invalid label: {label}")

            encoded = encode_issue(
                title=title,
                body=body,
                tokenizer=self.tokenizer,
                config=self.config,
            )

            encoded["labels"] = LABEL_TO_ID[label]

            yield encoded
