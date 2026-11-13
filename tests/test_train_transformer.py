import torch
from torch.utils.data import DataLoader

from src.models.train_transformer import train_epoch


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, input_ids, labels):
        logits = self.linear(input_ids.float())
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return type("Output", (), {"loss": loss, "logits": logits})()


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_gradient_accumulation():
    model = TinyClassifier()

    batches = [
        {
            "input_ids": torch.randn(2, 2),
            "labels": torch.tensor([0, 1]),
        }
        for _ in range(5)
    ]

    loader = DataLoader(batches, batch_size=None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = CountingScheduler()
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    result = train_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=torch.device("cpu"),
        grad_accum=2,
        max_batches=None,
    )

    assert result["rows"] == 10
    assert result["optimizer_steps"] == 3
    assert scheduler.steps == 3


def test_partial_batch_limit():
    model = TinyClassifier()

    batches = [
        {
            "input_ids": torch.randn(2, 2),
            "labels": torch.tensor([0, 1]),
        }
        for _ in range(10)
    ]

    loader = DataLoader(batches, batch_size=None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = CountingScheduler()
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    result = train_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=torch.device("cpu"),
        grad_accum=4,
        max_batches=5,
    )

    assert result["rows"] == 10
    assert result["optimizer_steps"] == 2
    assert scheduler.steps == 2
