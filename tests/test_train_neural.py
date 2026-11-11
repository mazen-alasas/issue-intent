import torch
from torch.utils.data import DataLoader

from src.data.neural_dataset import collate_issues
from src.models.textcnn import TextCNN
from src.models.train_neural import evaluate


def test_evaluate_smoke():
    model = TextCNN(20, embedding_dim=8, channels=4)
    samples = [
        ([2, 3, 4, 5, 6], 0),
        ([3, 4, 5, 6, 7], 1),
        ([4, 5, 6, 7, 8], 2),
        ([5, 6, 7, 8, 9], 3),
    ]
    loader = DataLoader(samples, batch_size=2, collate_fn=collate_issues)
    metrics = evaluate(model, loader, torch.device("cpu"))
    assert 0 <= metrics["micro_f1"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert all(
        label in metrics["per_class"]
        for label in ("bug", "documentation", "feature", "question")
    )
