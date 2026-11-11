import pytest
import torch

from src.models.textcnn import TextCNN


def test_forward_and_backward():
    model = TextCNN(vocab_size=20, embedding_dim=8, channels=4)
    inputs = torch.tensor([[2, 3, 4, 0, 0], [1, 5, 6, 7, 8]])
    logits = model(inputs)
    assert logits.shape == (2, 4)
    logits.sum().backward()
    assert model.embedding.weight.grad is not None


def test_reject_short_sequence():
    with pytest.raises(ValueError):
        TextCNN(10)(torch.ones((2, 4), dtype=torch.long))
