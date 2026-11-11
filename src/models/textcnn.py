"""Kim-style multi-kernel TextCNN classifier."""

import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 4,
        embedding_dim: int = 128,
        channels: int = 100,
        kernels: tuple[int, ...] = (3, 4, 5),
        dropout: float = 0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels, kernel_size=k) for k in kernels
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * len(kernels), num_classes)
        self.min_length = max(kernels)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.size(1) < self.min_length:
            raise ValueError("Input must be [batch, length] with sufficient length")
        embedded = self.embedding(tokens).transpose(1, 2)
        pooled = [torch.relu(conv(embedded)).amax(dim=2) for conv in self.convs]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))
