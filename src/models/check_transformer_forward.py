from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src.data.transformer_dataset import TransformerInputConfig
from src.data.transformer_streaming import TransformerIssueDataset


def main():
    model_name = "distilbert/distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    dataset = TransformerIssueDataset(
        path=Path("data/processed/structured_v1/train.parquet"),
        tokenizer=tokenizer,
        config=TransformerInputConfig(
            max_length=256,
            title_budget=48,
        ),
        shuffle=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
        collate_fn=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        ),
    )

    batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
    ).to(device)

    batch = {key: value.to(device) for key, value in batch.items()}

    model.eval()

    with torch.inference_mode():
        outputs = model(**batch)

    assert outputs.logits.shape == (4, 4)
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)

    print("Device:", device)
    print("Input shape:", tuple(batch["input_ids"].shape))
    print("Logits shape:", tuple(outputs.logits.shape))
    print("Loss:", outputs.loss.item())
    print("Transformer forward pass successful.")


if __name__ == "__main__":
    main()
