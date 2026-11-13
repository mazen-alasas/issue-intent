from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from src.data.transformer_dataset import TransformerInputConfig
from src.data.transformer_streaming import TransformerIssueDataset


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased",
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
        batch_size=8,
        num_workers=0,
        collate_fn=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
        ),
    )

    batch = next(iter(loader))

    for name, value in batch.items():
        print(name, tuple(value.shape))

    assert batch["input_ids"].shape[0] == 8
    assert batch["input_ids"].shape[1] <= 256
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["labels"].shape == (8,)

    print("Transformer batch check passed.")


if __name__ == "__main__":
    main()
