from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_stratified_sample(
    csv_path: Path,
    sample_size: int = 50_000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Load a reproducible stratified sample from a CSV dataset.

    Args:
        csv_path (Path): Path to the training CSV file.
        sample_size (int): Number of rows to sample.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Stratified sample containing titles and labels.

    Raises:
        ValueError: If the sample size is outside the valid range.
    """
    data = pd.read_csv(
        csv_path,
        usecols=["title", "labels"],
        dtype=str,
        keep_default_na=False,
    )

    if not 0 < sample_size <= len(data):
        raise ValueError("sample_size must be between 1 and the dataset size.")

    if sample_size == len(data):
        return data.reset_index(drop=True)

    sample, _ = train_test_split(
        data,
        train_size=sample_size,
        stratify=data["labels"],
        random_state=random_state,
    )

    return sample.reset_index(drop=True)
