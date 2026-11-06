from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def load_stratified_sample(
    csv_path: Path,
    sample_size: int = 50_000,
    random_state: int = RANDOM_STATE,
    columns: list[str] | None = None,
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
    if columns is None:
        columns = ["title", "labels"]
    if "labels" not in columns:
        raise ValueError("columns must include 'labels'.")

    data = pd.read_csv(
        csv_path,
        usecols=columns,
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


def load_nested_stratified_samples(
    csv_path: Path,
    sample_sizes: list[int],
    random_state: int = RANDOM_STATE,
    columns: list[str] | None = None,
) -> dict[int, pd.DataFrame]:
    """Create reproducible nested stratified training samples.

    Args:
        csv_path (Path): Path to the training dataset.
        sample_sizes (list[int]): Requested sample sizes.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict[int, pd.DataFrame]: Nested samples keyed by size.

    Raises:
        ValueError: If sample sizes are invalid.
    """
    if not sample_sizes or any(size <= 0 for size in sample_sizes):
        raise ValueError("sample_sizes must contain positive integers.")

    sizes = sorted(set(sample_sizes))

    data = pd.read_csv(
        csv_path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
    )

    if sizes[-1] > len(data):
        raise ValueError("Sample size exceeds dataset size.")

    samples = {}
    remaining = data

    for size in reversed(sizes):
        sample, _ = train_test_split(
            remaining,
            train_size=size,
            stratify=remaining["labels"],
            random_state=random_state,
        )

        samples[size] = sample.reset_index(drop=True)
        remaining = sample

    return samples
