from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "data/processed/issues_clean.csv"
TRAIN_PATH = ROOT / "data/processed/train.csv"
VALIDATION_PATH = ROOT / "data/processed/validation.csv"
REPORT_PATH = ROOT / "data/processed/split_report.json"

CHUNK_SIZE = 50_000
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42


def load_labels(input_path: Path) -> pd.Series:
    """Load class labels without loading issue text.

    Args:
        input_path (Path): Path to the prepared training CSV.

    Returns:
        pd.Series: Class labels in their original row order.

    Raises:
        FileNotFoundError: If the input CSV does not exist.
        ValueError: If the labels column is missing.
    """
    return pd.read_csv(input_path, usecols=["labels"])["labels"]


def create_split_indices(
    labels: pd.Series,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible stratified train and validation indices.

    Args:
        labels (pd.Series): Class labels in original dataset order.
        validation_size (float): Fraction assigned to validation.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray]: Training and validation
        row positions relative to the input dataset.

    Raises:
        ValueError: If stratified splitting is not possible.
    """
    indices = np.arange(len(labels))

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=random_state,
        stratify=labels,
    )

    return train_indices, validation_indices
