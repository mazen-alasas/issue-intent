import os
import tempfile
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


def write_split_files(
    input_path: Path,
    train_path: Path,
    validation_path: Path,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, int]:
    """Write train and validation CSV files using temporary outputs.

    Process the prepared dataset in chunks, preserving source row
    order within each split. Validate row counts before publishing
    the completed CSV files.

    Args:
        input_path (Path): Path to the prepared input CSV.
        train_path (Path): Destination for training rows.
        validation_path (Path): Destination for validation rows.
        train_indices (np.ndarray): Training row positions.
        validation_indices (np.ndarray): Validation row positions.
        chunk_size (int): Number of rows processed per chunk.

    Returns:
        dict[str, int]: Number of rows written to each split.

    Raises:
        ValueError: If indices are invalid or row counts disagree.
        OSError: If file operations fail.

    Side Effects:
        Creates or replaces the destination CSV files after
        successful processing.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)

    all_indices = np.concatenate((train_indices, validation_indices))
    total_rows = len(all_indices)

    if (
        np.any(all_indices < 0)
        or np.any(all_indices >= total_rows)
        or len(np.unique(all_indices)) != total_rows
    ):
        raise ValueError("Split indices must form a complete partition")

    train_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.parent.mkdir(parents=True, exist_ok=True)

    train_membership = np.zeros(total_rows, dtype=bool)
    train_membership[train_indices] = True

    written_train = 0
    written_validation = 0
    offset = 0

    with tempfile.TemporaryDirectory(dir=train_path.parent) as temp_dir:
        temp_dir = Path(temp_dir)
        temp_train = temp_dir / "train.csv"
        temp_validation = temp_dir / "validation.csv"

        with (
            temp_train.open("w", encoding="utf-8", newline="") as train_file,
            temp_validation.open("w", encoding="utf-8", newline="") as validation_file,
        ):
            for chunk_number, chunk in enumerate(
                pd.read_csv(input_path, chunksize=chunk_size)
            ):
                end = offset + len(chunk)

                if end > total_rows:
                    raise ValueError("Input row count exceeds split indices")

                mask = train_membership[offset:end]

                train_chunk = chunk.iloc[np.flatnonzero(mask)]
                validation_chunk = chunk.iloc[np.flatnonzero(~mask)]

                train_chunk.to_csv(
                    train_file,
                    index=False,
                    header=chunk_number == 0,
                )
                validation_chunk.to_csv(
                    validation_file,
                    index=False,
                    header=chunk_number == 0,
                )

                written_train += len(train_chunk)
                written_validation += len(validation_chunk)
                offset = end

        if offset != total_rows:
            raise ValueError("Input row count does not match split indices")

        os.replace(temp_train, train_path)
        os.replace(temp_validation, validation_path)

    return {
        "train_rows": written_train,
        "validation_rows": written_validation,
    }
