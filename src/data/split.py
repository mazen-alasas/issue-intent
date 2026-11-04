import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.prepare import make_text_hash

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
    return pd.read_csv(
        input_path,
        usecols=["labels"],
        keep_default_na=False,
    )["labels"]


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
                pd.read_csv(
                    input_path, chunksize=chunk_size, keep_default_na=False, dtype=str
                )
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


def compute_file_sha256(
    file_path: Path,
    block_size: int = 1024 * 1024,
) -> str:
    """Compute the SHA-256 digest of a file in binary chunks.

    Args:
        file_path (Path): Path to the file being hashed.
        block_size (int): Number of bytes read per iteration.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.

    Raises:
        ValueError: If block_size is not positive.
        OSError: If the file cannot be read.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")

    hasher = hashlib.sha256()

    with file_path.open("rb") as file:
        while block := file.read(block_size):
            hasher.update(block)

    return hasher.hexdigest()


def count_cross_split_text_overlap(
    train_path: Path,
    validation_path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """Count validation rows whose text also appears in training.

    Compare exact title-body pairs using the same SHA-256 text
    hashing logic as the data preparation stage.

    Args:
        train_path (Path): Path to the training CSV.
        validation_path (Path): Path to the validation CSV.
        chunk_size (int): Number of rows read per chunk.

    Returns:
        int: Number of validation rows with matching training text.

    Raises:
        ValueError: If chunk_size is not positive.
        OSError: If either CSV cannot be read.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    train_hashes = set()

    for chunk in pd.read_csv(
        train_path,
        usecols=["title", "body"],
        chunksize=chunk_size,
        keep_default_na=False,
    ):
        for title, body in chunk.itertuples(index=False, name=None):
            train_hashes.add(make_text_hash(title, body))

    overlap_count = 0

    for chunk in pd.read_csv(
        validation_path,
        usecols=["title", "body"],
        chunksize=chunk_size,
        keep_default_na=False,
    ):
        for title, body in chunk.itertuples(index=False, name=None):
            if make_text_hash(title, body) in train_hashes:
                overlap_count += 1

    return overlap_count


def compute_class_distribution(
    csv_path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> dict[str, dict[str, float | int]]:
    """Compute class counts and proportions from a CSV file.

    Read only the labels column in chunks to avoid loading
    the complete dataset into memory.

    Args:
        csv_path (Path): Path to the dataset CSV.
        chunk_size (int): Number of rows read per chunk.

    Returns:
        dict[str, dict[str, float | int]]: Per-class counts and
            proportions relative to the total number of rows.

    Raises:
        ValueError: If chunk_size is not positive.
        OSError: If the CSV cannot be read.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    counts: dict[str, int] = {}
    total_rows = 0

    for chunk in pd.read_csv(
        csv_path, usecols=["labels"], chunksize=chunk_size, keep_default_na=False
    ):
        chunk_counts = chunk["labels"].value_counts()

        for label, count in chunk_counts.items():
            label = str(label)
            counts[label] = counts.get(label, 0) + int(count)

        total_rows += len(chunk)

    if total_rows == 0:
        return {}

    return {
        label: {
            "count": count,
            "proportion": count / total_rows,
        }
        for label, count in sorted(counts.items())
    }


def build_split_report(
    source_path: Path,
    train_path: Path,
    validation_path: Path,
    source_rows: int,
    train_rows: int,
    validation_rows: int,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Build and validate the train-validation split report.

    Verify row counts, class distributions, cross-split text overlap,
    and complete source-row preservation before computing file hashes.

    Args:
        source_path (Path): Prepared source dataset.
        train_path (Path): Training CSV path.
        validation_path (Path): Validation CSV path.
        source_rows (int): Expected source row count.
        train_rows (int): Expected training row count.
        validation_rows (int): Expected validation row count.
        validation_size (float): Requested validation fraction.
        random_state (int): Random seed used for splitting.
        chunk_size (int): CSV rows processed per chunk.

    Returns:
        dict: Validated split metadata and integrity results.

    Raises:
        ValueError: If row counts or text integrity checks fail.
        OSError: If input files cannot be read.
    """
    source_distribution = compute_class_distribution(source_path, chunk_size)
    train_distribution = compute_class_distribution(train_path, chunk_size)
    validation_distribution = compute_class_distribution(validation_path, chunk_size)

    def distribution_total(distribution: dict) -> int:
        """Sum row counts across all classes.

        Args:
            distribution (dict): Per-class distribution.

        Returns:
            int: Total number of labeled rows.
        """
        return sum(item["count"] for item in distribution.values())

    actual_source = distribution_total(source_distribution)
    actual_train = distribution_total(train_distribution)
    actual_validation = distribution_total(validation_distribution)

    row_counts_match = (
        actual_source == source_rows
        and actual_train == train_rows
        and actual_validation == validation_rows
        and actual_train + actual_validation == actual_source
    )

    if not row_counts_match:
        raise ValueError("Split row counts do not match")

    overlap = count_cross_split_text_overlap(
        train_path,
        validation_path,
        chunk_size,
    )

    if overlap != 0:
        raise ValueError(f"Cross-split text overlap detected: {overlap}")

    all_labels = (
        set(source_distribution)
        | set(train_distribution)
        | set(validation_distribution)
    )

    for label in all_labels:
        source_count = source_distribution.get(label, {}).get("count", 0)
        train_count = train_distribution.get(label, {}).get("count", 0)
        validation_count = validation_distribution.get(label, {}).get("count", 0)

        if source_count != train_count + validation_count:
            raise ValueError(f"Class count mismatch for label: {label}")

    validate_source_row_integrity(
        source_path=source_path,
        train_path=train_path,
        validation_path=validation_path,
        chunk_size=chunk_size,
    )

    return {
        "split_config": {
            "validation_size": validation_size,
            "random_state": random_state,
            "stratified": True,
        },
        "row_counts": {
            "source": actual_source,
            "train": actual_train,
            "validation": actual_validation,
        },
        "integrity": {
            "row_counts_match": True,
            "cross_split_text_overlap": overlap,
            "source_rows_preserved": True,
        },
        "sha256": {
            "source": compute_file_sha256(source_path),
            "train": compute_file_sha256(train_path),
            "validation": compute_file_sha256(validation_path),
        },
        "class_distribution": {
            "source": source_distribution,
            "train": train_distribution,
            "validation": validation_distribution,
        },
    }


def save_split_report(
    report: dict,
    report_path: Path,
) -> None:
    """Write a split report atomically using a temporary file.

    Args:
        report (dict): Validated split metadata.
        report_path (Path): Destination JSON report path.

    Raises:
        ValueError: If row counts, class distributions, or row
            integrity checks fail.
        OSError: If input files cannot be read.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=report_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / "split_report.json"

        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
            file.write("\n")

        os.replace(temp_path, report_path)


def run_split_pipeline(
    input_path: Path = INPUT_PATH,
    train_path: Path = TRAIN_PATH,
    validation_path: Path = VALIDATION_PATH,
    report_path: Path = REPORT_PATH,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run the reproducible train-validation splitting pipeline.

    Generate stratified splits, validate their integrity, and
    publish a JSON report only after successful validation.

    Args:
        input_path (Path): Prepared source CSV path.
        train_path (Path): Training CSV output path.
        validation_path (Path): Validation CSV output path.
        report_path (Path): JSON report output path.
        validation_size (float): Fraction assigned to validation.
        random_state (int): Random seed for reproducibility.
        chunk_size (int): Number of CSV rows processed per chunk.

    Returns:
        dict: Validated split report.

    Raises:
        ValueError: If splitting or integrity validation fails.
        OSError: If file operations fail.
    """
    paths = [
        input_path.resolve(),
        train_path.resolve(),
        validation_path.resolve(),
        report_path.resolve(),
    ]

    if len(set(paths)) != len(paths):
        raise ValueError("Input and output paths must be distinct")

    if train_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("Train and validation files must share an output directory")

    labels = load_labels(input_path)

    train_indices, validation_indices = create_split_indices(
        labels,
        validation_size=validation_size,
        random_state=random_state,
    )

    # Invalidate previous success metadata before writing new outputs.
    report_path.unlink(missing_ok=True)

    counts = write_split_files(
        input_path=input_path,
        train_path=train_path,
        validation_path=validation_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
        chunk_size=chunk_size,
    )

    report = build_split_report(
        source_path=input_path,
        train_path=train_path,
        validation_path=validation_path,
        source_rows=len(labels),
        train_rows=counts["train_rows"],
        validation_rows=counts["validation_rows"],
        validation_size=validation_size,
        random_state=random_state,
        chunk_size=chunk_size,
    )

    save_split_report(report, report_path)

    return report


def compute_row_fingerprints(
    csv_path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> Counter[bytes]:
    """Count SHA-256 fingerprints of complete CSV rows.

    Preserve duplicate row multiplicities and use a canonical
    serialization for deterministic row hashing.

    Args:
        csv_path (Path): CSV file to inspect.
        chunk_size (int): Number of rows read per chunk.

    Returns:
        Counter[bytes]: Frequency of each complete-row digest.

    Raises:
        ValueError: If chunk_size is not positive.
        OSError: If the CSV cannot be read.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    fingerprints: Counter[bytes] = Counter()

    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        keep_default_na=False,
        dtype=str,
    ):
        for row in chunk.itertuples(index=False, name=None):
            payload = json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")

            digest = hashlib.sha256(payload).digest()
            fingerprints[digest] += 1

    return fingerprints


def validate_source_row_integrity(
    source_path: Path,
    train_path: Path,
    validation_path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    """Verify that splits preserve the complete source row multiset.

    Args:
        source_path (Path): Original prepared dataset.
        train_path (Path): Training split CSV.
        validation_path (Path): Validation split CSV.
        chunk_size (int): Number of rows processed per chunk.

    Raises:
        ValueError: If source and split row contents differ.
    """
    source_fingerprints = compute_row_fingerprints(
        source_path,
        chunk_size,
    )

    train_fingerprints = compute_row_fingerprints(
        train_path,
        chunk_size,
    )

    validation_fingerprints = compute_row_fingerprints(
        validation_path,
        chunk_size,
    )

    combined_fingerprints = train_fingerprints + validation_fingerprints

    if source_fingerprints != combined_fingerprints:
        raise ValueError(
            "Source row integrity check failed: split rows differ from source rows"
        )


def main() -> None:
    """Generate and validate the train-validation split."""
    report = run_split_pipeline()

    print("Train-validation split completed successfully.")
    print(f"Source rows: {report['row_counts']['source']:,}")
    print(f"Train rows: {report['row_counts']['train']:,}")
    print(f"Validation rows: {report['row_counts']['validation']:,}")
    print(
        "Cross-split text overlap:",
        report["integrity"]["cross_split_text_overlap"],
    )


if __name__ == "__main__":
    main()
