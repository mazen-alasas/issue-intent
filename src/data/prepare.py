import argparse
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data/raw/nlbse23-issue-classification-train.csv"
OUTPUT_PATH = ROOT / "data/processed/issues_clean.csv"
REPORT_PATH = ROOT / "data/processed/preparation_report.json"

CHUNK_SIZE = 50_000
COLUMNS = ["id", "labels", "title", "body", "author_association"]


def make_text_hash(title: str, body: str) -> bytes:
    """Generate a deterministic hash for an issue's title and body.

    The title and body are separated by a null character to prevent
    ambiguous concatenations. The resulting hash is used to identify
    exact-text duplicates and conflicting-label groups.

    Args:
        title (str): Issue title after missing-value normalization.
        body (str): Issue body after missing-value normalization.

    Returns:
        bytes: SHA-256 digest representing the title-body combination.
    """
    text = f"{title}\0{body}"
    return hashlib.sha256(text.encode("utf-8")).digest()


def normalize_text(chunk: pd.DataFrame) -> pd.DataFrame:
    """Normalize missing issue text fields within a DataFrame chunk.

    Convert the title and body columns to pandas string dtype and
    replace missing values with empty strings. Other columns remain
    unchanged, and the original DataFrame is not modified.

    Args:
        chunk (pd.DataFrame): DataFrame containing title and body columns.

    Returns:
        pd.DataFrame: A copy of the chunk with normalized text fields.
    """
    chunk = chunk.copy()
    for column in ["title", "body"]:
        chunk[column] = chunk[column].astype("string").fillna("")
    return chunk


def create_index(connection: sqlite3.Connection) -> None:
    """Create the SQLite index used for text deduplication.

    Initialize a table that stores each unique text hash, its first
    observed label, whether conflicting labels exist, and whether
    a representative row has already been written.

    Args:
        connection (sqlite3.Connection): Active SQLite database connection.

    Raises:
        sqlite3.Error: If the table cannot be created.
    """
    connection.execute(
        """
        CREATE TABLE text_groups (
            text_hash BLOB PRIMARY KEY,
            first_label TEXT NOT NULL,
            is_conflicting INTEGER NOT NULL DEFAULT 0,
            is_written INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def index_text_groups(
    connection: sqlite3.Connection,
    raw_path: Path,
    chunk_size: int,
) -> int:
    """Index issue text groups and detect conflicting labels.

    Read the raw training CSV in chunks, normalize missing text,
    and compute a SHA-256 hash for each title-body combination.

    For each unique hash, store the first observed label and flag
    the group if any subsequent row has a different label.

    Args:
        connection (sqlite3.Connection): Active SQLite connection
            containing the text_groups table.
        raw_path (Path): Path to the raw training CSV file.
        chunk_size (int): Number of rows to read per chunk.

    Returns:
        int: Total number of input rows processed.

    Raises:
        FileNotFoundError: If the input CSV does not exist.
        sqlite3.Error: If a database operation fails.
    """
    total_rows = 0

    for chunk in pd.read_csv(raw_path, chunksize=chunk_size):
        chunk = normalize_text(chunk)

        records = [
            (make_text_hash(title, body), label)
            for title, body, label in chunk[["title", "body", "labels"]].itertuples(
                index=False, name=None
            )
        ]

        connection.executemany(
            """
            INSERT INTO text_groups (text_hash, first_label)
            VALUES (?, ?)
            ON CONFLICT(text_hash) DO UPDATE SET
                is_conflicting = CASE
                    WHEN first_label != excluded.first_label THEN 1
                    ELSE is_conflicting
                END
            """,
            records,
        )

        total_rows += len(chunk)
        connection.commit()

    return total_rows


def write_prepared_data(
    connection: sqlite3.Connection,
    raw_path: Path,
    output_path: Path,
    chunk_size: int,
) -> dict:
    """Write the cleaned dataset using the indexed text groups.

    Process the raw training CSV in chunks and apply the agreed
    data quality rules:

    - Replace missing titles and bodies with empty strings.
    - Exclude all rows belonging to conflicting-label text groups.
    - Retain only the first occurrence of each consistent-label
      text group.
    - Preserve the selected original columns.
    - Collect preparation statistics and class counts.

    The function updates the SQLite index to track written groups
    and writes the prepared records to the output CSV.

    Args:
        connection (sqlite3.Connection): Active SQLite connection
            containing the indexed text groups.
        raw_path (Path): Path to the raw training CSV file.
        output_path (Path): Destination path for the prepared CSV.
        chunk_size (int): Number of rows processed per chunk.

    Returns:
        dict: Preparation statistics, including retained rows,
            removed duplicates, removed conflicting rows,
            missing-value counts, and class distribution.

    Raises:
        RuntimeError: If a text hash is missing from the index.
        sqlite3.Error: If a database operation fails.
    """
    kept_rows = 0
    conflicting_rows = 0
    duplicate_rows = 0
    missing_title_rows = 0
    missing_body_rows = 0
    class_counts = Counter()

    with output_path.open("w", encoding="utf-8", newline="") as output:
        # Write the CSV header exactly once, even if no rows are retained.
        pd.DataFrame(columns=COLUMNS).to_csv(output, index=False)

        for chunk in pd.read_csv(raw_path, chunksize=chunk_size):
            missing_title_rows += int(chunk["title"].isna().sum())
            missing_body_rows += int(chunk["body"].isna().sum())

            chunk = normalize_text(chunk)
            keep_indices = []

            for index, title, body in chunk[["title", "body"]].itertuples(
                index=True, name=None
            ):
                key = make_text_hash(title, body)

                row = connection.execute(
                    """
                    SELECT is_conflicting, is_written
                    FROM text_groups
                    WHERE text_hash = ?
                    """,
                    (key,),
                ).fetchone()

                if row is None:
                    raise RuntimeError("Text hash missing from index")

                is_conflicting, is_written = row

                if is_conflicting:
                    conflicting_rows += 1
                    continue

                if is_written:
                    duplicate_rows += 1
                    continue

                connection.execute(
                    """
                    UPDATE text_groups
                    SET is_written = 1
                    WHERE text_hash = ?
                    """,
                    (key,),
                )

                keep_indices.append(index)

            cleaned = chunk.loc[keep_indices, COLUMNS]

            if not cleaned.empty:
                # Append data rows only; the header already exists.
                cleaned.to_csv(output, index=False, header=False)

                class_counts.update(cleaned["labels"].value_counts().to_dict())
                kept_rows += len(cleaned)

            connection.commit()

    return {
        "kept_rows": kept_rows,
        "conflicting_rows_removed": conflicting_rows,
        "duplicate_rows_removed": duplicate_rows,
        "missing_title_rows": missing_title_rows,
        "missing_body_rows": missing_body_rows,
        "class_distribution": dict(sorted(class_counts.items())),
    }


def prepare_dataset(
    raw_path: Path,
    output_path: Path,
    report_path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run the complete data preparation pipeline.

    Build a temporary SQLite index, identify conflicting-label
    text groups, remove conflicting rows and redundant duplicates,
    and generate a cleaned CSV and JSON preparation report.

    The raw input dataset is preserved without modification.
    The temporary SQLite database is removed after processing.

    Args:
        raw_path (Path): Path to the original training CSV.
        output_path (Path): Destination for the cleaned dataset.
        report_path (Path): Destination for the JSON summary report.
        chunk_size (int): Number of rows per processing chunk.
            Defaults to CHUNK_SIZE.

    Returns:
        dict: Summary containing input and retained row counts,
            removal statistics, missing-value counts, conflicting
            group count, and final class distribution.

    Raises:
        AssertionError: If the preparation row counts do not
            reconcile with the total input row count.
        OSError: If required files cannot be read or written.
        sqlite3.Error: If a database operation fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        database_path = Path(temp_dir) / "text_index.sqlite"

        with sqlite3.connect(database_path) as connection:
            create_index(connection)

            total_rows = index_text_groups(connection, raw_path, chunk_size)

            stats = write_prepared_data(connection, raw_path, output_path, chunk_size)

            conflicting_groups = connection.execute(
                """
                SELECT COUNT(*)
                FROM text_groups
                WHERE is_conflicting = 1
                """
            ).fetchone()[0]

    stats["total_rows"] = total_rows
    stats["conflicting_groups"] = conflicting_groups

    assert (
        stats["kept_rows"]
        + stats["conflicting_rows_removed"]
        + stats["duplicate_rows_removed"]
        == total_rows
    )

    report_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=REPORT_PATH)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    args = parser.parse_args()

    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    report = prepare_dataset(
        raw_path=args.input,
        output_path=args.output,
        report_path=args.report,
        chunk_size=args.chunk_size,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
