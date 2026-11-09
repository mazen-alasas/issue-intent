import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.models.structured_ablation_experiment import EXPERIMENTS
from src.models.train_classical import ROOT
from src.preprocessing.structured import preprocess_structured

VERSION = "structured_v1"
EXPERIMENT = "E4"
CHUNK_SIZE = 10_000

SOURCE_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = SOURCE_DIR / VERSION

SOURCES = {
    "train": SOURCE_DIR / "train.csv",
    "validation": SOURCE_DIR / "validation.csv",
}

COLUMNS = ("title", "body", "labels")
SCHEMA = pa.schema(
    [
        pa.field("title", pa.string(), nullable=False),
        pa.field("body", pa.string(), nullable=False),
        pa.field("labels", pa.string(), nullable=False),
    ]
)


def display_path(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """Calculate SHA-256 without loading the whole file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Preserve title/label and preprocess only the body."""
    config = EXPERIMENTS[EXPERIMENT]

    processed = chunk.loc[:, list(COLUMNS)].copy()
    processed["body"] = processed["body"].map(
        lambda value: preprocess_structured(value, config)
    )

    return processed


def build_split(
    source: Path,
    destination: Path,
    *,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Write one CSV split to Parquet, using a temporary file."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not source.is_file():
        raise FileNotFoundError(source)

    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".parquet.tmp")

    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")

    row_count = 0
    raw_body_chars = 0
    processed_body_chars = 0
    writer = None
    start = time.perf_counter()

    try:
        reader = pd.read_csv(
            source,
            usecols=list(COLUMNS),
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
        )

        for chunk in reader:
            # Empty labels are invalid; empty bodies are allowed.
            if chunk["labels"].eq("").any():
                raise ValueError("Empty label found in source data")

            processed = process_chunk(chunk)

            if not chunk["title"].equals(processed["title"]):
                raise ValueError("Title changed during preprocessing")

            if not chunk["labels"].equals(processed["labels"]):
                raise ValueError("Labels changed during preprocessing")

            raw_body_chars += int(chunk["body"].str.len().sum())
            processed_body_chars += int(processed["body"].str.len().sum())

            table = pa.Table.from_pandas(
                processed,
                schema=SCHEMA,
                preserve_index=False,
                safe=True,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    SCHEMA,
                    compression="zstd",
                )

            writer.write_table(table)
            row_count += len(chunk)

        if writer is None:
            raise ValueError(f"Source contains no data rows: {source}")

        writer.close()
        writer = None

        # Validate the completed temporary artifact.
        metadata = pq.read_metadata(temporary)

        if metadata.num_rows != row_count:
            raise ValueError("Parquet row count mismatch")

        if metadata.schema.to_arrow_schema() != SCHEMA:
            raise ValueError("Parquet schema mismatch")

        if destination.exists():
            raise FileExistsError(f"Output already exists: {destination}")

        os.replace(temporary, destination)

    finally:
        if writer is not None:
            writer.close()

        if temporary.exists():
            temporary.unlink()

    return {
        "source": display_path(source),
        "output": display_path(destination),
        "rows": row_count,
        "source_sha256": sha256_file(source),
        "output_sha256": sha256_file(destination),
        "raw_body_characters": raw_body_chars,
        "processed_body_characters": processed_body_chars,
        "body_reduction_pct": round(
            100 * (raw_body_chars - processed_body_chars) / raw_body_chars,
            4,
        )
        if raw_body_chars
        else 0.0,
        "output_bytes": destination.stat().st_size,
        "processing_seconds": round(time.perf_counter() - start, 3),
    }


def build_dataset() -> dict:
    """Build both splits and write a reproducibility manifest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "manifest.json"

    # Avoid silently overwriting existing versioned artifacts.
    if manifest_path.exists():
        raise FileExistsError(manifest_path)

    for split in SOURCES:
        destination = OUTPUT_DIR / f"{split}.parquet"
        if destination.exists():
            raise FileExistsError(destination)

    config = EXPERIMENTS[EXPERIMENT]

    manifest = {
        "version": VERSION,
        "experiment": EXPERIMENT,
        "preprocessing_scope": "body_only",
        "configuration": {
            name: getattr(config, name) for name in config.__dataclass_fields__
        },
        "columns": list(COLUMNS),
        "chunk_size": CHUNK_SIZE,
        "splits": {},
    }

    for split, source in SOURCES.items():
        print(f"Building {split}...", flush=True)

        destination = OUTPUT_DIR / f"{split}.parquet"
        manifest["splits"][split] = build_split(
            source,
            destination,
        )

        print(
            f"  Rows: {manifest['splits'][split]['rows']:,}",
            flush=True,
        )

    temporary_manifest = OUTPUT_DIR / "manifest.json.tmp"

    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)

    return manifest


def main() -> None:
    manifest = build_dataset()

    print("\nDataset build complete")
    print("-" * 50)

    for split, result in manifest["splits"].items():
        print(
            f"{split}: {result['rows']:,} rows | "
            f"body reduction: {result['body_reduction_pct']:.2f}% | "
            f"file: {result['output']}"
        )

    print(f"Manifest: {OUTPUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
