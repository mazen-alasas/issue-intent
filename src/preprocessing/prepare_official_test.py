import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.models.structured_ablation_experiment import EXPERIMENTS
from src.preprocessing.structured import preprocess_structured

COLUMNS = ("id", "labels", "title", "body", "author_association")
SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("labels", pa.string()),
        pa.field("title", pa.string()),
        pa.field("body", pa.string()),
        pa.field("author_association", pa.string()),
    ]
)


def prepare_official_test(
    source: Path,
    destination: Path,
    chunk_size: int = 10_000,
) -> int:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    if not source.is_file():
        raise FileNotFoundError(source)

    if destination.exists():
        raise FileExistsError(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")

    if temporary.exists():
        raise FileExistsError(temporary)

    config = EXPERIMENTS["E4"]
    writer = None
    rows = 0

    try:
        reader = pd.read_csv(
            source,
            usecols=list(COLUMNS),
            dtype={
                "id": "int64",
                "labels": str,
                "title": str,
                "body": str,
                "author_association": str,
            },
            keep_default_na=False,
            chunksize=chunk_size,
        )

        for chunk in reader:
            chunk = chunk.loc[:, list(COLUMNS)].copy()

            if chunk["labels"].eq("").any():
                raise ValueError("Empty labels in official test")

            chunk["body"] = chunk["body"].map(
                lambda value: preprocess_structured(value, config)
            )

            table = pa.Table.from_pandas(
                chunk,
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
            rows += len(chunk)

        if rows == 0:
            raise ValueError("Official test contains no rows")

        writer.close()
        writer = None

        metadata = pq.read_metadata(temporary)

        if metadata.num_rows != rows:
            raise ValueError("Output row count mismatch")

        if metadata.schema.to_arrow_schema() != SCHEMA:
            raise ValueError("Output schema mismatch")

        if destination.exists():
            raise FileExistsError(destination)

        os.replace(temporary, destination)

    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    args = parser.parse_args()

    rows = prepare_official_test(
        source=args.source,
        destination=args.output,
        chunk_size=args.chunk_size,
    )

    print(f"Prepared {rows:,} official test rows")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
