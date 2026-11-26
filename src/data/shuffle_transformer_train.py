import argparse
import hashlib
import json
import shutil
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

COLUMNS = ("title", "body", "labels")
MODULUS = 1 << 256


def parse_args():
    parser = argparse.ArgumentParser(
        description="Globally shuffle transformer training data."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-limit", default="2GB")
    parser.add_argument("--row-group-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--temp-dir", type=Path)
    return parser.parse_args()


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def fingerprint(path, batch_size):
    parquet = pq.ParquetFile(path)

    if set(parquet.schema_arrow.names) != set(COLUMNS):
        raise ValueError("Unexpected Parquet columns")

    count = 0
    digest_sum = 0
    digest_xor = 0
    label_counts = {}

    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=list(COLUMNS),
    ):
        data = batch.to_pydict()

        for title, body, label in zip(
            data["title"],
            data["body"],
            data["labels"],
            strict=True,
        ):
            payload = json.dumps(
                [title, body, label],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")

            digest = int.from_bytes(
                hashlib.sha256(payload).digest(),
                "big",
            )

            digest_sum = (digest_sum + digest) % MODULUS
            digest_xor ^= digest
            label_counts[label] = label_counts.get(label, 0) + 1
            count += 1

    return {
        "rows": count,
        "label_counts": dict(sorted(label_counts.items())),
        "digest_sum": f"{digest_sum:064x}",
        "digest_xor": f"{digest_xor:064x}",
    }


def shuffle_parquet(
    input_path,
    output_path,
    temp_dir,
    seed,
    memory_limit,
    row_group_size,
):
    connection = duckdb.connect()

    try:
        connection.execute(f"SET memory_limit = {sql_literal(memory_limit)}")
        connection.execute(f"SET temp_directory = {sql_literal(temp_dir)}")
        connection.execute("SET threads = 2")
        connection.execute("SET preserve_insertion_order = true")

        source = sql_literal(input_path)
        destination = sql_literal(output_path)

        query = f"""
            COPY (
                SELECT title, body, labels
                FROM (
                    SELECT
                        title,
                        body,
                        labels,
                        row_number() OVER (
                            ORDER BY title, body, labels
                        ) AS row_id
                    FROM read_parquet({source})
                )
                ORDER BY hash(row_id, {seed}), row_id
            )
            TO {destination}
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE {row_group_size}
            )
        """

        connection.execute(query)
    finally:
        connection.close()


def main():
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if input_path == output_path:
        raise ValueError("Input and output must be different")

    if args.row_group_size < 2048:
        raise ValueError("row-group-size must be at least 2048")

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = (
        args.temp_dir.resolve()
        if args.temp_dir
        else output_path.parent / "shuffle_temp"
    )

    if temp_dir == input_path or temp_dir == output_path:
        raise ValueError("Invalid temporary directory")

    partial_output = output_path.with_name(output_path.name + ".partial.parquet")

    if partial_output.exists():
        raise FileExistsError(partial_output)

    if output_path.exists():
        raise FileExistsError(f"Output already exists: {output_path}")

    temp_dir.mkdir(parents=True, exist_ok=False)

    try:
        print("Computing source fingerprint...", flush=True)
        source_fingerprint = fingerprint(
            input_path,
            args.batch_size,
        )

        print("Running external shuffle...", flush=True)

        shuffle_parquet(
            input_path=input_path,
            output_path=partial_output,
            temp_dir=temp_dir,
            seed=args.seed,
            memory_limit=args.memory_limit,
            row_group_size=args.row_group_size,
        )

        print("Verifying shuffled output...", flush=True)
        output_fingerprint = fingerprint(
            partial_output,
            args.batch_size,
        )

        if source_fingerprint != output_fingerprint:
            raise RuntimeError("Data integrity verification failed")

        partial_output.rename(output_path)

        report = {
            "input": str(input_path),
            "output": str(output_path),
            "seed": args.seed,
            "memory_limit": args.memory_limit,
            "row_group_size": args.row_group_size,
            "duckdb_version": duckdb.__version__,
            "source": source_fingerprint,
            "output_fingerprint": output_fingerprint,
            "verified": True,
        }

        report_path = output_path.with_suffix(".shuffle_report.json")

        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(report, indent=2),
            flush=True,
        )
        print("Shuffle completed and verified.", flush=True)

    finally:
        if partial_output.exists():
            partial_output.unlink()

        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
