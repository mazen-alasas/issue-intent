import json

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data import shuffle_transformer_train as shuffle_module


@pytest.fixture
def input_parquet(tmp_path):
    path = tmp_path / "input.parquet"

    table = pa.table(
        {
            "title": [f"Issue {i}" for i in range(20)],
            "body": [f"Description {i}" for i in range(20)],
            "labels": [
                ["bug", "feature", "question", "documentation"][i % 4]
                for i in range(20)
            ],
        }
    )

    pq.write_table(table, path)
    return path


def read_rows(path):
    table = pq.read_table(path)
    return list(
        zip(
            table["title"].to_pylist(),
            table["body"].to_pylist(),
            table["labels"].to_pylist(),
            strict=True,
        )
    )


def run_shuffle(input_path, output_path, temp_dir, seed):
    temp_dir.mkdir()

    shuffle_module.shuffle_parquet(
        input_path=input_path,
        output_path=output_path,
        temp_dir=temp_dir,
        seed=seed,
        memory_limit="256MB",
        row_group_size=2048,
    )


def test_same_seed_produces_same_order(input_parquet, tmp_path):
    output_a = tmp_path / "output_a.parquet"
    output_b = tmp_path / "output_b.parquet"

    run_shuffle(
        input_parquet,
        output_a,
        tmp_path / "temp_a",
        seed=42,
    )

    run_shuffle(
        input_parquet,
        output_b,
        tmp_path / "temp_b",
        seed=42,
    )

    assert read_rows(output_a) == read_rows(output_b)


def test_different_seeds_produce_different_order(input_parquet, tmp_path):
    output_a = tmp_path / "output_a.parquet"
    output_b = tmp_path / "output_b.parquet"

    run_shuffle(
        input_parquet,
        output_a,
        tmp_path / "temp_a",
        seed=42,
    )

    run_shuffle(
        input_parquet,
        output_b,
        tmp_path / "temp_b",
        seed=123,
    )

    assert read_rows(output_a) != read_rows(output_b)


def test_shuffle_preserves_data(input_parquet, tmp_path):
    output = tmp_path / "output.parquet"

    run_shuffle(
        input_parquet,
        output,
        tmp_path / "temp",
        seed=42,
    )

    source_fingerprint = shuffle_module.fingerprint(input_parquet, 5)
    output_fingerprint = shuffle_module.fingerprint(output, 5)

    assert source_fingerprint == output_fingerprint


def test_main_writes_report_and_cleans_temp_dir(
    input_parquet,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output.parquet"
    temp_dir = tmp_path / "shuffle_temp"

    args = shuffle_module.argparse.Namespace(
        input=input_parquet,
        output=output,
        seed=42,
        memory_limit="256MB",
        row_group_size=2048,
        batch_size=5,
        temp_dir=temp_dir,
    )

    monkeypatch.setattr(shuffle_module, "parse_args", lambda: args)

    shuffle_module.main()

    report_path = output.with_suffix(".shuffle_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["verified"] is True
    assert report["seed"] == 42
    assert report["duckdb_version"] == duckdb.__version__
    assert output.exists()
    assert report_path.exists()
    assert not temp_dir.exists()


def test_main_cleans_temp_dir_after_failure(
    input_parquet,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "output.parquet"
    temp_dir = tmp_path / "shuffle_temp"

    args = shuffle_module.argparse.Namespace(
        input=input_parquet,
        output=output,
        seed=42,
        memory_limit="256MB",
        row_group_size=2048,
        batch_size=5,
        temp_dir=temp_dir,
    )

    monkeypatch.setattr(shuffle_module, "parse_args", lambda: args)

    def fail_shuffle(**kwargs):
        raise RuntimeError("Simulated shuffle failure")

    monkeypatch.setattr(
        shuffle_module,
        "shuffle_parquet",
        fail_shuffle,
    )

    with pytest.raises(RuntimeError, match="Simulated shuffle failure"):
        shuffle_module.main()

    assert not temp_dir.exists()
    assert not output.exists()
