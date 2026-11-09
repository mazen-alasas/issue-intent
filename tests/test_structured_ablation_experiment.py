import pandas as pd
import pytest

from src.models.structured_ablation_experiment import (
    EXPERIMENTS,
    build_experiment_text,
    result_path,
    run_experiment,
)


def test_experiment_configurations():
    assert set(EXPERIMENTS) == {"E2", "E3", "E4", "E5"}

    assert EXPERIMENTS["E2"].replace_fenced_code
    assert EXPERIMENTS["E2"].compress_repeated_blocks

    assert not EXPERIMENTS["E3"].replace_fenced_code
    assert EXPERIMENTS["E3"].compress_repeated_blocks

    assert EXPERIMENTS["E4"].replace_fenced_code
    assert not EXPERIMENTS["E4"].compress_repeated_blocks

    assert EXPERIMENTS["E5"].selective_code_compression
    assert not EXPERIMENTS["E5"].compress_repeated_blocks
    assert EXPERIMENTS["E5"].replace_fenced_code


def test_build_experiment_text_preserves_title():
    data = pd.DataFrame(
        {
            "title": ["## Application crashes"],
            "body": ["## Error\n`ValueError`"],
        }
    )

    result = build_experiment_text(data, EXPERIMENTS["E2"])

    assert result.iloc[0] == ("## Application crashes\n\nError\nValueError")


def test_code_replacement_ablation():
    fence = "`" * 3
    data = pd.DataFrame(
        {
            "title": ["Crash"],
            "body": [f"{fence}python\nraise ValueError()\n{fence}\n"],
        }
    )

    full = build_experiment_text(data, EXPERIMENTS["E2"])
    retained = build_experiment_text(data, EXPERIMENTS["E3"])

    assert "CODE" in full.iloc[0]
    assert "raise ValueError()" not in full.iloc[0]
    assert "raise ValueError()" in retained.iloc[0]


def test_repetition_ablation():
    data = pd.DataFrame(
        {
            "title": ["Timeout"],
            "body": ["Connection timeout\n" * 6],
        }
    )

    full = build_experiment_text(data, EXPERIMENTS["E2"])
    uncompressed = build_experiment_text(data, EXPERIMENTS["E4"])

    assert "[REPEATED_LINES]" in full.iloc[0]
    assert "[REPEATED_LINES]" not in uncompressed.iloc[0]
    assert uncompressed.iloc[0].count("Connection timeout") == 6


def test_invalid_experiment_rejected():
    with pytest.raises(ValueError, match="Unsupported experiment"):
        run_experiment("INVALID")


def test_result_paths_are_distinct():
    paths = [result_path(name) for name in EXPERIMENTS]

    assert len(set(paths)) == len(EXPERIMENTS)
    assert all(path.suffix == ".json" for path in paths)
