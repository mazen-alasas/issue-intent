import pandas as pd
import pytest

from src.data.sampling import load_nested_stratified_samples, load_stratified_sample


def test_load_stratified_sample_is_reproducible(tmp_path) -> None:
    """Verify stratified sampling produces reproducible results."""
    csv_path = tmp_path / "train.csv"

    data = pd.DataFrame(
        {
            "title": [f"Issue {i}" for i in range(100)],
            "labels": ["bug"] * 60 + ["feature"] * 40,
        }
    )
    data.to_csv(csv_path, index=False)

    first = load_stratified_sample(csv_path, sample_size=20)
    second = load_stratified_sample(csv_path, sample_size=20)

    pd.testing.assert_frame_equal(first, second)

    assert len(first) == 20
    assert first["labels"].value_counts().to_dict() == {
        "bug": 12,
        "feature": 8,
    }


@pytest.mark.parametrize("sample_size", [0, -1, 101])
def test_load_stratified_sample_rejects_invalid_size(
    tmp_path,
    sample_size: int,
) -> None:
    """Verify invalid sample sizes are rejected."""
    csv_path = tmp_path / "train.csv"

    pd.DataFrame(
        {
            "title": ["A", "B", "C", "D"],
            "labels": ["bug", "bug", "feature", "feature"],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="sample_size"):
        load_stratified_sample(csv_path, sample_size=sample_size)


def test_nested_samples_preserve_subset_relationship(tmp_path) -> None:
    """Verify smaller samples are subsets of larger samples."""
    csv_path = tmp_path / "train.csv"

    data = pd.DataFrame(
        {
            "title": [f"Issue {i}" for i in range(100)],
            "labels": ["bug"] * 60 + ["feature"] * 40,
        }
    )
    data.to_csv(csv_path, index=False)

    samples = load_nested_stratified_samples(
        csv_path,
        sample_sizes=[10, 20, 40],
    )

    assert {size: len(sample) for size, sample in samples.items()} == {
        10: 10,
        20: 20,
        40: 40,
    }

    titles_10 = set(samples[10]["title"])
    titles_20 = set(samples[20]["title"])
    titles_40 = set(samples[40]["title"])

    assert titles_10.issubset(titles_20)
    assert titles_20.issubset(titles_40)
