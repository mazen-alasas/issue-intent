from src.models.structured_ablation_experiment import (
    EXPERIMENTS as LEGACY_EXPERIMENTS,
)
from src.preprocessing.experiment_configs import EXPERIMENTS


def test_expected_experiments_exist():
    assert set(EXPERIMENTS) == {"E2", "E3", "E4", "E5"}


def test_e4_preserves_training_preprocessing():
    config = EXPERIMENTS["E4"]

    assert config.replace_fenced_code
    assert not config.compress_repeated_blocks


def test_legacy_experiments_reference_same_configuration():
    assert LEGACY_EXPERIMENTS is EXPERIMENTS
    assert LEGACY_EXPERIMENTS["E4"] is EXPERIMENTS["E4"]
