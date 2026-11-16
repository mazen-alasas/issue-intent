from src.data.labels import LABEL_TO_ID, LABELS
from src.data.neural_dataset import (
    LABEL_TO_ID as LEGACY_LABEL_TO_ID,
)
from src.data.neural_dataset import (
    LABELS as LEGACY_LABELS,
)


def test_label_order():
    assert LABELS == (
        "bug",
        "documentation",
        "feature",
        "question",
    )


def test_label_mapping():
    assert LABEL_TO_ID == {
        "bug": 0,
        "documentation": 1,
        "feature": 2,
        "question": 3,
    }


def test_legacy_imports_remain_compatible():
    assert LEGACY_LABELS is LABELS
    assert LEGACY_LABEL_TO_ID is LABEL_TO_ID
