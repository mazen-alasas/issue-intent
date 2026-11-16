"""Shared structured preprocessing experiment configurations."""

from src.preprocessing.structured import StructuredPreprocessingConfig

EXPERIMENTS = {
    "E2": StructuredPreprocessingConfig(),
    "E3": StructuredPreprocessingConfig(
        replace_fenced_code=False,
    ),
    "E4": StructuredPreprocessingConfig(
        compress_repeated_blocks=False,
    ),
    "E5": StructuredPreprocessingConfig(
        compress_repeated_blocks=False,
        selective_code_compression=True,
    ),
}
