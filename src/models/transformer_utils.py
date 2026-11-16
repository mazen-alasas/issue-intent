import torch

from src.data.labels import LABELS


def resolve_device(name: str) -> torch.device:
    """Resolve the requested PyTorch device."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"

    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    return torch.device(name)


def validate_model_labels(model, *, allow_generic_labels: bool = False) -> None:
    """Validate checkpoint labels against the training label order."""
    expected = {index: label for index, label in enumerate(LABELS)}

    actual = {int(index): str(label) for index, label in model.config.id2label.items()}

    if actual == expected:
        return

    generic = {index: f"LABEL_{index}" for index in range(len(LABELS))}

    if actual == generic and allow_generic_labels:
        return

    if actual == generic:
        raise ValueError(
            "Checkpoint uses generic LABEL_N names. "
            "Verify the training label mapping, then explicitly "
            "enable allow_generic_labels."
        )

    raise ValueError(f"Checkpoint label mapping mismatch: {actual} != {expected}")
