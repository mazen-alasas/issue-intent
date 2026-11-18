import argparse
from collections import Counter
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

DEFAULT_MODEL_DIR = Path.home() / "issue-intent-artifacts/models/distilbert/onnx"


def quantize_model(
    source: Path,
    destination: Path,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"FP32 model not found: {source}")

    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must be different")

    destination.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(source),
        model_output=str(destination),
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=False,
        weight_type=QuantType.QInt8,
    )

    model = onnx.load(destination)
    onnx.checker.check_model(model)

    session = ort.InferenceSession(
        str(destination),
        providers=["CPUExecutionProvider"],
    )

    expected_inputs = {"input_ids", "attention_mask"}
    actual_inputs = {item.name for item in session.get_inputs()}

    if actual_inputs != expected_inputs:
        raise ValueError(f"Unexpected inputs: {actual_inputs}")

    outputs = [item.name for item in session.get_outputs()]

    if outputs != ["logits"]:
        raise ValueError(f"Unexpected outputs: {outputs}")

    source_size = source.stat().st_size / (1024**2)
    destination_size = destination.stat().st_size / (1024**2)

    operators = Counter(node.op_type for node in model.graph.node)

    print(f"Source: {source}")
    print(f"Destination: {destination}")
    print(f"FP32 size: {source_size:.2f} MiB")
    print(f"INT8 size: {destination_size:.2f} MiB")
    print(f"Size reduction: {(1 - destination_size / source_size) * 100:.2f}%")
    print(f"Quantized operators: {dict(operators)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_MODEL_DIR / "model_fp32.onnx",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_MODEL_DIR / "model_int8.onnx",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    quantize_model(args.source, args.destination)
