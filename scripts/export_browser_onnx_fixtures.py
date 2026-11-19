"""Export representative ONNX inputs and reference logits for the browser smoke test.

Does not copy model weights into the repository or re-run dataset evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.api.onnx_inference import OnnxIssueClassifier
from src.api.shared import LABEL_NAMES
from src.data.transformer_dataset import encode_issue
from src.preprocessing.structured import preprocess_structured

EXAMPLES = [
    {
        "name": "crash",
        "title": "Application crashes on startup",
        "body": "RuntimeError after updating.",
    },
    {
        "name": "feature",
        "title": "Add an option to export results as CSV",
        "body": "It would be helpful to download the classification results as CSV.",
    },
    {
        "name": "markdown_code",
        "title": "HTTP 500 after deploying to server",
        "body": (
            "## Steps to reproduce\n"
            "1. Start the server\n"
            "2. Send a request to https://example.com/api?token=abc\n"
            "```python\nraise RuntimeError('bad config')\n```\n"
            "See /var/log/app/error.log for details."
        ),
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frontend/public/browser-fixtures.json"),
    )
    args = parser.parse_args()

    classifier = OnnxIssueClassifier()
    fixtures = []
    for item in EXAMPLES:
        clean_body = preprocess_structured(
            item["body"], classifier.preprocessing_config
        )
        encoded = encode_issue(
            item["title"], clean_body, classifier.tokenizer, classifier.input_config
        )
        input_tensors = {
            key: np.asarray([encoded[key]], dtype=np.int64)
            for key in ("input_ids", "attention_mask")
        }
        logits = np.asarray(classifier.session.run(["logits"], input_tensors)[0])[0]
        shifted = logits.astype(np.float64) - float(logits.max())
        exp = np.exp(shifted)
        probs = exp / exp.sum()
        fixtures.append(
            {
                "name": item["name"],
                "title": item["title"],
                "body": item["body"],
                "processed_body": clean_body,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "reference_logits": [float(x) for x in logits],
                "reference_probabilities": [float(x) for x in probs],
                "reference_label": LABEL_NAMES[int(probs.argmax())],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"labels": list(LABEL_NAMES), "fixtures": fixtures}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(fixtures)} reference fixtures to {args.output}")
    print("Reference labels:", ", ".join(f["reference_label"] for f in fixtures))


if __name__ == "__main__":
    main()
