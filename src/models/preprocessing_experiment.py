import json

from src.models.train_classical import ROOT
from src.models.train_title_body import run_experiment

RESULTS_PATH = ROOT / "data/processed/title_body_minimal_logreg_200k_metrics.json"


def main() -> None:
    """Run and save the minimal preprocessing experiment."""
    results = run_experiment(preprocessing="minimal")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metrics = results["metrics"]

    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Training: {results['training_seconds']:.2f}s")
    print(f"Prediction: {results['prediction_seconds']:.2f}s")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
