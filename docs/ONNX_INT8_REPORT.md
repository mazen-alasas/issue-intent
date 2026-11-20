# ONNX INT8 Quantization and Partial Evaluation

**Date:** 2026-11-18  
**Branch:** `onnx-browser-inference`  
**Status (2026-11-19):** ONNX INT8 inference implemented and validated in Docker and browser environments. The public Cloudflare Pages application was deployed and manually smoke-tested. Full INT8 official-test evaluation remains incomplete.

## Scope and provenance

The selected DistilBERT checkpoint was exported to ONNX FP32 and dynamically quantized to ONNX INT8. The FP32 model and original PyTorch checkpoint remain reference artifacts. The partial evaluation used saved logits from the **first 14,040 rows** of the 142,320-row official test dataset, not a random/stratified draw. Although the observed label proportions are close to the complete test, representativeness is **not guaranteed**. The partial result must not be reported as the full official-test score.

Artifacts (not tracked in Git):

- FP32: `~/issue-intent-artifacts/models/distilbert/onnx/model_fp32.onnx`
- INT8: `~/issue-intent-artifacts/models/distilbert/onnx/model_int8.onnx`
- Saved partial logits: `~/issue-intent-artifacts/models/distilbert/onnx/int8_full_evaluation.progress.npz`
- Partial report: `~/issue-intent-artifacts/models/distilbert/onnx/int8_partial_evaluation.json`

## Size and preliminary throughput

| Measure | FP32 | INT8 |
|---|---:|---:|
| ONNX model size | 255.54 MiB | 132.44 MiB |
| CPU throughput (earlier batch-4 comparison) | 8.63 issues/s | 12.30 issues/s |

Size reduction: **48.17%**. Observed CPU throughput ratio: **1.43×** under the measured setup. Throughput excludes model loading, preprocessing, tokenization and HTTP overhead; the comparison is not a production latency guarantee. Benchmark results depend on hardware, sequence lengths, batch size and threading.

## Partial quality results

| Metric | ONNX FP32 | ONNX INT8 |
|---|---:|---:|
| Micro F1 | 0.881054 | 0.881339 |
| Macro F1 | 0.783052 | 0.781291 |
| Weighted F1 | 0.878794 | 0.878682 |
| Bug F1 (n=7,290) | 0.916846 | 0.917535 |
| Documentation F1 (n=645) | 0.704603 | 0.701169 |
| Feature F1 (n=5,261) | 0.888593 | 0.888423 |
| Question F1 (n=844) | 0.622165 | 0.618037 |

Coverage: **14,040/142,320 (9.87%)**. Prediction agreement: **98.2906%** (13,800 unchanged, 240 changed). Among changed predictions, 106 were FP32-correct/INT8-wrong, 110 were FP32-wrong/INT8-correct, and 24 were wrong under both with different labels. INT8's Micro F1 is higher by 0.000285, while Macro F1 is lower by 0.001761 on this partial prefix; neither difference establishes a general superiority claim.

Partial label fractions versus complete official test: bug 51.92% vs 52.54%; documentation 4.59% vs 4.39%; feature 37.47% vs 37.10%; question 6.01% vs 5.97%.

## Engineering decision

The selected DistilBERT checkpoint was exported to ONNX FP32 and dynamically quantized to INT8, reducing model size while preserving the PyTorch and FP32 reference artifacts. ONNX INT8 was integrated into the CPU Docker backend and browser-based inference system, with the public React application deployed on Cloudflare Pages. The INT8 evaluation covers only the first 14,040 official-test examples and must not be interpreted as full-test performance. See [Docker Deployment](DOCKER_DEPLOYMENT.md), [HTTP Benchmark Report](HTTP_BENCHMARK_REPORT.md), and [Browser Compatibility](BROWSER_INFERENCE_COMPATIBILITY.md).

## Reproduction and checks

```bash
uv run python -m scripts.report_partial_onnx_evaluation
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The partial report command requires the saved `.progress.npz` checkpoint and official test Parquet. Do not run full evaluation unless its additional value justifies the runtime.


## Serving and browser validation — November 19, 2026

- Optimized ONNX-only Docker image: **556 MB**, compared with the earlier 1.72 GB transitional image; PyTorch confirmed absent.
- Docker `/health` and `/predict` smoke tests passed; fixed short-input HTTP benchmark: **56.986 req/s (C1)** and **136.316 req/s (C4)** over 100 measured requests each.
- ONNX Runtime Web graph: **3/3** reference cases matched expected labels; browser E4, token IDs and attention masks: **3/3** Python fixtures matched.
- Local production preview used the INT8 graph hosted on Hugging Face and returned a prediction in the existing React UI.
- The public Cloudflare Pages demo was successfully smoke-tested on real GitHub issues. Browser parity fixtures provide limited compatibility evidence and do not replace full INT8 evaluation or comprehensive cross-browser testing.
