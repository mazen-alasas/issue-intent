# FP32 ONNX Export and Validation

## Scope

Export the selected DistilBERT sequence classifier to ONNX FP32, preserving the existing E4 structured-body preprocessing, title/body token allocation, label mapping, and model weights. This stage does **not** implement INT8 quantization, ONNX Runtime Web, or public deployment.

## Artifact and configuration

- Checkpoint: `~/issue-intent-artifacts/models/distilbert/best`
- ONNX graph: `~/issue-intent-artifacts/models/distilbert/onnx/model_fp32.onnx`
- ONNX opset: 17; PyTorch legacy exporter (`dynamo=False`)
- Inputs: `input_ids`, `attention_mask`; output: `logits`
- Dynamic batch and sequence dimensions; maximum sequence length of 256 is enforced by the application encoding pipeline, not the graph
- Label order: bug, documentation, feature, question
- ONNX size: 255.54 MiB; original checkpoint weights: 255.43 MiB
- Model: DistilBERT for sequence classification, 66,956,548 parameters

## Validation evidence (2026-11-17)

| Test | Result |
|---|---|
| ONNX graph checker | Passed |
| Synthetic parity | 9/9 batches passed; max absolute logit difference 0.00000215 |
| Real-data parity (200 issues) | 200/200 predictions agree; 200/200 logits close; max absolute difference 0.00000650 |
| Real-data parity (40 issues) | 40/40 predictions agree; 40/40 logits close; max absolute difference 0.00000407 |
| Tolerance | `atol=1e-4`, `rtol=1e-4` |

Parity is demonstrated only for the tested cases. It is not a full-test-set accuracy re-evaluation or a browser-runtime parity test.

## Preliminary CPU benchmark

The benchmark used a stratified sample of 40 issues from `data/raw/nlbse23-test.parquet`, seed 42, batch size 4, two CPU threads, one warmup round, and three measured rounds. Each runtime completed 30 measured batch calls. Model loading, E4 preprocessing, and tokenization were excluded.

| Metric | PyTorch FP32 | ONNX FP32 |
|---|---:|---:|
| Median batch latency | 502.38 ms | 468.37 ms |
| P95 batch latency | 584.03 ms | 565.02 ms |
| Mean batch latency | 450.78 ms | 427.93 ms |
| Throughput | 8.87 issues/s | 9.35 issues/s |

Observed ONNX throughput speedup: **1.053×** (~5.3%) in this run. These are preliminary, sequentially measured CPU results; they are not a general performance guarantee or end-to-end HTTP latency. The sampled accuracy of 80% is not representative of the full official test set (the previously evaluated model achieved ~88.66% there).

The 40-issue sample contained no null bodies. The complete test data has 466 null bodies, which the sampling pipeline normalizes to empty strings when selected. Explicit end-to-end null/empty-body regression coverage should be added before browser parity claims.

## Reproduce

```bash
uv run python -m scripts.export_onnx
uv run python -m scripts.validate_onnx
uv run python -u -m scripts.benchmark_onnx --sample-size 40 --benchmark-sample-size 40 --seed 42 --batch-size 4 --warmup 1 --repeats 3 --threads 2 --output ~/issue-intent-artifacts/models/distilbert/onnx/int8_comparison.json
```

The current comparison benchmark writes `~/issue-intent-artifacts/models/distilbert/onnx/int8_comparison.json`, including sample issue IDs and a SHA-256 digest of their ordered list. Larger samples and repeated interleaved runs are recommended before drawing stronger latency conclusions.

## Historical scope note

This document records the FP32-only milestone. INT8 results are recorded separately in `docs/ONNX_INT8_REPORT.md`; the benchmark script has since been extended to require both FP32 and INT8 graphs.

## Subsequent stages (status as of November 19, 2026)

INT8 quantization, ONNX-only CPU Docker, browser E4/tokenizer parity, and ONNX Runtime Web integration were subsequently implemented and locally validated. The original FP32 validation measurements above remain unchanged. See [INT8 Report](ONNX_INT8_REPORT.md), [Browser Compatibility](BROWSER_INFERENCE_COMPATIBILITY.md) and [Static Deployment](STATIC_DEPLOYMENT.md). Public Cloudflare Pages deployment is still pending.
