# issue-intent

**GitHub issue classification · NLP experimentation · ONNX browser inference**

`issue-intent` classifies English GitHub issue reports into **bug**, **feature**, **question**, and **documentation**. The project follows the full ML engineering lifecycle: dataset audits, reproducible splits, classical and neural baselines, DistilBERT fine-tuning, held-out evaluation, CPU serving, ONNX quantization, and a GitHub-inspired React demo.

**Status (November 19, 2026):** [Live browser demo](https://issue-intent.pages.dev/) deployed on Cloudflare Pages. The static React application runs INT8 ONNX inference in the browser and fetches the public model from [Hugging Face](https://huggingface.co/mazenalasas/issue-intent-onnx); no Python inference server is required for the demo. The owner successfully tested predictions against several real GitHub issues, including issues from external repositories. This is a manual smoke test, **not** a new benchmark or measured generalization score. See the [static deployment guide](docs/STATIC_DEPLOYMENT.md).

## Results

The selected checkpoint is DistilBERT (epoch 3, chosen by validation Macro F1). The official NLBSE'23 test set contains **142,320** issues.

| Model | Split | Micro F1 | Macro F1 |
| --- | --- | ---: | ---: |
| TF-IDF + Logistic Regression | Validation | 0.842650 | 0.706409 |
| TextCNN | Validation | 0.852650 | 0.724730 |
| DistilBERT | Validation | 0.883665 | 0.788362 |
| **DistilBERT (selected PyTorch checkpoint)** | **Official test** | **0.886558** | **0.791175** |

The official test recorded **126,175 correct** and **16,145 incorrect** predictions, weighted F1 **0.884702**. The first three rows use validation rather than the official test and are **not a same-split comparison**. `question` and `documentation` remain more difficult classes. See [final evaluation](docs/FINAL_EVALUATION_REPORT.md) and [model card](docs/MODEL_CARD.md).

The **INT8 ONNX** graph was evaluated separately on the **first 14,040 test rows** (9.87%, non-random): Micro F1 **0.881339**, Macro F1 **0.781291**. These partial scores must not be presented as the full official-test performance of the deployed graph. [INT8 report](docs/ONNX_INT8_REPORT.md).

## Dataset and modeling

The [NLBSE'23 issue classification dataset](https://github.com/nlbse2023/issue-report-classification) contains **1,418,201** labeled issues: **1,275,881** official training and **142,320** official test examples. After exact title/body deduplication and conflicting-label exclusion, **1,221,982** training-source examples remained. Stratified 85/15 splitting (`random_state=42`) produced **1,038,684** training and **183,298** validation examples. The official test remained held out.

Experiments progressed from title-only and title+body TF-IDF to preprocessing ablations, a PyTorch TextCNN, and DistilBERT. The selected E4 preprocessing modifies the issue **body**, preserving the title. Transformer input uses **256 tokens maximum**, a **48-token title budget**, and head/tail allocation for the body. Details: [project plan](docs/PROJECT_PLAN.md) and [progress log](docs/PROGRESS.md).

## Demo architecture

```text
Static React / TypeScript / Vite site
  ├── GitHub-inspired issue editor and Markdown preview
  ├── E4 body preprocessing + WordPiece tokenizer (browser)
  ├── ONNX Runtime Web (WASM) ── fetches INT8 graph from Hugging Face
  └── Predicted class, confidence, four class probabilities

Independent server path (preserved in this repository)
  FastAPI /predict → PyTorch or ONNX INT8 CPU backend
  Docker ONNX-only CPU image (556 MB; no PyTorch)
```

The demo **does not create or modify GitHub issues**. The browser model loads on first use; subsequent requests benefit from caching and session reuse. Confidence is an uncalibrated softmax score, not a guarantee of correctness.

## Browser validation and deployment status

| Evidence | Verified result |
| --- | --- |
| Browser ONNX graph compatibility | 3/3 reference cases matched expected labels |
| Browser E4, token IDs, attention masks vs Python | 3/3 fixtures passed |
| Local production preview | Model fetched from Hugging Face; real UI prediction succeeded |
| Frontend checks | Vite build, ESLint, 19 Vitest tests passed |
| Python checks | 372 pytest tests passed |
| Static artifact preparation | 12 files, each within 25 MiB; local ONNX graph and unused WASM excluded |
| Public Cloudflare Pages site | **Deployed** — owner verified UI and issue predictions on November 19, 2026; no automated cross-browser validation claimed |

These small parity checks do not constitute full browser accuracy or cross-browser testing. [Browser compatibility](docs/BROWSER_INFERENCE_COMPATIBILITY.md) · [Static deployment](docs/STATIC_DEPLOYMENT.md).

## Local development

Install the Python development environment and run the backend quality gate:

```bash
uv sync
uv run ruff check --fix .
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

For the **browser-only frontend** (no running Python API required), ensure the public model and tokenizer assets are available as described in the [frontend guide](frontend/README.md):

```bash
cd frontend
npm ci
npm run dev
```

For a deployment-equivalent local preview:

```bash
cd frontend
npm run build
node scripts/prepare-static-deploy.mjs
npm run preview -- --host 127.0.0.1
```

Open `http://127.0.0.1:4173/`. The same frontend is publicly deployed at [issue-intent.pages.dev](https://issue-intent.pages.dev/). To run the optional **FastAPI backend** locally, see the [frontend guide](frontend/README.md) and [Docker deployment](docs/DOCKER_DEPLOYMENT.md). Training data and model checkpoints are not included in Git.

## ONNX and serving measurements

| Measure | Observed result |
| --- | --- |
| FP32 → INT8 ONNX size | 255.54 → 132.44 MiB (48.17% reduction) |
| Preliminary model-only CPU throughput | 8.63 → 12.30 issues/s (FP32 → INT8, batch 4) |
| ONNX-only Docker image | 556 MB (vs 1.72 GB transitional image) |
| Local HTTP benchmark, concurrency 1 | 56.986 requests/s; p95 21.388 ms |
| Local HTTP benchmark, concurrency 4 | 136.316 requests/s; p95 35.309 ms |

The HTTP benchmark used a repeated short issue, 100 measured requests and 10 warmups per configuration. It is not a production load test or comparable directly with model-only throughput. See [FP32 report](docs/ONNX_FP32_REPORT.md), [INT8 report](docs/ONNX_INT8_REPORT.md), and [HTTP benchmark report](docs/HTTP_BENCHMARK_REPORT.md).

## Project status

| Milestone | State |
| --- | --- |
| Data audit, evaluation protocol, classical baselines | Complete |
| TextCNN and DistilBERT training | Complete |
| Official test evaluation and artifact audit | Complete |
| FastAPI, Docker, CI | Implemented and locally validated |
| ONNX FP32 export, INT8 quantization, CPU serving | Complete with documented evaluation limits |
| React application and browser inference | Implemented and locally validated |
| Static build and Hugging Face model hosting | Validated locally / model published |
| Cloudflare Pages public deployment | **Live; owner smoke-tested** |
| Portfolio release and architecture diagrams | Finalization in progress |

## Documentation

- [Frontend setup and inference modes](frontend/README.md)
- [Static deployment](docs/STATIC_DEPLOYMENT.md)
- [Project plan](docs/PROJECT_PLAN.md) and [progress history](docs/PROGRESS.md)
- [Model card](docs/MODEL_CARD.md) and [official evaluation](docs/FINAL_EVALUATION_REPORT.md)
- [ONNX FP32](docs/ONNX_FP32_REPORT.md) and [ONNX INT8](docs/ONNX_INT8_REPORT.md)
- [Browser compatibility](docs/BROWSER_INFERENCE_COMPATIBILITY.md)
- [Docker](docs/DOCKER_DEPLOYMENT.md) and [HTTP benchmark](docs/HTTP_BENCHMARK_REPORT.md)

## Limitations and licensing

This project does not establish generalization to private repositories, other issue trackers, or non-English issues. Some labels are ambiguous; near-duplicate and repository leakage risks are not fully eliminated. Token budgeting can omit important context. Confidence is not calibrated. The model weights are hosted separately; see the [Hugging Face model card](https://huggingface.co/mazenalasas/issue-intent-onnx) for attribution and upstream-license discussion. **A repository code license has not yet been selected.** Do not assume the base model's license automatically covers all dataset-derived artifacts.
