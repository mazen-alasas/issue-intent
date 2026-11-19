# Static Portfolio Deployment — Issue Intent

**Status:** [Public Cloudflare Pages demo](https://issue-intent.pages.dev/) deployed and manually smoke-tested by the project owner on **November 19, 2026**. Local production preview, browser fixtures, and build-size checks were separately verified.

## Goal and architecture

A recruiter-friendly public portfolio demo with no always-on Python server:

1. React + TypeScript + Vite served from Cloudflare Pages (static assets).
2. Browser E4 structured preprocessing and compatible DistilBERT WordPiece tokenizer.
3. ONNX Runtime Web using WASM, with lazy model loading and session reuse.
4. INT8 ONNX graph fetched on first use from public [Hugging Face model repository](https://huggingface.co/mazenalasas/issue-intent-onnx) (`/resolve/main/model_int8.onnx`).
5. The existing prediction panel displays category, confidence, and four probabilities. FastAPI and the ONNX-only Docker image remain available as an independent server inference path in the repository.

There is no recurring model-serving bill in this architecture, but neither third-party free tiers nor uptime are guaranteed permanently. First use downloads the graph (~132 MiB) and may be slow; browser caching and reusing the same session improve subsequent requests. Actual cold-load time varies by browser/network/device.

## Verification recorded on 2026-11-19

- Browser ONNX Runtime Web compatibility: **3/3 reference examples matched expected classes**; small numeric differences from Python ONNX Runtime observed.
- Browser preprocessing and tokenization parity: **3/3** E4 / `input_ids` / `attention_mask` fixtures matched Python.
- Live prediction via the existing React UI was observed locally. Local production preview loaded the public Hugging Face graph and predicted `bug` on a sample issue.
- `npm run build`, `npm run lint`, `npm run test`: all passed (**19 frontend tests**). Python suite: **372 passed**.
- Build preparation excluded local `model_int8.onnx`, unused `ort-wasm-simd-threaded.asyncify*.wasm`, and browser fixtures from `dist/`. Result: **12 files**, each within **25 MiB**, for Cloudflare Pages compatibility.

Fixture parity is limited evidence; it does not prove numerical equivalence for every issue, full INT8 test accuracy, or public-site uptime.

## Repository and model files

The ONNX weights are hosted separately (do not commit into Git):

`https://huggingface.co/mazenalasas/issue-intent-onnx/resolve/main/model_int8.onnx`

Small tokenizer assets in `frontend/public/models/tokenizer/` are intended to be tracked in the repository. The original full model checkpoint, PyTorch weights, and training/test datasets remain external and must not be checked in. Confirm license/attribution notes in the Hugging Face model card; the base checkpoint is `distilbert/distilbert-base-uncased` (Apache 2.0), but this does not automatically license every training issue or the derived weights.

## Build and test locally

```bash
cd frontend
npm ci
npm run build
node scripts/prepare-static-deploy.mjs
npm run lint
npm run test
npm run preview -- --host 127.0.0.1
```

Open `http://127.0.0.1:4173/` and test a prediction. Inspect DevTools **Network** to verify the ONNX graph is fetched from Hugging Face, not `localhost`, and that tokenizer assets load from `/models/tokenizer/`. Test once in a clean profile/incognito window to avoid relying on a previous model cache.

## Cloudflare Pages configuration (deployed)

The `static-deployment` branch was merged into `main` via PR #10 (`735dcbd`). The production deployment uses these settings:

| Setting | Value |
| --- | --- |
| Project root directory | `frontend` |
| Production branch | `main` |
| Framework | React (Vite) |
| Build command | `npm run build && node scripts/prepare-static-deploy.mjs` |
| Output directory | `dist` |
| Inference backend | Browser mode is the built-in production default; `VITE_INFERENCE_BACKEND=browser` may be set explicitly |

The output directory is relative to the configured `frontend` root. Node.js must meet the Vite version's engine requirements. No Hugging Face token is required for the public model.

**Observed manual public check (November 19, 2026):** the project owner opened the deployed site and classified multiple existing GitHub issues, including external-repository issues; the predicted labels agreed with the issues' existing labels in those examples. This confirms working end-to-end predictions in the tested browser but is **not** a controlled metric, automated cross-browser test, or uptime audit.

**Recommended release smoke check:** revisit from a fresh browser session; confirm the JS/WASM and tokenizer assets load, model weights resolve from Hugging Face, no `/predict` API call is required, and both predicted label and probability breakdown render. Cold-load timings and broad cross-browser behavior have not been measured.

## Failure cases

- **404 tokenizer:** verify `frontend/public/models/tokenizer/tokenizer.json` and `tokenizer_config.json` are present in Git and copied into the output.
- **Model download/CORS failure:** inspect the actual network request and CDN redirects to the Hugging Face `/resolve/main/` URL. Do not use `/blob/main/` for direct graph download.
- **Oversized files rejected:** ensure the post-build preparation script actually runs on Cloudflare and model weights are not included in static output.
- **Slow first prediction:** expected for a large first download and WASM session startup; distinguish initial download from subsequent cached requests.
- **Server unavailable:** the browser demo should not depend on FastAPI. Leave `VITE_INFERENCE_BACKEND=browser` in Cloudflare configuration.

## Remaining release tasks

Complete repository cleanup, updated CI coverage, and portfolio/architecture documentation. Keep release checks distinct from the already completed first public deployment; do not claim unmeasured cross-browser support, long-term availability, or new accuracy metrics.
