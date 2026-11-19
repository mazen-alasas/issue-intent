# Static Portfolio Deployment — Issue Intent

**Status:** Local production preview verified on **November 19, 2026**. **No public Cloudflare Pages deployment has yet been confirmed.**

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

## Cloudflare Pages configuration (planned, not yet deployed)

After reviewing and merging `static-deployment` to `main`, connect the GitHub repository to Cloudflare Pages and configure:

| Setting | Value |
| --- | --- |
| Project root directory | `frontend` |
| Production branch | `main` |
| Framework | React (Vite) / None |
| Build command | `npm run build && node scripts/prepare-static-deploy.mjs` |
| Output directory | `dist` |
| Environment variable | `VITE_INFERENCE_BACKEND=browser` |

Check Cloudflare's UI if the build directory is specified relative to repository root rather than configured project root; in that case use `frontend/dist` as appropriate. Node.js must meet the project's Vite 8 engine requirements. Never add Hugging Face tokens for this public model.

Once published, verify using a brand-new browser session: site loads, JS/WASM assets load successfully, `model_int8.onnx` downloads via Hugging Face without CORS error, and `/predict` is **not** called for browser inference. Confirm prediction and probability display, then record the actual public URL and deployment date in `README.md` and `docs/PROGRESS.md`.

## Failure cases

- **404 tokenizer:** verify `frontend/public/models/tokenizer/tokenizer.json` and `tokenizer_config.json` are present in Git and copied into the output.
- **Model download/CORS failure:** inspect the actual network request and CDN redirects to the Hugging Face `/resolve/main/` URL. Do not use `/blob/main/` for direct graph download.
- **Oversized files rejected:** ensure the post-build preparation script actually runs on Cloudflare and model weights are not included in static output.
- **Slow first prediction:** expected for a large first download and WASM session startup; distinguish initial download from subsequent cached requests.
- **Server unavailable:** the browser demo should not depend on FastAPI. Leave `VITE_INFERENCE_BACKEND=browser` in Cloudflare configuration.

## Planned closure

After first successful public deployment, document the public link, CI build, fresh-browser smoke test and any observed cold-start behavior, then proceed to portfolio release and architecture diagrams. Do not mark these as verified until they have occurred.
