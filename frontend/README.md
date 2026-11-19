# issue-intent frontend

React + TypeScript + Vite application with a GitHub-inspired issue editor, Markdown preview, and a four-class DistilBERT prediction panel.

**Status (2026-11-19):** [Public Cloudflare Pages demo](https://issue-intent.pages.dev/) is live and was manually tested on real external GitHub issues by the project owner. The browser loads the ONNX INT8 graph from [Hugging Face](https://huggingface.co/mazenalasas/issue-intent-onnx), without a running Python server. This manual check is not a formal accuracy or cross-browser evaluation. FastAPI remains an optional alternative.

## Features

- GitHub-inspired dark issue editor with title and Markdown description, Write/Preview tabs and formatting toolbar.
- Client-side limits: title up to **500** characters, body up to **20,000** characters, at least one nonblank field.
- Predicted label, softmax confidence, and probabilities for `bug`, `documentation`, `feature`, `question`.
- Loading/error states and cancellation of superseded requests.
- **Browser mode:** E4 preprocessing, WordPiece tokenizer, 256-token limit / 48-token title budget, ONNX Runtime Web (WASM), lazy model loading and session reuse.
- **API mode:** typed `POST /predict` client, configurable API base URL, independent FastAPI serving.

The app does **not** authenticate with GitHub or post issues. Model confidence is not calibrated.

## Install and run in browser mode

```bash
cd frontend
npm ci
npm run dev
```

Set `VITE_INFERENCE_BACKEND=browser` in your local Vite environment if browser mode is not already selected. The model is fetched from Hugging Face for the production configuration; the browser also needs the tokenizer files under `public/models/tokenizer/`. Do not store credentials in `VITE_*` variables: Vite exposes them to the browser.

First inference downloads approximately **132 MiB** of model weights and initializes WASM. Later predictions can be much faster due to cache/session reuse. A slow first request is not necessarily an error.

## Production-equivalent preview

```bash
cd frontend
npm run build
node scripts/prepare-static-deploy.mjs
npm run lint
npm run test
npm run preview -- --host 127.0.0.1
```

Open `http://127.0.0.1:4173/` and test **Analyze issue**. In browser DevTools, confirm the model downloads from Hugging Face (`/resolve/main/model_int8.onnx`) and tokenizer files load from the site. The preparation script removes local ONNX weights, browser fixtures and an unused oversized WASM asset from `dist/`, then checks the static host's 25 MiB-per-file limit. The reported November 19 run retained **12 files**, all within that limit.

For Cloudflare Pages settings, Git hygiene, and public-site checks, see [Static Deployment](../docs/STATIC_DEPLOYMENT.md).

## Optional FastAPI mode

Set `VITE_INFERENCE_BACKEND=api` and `VITE_API_BASE_URL=http://localhost:8000` in `.env.local`, then run the backend from the repository root with a local checkpoint:

```bash
ISSUE_INTENT_CHECKPOINT="$HOME/issue-intent-artifacts/models/distilbert/best" \
  uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

In another terminal run `npm run dev`. The API implements `GET /health` and `POST /predict`. A request contains `title` and `body`; the response contains `label`, `confidence`, and a `probabilities` object for the four classes. The checkpoint path is machine-specific and is **not** included in Git. For ONNX-only Docker deployment see [Docker guide](../docs/DOCKER_DEPLOYMENT.md).

## Quality checks

```bash
npm run format
npm run format:check
npm run lint
npm run test
npm run build
```

On November 19, the Vite build and ESLint passed and **19 Vitest tests** passed across three files. This test count does not include the manual browser fixture parity runs.

## Code map

| Responsibility                           | Path                                              |
| ---------------------------------------- | ------------------------------------------------- |
| Main application                         | `src/App.tsx`                                     |
| UI and prediction display                | `src/features/classification/PredictionPanel.tsx` |
| Inference selection and API contract     | `src/features/classification/api.ts`              |
| Browser inference                        | `src/browser/inference.ts`                        |
| E4 preprocessing                         | `src/browser/e4.ts`                               |
| Tokenization and input encoding          | `src/browser/encoding.ts`                         |
| Browser compatibility / parity harnesses | `src/browser-smoke.ts`, `src/browser-parity.ts`   |
| Markdown editor                          | `src/features/issue-editor/`                      |
| Static build cleanup                     | `scripts/prepare-static-deploy.mjs`               |

## Troubleshooting

- **First prediction slow:** the graph must download and the ONNX WASM session must initialize; compare first and subsequent predictions.
- **Model download blocked:** inspect Hugging Face redirects and CORS in DevTools; use `/resolve/main/`, not `/blob/main/`.
- **Tokenizer 404:** check `public/models/tokenizer/tokenizer.json`, `tokenizer_config.json` and static output.
- **Cloudflare oversized asset:** run the post-build preparation script; never upload the local ONNX graph in `dist/`.
- **API mode fails:** check the backend health endpoint, `VITE_API_BASE_URL`, CORS and checkpoint path.

## Validation scope

Three browser ONNX fixtures matched expected labels; three browser E4/IDs/masks fixtures matched Python. A local production preview successfully predicted from the Hugging Face-hosted graph without FastAPI. These are representative checks, not full INT8 official-test evaluation or cross-browser performance guarantees. The public Cloudflare Pages deployment was successfully smoke-tested by the owner on November 19, 2026.
