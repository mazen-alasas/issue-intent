# issue-intent Frontend

React + TypeScript + Vite web interface for the `issue-intent` GitHub Issue Type Classifier.

The frontend provides a GitHub-inspired dark New Issue experience. Visitors can write a title and description, preview GitHub-flavored Markdown, and classify the issue using the existing FastAPI/DistilBERT backend. The browser currently **does not run the model locally**.

## Features

- Responsive dark layout and GitHub-inspired issue editor.
- Title and body inputs with client-side validation.
- Write and Preview tabs, Markdown editing toolbar, and GitHub-flavored Markdown output.
- Typed POST `/predict` requests to a configurable API base URL.
- Prediction label, confidence, and probabilities for `bug`, `documentation`, `feature`, and `question`.
- Loading, success, error, and input-validation feedback.
- Cancellation of superseded requests so stale responses do not replace newer predictions.

The application **does not post issues to GitHub** and requires no GitHub account. Model confidence is **not calibrated** and should not be interpreted as a guarantee of correctness.

## Prerequisites

- Node.js and npm installed locally (Node.js 26.13.1 was used during development).
- Running `issue-intent` FastAPI backend and access to the selected local DistilBERT checkpoint.
- The Python development environment managed with `uv`; see the [root README](../README.md).

## Start the Backend

From the repository root, start the API in a separate terminal:

```bash
cd ~/issue-intent
ISSUE_INTENT_CHECKPOINT="$HOME/issue-intent-artifacts/models/distilbert/best" \
  uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

Check readiness:

```bash
curl http://127.0.0.1:8000/health
```

The checkpoint directory shown above is specific to the original development machine. Model weights are not committed to Git. The API loads the checkpoint at startup, so a missing checkpoint prevents successful startup.

## Install and Run the Frontend

In another terminal:

```bash
cd ~/issue-intent/frontend
npm ci
cp -n .env.example .env.local
npm run dev
```

Open the URL printed by Vite, usually `http://localhost:5173`.

`frontend/.env.example` provides the local API configuration:

```dotenv
VITE_API_BASE_URL=http://localhost:8000
```

Override `VITE_API_BASE_URL` in `.env.local` if the backend runs at a different location. Restart Vite after editing the environment file. Do not put secrets in `VITE_*` environment variables: Vite embeds these values in browser-delivered code.

## API Contract

`POST /predict` accepts JSON:

```json
{
  "title": "Application crashes on startup",
  "body": "The application raises a RuntimeError during initialization."
}
```

The backend enforces a 500-character title limit, a 20,000-character body limit, and rejects requests in which both fields are blank. Successful responses have this shape:

```json
{
  "label": "bug",
  "confidence": 0.9969,
  "probabilities": {
    "bug": 0.9969,
    "documentation": 0.0001,
    "feature": 0.0018,
    "question": 0.0012
  }
}
```

The response shown is an **illustrative rounded example**, not a fixed output contract for every input. The UI displays values returned by the model; it does not generate or calibrate its own probabilities.

## Scripts and Quality Checks

```bash
npm run format
npm run format:check
npm run lint
npm run test
npm run build
```

- `format` applies Prettier to the frontend.
- `format:check` verifies formatting without changing files.
- `lint` runs ESLint.
- `test` runs the configured Vitest test suite.
- `build` runs the production Vite/TypeScript build.

**Last reported result (2026-11-17):** 19 frontend tests passed across three test files; formatting, linting, and build passed. Rerun the commands before merging or releasing. A local browser-to-FastAPI test also produced a successful DistilBERT prediction.

## Key Source Files

| Responsibility | Path |
|---|---|
| Application orchestration | `src/App.tsx` |
| UI and layout styling | `src/App.css`, `src/styles/theme.css`, `src/index.css` |
| Shared layout | `src/components/layout/` |
| Markdown issue editor | `src/features/issue-editor/` |
| Prediction display and API client | `src/features/classification/` |
| Application tests | `src/App.test.tsx` |
| API client tests | `src/features/classification/api.test.ts` |
| Editor tests | `src/features/issue-editor/IssueEditor.test.tsx` |

## Troubleshooting

- **Network error:** Confirm FastAPI is listening on the configured host/port and the `VITE_API_BASE_URL` matches it.
- **CORS or private-network preflight error:** Check that the API allows the Vite origin; `ISSUE_INTENT_CORS_ORIGINS` is used by the backend. Default local origin: `http://localhost:5173`.
- **503 / unavailable model:** Check API startup logs and the `ISSUE_INTENT_CHECKPOINT` directory.
- **422 validation error:** Ensure at least one nonblank field and stay within the field limits.
- **Vite environment changes not reflected:** Restart the development server after modifying `.env.local`.

## Deployment Status and Next Steps

The current application has been validated locally. There is **no verified public deployment** and no completed browser-side model inference. A future separate branch will assess FP32 ONNX export, parity validation, INT8 quantization, porting preprocessing and tokenization to JavaScript, and static web hosting. The tested FastAPI and Docker implementations remain available independently of that planned work.
