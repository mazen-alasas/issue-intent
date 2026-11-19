# Browser inference: first compatibility gate

The purpose of this test is to establish that **the existing INT8 ONNX graph**, without conversion or retraining, can execute with ONNX Runtime Web (WASM). The Python exporter creates three realistic issue inputs, with the exact E4 preprocessing and tokenizer ID budgeting already used by the ONNX CPU API. The browser executes those same input IDs to isolate model-graph compatibility from the JavaScript preprocessing/tokenization port.

## Local workflow

1. Unpack the delivered files at the repository root.
2. From `frontend/`, run `npm install onnxruntime-web` to add the browser runtime dependency, then `npm run build`, `npm run test`, and `npm run lint`.
3. From the repository root, run `uv run python scripts/export_browser_onnx_fixtures.py`. This writes `frontend/public/browser-fixtures.json` (no model weights).
4. Copy (do not commit) `$HOME/issue-intent-artifacts/models/distilbert/onnx/model_int8.onnx` to `frontend/public/models/model_int8.onnx`.
5. From `frontend/`, run `npm run dev -- --host 127.0.0.1`, then visit `http://127.0.0.1:5173/browser-smoke.html` and click **Run compatibility test**.
6. Expect 3/3 PASS; send the complete browser log if an operator fails or scores drift.

If graph execution works, the **next stage** is porting E4 preprocessing and tokenization, checking Python↔JavaScript token-ID parity, and integrating the result with the existing `predictIssue` API interface (keeping the current FastAPI path for local development). After that, host the website on free static hosting and model artifacts on a static file host supporting ~132 MiB objects and cross-origin downloads.

## Security and repository hygiene

- Never commit the ONNX model, checkpoints, or `frontend/public/models/` directory.
- Remove `frontend/public/models/model_int8.onnx` before publishing or building a public artifact; the model must eventually be downloaded from a separately hosted artifact URL rather than embedded in Cloudflare Pages assets.
- Browser test fixtures are reproducible and safe to version if desired; the model is not.
- This test does **not** prove cross-language preprocessing/tokenization parity, cross-browser compatibility, or final deployment performance.
- Model download time, browser RAM, browser support for quantized operators, CORS, and caching remain to be validated.
