# Browser ONNX compatibility and tokenizer parity

**Status (November 19, 2026):** Both browser compatibility gates passed on three reference fixtures each, and browser inference is integrated into the existing React interface. This is **not** a full INT8 official-test evaluation or proof of cross-browser compatibility.

## What was tested

1. **Graph compatibility:** the exported INT8 ONNX graph executed with ONNX Runtime Web (WASM) using Python-generated `input_ids` and `attention_mask`. This isolates runtime/operator compatibility from browser preprocessing.
2. **Preprocessing and tokenizer parity:** browser E4 processing, token IDs and attention masks were compared against Python-generated fixtures.
3. **Integrated UI:** the existing issue editor and prediction panel produced a real prediction locally, including in a production preview with the model fetched from Hugging Face rather than bundled into the static site.

| Check | Observed result |
| --- | --- |
| Browser ONNX graph compatibility | 3/3 labels passed |
| Browser E4 preprocessing | 3/3 fixtures passed |
| Browser `input_ids` parity | 3/3 fixtures passed |
| Browser `attention_mask` parity | 3/3 fixtures passed |
| Local production preview | Successful prediction without FastAPI |

The graph fixture checks observed small logit/probability differences from Python ONNX Runtime; label agreement on three examples is not proof of universal numerical parity. These tests used representative cases including an issue with Markdown/code content.

## Local compatibility harnesses

From the repository root, Python fixtures can be generated using the checked-in script and a locally available model checkpoint:

```bash
uv run python -m scripts.export_browser_onnx_fixtures
```

For the **local graph compatibility harness only**, the ONNX file can be copied into `frontend/public/models/model_int8.onnx` (do **not** commit it). From `frontend/`, run:

```bash
npm ci
npm run dev -- --host 127.0.0.1
```

Open `/browser-smoke.html` for graph execution and `/browser-parity.html` for E4/tokenizer parity. The static production demo instead fetches the graph from the public [Hugging Face model repository](https://huggingface.co/mazenalasas/issue-intent-onnx). Browser fixtures and local model weights are excluded from production output by the static preparation script.

## Repository hygiene and limitations

- **Never commit** `frontend/public/models/model_int8.onnx`, full checkpoints, or datasets.
- **Do track** the small compatible tokenizer assets under `frontend/public/models/tokenizer/` for the public static site.
- Do not confuse `/blob/main/model_int8.onnx` with the direct-download `/resolve/main/model_int8.onnx` URL.
- First-load time, model RAM use, CORS, network and device capabilities vary. The observed fast repeated predictions benefit from cache/session reuse.
- The owner verified the public [Cloudflare Pages demo](https://issue-intent.pages.dev/) with multiple real GitHub issue predictions on November 19, 2026. This does not establish automated cross-browser compatibility, uncached cold-load timing, or full INT8 accuracy. See [Static Deployment](STATIC_DEPLOYMENT.md).
