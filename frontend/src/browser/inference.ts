import type { PredictRequest, PredictResponse, IssueLabel } from '../features/classification/api'
import { encodeBrowserIssue } from './encoding'

const LABELS: IssueLabel[] = ['bug', 'documentation', 'feature', 'question']
// Development uses a local model; production downloads the public quantized graph.
// Override with VITE_BROWSER_MODEL_URL for a pinned Hugging Face revision or another host.
const MODEL_URL =
  import.meta.env.VITE_BROWSER_MODEL_URL ||
  (import.meta.env.PROD
    ? 'https://huggingface.co/mazenalasas/issue-intent-onnx/resolve/main/model_int8.onnx'
    : '/models/model_int8.onnx')

type Session = import('onnxruntime-web/wasm').InferenceSession
let sessionPromise: Promise<Session> | undefined

async function getSession(): Promise<Session> {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      const ort = await import('onnxruntime-web/wasm')
      ort.env.wasm.numThreads = 1
      return ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      })
    })().catch((error: unknown) => {
      sessionPromise = undefined
      throw error
    })
  }
  return sessionPromise
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits)
  const exps = logits.map((value) => Math.exp(value - max))
  const sum = exps.reduce((acc, value) => acc + value, 0)
  return exps.map((value) => value / sum)
}

function checkAbort(signal?: AbortSignal) {
  if (signal?.aborted) {
    throw new DOMException('Prediction cancelled', 'AbortError')
  }
}

export async function predictBrowserIssue(
  payload: PredictRequest,
  signal?: AbortSignal,
): Promise<PredictResponse> {
  checkAbort(signal)
  // Load model and tokenizer only after the first Analyze click.
  const [session, encoding, ort] = await Promise.all([
    getSession(),
    encodeBrowserIssue(payload.title, payload.body),
    import('onnxruntime-web/wasm'),
  ])
  checkAbort(signal)

  const length = encoding.input_ids.length
  if (length !== encoding.attention_mask.length || length === 0) {
    throw new Error('Invalid browser tokenization output.')
  }
  const feeds = {
    input_ids: new ort.Tensor('int64', BigInt64Array.from(encoding.input_ids, BigInt), [1, length]),
    attention_mask: new ort.Tensor('int64', BigInt64Array.from(encoding.attention_mask, BigInt), [
      1,
      length,
    ]),
  }
  const outputs = await session.run(feeds)
  checkAbort(signal)
  const logitsTensor = outputs.logits
  if (!logitsTensor || !(logitsTensor.data instanceof Float32Array)) {
    throw new Error('The browser model did not return Float32 logits.')
  }
  const logits = Array.from(logitsTensor.data)
  if (logits.length !== LABELS.length || logits.some((value) => !Number.isFinite(value))) {
    throw new Error('The browser model returned invalid prediction scores.')
  }
  const values = softmax(logits)
  const bestIndex = values.indexOf(Math.max(...values))
  const probabilities = Object.fromEntries(
    LABELS.map((label, index) => [label, values[index]]),
  ) as Record<IssueLabel, number>
  return { label: LABELS[bestIndex], confidence: values[bestIndex], probabilities }
}
