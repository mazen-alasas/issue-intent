export type IssueLabel = 'bug' | 'documentation' | 'feature' | 'question'

export type PredictRequest = {
  title: string
  body: string
}

export type PredictResponse = {
  label: IssueLabel
  confidence: number
  probabilities: Record<IssueLabel, number>
}

const LABELS: IssueLabel[] = ['bug', 'documentation', 'feature', 'question']

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '')

// Production static demo uses browser inference; local development preserves the API workflow.
// Override with VITE_INFERENCE_BACKEND=api or browser when needed.
export const INFERENCE_BACKEND = import.meta.env.VITE_INFERENCE_BACKEND ||
  (import.meta.env.PROD ? 'browser' : 'api')

export class ApiError extends Error {
  readonly status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function isPredictResponse(value: unknown): value is PredictResponse {
  if (typeof value !== 'object' || value === null) return false

  const result = value as Record<string, unknown>
  const probabilities = result.probabilities

  if (
    !LABELS.includes(result.label as IssueLabel) ||
    typeof result.confidence !== 'number' ||
    !Number.isFinite(result.confidence) ||
    result.confidence < 0 ||
    result.confidence > 1 ||
    typeof probabilities !== 'object' ||
    probabilities === null
  ) {
    return false
  }

  const scores = probabilities as Record<string, unknown>

  return LABELS.every(
    (label) =>
      typeof scores[label] === 'number' &&
      Number.isFinite(scores[label]) &&
      (scores[label] as number) >= 0 &&
      (scores[label] as number) <= 1,
  )
}

export async function predictIssue(
  payload: PredictRequest,
  signal?: AbortSignal,
): Promise<PredictResponse> {
  if (INFERENCE_BACKEND === 'browser') {
    const { predictBrowserIssue } = await import('../../browser/inference')
    return predictBrowserIssue(payload, signal)
  }
  if (INFERENCE_BACKEND !== 'api') {
    throw new ApiError(`Unsupported inference backend: ${INFERENCE_BACKEND}`)
  }

  let response: Response

  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
      signal,
    })
  } catch (error) {
    if (signal?.aborted) {
      throw error
    }

    throw new ApiError('Cannot connect to the classification API. Please try again.')
  }

  if (!response.ok) {
    if (response.status === 422) {
      throw new ApiError('Invalid issue content. Check the title and description.', 422)
    }

    if (response.status === 503) {
      throw new ApiError('The classification model is currently unavailable.', 503)
    }

    if (response.status === 429) {
      throw new ApiError('Too many requests. Please try again later.', 429)
    }

    throw new ApiError(`Prediction failed (HTTP ${response.status}).`, response.status)
  }

  let data: unknown

  try {
    data = await response.json()
  } catch {
    throw new ApiError('The API returned an invalid JSON response.')
  }

  if (!isPredictResponse(data)) {
    throw new ApiError('The API returned an unexpected prediction format.')
  }

  return data
}
