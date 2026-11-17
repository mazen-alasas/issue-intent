import { afterEach, describe, expect, it, vi } from 'vitest'
import { predictIssue } from './api'

const prediction = {
  label: 'bug',
  confidence: 0.989,
  probabilities: {
    bug: 0.989,
    documentation: 0.001,
    feature: 0.003,
    question: 0.007,
  },
}

const issue = {
  title: 'Application crashes on startup',
  body: 'RuntimeError during initialization',
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('predictIssue', () => {
  it('sends the issue to the prediction endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => prediction,
    })

    vi.stubGlobal('fetch', fetchMock)

    const result = await predictIssue(issue)

    expect(result).toEqual(prediction)
    expect(fetchMock).toHaveBeenCalledOnce()

    const [url, options] = fetchMock.mock.calls[0]

    expect(url).toMatch(/\/predict$/)
    expect(options.method).toBe('POST')
    expect(options.headers).toEqual({
      'Content-Type': 'application/json',
    })
    expect(JSON.parse(options.body)).toEqual(issue)
  })

  it('handles validation errors', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 422,
      }),
    )

    await expect(predictIssue(issue)).rejects.toThrow('Invalid issue content')
  })

  it('handles unavailable models', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
      }),
    )

    await expect(predictIssue(issue)).rejects.toThrow(
      'classification model is currently unavailable',
    )
  })

  it('handles network failures', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')))

    await expect(predictIssue(issue)).rejects.toThrow('Cannot connect to the classification API')
  })

  it('rejects invalid API responses', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          label: 'unknown',
          confidence: 0.9,
          probabilities: {},
        }),
      }),
    )

    await expect(predictIssue(issue)).rejects.toThrow('unexpected prediction format')
  })

  it('passes an AbortSignal to fetch', async () => {
    const controller = new AbortController()
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => prediction,
    })

    vi.stubGlobal('fetch', fetchMock)

    await predictIssue(issue, controller.signal)

    expect(fetchMock.mock.calls[0][1].signal).toBe(controller.signal)
  })
})
