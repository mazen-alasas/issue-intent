import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

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

function mockSuccessfulResponse() {
  return {
    ok: true,
    json: async () => prediction,
  }
}

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('App integration', () => {
  it('displays the prediction returned by the API', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn().mockResolvedValue(mockSuccessfulResponse())

    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await user.click(screen.getByRole('button', { name: /analyze issue/i }))

    expect(await screen.findAllByText('98.9%')).toHaveLength(2)
    expect(screen.getAllByText('Bug').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('Live inference')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledOnce()
  })

  it('shows loading while the prediction is pending', async () => {
    const user = userEvent.setup()

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation(() => new Promise(() => {})),
    )

    render(<App />)

    await user.click(screen.getByRole('button', { name: /analyze issue/i }))

    expect(screen.getByRole('status')).toHaveTextContent('Analyzing issue with DistilBERT')

    expect(screen.getByRole('button', { name: /analyzing/i })).toBeDisabled()
  })

  it('shows an error and allows retrying', async () => {
    const user = userEvent.setup()
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValueOnce(mockSuccessfulResponse())

    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await user.click(screen.getByRole('button', { name: /analyze issue/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Cannot connect to the classification API',
    )

    await user.click(screen.getByRole('button', { name: /analyze issue/i }))

    expect(await screen.findAllByText('98.9%')).toHaveLength(2)
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('clears the prediction when the issue changes', async () => {
    const user = userEvent.setup()

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(mockSuccessfulResponse()))

    render(<App />)

    await user.click(screen.getByRole('button', { name: /analyze issue/i }))

    expect(await screen.findAllByText('98.9%')).toHaveLength(2)

    await user.type(screen.getByRole('textbox', { name: /title/i }), ' updated')

    expect(screen.queryAllByText('98.9%')).toHaveLength(0)
    expect(screen.getByText('Ready to classify your issue.')).toBeInTheDocument()
  })

  it('ignores an outdated prediction after editing', async () => {
    const user = userEvent.setup()
    let resolveFetch!: (value: ReturnType<typeof mockSuccessfulResponse>) => void

    const fetchMock = vi.fn().mockImplementation(
      () =>
        new Promise<ReturnType<typeof mockSuccessfulResponse>>((resolve) => {
          resolveFetch = resolve
        }),
    )

    vi.stubGlobal('fetch', fetchMock)

    render(<App />)

    await user.click(screen.getByRole('button', { name: /analyze issue/i }))

    await user.type(screen.getByRole('textbox', { name: /title/i }), ' updated')

    await act(async () => {
      resolveFetch(mockSuccessfulResponse())
    })

    await waitFor(() => {
      expect(screen.getByText('Ready to classify your issue.')).toBeInTheDocument()
    })

    expect(screen.queryAllByText('98.9%')).toHaveLength(0)
  })
})
