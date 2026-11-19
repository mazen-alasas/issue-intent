import * as ort from 'onnxruntime-web'

type Fixture = {
  name: string
  input_ids: number[]
  attention_mask: number[]
  reference_logits: number[]
  reference_probabilities: number[]
  reference_label: string
}

type FixtureDocument = {
  labels: string[]
  fixtures: Fixture[]
}

const runButton = document.getElementById('run') as HTMLButtonElement
const output = document.getElementById('output') as HTMLPreElement

function log(message: string) {
  output.textContent += `${message}\n`
}

function softmax(values: number[]): number[] {
  const largest = Math.max(...values)
  const exp = values.map((value) => Math.exp(value - largest))
  const denominator = exp.reduce((total, value) => total + value, 0)
  return exp.map((value) => value / denominator)
}

async function run() {
  runButton.disabled = true
  output.textContent = ''

  try {
    ort.env.wasm.numThreads = 1
    log('Loading Python-generated fixtures…')
    const fixturesResponse = await fetch('/browser-fixtures.json')
    if (!fixturesResponse.ok) throw new Error(`Fixtures HTTP ${fixturesResponse.status}`)
    const data = (await fixturesResponse.json()) as FixtureDocument
    if (!Array.isArray(data.fixtures) || !Array.isArray(data.labels)) {
      throw new Error('Invalid fixture format')
    }

    log('Loading model_int8.onnx with WASM… (first load may take time)')
    const started = performance.now()
    const session = await ort.InferenceSession.create('/models/model_int8.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
    log(`Model loaded in ${Math.round(performance.now() - started)} ms`)
    log(`Model inputs: ${session.inputNames.join(', ')}`)
    log(`Model outputs: ${session.outputNames.join(', ')}`)

    let passing = 0
    for (const fixture of data.fixtures) {
      const length = fixture.input_ids.length
      if (length !== fixture.attention_mask.length) throw new Error('Mask length mismatch')
      const feeds = {
        input_ids: new ort.Tensor(
          'int64',
          BigInt64Array.from(fixture.input_ids, (id) => BigInt(id)),
          [1, length],
        ),
        attention_mask: new ort.Tensor(
          'int64',
          BigInt64Array.from(fixture.attention_mask, (id) => BigInt(id)),
          [1, length],
        ),
      }
      const start = performance.now()
      const result = await session.run(feeds)
      const tensor = result.logits
      if (!tensor) throw new Error('Missing logits output')
      if (tensor.type !== 'float32') {
        throw new Error(`Unexpected logits type: ${tensor.type}`)
      }
      const logits: number[] = Array.from(tensor.data as Float32Array)
      if (logits.length !== data.labels.length) throw new Error('Unexpected logits shape')
      const probabilities = softmax(logits)
      const maximumIndex = probabilities.indexOf(Math.max(...probabilities))
      const label = data.labels[maximumIndex]
      const maxLogitDelta = Math.max(
        ...logits.map((value, index) => Math.abs(value - fixture.reference_logits[index])),
      )
      const maxProbabilityDelta = Math.max(
        ...probabilities.map((value, index) =>
          Math.abs(value - fixture.reference_probabilities[index]),
        ),
      )
      const labelMatches = label === fixture.reference_label
      const numericalMatch = maxProbabilityDelta <= 0.01
      const pass = labelMatches && numericalMatch
      if (pass) passing += 1
      log(
        `${pass ? 'PASS' : 'FAIL'} ${fixture.name}: ${label} (expected ${fixture.reference_label}), ` +
          `max |Δlogit|=${maxLogitDelta.toFixed(5)}, ` +
          `max |Δprob|=${maxProbabilityDelta.toFixed(5)}, ` +
          `${(performance.now() - start).toFixed(1)} ms`,
      )
    }
    log(`\nResult: ${passing}/${data.fixtures.length} passed`)
    if (passing !== data.fixtures.length) log('Do not proceed with browser tokenization until differences are investigated.')
  } catch (error) {
    log(`ERROR: ${error instanceof Error ? error.stack ?? error.message : String(error)}`)
  } finally {
    runButton.disabled = false
  }
}

runButton.addEventListener('click', () => {
  void run()
})
