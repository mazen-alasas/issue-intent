import { encodeBrowserIssue } from './browser/encoding'

type Fixture = {
  name: string
  title: string
  body: string
  processed_body: string
  input_ids: number[]
  attention_mask: number[]
}
const button = document.querySelector<HTMLButtonElement>('#run')!
const output = document.querySelector<HTMLPreElement>('#output')!
function log(s: string) { output.textContent += `${s}\n` }
function mismatch(expected: number[], actual: number[]): string {
  if (expected.length !== actual.length) return `length ${actual.length} vs expected ${expected.length}`
  const first = expected.findIndex((id, i) => id !== actual[i])
  return first === -1 ? 'none' : `index ${first}: ${actual[first]} vs expected ${expected[first]}`
}
async function run() {
  button.disabled = true
  output.textContent = ''
  try {
    const response = await fetch('/browser-fixtures.json')
    if (!response.ok) throw new Error(`Fixture fetch failed: ${response.status}`)
    const document = await response.json() as { fixtures: Fixture[] }
    let passing = 0
    for (const fixture of document.fixtures) {
      const actual = await encodeBrowserIssue(fixture.title, fixture.body)
      const bodyMatch = actual.processed_body === fixture.processed_body
      const idsMatch = mismatch(fixture.input_ids, actual.input_ids) === 'none'
      const maskMatch = mismatch(fixture.attention_mask, actual.attention_mask) === 'none'
      const pass = bodyMatch && idsMatch && maskMatch
      if (pass) passing++
      log(`${pass ? 'PASS' : 'FAIL'} ${fixture.name}: E4=${bodyMatch}, IDs=${idsMatch}, Mask=${maskMatch}`)
      if (!idsMatch) log(`  first ID mismatch: ${mismatch(fixture.input_ids, actual.input_ids)}`)
      if (!bodyMatch) log(`  E4 actual: ${JSON.stringify(actual.processed_body)}\n  E4 python: ${JSON.stringify(fixture.processed_body)}`)
    }
    log(`\nResult: ${passing}/${document.fixtures.length} passed`)
  } catch (error) {
    log(`ERROR: ${error instanceof Error ? error.stack || error.message : String(error)}`)
  } finally { button.disabled = false }
}
button.addEventListener('click', () => { void run() })
