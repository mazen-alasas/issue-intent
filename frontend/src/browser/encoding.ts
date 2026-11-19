import { AutoTokenizer, env } from '@huggingface/transformers'
import { preprocessE4 } from './e4'

export type IssueEncoding = { input_ids: number[]; attention_mask: number[]; processed_body: string }

type Tokenizer = Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>
let cachedTokenizer: Promise<Tokenizer> | undefined

async function loadTokenizer(): Promise<Tokenizer> {
  // In browsers, Transformers.js disables local model loading by default.
  // The tokenizer is served by Vite from frontend/public/models/tokenizer/.
  env.allowLocalModels = true
  env.localModelPath = '/models/'
  env.allowRemoteModels = false

  const configUrl = '/models/tokenizer/tokenizer_config.json'
  const configResponse = await fetch(configUrl)
  if (!configResponse.ok) {
    throw new Error(
      `Tokenizer configuration unavailable (${configResponse.status}) at ${configUrl}. ` +
      'Copy tokenizer_config.json and tokenizer.json into frontend/public/models/tokenizer/.'
    )
  }

  return AutoTokenizer.from_pretrained('tokenizer', { local_files_only: true })
}

export function getBrowserTokenizer(): Promise<Tokenizer> {
  if (!cachedTokenizer) {
    cachedTokenizer = loadTokenizer().catch((error: unknown) => {
      cachedTokenizer = undefined
      throw error
    })
  }
  return cachedTokenizer
}

function idsForText(tokenizer: Tokenizer, text: string): number[] {
  const output = tokenizer(text, { add_special_tokens: false, truncation: false, padding: false })
  return Array.from(output.input_ids.data, (id) => Number(id))
}

export async function encodeBrowserIssue(title: string, body: string): Promise<IssueEncoding> {
  const tokenizer = await getBrowserTokenizer()
  const processed_body = preprocessE4(body)
  const titleIds = idsForText(tokenizer, title).slice(0, 48)
  let bodyIds = idsForText(tokenizer, processed_body)
  const available = 256 - titleIds.length - 3
  if (bodyIds.length > available) {
    const head = Math.floor((available * 3) / 4)
    const tail = available - head
    bodyIds = tail > 0 ? [...bodyIds.slice(0, head), ...bodyIds.slice(-tail)] : bodyIds.slice(0, head)
  }
  // IDs [CLS]=101 and [SEP]=102 for the evaluated uncased DistilBERT tokenizer.
  const input_ids = [101, ...titleIds, 102, ...bodyIds, 102]
  return { input_ids, attention_mask: input_ids.map(() => 1), processed_body }
}
