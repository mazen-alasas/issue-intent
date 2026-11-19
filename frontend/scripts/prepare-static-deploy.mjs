// Remove only known oversized development assets from Cloudflare Pages builds.
// Run AFTER `npm run build`: node scripts/prepare-static-deploy.mjs
import { readdir, rm, stat } from 'node:fs/promises'

const dist = new URL('../dist/', import.meta.url)
const limit = 25 * 1024 * 1024
const excluded = ['models/model_int8.onnx', 'browser-fixtures.json']

async function walk(dir, prefix = '') {
  const files = []
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const name = prefix ? `${prefix}/${entry.name}` : entry.name
    if (entry.isDirectory()) files.push(...(await walk(new URL(`${name}/`, dist), name)))
    else if (entry.isFile()) files.push(name)
  }
  return files
}

const before = await walk(dist)
for (const name of before) {
  if (
    excluded.includes(name) ||
    (name.startsWith('assets/') && /^ort-wasm.*asyncify.*\.wasm$/.test(name.split('/').at(-1)))
  ) {
    await rm(new URL(name, dist))
    console.log(`Excluded from static deploy: ${name}`)
  }
}
const files = await walk(dist)
if (
  !files.includes('models/tokenizer/tokenizer.json') ||
  !files.includes('models/tokenizer/tokenizer_config.json')
) {
  throw new Error(
    'Tokenizer assets missing from dist/models/tokenizer/. Copy them into frontend/public/models/tokenizer/ before building.',
  )
}
for (const name of files) {
  const size = (await stat(new URL(name, dist))).size
  if (size > limit)
    throw new Error(`File exceeds Cloudflare Pages 25 MiB limit: ${name} (${size} bytes)`)
}
console.log(`Static deploy validated: ${files.length} files, all <= 25 MiB`)
