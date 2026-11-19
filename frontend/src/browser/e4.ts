/** Browser equivalent of Python EXPERIMENTS["E4"] for issue bodies. */
function normalizeProse(text: string): string {
  text = text.replace(/<!--[\s\S]*?-->/g, ' ')
  text = text.replace(/^[ \t]*[-*+][ \t]+\[[ \t]*\][ \t]*(?:\r?\n|$)/gm, '')
  text = text.replace(/^[ \t]{0,3}#{1,6}[ \t]+/gm, '')
  text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
  text = text.replace(/\*\*([^*\n]+)\*\*/g, '$1')
  text = text.replace(/(^|[^\w])\*([^*\n]+)\*(?!\w)/g, '$1$2')
  text = text.replace(/`([^`\n]+)`/g, '$1')
  text = text.replace(/^[ \t]*[-*+][ \t]+/gm, '')
  // Process paths only outside URL spans, as in Python normalize_urls_paths.
  const urlPattern = /https?:\/\/[^\s<>"']+/g
  const replacePaths = (part: string) => part.replace(/(^|[^\w/])\/(?:[\w.-]+\/)+[\w.-]+/g, (_whole, prefix: string, offset: number, source: string) => {
    // Extract filename from matched path without the prefix.
    const path = source.slice(offset + prefix.length).match(/^\/(?:[\w.-]+\/)+[\w.-]+/)?.[0]
    return path ? `${prefix}PATH/${path.split('/').at(-1)}` : _whole
  })
  let result = ''
  let cursor = 0
  for (const match of text.matchAll(urlPattern)) {
    const start = match.index
    result += replacePaths(text.slice(cursor, start))
    const whole = match[0]
    const clean = whole.replace(/[.,;!]+$/, '')
    const suffix = whole.slice(clean.length)
    try {
      const parsed = new URL(clean)
      result += `${parsed.protocol}//${parsed.host}${parsed.pathname}${suffix}`
    } catch {
      result += whole
    }
    cursor = start + whole.length
  }
  result += replacePaths(text.slice(cursor))
  return result
}

function normalizeWhitespace(text: string): string {
  text = text.replace(/\r\n?/g, '\n')
  const endedWithNewline = text.endsWith('\n')
  text = text.split('\n').map((line) => line.replace(/[ \t]+$/g, '')).join('\n')
  text = text.replace(/\n{3,}/g, '\n\n').trim()
  return text && endedWithNewline ? `${text}\n` : text
}

export function preprocessE4(text: string): string {
  const lines = text.match(/[^\n]*\n|[^\n]+$/g) ?? []
  let prose = ''
  let result = ''
  let index = 0
  const opening = /^[ \t]{0,3}(`{3,}|~{3,})[^\n]*$/
  const closing = /^[ \t]{0,3}(`{3,}|~{3,})[ \t]*$/
  const flush = () => {
    result += normalizeProse(prose)
    prose = ''
  }
  while (index < lines.length) {
    const candidate = lines[index].replace(/[\r\n]+$/, '')
    const open = opening.exec(candidate)
    if (!open) {
      prose += lines[index++]
      continue
    }
    let end = -1
    for (let j = index + 1; j < lines.length; j++) {
      const close = closing.exec(lines[j].replace(/[\r\n]+$/, ''))
      if (close && close[1][0] === open[1][0] && close[1].length >= open[1].length) {
        end = j
        break
      }
    }
    if (end < 0) {
      prose += lines.slice(index).join('')
      break
    }
    flush()
    result += 'CODE\n'
    index = end + 1
  }
  flush()
  return normalizeWhitespace(result)
}
