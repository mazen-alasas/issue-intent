import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type Props = {
  content: string
}

export function MarkdownPreview({ content }: Props) {
  if (!content.trim()) {
    return <div className="markdown-preview markdown-empty">Nothing to preview</div>
  }

  return (
    <div className="markdown-preview markdown-body">
      <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>
        {content}
      </ReactMarkdown>
    </div>
  )
}
