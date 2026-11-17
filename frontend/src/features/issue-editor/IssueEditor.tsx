import {
  AtSign,
  Bold,
  CheckSquare,
  Code2,
  Heading,
  Italic,
  Link2,
  List,
  ListOrdered,
  Maximize2,
  Minimize2,
  Strikethrough,
} from 'lucide-react'
import { useRef, useState } from 'react'
import { MarkdownPreview } from './MarkdownPreview'

type Props = {
  title: string
  body: string
  onTitleChange: (value: string) => void
  onBodyChange: (value: string) => void
  onDemo: () => void
}

type EditorTab = 'write' | 'preview'

const toolbar = [
  { name: 'Heading', Icon: Heading },
  { name: 'Bold', Icon: Bold },
  { name: 'Italic', Icon: Italic },
  { name: 'Strikethrough', Icon: Strikethrough },
  { name: 'Code', Icon: Code2 },
  { name: 'Link', Icon: Link2 },
  { name: 'Bulleted list', Icon: List },
  { name: 'Numbered list', Icon: ListOrdered },
  { name: 'Task list', Icon: CheckSquare },
  { name: 'Mention', Icon: AtSign },
]

export function IssueEditor({ title, body, onTitleChange, onBodyChange, onDemo }: Props) {
  const [activeTab, setActiveTab] = useState<EditorTab>('write')
  const [expanded, setExpanded] = useState(false)
  const bodyRef = useRef<HTMLTextAreaElement>(null)

  function switchTab(tab: EditorTab) {
    setActiveTab(tab)

    if (tab === 'write') {
      requestAnimationFrame(() => bodyRef.current?.focus())
    }
  }

  function insertMarkdown(name: string) {
    const element = bodyRef.current
    if (!element || activeTab !== 'write') return

    const start = element.selectionStart
    const end = element.selectionEnd
    const selected = body.slice(start, end)

    const wrappers: Record<string, [string, string, string]> = {
      Bold: ['**', '**', 'bold text'],
      Italic: ['*', '*', 'italic text'],
      Strikethrough: ['~~', '~~', 'strikethrough text'],
      Code: ['`', '`', 'code'],
      Link: ['[', '](https://example.com)', 'link text'],
      Mention: ['@', '', 'username'],
    }

    if (name in wrappers) {
      const [prefix, suffix, placeholder] = wrappers[name]
      const content = selected || placeholder

      const replacement = prefix + content + suffix

      onBodyChange(body.slice(0, start) + replacement + body.slice(end))

      requestAnimationFrame(() => {
        element.focus()
        element.setSelectionRange(start + prefix.length, start + prefix.length + content.length)
      })

      return
    } else if (
      name === 'Heading' ||
      name === 'Bulleted list' ||
      name === 'Numbered list' ||
      name === 'Task list'
    ) {
      // Line-based formatting should operate on complete lines.
      const lineStart = body.lastIndexOf('\n', start - 1) + 1
      const lineEndIndex = body.indexOf('\n', end)
      const lineEnd = lineEndIndex === -1 ? body.length : lineEndIndex
      const block = body.slice(lineStart, lineEnd)
      const lines = block.split('\n')

      const formatted = lines.map((line, index) => {
        if (name === 'Heading') return `### ${line}`
        if (name === 'Bulleted list') return `- ${line}`
        if (name === 'Task list') return `- [ ] ${line}`
        return `${index + 1}. ${line}`
      })

      const updatedBlock = formatted.join('\n')

      onBodyChange(body.slice(0, lineStart) + updatedBlock + body.slice(lineEnd))

      requestAnimationFrame(() => {
        element.focus()
        element.setSelectionRange(lineStart, lineStart + updatedBlock.length)
      })

      return
    } else {
      return
    }
  }

  return (
    <section
      className={`issue-card ${expanded ? 'issue-card-expanded' : ''}`}
      aria-labelledby="issue-editor-heading"
    >
      <h2 id="issue-editor-heading" className="visually-hidden">
        Issue editor
      </h2>

      <label className="field-label" htmlFor="issue-title">
        Title <span className="required">*</span>
      </label>

      <input
        id="issue-title"
        className="title-input"
        value={title}
        onChange={(event) => onTitleChange(event.target.value)}
        placeholder="Title"
        maxLength={300}
        required
      />

      <label className="field-label description-label" htmlFor="issue-description">
        Description
      </label>

      <div className="markdown-box">
        <div className="editor-toolbar">
          <div className="editor-tabs" role="tablist" aria-label="Editor mode">
            <button
              type="button"
              role="tab"
              id="write-tab"
              aria-selected={activeTab === 'write'}
              aria-controls="write-panel"
              className={`editor-tab ${activeTab === 'write' ? 'selected' : 'inactive'}`}
              onClick={() => switchTab('write')}
            >
              Write
            </button>

            <button
              type="button"
              role="tab"
              id="preview-tab"
              aria-selected={activeTab === 'preview'}
              aria-controls="preview-panel"
              className={`editor-tab ${activeTab === 'preview' ? 'selected' : 'inactive'}`}
              onClick={() => switchTab('preview')}
            >
              Preview
            </button>
          </div>

          <div className="format-tools" aria-label="Markdown formatting tools">
            {activeTab === 'write' &&
              toolbar.map(({ name, Icon }, index) => (
                <button
                  key={name}
                  type="button"
                  className={`tool-button ${index === 6 || index === 9 ? 'tool-divider' : ''}`}
                  title={name}
                  aria-label={name}
                  onClick={() => insertMarkdown(name)}
                >
                  <Icon size={17} />
                </button>
              ))}

            <button
              type="button"
              className="tool-button expand-button"
              aria-label={expanded ? 'Collapse editor' : 'Expand editor'}
              title={expanded ? 'Collapse editor' : 'Expand editor'}
              aria-pressed={expanded}
              onClick={() => setExpanded((value) => !value)}
            >
              {expanded ? <Minimize2 size={17} /> : <Maximize2 size={17} />}
            </button>
          </div>
        </div>

        <div
          id="write-panel"
          role="tabpanel"
          aria-labelledby="write-tab"
          hidden={activeTab !== 'write'}
        >
          <textarea
            ref={bodyRef}
            id="issue-description"
            className="description-input"
            value={body}
            onChange={(event) => onBodyChange(event.target.value)}
            spellCheck={false}
            aria-label="Issue description in Markdown"
          />
        </div>

        <div
          id="preview-panel"
          role="tabpanel"
          aria-labelledby="preview-tab"
          hidden={activeTab !== 'preview'}
        >
          {activeTab === 'preview' && <MarkdownPreview content={body} />}
        </div>
      </div>

      <div className="editor-actions">
        <span className="prototype-note">UI preview · Classification API not connected yet</span>

        <button
          className="analyze-button"
          type="button"
          onClick={onDemo}
          disabled={!title.trim()}
          title="Show illustrative prediction only"
        >
          ✧ <span>Show demo analysis</span>
        </button>
      </div>
    </section>
  )
}
