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
  Strikethrough,
} from 'lucide-react'
import { useRef } from 'react'

type Props = {
  title: string
  body: string
  onTitleChange: (value: string) => void
  onBodyChange: (value: string) => void
  onDemo: () => void
}

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
  { name: 'Expand', Icon: Maximize2 },
]

export function IssueEditor({ title, body, onTitleChange, onBodyChange, onDemo }: Props) {
  const bodyRef = useRef<HTMLTextAreaElement>(null)
  function insertMarkdown(name: string) {
    const el = bodyRef.current
    if (!el) return
    const start = el.selectionStart
    const end = el.selectionEnd
    const selected = body.slice(start, end)
    const wrappers: Record<string, [string, string]> = {
      Heading: ['### ', ''],
      Bold: ['**', '**'],
      Italic: ['*', '*'],
      Strikethrough: ['~~', '~~'],
      Code: ['`', '`'],
      Link: ['[', '](https://example.com)'],
      'Bulleted list': ['- ', ''],
      'Numbered list': ['1. ', ''],
      'Task list': ['- [ ] ', ''],
      Mention: ['@', ''],
    }
    if (name === 'Expand') {
      el.focus()
      return
    }
    const [before, after] = wrappers[name]
    const replacement = before + (selected || 'text') + after
    onBodyChange(body.slice(0, start) + replacement + body.slice(end))
    requestAnimationFrame(() => {
      el.focus()
      el.setSelectionRange(start + before.length, start + replacement.length - after.length)
    })
  }

  return (
    <section className="issue-card" aria-labelledby="issue-title">
      <label className="field-label" htmlFor="issue-title">
        Title <span className="required">*</span>
      </label>
      <input
        id="issue-title"
        className="title-input"
        value={title}
        onChange={(e) => onTitleChange(e.target.value)}
        placeholder="Title"
        maxLength={300}
      />
      <label className="field-label description-label" htmlFor="issue-description">
        Description
      </label>
      <div className="markdown-box">
        <div className="editor-toolbar">
          <div className="editor-tabs">
            <span className="editor-tab selected">Write</span>
            <span
              className="editor-tab inactive"
              title="Markdown preview is coming in the next stage"
            >
              Preview
            </span>
          </div>
          <div className="format-tools" aria-label="Markdown formatting tools">
            {toolbar.map(({ name, Icon }, i) => (
              <button
                key={name}
                type="button"
                className={`tool-button ${i === 6 || i === 9 ? 'tool-divider' : ''}`}
                title={name}
                aria-label={name}
                onClick={() => insertMarkdown(name)}
              >
                <Icon size={17} />
              </button>
            ))}
          </div>
        </div>
        <textarea
          ref={bodyRef}
          id="issue-description"
          className="description-input"
          value={body}
          onChange={(e) => onBodyChange(e.target.value)}
          spellCheck={false}
          aria-label="Issue description in Markdown"
        />
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
