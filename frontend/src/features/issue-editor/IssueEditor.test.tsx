import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { IssueEditor } from './IssueEditor'

function TestEditor({ initialTitle = '' }: { initialTitle?: string }) {
  const [title, setTitle] = useState(initialTitle)
  const [body, setBody] = useState('')
  const onDemo = vi.fn()

  return (
    <IssueEditor
      title={title}
      body={body}
      onTitleChange={setTitle}
      onBodyChange={setBody}
      onDemo={onDemo}
    />
  )
}

describe('IssueEditor', () => {
  it('renders title and description fields', () => {
    render(<TestEditor />)

    expect(screen.getByRole('textbox', { name: /title/i })).toBeInTheDocument()
    expect(
      screen.getByRole('textbox', { name: /issue description in markdown/i }),
    ).toBeInTheDocument()
  })

  it('disables demo analysis when title is empty', async () => {
    const user = userEvent.setup()
    render(<TestEditor />)

    const button = screen.getByRole('button', { name: /Analyze issue/i })

    expect(button).toBeDisabled()

    await user.type(screen.getByRole('textbox', { name: /title/i }), 'App crashes')

    expect(button).toBeEnabled()
  })

  it('preserves Markdown when switching between Write and Preview', async () => {
    const user = userEvent.setup()
    render(<TestEditor />)

    const editor = screen.getByRole('textbox', {
      name: /issue description in markdown/i,
    })

    await user.type(editor, '# Application crashes')

    await user.click(screen.getByRole('tab', { name: 'Preview' }))

    const preview = screen.getByRole('tabpanel', { name: 'Preview' })

    expect(
      within(preview).getByRole('heading', { name: 'Application crashes' }),
    ).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: 'Write' }))

    expect(editor).toHaveValue('# Application crashes')
  })
  it('renders GitHub Flavored Markdown task lists', async () => {
    const user = userEvent.setup()
    render(<TestEditor />)

    const editor = screen.getByRole('textbox', {
      name: /issue description in markdown/i,
    })

    await user.click(editor)
    await user.paste('- [x] Reproduced issue')

    expect(editor).toHaveValue('- [x] Reproduced issue')

    await user.click(screen.getByRole('tab', { name: 'Preview' }))

    const preview = screen.getByRole('tabpanel', { name: 'Preview' })
    const checkbox = within(preview).getByRole('checkbox')

    expect(checkbox).toBeChecked()
    expect(checkbox).toBeDisabled()
  })

  it('inserts bold Markdown around selected text', async () => {
    const user = userEvent.setup()
    render(<TestEditor />)

    const editor = screen.getByRole('textbox', {
      name: /issue description in markdown/i,
    }) as HTMLTextAreaElement

    await user.type(editor, 'Important bug')

    editor.focus()
    editor.setSelectionRange(0, 9)

    await user.click(screen.getByRole('button', { name: 'Bold' }))

    expect(editor).toHaveValue('**Important** bug')
  })

  it('formats multiple lines as a numbered list', async () => {
    const user = userEvent.setup()
    render(<TestEditor />)

    const editor = screen.getByRole('textbox', {
      name: /issue description in markdown/i,
    }) as HTMLTextAreaElement

    await user.type(editor, 'First step{enter}Second step')

    editor.focus()
    editor.setSelectionRange(0, editor.value.length)

    await user.click(screen.getByRole('button', { name: 'Numbered list' }))

    expect(editor).toHaveValue('1. First step\n2. Second step')
  })

  it('expands and collapses the editor', async () => {
    const user = userEvent.setup()
    render(<TestEditor />)

    const section = screen.getByRole('region', { name: 'Issue editor' })

    expect(section).not.toHaveClass('issue-card-expanded')

    await user.click(screen.getByRole('button', { name: 'Expand editor' }))

    expect(section).toHaveClass('issue-card-expanded')

    await user.click(screen.getByRole('button', { name: 'Collapse editor' }))

    expect(section).not.toHaveClass('issue-card-expanded')
  })

  it('shows an empty preview message', async () => {
    const user = userEvent.setup()
    render(<TestEditor />)

    await user.click(screen.getByRole('tab', { name: 'Preview' }))

    expect(screen.getByText('Nothing to preview')).toBeInTheDocument()
  })
})
