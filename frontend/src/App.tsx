import { useState } from 'react'
import { Header } from './components/layout/Header'
import { RepositoryNav } from './components/layout/RepositoryNav'
import { IssueEditor } from './features/issue-editor/IssueEditor'
import { PredictionPanel } from './features/classification/PredictionPanel'
import './App.css'

export default function App() {
  const [title, setTitle] = useState('Application crashes on startup')
  const [body, setBody] = useState(
    'After updating the application to version **v2.3.0**, it crashes immediately on startup.\n\nA **RuntimeError** is raised during the initialization phase related to the configuration loader.\n\n### Steps to reproduce\n1. Update to version `v2.3.0`.\n2. Run the application using `python app.py`.\n3. Observe that it crashes during startup.\n\n### Error log\n```text\nRuntimeError: Failed to load configuration file: config.yaml\n  File "app.py", line 42, in <module>\n    app = create_app()\n  File "app/core/init.py", line 18, in create_app\n    raise RuntimeError("Failed to load configuration file: config.yaml")\n```',
  )
  const [showExample, setShowExample] = useState(true)

  function updateTitle(value: string) {
    setTitle(value)
    setShowExample(false)
  }
  function updateBody(value: string) {
    setBody(value)
    setShowExample(false)
  }
  function showSample() {
    setShowExample(true)
  }

  return (
    <div className="app-shell">
      <Header />
      <RepositoryNav />
      <main className="page-content">
        <h1 className="page-title">New issue</h1>
        <div className="workspace">
          <IssueEditor
            title={title}
            body={body}
            onTitleChange={updateTitle}
            onBodyChange={updateBody}
            onDemo={showSample}
          />
          <PredictionPanel showExample={showExample} />
        </div>
      </main>
    </div>
  )
}
