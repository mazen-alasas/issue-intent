import { useEffect, useRef, useState } from 'react'
import { Header } from './components/layout/Header'
import { RepositoryNav } from './components/layout/RepositoryNav'
import { predictIssue, type PredictResponse } from './features/classification/api'
import { PredictionPanel } from './features/classification/PredictionPanel'
import { IssueEditor } from './features/issue-editor/IssueEditor'
import './App.css'

type AnalysisState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; result: PredictResponse }
  | { status: 'error'; message: string }

const INITIAL_BODY =
  'After updating the application to version **v2.3.0**, it crashes immediately on startup.\n\n' +
  'A **RuntimeError** is raised during the initialization phase related to the configuration loader.\n\n' +
  '### Steps to reproduce\n' +
  '1. Update to version `v2.3.0`.\n' +
  '2. Run the application using `python app.py`.\n' +
  '3. Observe that it crashes during startup.\n\n' +
  '### Error log\n' +
  '```text\n' +
  'RuntimeError: Failed to load configuration file: config.yaml\n' +
  '  File "app.py", line 42, in <module>\n' +
  '    app = create_app()\n' +
  '  File "app/core/init.py", line 18, in create_app\n' +
  '    raise RuntimeError("Failed to load configuration file: config.yaml")\n' +
  '```'

export default function App() {
  const [title, setTitle] = useState('Application crashes on startup')
  const [body, setBody] = useState(INITIAL_BODY)
  const [analysis, setAnalysis] = useState<AnalysisState>({ status: 'idle' })

  const controllerRef = useRef<AbortController | null>(null)
  const requestIdRef = useRef(0)

  useEffect(() => {
    return () => {
      controllerRef.current?.abort()
    }
  }, [])

  function clearAnalysis() {
    requestIdRef.current += 1
    controllerRef.current?.abort()
    controllerRef.current = null
    setAnalysis({ status: 'idle' })
  }

  function updateTitle(value: string) {
    setTitle(value)
    clearAnalysis()
  }

  function updateBody(value: string) {
    setBody(value)
    clearAnalysis()
  }

  async function analyzeIssue() {
    if (!title.trim() && !body.trim()) return
    if (title.length > 500 || body.length > 20_000) return

    controllerRef.current?.abort()

    const controller = new AbortController()
    controllerRef.current = controller

    const requestId = ++requestIdRef.current

    setAnalysis({ status: 'loading' })

    try {
      const result = await predictIssue({ title, body }, controller.signal)

      if (controller.signal.aborted || requestId !== requestIdRef.current) {
        return
      }

      setAnalysis({ status: 'success', result })
    } catch (error) {
      if (controller.signal.aborted || requestId !== requestIdRef.current) {
        return
      }

      setAnalysis({
        status: 'error',
        message: error instanceof Error ? error.message : 'An unexpected error occurred.',
      })
    } finally {
      if (requestId === requestIdRef.current) {
        controllerRef.current = null
      }
    }
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
            onDemo={analyzeIssue}
            isLoading={analysis.status === 'loading'}
          />

          <PredictionPanel analysis={analysis} />
        </div>
      </main>
    </div>
  )
}
