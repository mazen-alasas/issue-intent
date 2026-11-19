import { BookOpen, Bug, CircleHelp, Cpu, Info, Lightbulb, LoaderCircle } from 'lucide-react'
import { INFERENCE_BACKEND, type IssueLabel, type PredictResponse } from './api'

type AnalysisState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; result: PredictResponse }
  | { status: 'error'; message: string }

type Props = {
  analysis: AnalysisState
}

const classes = [
  { key: 'bug', label: 'Bug', color: '#f85149' },
  { key: 'feature', label: 'Feature', color: '#a371f7' },
  { key: 'question', label: 'Question', color: '#f2cc60' },
  { key: 'documentation', label: 'Documentation', color: '#388bfd' },
] as const

const labelNames: Record<IssueLabel, string> = {
  bug: 'Bug',
  feature: 'Feature',
  question: 'Question',
  documentation: 'Documentation',
}

const icons = {
  bug: Bug,
  feature: Lightbulb,
  question: CircleHelp,
  documentation: BookOpen,
}

function PredictionResult({ result }: { result: PredictResponse }) {
  const Icon = icons[result.label]

  return (
    <div className="analysis-content">
      <div className="muted-label">Predicted category</div>

      <div className="prediction-summary">
        <div className="prediction-identity">
          <div className="bug-icon">
            <Icon size={26} />
          </div>

          <strong>{labelNames[result.label]}</strong>
        </div>

        <div className="confidence">
          <strong>{(result.confidence * 100).toFixed(1)}%</strong>
          <span>model confidence</span>
        </div>
      </div>

      <div className="analysis-divider" />

      <div className="probability-heading">Class probabilities</div>

      <div className="probabilities">
        {classes.map((item) => {
          const percent = result.probabilities[item.key] * 100

          return (
            <div className="probability-row" key={item.key}>
              <span className="probability-label">{item.label}</span>

              <div
                className="probability-track"
                role="progressbar"
                aria-label={`${item.label} probability`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={percent}
              >
                <div
                  className="probability-fill"
                  style={{
                    width: `${percent}%`,
                    backgroundColor: item.color,
                  }}
                />
              </div>

              <span className="probability-value">{percent.toFixed(1)}%</span>
            </div>
          )
        })}
      </div>

      <div className="analysis-notice">
        <Info size={18} />
        <span>Model predictions are estimates and should be reviewed before labeling.</span>
      </div>

      <div className="analysis-divider footer-divider" />

      <div className="analysis-footer">
        <span>
          <Cpu size={17} /> Powered by DistilBERT
        </span>
        <span className="version-tag">
          {INFERENCE_BACKEND === 'browser' ? 'In-browser inference' : 'Live inference'}
        </span>
      </div>
    </div>
  )
}

export function PredictionPanel({ analysis }: Props) {
  return (
    <aside className="analysis-card" aria-label="AI Issue Analysis">
      <div className="analysis-header">
        <h2>AI Issue Analysis</h2>
        <Info size={18} aria-hidden="true" />
      </div>

      {analysis.status === 'success' ? (
        <PredictionResult result={analysis.result} />
      ) : (
        <div className="analysis-content analysis-empty" aria-live="polite">
          {analysis.status === 'idle' && (
            <>
              <p>Ready to classify your issue.</p>
              <p>
                Enter a title or description, then select <strong>Analyze issue</strong>.
              </p>
            </>
          )}

          {analysis.status === 'loading' && (
            <div className="analysis-loading" role="status">
              <LoaderCircle size={20} className="loading-spinner" />
              <span>
                {INFERENCE_BACKEND === 'browser'
                  ? 'Analyzing with DistilBERT in your browser… First use downloads the model.'
                  : 'Analyzing issue with DistilBERT...'}
              </span>
            </div>
          )}

          {analysis.status === 'error' && (
            <div role="alert" className="analysis-error">
              <strong>Analysis failed</strong>
              <p>{analysis.message}</p>
              <p>You can retry using the Analyze issue button.</p>
            </div>
          )}
        </div>
      )}
    </aside>
  )
}
