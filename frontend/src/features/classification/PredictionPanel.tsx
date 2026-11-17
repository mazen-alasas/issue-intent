import { Bug, Cpu, Info } from 'lucide-react'

type Props = { showExample: boolean }
const classes = [
  { label: 'Bug', percent: 88.7, color: '#f85149' },
  { label: 'Feature', percent: 6.8, color: '#a371f7' },
  { label: 'Question', percent: 3.2, color: '#f2cc60' },
  { label: 'Documentation', percent: 1.3, color: '#388bfd' },
]

export function PredictionPanel({ showExample }: Props) {
  return (
    <aside className="analysis-card" aria-label="AI Issue Analysis">
      <div className="analysis-header">
        <h2>AI Issue Analysis</h2>
        <Info size={18} aria-label="Demo preview" />
      </div>
      {showExample ? (
        <div className="analysis-content">
          <div className="demo-tag">Illustrative result — not model output</div>
          <div className="muted-label">Predicted category</div>
          <div className="prediction-summary">
            <div className="prediction-identity">
              <div className="bug-icon">
                <Bug size={26} />
              </div>
              <strong>Bug</strong>
            </div>
            <div className="confidence">
              <strong>88.7%</strong>
              <span>confidence</span>
            </div>
          </div>
          <div className="analysis-divider" />
          <div className="probability-heading">Class probabilities</div>
          <div className="probabilities">
            {classes.map((item) => (
              <div className="probability-row" key={item.label}>
                <span className="probability-label">{item.label}</span>
                <div className="probability-track">
                  <div
                    className="probability-fill"
                    style={{ width: `${item.percent}%`, backgroundColor: item.color }}
                  />
                </div>
                <span className="probability-value">{item.percent.toFixed(1)}%</span>
              </div>
            ))}
          </div>
          <div className="analysis-notice">
            <Info size={18} />
            <span>AI-generated predictions should be reviewed before labeling.</span>
          </div>
          <div className="analysis-divider footer-divider" />
          <div className="analysis-footer">
            <span>
              <Cpu size={17} /> Powered by DistilBERT
            </span>
            <span className="version-tag">UI preview</span>
          </div>
        </div>
      ) : (
        <div className="analysis-content analysis-empty">
          <p>Issue edited. Previous demo result cleared.</p>
          <p>
            Use <strong>Show demo analysis</strong> to preview the layout. Live classification will
            be enabled when the API is connected.
          </p>
        </div>
      )}
    </aside>
  )
}
