import { CircleDot, Code2, GitPullRequest, PlayCircle, PanelsTopLeft } from 'lucide-react'

const items = [
  { label: 'Code', Icon: Code2 },
  { label: 'Issues', Icon: CircleDot },
  { label: 'Pull requests', Icon: GitPullRequest },
  { label: 'Actions', Icon: PlayCircle },
  { label: 'Projects', Icon: PanelsTopLeft },
]

export function RepositoryNav() {
  return (
    <nav className="repo-nav" aria-label="Repository navigation preview">
      {items.map(({ label, Icon }) => (
        <span
          key={label}
          className={`repo-nav-item ${label === 'Issues' ? 'active' : ''}`}
          aria-current={label === 'Issues' ? 'page' : undefined}
        >
          <Icon size={18} aria-hidden="true" />
          <span>{label}</span>
        </span>
      ))}
    </nav>
  )
}
