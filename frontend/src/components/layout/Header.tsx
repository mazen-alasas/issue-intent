import { Bell, ChevronDown, GitBranch, Plus, Search } from 'lucide-react'
import avatar from '../../assets/avatar.png'

export function Header() {
  return (
    <header className="site-header">
      <div className="site-brand">
        <GitBranch size={32} strokeWidth={1.8} aria-hidden="true" />
        <span className="repo-name">issue-intent</span>
        <span className="brand-divider">/</span>
        <span className="brand-section">Issues</span>
      </div>
      <div className="header-right">
        <div className="search-display" aria-label="Search unavailable in this prototype">
          <Search size={18} aria-hidden="true" />
          <span>
            Type <kbd>/</kbd> to search
          </span>
        </div>
        <span className="header-symbol" title="Notifications (visual preview)">
          <Bell size={20} />
        </span>
        <span className="header-symbol plus-symbol" title="Create menu (visual preview)">
          <Plus size={19} />
          <ChevronDown size={12} />
        </span>
        <img className="user-avatar" src={avatar} alt="Profile avatar" />
      </div>
    </header>
  )
}
