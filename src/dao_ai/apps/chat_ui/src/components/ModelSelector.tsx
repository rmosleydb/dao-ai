import { UCModel } from '../hooks/useModels'

interface Props {
  models: UCModel[]
  loading: boolean
  error: string | null
  value: string
  onChange: (value: string) => void
}

export function ModelSelector({ models, loading, error, value, onChange }: Props) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <label style={{ color: 'var(--text-muted)', whiteSpace: 'nowrap', fontSize: 13 }}>
        Agent
      </label>
      {error ? (
        <span style={{ color: 'var(--accent)', fontSize: 12 }}>Failed to load models</span>
      ) : (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={loading}
          style={{
            padding: '6px 10px',
            minWidth: 280,
            background: 'var(--surface-2)',
            color: loading ? 'var(--text-muted)' : 'var(--text)',
          }}
        >
          <option value="">{loading ? 'Loading models…' : 'Select a model…'}</option>
          {models.map((m) => (
            <option key={m.name} value={m.name}>
              {m.name}
            </option>
          ))}
        </select>
      )}
    </div>
  )
}
