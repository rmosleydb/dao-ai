import { useState } from 'react'
import { ChatInput } from './components/ChatInput'
import { ChatWindow } from './components/ChatWindow'
import { ModelSelector } from './components/ModelSelector'
import { useChat } from './hooks/useChat'
import { useModels } from './hooks/useModels'

export default function App() {
  const [selectedModel, setSelectedModel] = useState('')
  const { models, loading, error: modelsError } = useModels()
  const { messages, streaming, error: chatError, send, reset } = useChat(selectedModel)

  const handleModelChange = (model: string) => {
    setSelectedModel(model)
    reset()
  }

  return (
    <div
      style={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        background: 'var(--bg)',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 20px',
          borderBottom: '1px solid var(--border)',
          background: 'var(--surface)',
          gap: 16,
          flexWrap: 'wrap',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          {/* Databricks wordmark color */}
          <span style={{ fontWeight: 700, fontSize: 16, color: 'var(--accent)', letterSpacing: '-0.3px' }}>
            DAO AI
          </span>
          <span style={{ color: 'var(--border)' }}>|</span>
          <ModelSelector
            models={models}
            loading={loading}
            error={modelsError}
            value={selectedModel}
            onChange={handleModelChange}
          />
        </div>

        <button
          onClick={reset}
          disabled={messages.length === 0}
          style={{
            padding: '6px 14px',
            borderRadius: 8,
            background: 'transparent',
            border: '1px solid var(--border)',
            color: messages.length === 0 ? 'var(--text-muted)' : 'var(--text)',
            fontSize: 13,
          }}
        >
          New conversation
        </button>
      </div>

      {/* Chat area */}
      <ChatWindow messages={messages} streaming={streaming} />

      {/* Error banner */}
      {chatError && (
        <div
          style={{
            padding: '10px 20px',
            background: '#3b1212',
            color: '#ff8080',
            borderTop: '1px solid #5c1f1f',
            fontSize: 13,
          }}
        >
          {chatError}
        </div>
      )}

      {/* Input */}
      <ChatInput onSend={send} disabled={streaming || !selectedModel} />
    </div>
  )
}
