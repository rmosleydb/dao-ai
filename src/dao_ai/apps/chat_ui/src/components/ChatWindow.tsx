import { useEffect, useRef } from 'react'
import { Message } from '../hooks/useChat'

interface Props {
  messages: Message[]
  streaming: boolean
}

export function ChatWindow({ messages, streaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div
        style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'var(--text-muted)',
          fontSize: 15,
        }}
      >
        Select a model and start chatting
      </div>
    )
  }

  return (
    <div
      style={{
        flex: 1,
        overflowY: 'auto',
        padding: '24px 16px',
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
      }}
    >
      {messages.map((msg) => (
        <div
          key={msg.id}
          style={{
            display: 'flex',
            justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
          }}
        >
          <div
            style={{
              maxWidth: '75%',
              padding: '12px 16px',
              borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
              background: msg.role === 'user' ? 'var(--user-bubble)' : 'var(--assistant-bubble)',
              border: msg.role === 'assistant' ? '1px solid var(--border)' : 'none',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              lineHeight: 1.65,
            }}
          >
            {msg.content || (msg.pending && streaming ? <Cursor /> : null)}
          </div>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

function Cursor() {
  return (
    <span
      style={{
        display: 'inline-block',
        width: 2,
        height: '1em',
        background: 'var(--text-muted)',
        verticalAlign: 'text-bottom',
        animation: 'blink 1s step-end infinite',
      }}
    />
  )
}
