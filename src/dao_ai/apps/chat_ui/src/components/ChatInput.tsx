import { KeyboardEvent, useRef, useState } from 'react'

interface Props {
  onSend: (content: string) => void
  disabled: boolean
}

export function ChatInput({ onSend, disabled }: Props) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const onInput = () => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }

  return (
    <div
      style={{
        display: 'flex',
        gap: 8,
        padding: '12px 16px',
        borderTop: '1px solid var(--border)',
        background: 'var(--surface)',
        alignItems: 'flex-end',
      }}
    >
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        onInput={onInput}
        placeholder={disabled ? 'Waiting for response…' : 'Message (Enter to send, Shift+Enter for newline)'}
        disabled={disabled}
        rows={1}
        style={{
          flex: 1,
          resize: 'none',
          padding: '10px 14px',
          borderRadius: 10,
          overflow: 'hidden',
          background: disabled ? 'var(--surface)' : 'var(--surface-2)',
          color: disabled ? 'var(--text-muted)' : 'var(--text)',
        }}
      />
      <button
        onClick={submit}
        disabled={disabled || !value.trim()}
        style={{
          padding: '10px 18px',
          borderRadius: 10,
          background: disabled || !value.trim() ? 'var(--surface-2)' : 'var(--accent)',
          color: disabled || !value.trim() ? 'var(--text-muted)' : '#fff',
          fontWeight: 600,
          transition: 'background 0.15s',
          flexShrink: 0,
        }}
      >
        Send
      </button>
    </div>
  )
}
