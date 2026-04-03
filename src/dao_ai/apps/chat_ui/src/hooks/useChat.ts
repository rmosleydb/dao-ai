import { useCallback, useRef, useState } from 'react'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  pending?: boolean
}

let msgCounter = 0
const uid = () => String(++msgCounter)

export function useChat(daoModel: string) {
  const [messages, setMessages] = useState<Message[]>([])
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const threadId = useRef<string>(crypto.randomUUID())

  const send = useCallback(
    async (content: string) => {
      if (!daoModel || !content.trim() || streaming) return
      setError(null)

      const userMsg: Message = { id: uid(), role: 'user', content }
      const assistantMsg: Message = { id: uid(), role: 'assistant', content: '', pending: true }

      setMessages((prev) => [...prev, userMsg, assistantMsg])
      setStreaming(true)

      const history = [...messages, userMsg].map((m) => ({
        role: m.role,
        content: m.content,
      }))

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            dao_model: daoModel,
            messages: history,
            thread_id: threadId.current,
          }),
        })

        if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`)

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let fullResponse = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() ?? ''

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue
            const raw = line.slice(6).trim()
            if (!raw) continue
            try {
              const event = JSON.parse(raw)
              if (event.type === 'delta') {
                fullResponse += event.content
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsg.id
                      ? { ...m, content: fullResponse }
                      : m
                  )
                )
              } else if (event.type === 'error') {
                setError(event.error)
              }
            } catch {}
          }
        }

        // Mark as no longer pending
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMsg.id ? { ...m, pending: false } : m
          )
        )
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e)
        setError(msg)
        setMessages((prev) => prev.filter((m) => m.id !== assistantMsg.id))
      } finally {
        setStreaming(false)
      }
    },
    [daoModel, messages, streaming]
  )

  const reset = useCallback(() => {
    setMessages([])
    setError(null)
    threadId.current = crypto.randomUUID()
  }, [])

  return { messages, streaming, error, send, reset }
}
