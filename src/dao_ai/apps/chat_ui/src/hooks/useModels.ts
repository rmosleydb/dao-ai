import { useEffect, useState } from 'react'

export interface UCModel {
  name: string
  latest_versions: { version: string; aliases: string[] }[]
}

export function useModels() {
  const [models, setModels] = useState<UCModel[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/models')
      .then((r) => r.json())
      .then((data) => {
        if (data.error) throw new Error(data.error)
        setModels(data.models ?? [])
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return { models, loading, error }
}
