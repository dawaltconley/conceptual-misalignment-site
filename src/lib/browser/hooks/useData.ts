import { useState, useEffect } from 'react'

interface DataStateLoading {
  status: 'loading'
  data: null
  errorMessage?: undefined
}
interface DataStateSuccess<T> {
  status: 'success'
  data: T
  errorMessage?: undefined
}
interface DataStateError {
  status: 'error'
  data: null
  errorMessage: string
}

type DataState<T> = DataStateLoading | DataStateError | DataStateSuccess<T>

export default function useData<T>(
  path: string,
  validator: (data: any) => T,
): DataState<T> {
  const [data, setData] = useState<T | null>(null)
  const [errorMessage, setErrorMessage] = useState<string>()

  useEffect(() => {
    fetch(path)
      .then(async (res) => {
        const data = await res.json()
        setData(validator(data))
      })
      .catch((e) => setErrorMessage(e?.message || 'Error fetching data'))
  }, [path])

  if (errorMessage) {
    return { status: 'error', data: null, errorMessage }
  }
  if (data) {
    return { status: 'success', data }
  }

  return { status: 'loading', data: null }
}
