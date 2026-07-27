import { useCallback, useEffect, useRef, useState } from 'react'
import type { TsneProgress, TsneRequest } from '@lib/tsne.messages'

export interface TsneOptions {
  perplexity: number
  maxIter?: number
  stepsPerPost?: number
}

export interface TsneState {
  coords: number[][]
  steps: number
  done: boolean
}

export interface UseTsne extends TsneState {
  run: (data: number[][], opts: TsneOptions) => void
  stop: () => void
}

const EMPTY: TsneState = { coords: [], steps: 0, done: true }

/**
 * Runs t-SNE in a web worker so the iteration never blocks the main thread.
 * Progress is coalesced to one state update per animation frame, so the worker's
 * post rate can't flood React with re-renders. Each `run` supersedes the last;
 * stale progress (by run id) is ignored.
 */
export default function useTsne(): UseTsne {
  const workerRef = useRef<Worker | null>(null)
  const runIdRef = useRef(0)
  const latest = useRef<TsneProgress | null>(null)
  const rafRef = useRef(0)
  const [state, setState] = useState<TsneState>(EMPTY)

  useEffect(() => {
    const worker = new Worker(new URL('../../tsne.worker.ts', import.meta.url), {
      type: 'module',
    })
    workerRef.current = worker
    worker.onmessage = (e: MessageEvent<TsneProgress>) => {
      const msg = e.data
      if (msg.type !== 'progress' || msg.runId !== runIdRef.current) return
      latest.current = msg
      if (!rafRef.current) {
        rafRef.current = requestAnimationFrame(() => {
          rafRef.current = 0
          const m = latest.current
          if (m) setState({ coords: m.coords, steps: m.steps, done: m.done })
        })
      }
    }
    return () => {
      worker.terminate()
      workerRef.current = null
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  const run = useCallback((data: number[][], opts: TsneOptions) => {
    const worker = workerRef.current
    if (!worker || data.length < 2) return
    const runId = ++runIdRef.current
    setState({ coords: [], steps: 0, done: false })
    const req: TsneRequest = {
      type: 'start',
      runId,
      data,
      perplexity: opts.perplexity,
      maxIter: opts.maxIter ?? 500,
      stepsPerPost: opts.stepsPerPost ?? 2,
    }
    worker.postMessage(req)
  }, [])

  const stop = useCallback(() => {
    runIdRef.current++ // invalidate any in-flight progress
    const req: TsneRequest = { type: 'stop' }
    workerRef.current?.postMessage(req)
    setState(EMPTY)
  }, [])

  return { ...state, run, stop }
}
