import { TSNE } from '@keckelt/tsne'
import type { TsneRequest, TsneProgress } from './tsne.messages'

// Minimal typed view of the worker global (avoids pulling in the whole webworker
// lib, which conflicts with the project's DOM lib).
const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<TsneRequest>) => void) | null
  postMessage: (message: TsneProgress) => void
}

// The run currently allowed to post. A new `start` (or a `stop`) supersedes any
// in-flight run: its next scheduled batch sees the id no longer matches and exits.
let currentRunId = -1

ctx.onmessage = (e) => {
  const msg = e.data
  if (msg.type === 'stop') {
    currentRunId = -1
    return
  }

  const { runId, data, perplexity, maxIter, stepsPerPost } = msg
  currentRunId = runId

  const tsne = new TSNE({ epsilon: 10, perplexity, dim: 2 })
  tsne.initDataRaw(data)
  let steps = 0

  const run = (): void => {
    if (runId !== currentRunId) return // superseded or stopped
    const target = Math.min(steps + stepsPerPost, maxIter)
    while (steps < target) {
      tsne.step()
      steps++
    }
    const solution = tsne.getSolution() as number[][]
    const done = steps >= maxIter
    ctx.postMessage({
      type: 'progress',
      runId,
      coords: solution.map((c) => [c[0], c[1]]),
      steps,
      done,
    })
    // Yield between batches so incoming start/stop messages are processed.
    if (!done) setTimeout(run, 0)
  }

  run()
}
