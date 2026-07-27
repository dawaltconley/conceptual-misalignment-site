/** Message contract between the main thread and the t-SNE web worker. */

export interface TsneStart {
  type: 'start'
  runId: number
  data: number[][]
  perplexity: number
  maxIter: number
  stepsPerPost: number
}

export interface TsneStop {
  type: 'stop'
}

/** Main thread → worker. */
export type TsneRequest = TsneStart | TsneStop

/** Worker → main thread: the current 2-D solution and progress. */
export interface TsneProgress {
  type: 'progress'
  runId: number
  coords: number[][]
  steps: number
  done: boolean
}
