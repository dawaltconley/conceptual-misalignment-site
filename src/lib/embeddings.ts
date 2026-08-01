import { z } from 'zod'
import { SourceSchema } from './terms'

/**
 * One node in a per-corpus embedding dataset: a PCA-reduced vector plus metadata.
 * The client derives both projections from `vec` — PCA is columns 0/1 (the
 * reduced columns are variance-ordered), t-SNE is run over the full vector.
 */
export const EmbeddingNodeSchema = z.object({
  id: z.string(),
  target: z.boolean(),
  community: z.number(),
  vec: z.array(z.number()),
  /** How many documents (sources) in the corpus this word appears in. */
  doc_freq: z.number().default(0),
  /** Weighted degree in the similarity graph (sum of incident edge weights). */
  strength: z.number().default(0),
  /** PageRank on the weighted similarity graph (global importance). */
  pagerank: z.number().default(0),
})

export const EmbeddingDatasetSchema = z.object({
  /** The corpus this dataset came from (the pipeline's reduced `lib.Source`). */
  source: SourceSchema.nullable(),
  dims: z.number(),
  /** Total documents (sources) in the corpus, for relative doc-freq. */
  documents: z.number().default(0),
  nodes: z.array(EmbeddingNodeSchema),
})

export type EmbeddingNode = z.infer<typeof EmbeddingNodeSchema>
export type EmbeddingDataset = z.infer<typeof EmbeddingDatasetSchema>
