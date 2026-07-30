import { z } from 'zod'

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
})

/** The corpus this dataset came from (the pipeline's reduced `lib.Source`). */
const SourceRefSchema = z.object({
  id: z.string().nullable(),
  url: z.string().nullable(),
  title: z.string().nullable(),
  description: z.string().nullable(),
})

export const EmbeddingDatasetSchema = z.object({
  source: SourceRefSchema.nullable(),
  dims: z.number(),
  nodes: z.array(EmbeddingNodeSchema),
})

export type EmbeddingNode = z.infer<typeof EmbeddingNodeSchema>
export type EmbeddingDataset = z.infer<typeof EmbeddingDatasetSchema>
