import { WeightedNodeLinkDataSchema } from './networkx'
import { z } from 'zod'

/**
 * A single per-(term, source) network file emitted by the pipeline's
 * `lib.NetworkData`: the term, the source it belongs to, and the graph itself
 * (null when the term never occurs in that source).
 */
const TermDataSchema = z.object({
  label: z.string(),
  renderings: z.string().array(),
  occurrences: z.number(),
  variants: z.array(z.string()).default([]),
})

const SourceRefSchema = z.object({
  id: z.string().nullable(),
  url: z.string().nullable(),
  title: z.string().nullable(),
  description: z.string().nullable(),
})

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

export const EmbeddingDatasetSchema = z.object({
  source: SourceRefSchema.nullable(),
  dims: z.number(),
  nodes: z.array(EmbeddingNodeSchema),
})

export type EmbeddingNode = z.infer<typeof EmbeddingNodeSchema>
export type EmbeddingDataset = z.infer<typeof EmbeddingDatasetSchema>

/**
 * Every data file ultimately returns this structure
 */

export const NetworkDataSchema = z.object({
  term: TermDataSchema,
  source: SourceRefSchema,
  cooccurrence: WeightedNodeLinkDataSchema.nullish(),
  similarity: WeightedNodeLinkDataSchema.nullish(),
  embeddings: EmbeddingDatasetSchema.nullish(),
})

export type NetworkData = z.infer<typeof NetworkDataSchema>
