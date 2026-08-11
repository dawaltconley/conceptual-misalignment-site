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
  /** Raw occurrence count over the whole corpus — how often the word is said, as
   * opposed to how widely it is spread (`doc_freq`). `doc_freq` saturates at
   * `documents` so most of the vocabulary ties; occurrences are heavy-tailed, so
   * log or rank this before using it to sort. */
  freq: z.number().default(0),
  /** L2 norm in the centered/debiased space (≈ distance from the corpus centroid);
   * the radial signal that `vec` (L2-normalized, direction-only) drops. */
  norm: z.number().default(0),
  /** Weighted degree in the similarity graph (sum of incident edge weights). */
  strength: z.number().default(0),
  /** PageRank on the weighted similarity graph (global importance). */
  pagerank: z.number().default(0),
  /** Eigenvector centrality on the weighted similarity graph. */
  eigenvector: z.number().default(0),
  /** Derivational variants folded into this node (`inspiration` carrying
   * `inspire`). Empty when the pipeline's `merge_variants` is off, and always
   * empty for targets, which never merge. */
  variants: z.array(z.string()).default([]),
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
