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
  /** The other words this node stands for: derivational variants folded in by
   * the merge (`inspiration` carrying `inspire`), or, for a target, the lemmas
   * its rendering matched across the corpus (`wisdom` carrying `wise`,
   * `wisely`). Empty for an unmerged node, or a target used only under its own
   * label. */
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
