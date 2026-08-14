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

/**
 * A 2-D projection the pipeline already computed (`models.Layout`) — today the
 * t-SNE perplexity sweep, which is the scatter's default view. `coords` is keyed
 * by node id rather than parallel to `nodes`, so a node the layout doesn't cover
 * is simply not plotted instead of silently taking another word's position.
 *
 * PCA needs no layout: the exported vectors are variance-ordered, so its
 * coordinates are `vec[0]`/`vec[1]`.
 */
export const LayoutSchema = z.object({
  id: z.string(),
  method: z.string(),
  /** What the layout picker shows ("t-SNE · perplexity 30"). */
  label: z.string(),
  /**
   * The settings that produced it. `perplexity` and `dims` are first-class
   * because the scatter shows them — the slider keys on the perplexity, and the
   * dimensionality is how you tell a layout of the untruncated vectors from one
   * the browser could have computed itself. The rest (epsilon, iterations, seed,
   * source) ride along for provenance.
   */
  params: z
    .looseObject({
      perplexity: z.number().optional(),
      dims: z.number().optional(),
    })
    .default({ perplexity: undefined, dims: undefined }),
  coords: z.record(z.string(), z.tuple([z.number(), z.number()])),
})

export const EmbeddingDatasetSchema = z.object({
  /** The corpus this dataset came from (the pipeline's reduced `lib.Source`). */
  source: SourceSchema.nullable(),
  dims: z.number(),
  /** Total documents (sources) in the corpus, for relative doc-freq. */
  documents: z.number().default(0),
  nodes: z.array(EmbeddingNodeSchema),
  /** Precomputed projections. Empty is valid — the view falls back to PCA. */
  layouts: z.array(LayoutSchema).default([]),
})

export type EmbeddingNode = z.infer<typeof EmbeddingNodeSchema>
export type EmbeddingDataset = z.infer<typeof EmbeddingDatasetSchema>
export type EmbeddingLayout = z.infer<typeof LayoutSchema>
