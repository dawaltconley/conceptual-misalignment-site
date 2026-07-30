import { z } from 'zod'

/**
 * The master term index (`src/data/terms.json`, emitted by the Python pipeline's
 * `build_master`): each term, its sources, and the paths to the per-source
 * co-occurrence / similarity / embedding JSON files. Passed from Astro to the
 * co-occurrence component (not fetched at runtime).
 */

/** One source of a term's corpus: a chapter, an article, or the whole corpus. */
const MasterSourceSchema = z.object({
  id: z.string(),
  title: z.string(),
  /** Web path to this source's co-occurrence NetworkData JSON. */
  cooccurrence: z.string(),
})

const CorpusSideSchema = z.object({
  corpus: z.string(),
  /** Web path to the corpus embedding dataset. */
  embeddings: z.string(),
  /** Web path to the term's similarity NetworkData JSON (or null). */
  similarity: z.string().nullable(),
  sources: MasterSourceSchema.array(),
})

/** An English rendering of a term (its own SEP sub-corpus). */
const EnglishRenderingSchema = CorpusSideSchema.extend({
  label: z.string(),
})

export const MasterTermSchema = z.object({
  hanzi: z.string(),
  renderings: z.string().array(),
  chinese: CorpusSideSchema,
  english: EnglishRenderingSchema.array(),
})

export const MasterSchema = z.object({
  terms: MasterTermSchema.array(),
})

export type MasterSource = z.infer<typeof MasterSourceSchema>
export type CorpusSide = z.infer<typeof CorpusSideSchema>
export type EnglishRendering = z.infer<typeof EnglishRenderingSchema>
export type MasterTerm = z.infer<typeof MasterTermSchema>
export type Master = z.infer<typeof MasterSchema>
