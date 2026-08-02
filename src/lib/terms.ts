import { z } from 'zod'

/**
 * The master term index (`src/data/terms.json`, emitted by the Python pipeline's
 * `build_master`): each term, its sources, and the paths to the per-source
 * co-occurrence / similarity / embedding JSON files. Passed from Astro to the
 * co-occurrence component (not fetched at runtime).
 */

export const TermSchema = z.object({
  label: z.string(),
  variants: z.array(z.string()).default([]),
})

/**
 * One source: a chapter, an article, the whole corpus, or a corpus stand-in.
 * `occurrences` and `data` are nullish because they don't apply in every context
 * — the per-corpus embedding source has no term `occurrences`, and a per-file
 * `NetworkData.source` has no `data` path to itself; only the master index sets both.
 */
export const SourceSchema = z.object({
  id: z.string(),
  url: z.string(),
  title: z.string(),
  description: z.string().nullable(),
  /** Term occurrences in this source (absent on the corpus embedding source). */
  occurrences: z.number().nullish(),
  /** Web path to this source's dataset (set only in the master index). */
  data: z.string().nullish(),
})

const CorporaSchema = z.literal(['mengzi', 'sep'])
export type Corpora = z.infer<typeof CorporaSchema>

const CorpusSideSchema = z.object({
  corpus: CorporaSchema,
  term: TermSchema,
  /** term occurrences across the whole corpora */
  occurrences: z.number(),
  /** Each source's `data` returns an embedding dataset (usually one, per corpus). */
  embeddings: SourceSchema.array(),
  /** Each source's `data` returns a similarity NetworkData JSON. */
  similarity: SourceSchema.array(),
  /** Each source's `data` returns a co-occurrence NetworkData JSON (one per source). */
  cooccurrence: SourceSchema.array(),
})

export const MasterTermSchema = z.object({
  hanzi: z.string(),
  renderings: z.string().array(),
  chinese: CorpusSideSchema,
  english: CorpusSideSchema.array(),
})

export const MasterSchema = z.object({
  terms: MasterTermSchema.array(),
})

export type MasterSource = z.infer<typeof SourceSchema>
export type CorpusSide = z.infer<typeof CorpusSideSchema>
export type MasterTerm = z.infer<typeof MasterTermSchema>
export type Master = z.infer<typeof MasterSchema>

export function getTermFrequencies(terms: MasterTerm[]): Map<string, number> {
  const map = new Map<string, number>()
  terms
    .flatMap((t) => [t.chinese, ...t.english])
    .forEach((c) => {
      const k = c.term.label
      const o = map.get(k) || 0
      map.set(k, o + c.occurrences)
    })
  return map
}
