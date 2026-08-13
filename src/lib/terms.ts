import { z } from 'zod'

/**
 * The master term index (`src/data/terms.json`, emitted by the Python pipeline's
 * `build_master`): each term, its sources, and the paths to the per-source
 * co-occurrence / similarity / embedding JSON files. Passed from Astro to the
 * co-occurrence component (not fetched at runtime).
 */

export const TermSchema = z.object({
  label: z.string(),
  /** The other words this term matched — `wisdom` also said as `wise`/`wisely`.
   * Scoped to the enclosing file: one source on a per-source network, the whole
   * corpus here in the master index. Chinese terms are their own lemma, so `[]`. */
  variants: z.array(z.string()).default([]),
})

/**
 * One source: a chapter, an article, the whole corpus, or a corpus stand-in.
 * `occurrences` is nullish because the per-corpus embedding source has no term
 * `occurrences`.
 */
export const SourceSchema = z.object({
  id: z.string(),
  url: z.string(),
  title: z.string(),
  description: z.string().nullable(),
  /** Term occurrences in this source (absent on the corpus embedding source). */
  occurrences: z.number().nullish(),
})
export type Source = z.infer<typeof SourceSchema>

/**
 * A per-file `NetworkData.source` has no `data` path to
 * itself; only the master index sets the data path.
 */
export const MasterSourceSchema = SourceSchema.extend({
  /** Web path to this source's dataset (set only in the master index). */
  data: z.string().nullish(),
})
export type MasterSource = z.infer<typeof MasterSourceSchema>

const CorporaSchema = z.literal(['mengzi', 'sep'])
export type Corpora = z.infer<typeof CorporaSchema>

const CorpusSideSchema = z.object({
  corpus: CorporaSchema,
  term: TermSchema,
  /** Term occurrences across the whole corpora — grand total, including
   * `chinesePhilosophyOccurrences`. */
  totalOccurrences: z.number(),
  /** Occurrences within Chinese-philosophy SEP articles (excluded from the
   * analyzed corpus); always 0 on the Mengzi side. */
  chinesePhilosophyOccurrences: z.number().default(0),
  /** Each source's `data` returns an embedding dataset (usually one, per corpus). */
  embeddings: MasterSourceSchema.array(),
  /** Each source's `data` returns a similarity NetworkData JSON. */
  similarity: MasterSourceSchema.array(),
  /** Each source's `data` returns a co-occurrence NetworkData JSON (one per source). */
  cooccurrence: MasterSourceSchema.array(),
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
      map.set(k, o + c.totalOccurrences)
    })
  return map
}
