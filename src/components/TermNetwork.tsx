import type { Master, CorpusSide } from '@lib/terms'
import type { Dictionary } from '@build/cedict'
import { useState } from 'react'
import MultiNetwork, { type NetworkRef } from './MultiNetwork'
import Select from './Select'

type NetworkKind = 'cooccurrence' | 'similarity'

interface TermNetworkProps {
  /** The master term index (passed from Astro, not fetched). */
  data: Master
  /**
   * Which networks to show: per-source co-occurrence (a sidebar of sources) or
   * the single whole-corpus similarity network (no sidebar). Default co-occurrence.
   */
  kind?: NetworkKind
  /** Dictionary for the Chinese hanzi nodes (pinyin + definitions). */
  dictionary?: Dictionary
}

/** The networks to feed one MultiNetwork for a corpus side, per kind. */
function refsFor(side: CorpusSide, kind: NetworkKind): NetworkRef[] {
  const sources = kind === 'similarity' ? side.similarity : side.cooccurrence
  // `data` is nullish on the shared Source type; keep only sources with a path.
  return sources.flatMap((s) =>
    s.data ? [{ id: s.id, title: s.title, path: s.data }] : [],
  )
}

/**
 * Side-by-side networks for a Chinese term and one of its English renderings.
 * Two dropdowns pick the term (the English options repopulate when the Chinese
 * term changes); `kind` selects co-occurrence (per-source) or similarity (single).
 */
export default function TermNetwork({
  data,
  kind = 'cooccurrence',
  dictionary,
}: TermNetworkProps): JSX.Element {
  const terms = data.terms
  const [hanzi, setHanzi] = useState(terms[0]?.hanzi ?? '')
  const term = terms.find((t) => t.hanzi === hanzi) ?? terms[0]

  const [renderingLabel, setRenderingLabel] = useState(
    term?.renderings[0] ?? '',
  )
  const english =
    term?.english.find((e) => e.term.label === renderingLabel) ??
    term?.english[0]

  function selectHanzi(next: string): void {
    setHanzi(next)
    const t = terms.find((t) => t.hanzi === next)
    if (t) setRenderingLabel(t.renderings[0]) // repopulate the English options
  }

  if (!term || !english) return <div>No term data.</div>

  return (
    <div className="items-start justify-center xl:flex xl:gap-6">
      <div className="xl:w-1/2">
        <Select
          label="Chinese term"
          value={term.hanzi}
          options={terms.map((t) => t.hanzi)}
          onChange={selectHanzi}
          className="mb-3"
          triggerClassName="text-lg"
        />
        <MultiNetwork
          key={`${kind}:${term.hanzi}`}
          sources={refsFor(term.chinese, kind)}
          centralNodeId={term.hanzi}
          dictionary={dictionary}
          sourceAlign="left"
        />
      </div>
      <div className="mt-8 xl:mt-0 xl:w-1/2">
        <Select
          label="English rendering"
          value={english.term.label}
          options={term.renderings}
          onChange={setRenderingLabel}
          className="mb-3"
          triggerClassName="text-lg"
        />
        <MultiNetwork
          key={`${kind}:${term.hanzi}:${english.term.label}`}
          sources={refsFor(english, kind)}
          centralNodeId={english.term.label}
          sourceAlign="right"
        />
      </div>
    </div>
  )
}
