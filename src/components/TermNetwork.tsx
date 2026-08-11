import type { Master, CorpusSide } from '@lib/terms'
import type { Dictionary } from '@build/cedict'
import { useState } from 'react'
import MultiNetwork from './MultiNetwork'
import Select from './Select'
import { getHeading, type Heading } from '@lib/headings'

type NetworkKind = 'cooccurrence' | 'similarity'

interface TermNetworkProps {
  title: string

  /** The master term index (passed from Astro, not fetched). */
  data: Master
  /**
   * Which networks to show: per-source co-occurrence (a sidebar of sources) or
   * the single whole-corpus similarity network (no sidebar). Default co-occurrence.
   */
  kind?: NetworkKind
  /** Dictionary for the Chinese hanzi nodes (pinyin + definitions). */
  dictionary?: Dictionary

  headingLevel?: Heading
}

/** The networks to feed one MultiNetwork for a corpus side, per kind. */
function refsFor(side: CorpusSide, kind: NetworkKind) {
  return kind === 'similarity' ? side.similarity : side.cooccurrence
}

/**
 * Side-by-side networks for a Chinese term and one of its English renderings.
 * Two dropdowns pick the term (the English options repopulate when the Chinese
 * term changes); `kind` selects co-occurrence (per-source) or similarity (single).
 */
export default function TermNetwork({
  title,
  data,
  kind = 'cooccurrence',
  dictionary,
  headingLevel,
}: TermNetworkProps): JSX.Element {
  const H = headingLevel ? getHeading(headingLevel) : 'p'

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

  const chinesePercentage = (
    (english.chinesePhilosophyOccurrences / english.totalOccurrences) *
    100
  ).toFixed(2)

  return (
    <div className="rounded-md border border-gray-100">
      <div className="grid divide-x divide-gray-100 2xl:grid-cols-2">
        <div className="p-4">
          <H className="mb-4 text-xl font-bold">{title}</H>
          <Select
            label="Chinese term"
            value={term.hanzi}
            options={terms.map((t) => t.hanzi)}
            onChange={selectHanzi}
            triggerClassName="text-lg"
          />
        </div>
        <div className="flex justify-between gap-4 p-4 align-baseline">
          <Select
            label="English rendering"
            value={english.term.label}
            options={term.renderings}
            onChange={setRenderingLabel}
            className="mt-auto shrink-0"
            triggerClassName="text-lg"
          />
          <div className="mt-auto max-w-64 text-sm leading-5 text-gray-700">
            <span className="align-baseline text-lg font-bold leading-5 text-gray-900">
              {chinesePercentage}%
            </span>{' '}
            of sampled usage occurs in articles about Chinese philosophy
          </div>
        </div>
        <div className="p2">
          <MultiNetwork
            key={`${kind}:${term.hanzi}`}
            sources={refsFor(term.chinese, kind)}
            centralNodeId={term.hanzi}
            dictionary={dictionary}
            sourceAlign="left"
          />
        </div>
        <div className="p-2">
          <MultiNetwork
            key={`${kind}:${term.hanzi}:${english.term.label}`}
            sources={refsFor(english, kind)}
            centralNodeId={english.term.label}
            sourceAlign="right"
          />
        </div>
      </div>
    </div>
  )
}
